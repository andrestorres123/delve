"""Node for labeling documents using the generated taxonomy."""

import re
from typing import Dict, Any, List
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

from taxonomy_generator.state import State, Doc
from taxonomy_generator.utils import load_chat_model
from taxonomy_generator.configuration import Configuration
from taxonomy_generator.prompts import LABELER_PROMPT

def _parse_labels(output_text: str) -> Dict[str, str]:
    """Parse the generated labels from the predictions."""
    category_matches = re.findall(
        r"\s*<category>(.*?)</category>.*",
        output_text,
        re.DOTALL,
    )
    categories = [{"category": category.strip()} for category in category_matches]
    
    if len(categories) > 1:
        print(f"Warning: Multiple selected categories: {categories}")
    
    if categories:
        label = categories[0]
        stripped = re.sub(r"^\d+\.\s*", "", label["category"]).strip()
        return {"category": stripped}
    
    return {"category": "Other"}


def _format_taxonomy(clusters: List[Dict[str, str]]) -> str:
    """Format taxonomy clusters as XML."""
    
    xml = "<cluster_table>\n"
    
    if clusters and isinstance(clusters[0], list):
        clusters = clusters[0]
    
    if isinstance(clusters, dict):
        clusters = [clusters]
    
    for cluster in clusters:
        xml += "  <cluster>\n"
        if isinstance(cluster, dict):
            xml += f'    <id>{cluster["id"]}</id>\n'
            xml += f'    <name>{cluster["name"]}</name>\n'
            xml += f'    <description>{cluster["description"]}</description>\n'
        else:
            xml += f'    <id>{getattr(cluster, "id", "")}</id>\n'
            xml += f'    <name>{getattr(cluster, "name", "")}</name>\n'
            xml += f'    <description>{getattr(cluster, "description", "")}</description>\n'
        xml += "  </cluster>\n"
    xml += "</cluster_table>"
    return xml


def _format_results(docs: List[Doc]) -> str:
    """Format labeled documents in a readable way.
    
    Args:
        docs: List of labeled documents
        
    Returns:
        str: Formatted string showing document previews and their labels
    """
    result = " Document Classification Results:\n\n"
    for doc in docs:
        # Get first 200 chars of content, clean it up
        preview = doc.content[:200].replace('\n', ' ').strip()
        if len(doc.content) > 200:
            preview += "..."
            
        # Add document preview and its category
        result += f"🏷️  Category: {doc.category}\n"
        result += f"📄 Document: {preview}\n"
        result += "─" * 80 + "\n\n"
    
    return result


def _setup_classification_chain(configuration: Configuration):
    """Set up the chain for document labeling."""
    model = load_chat_model(configuration.fast_llm)

    return (
        LABELER_PROMPT
        | model
        | StrOutputParser()
        | _parse_labels
    ).with_config(run_name="LabelDocs")


async def label_documents(
    state: State,
    config: RunnableConfig,
    model_name: str = "claude-3-haiku-20240307",
    max_tokens: int = 2000,
) -> dict:
    """Label documents using the generated taxonomy."""
    
    configuration = Configuration.from_runnable_config(config)
    # Set up the chain
    labeling_chain = _setup_classification_chain(configuration)
    
    # Get configuration
    batch_size = configuration.batch_size
    
    # Get latest complete set of clusters
    latest_clusters = None
    for clusters in reversed(state.clusters):
        if isinstance(clusters, list) and clusters:
            latest_clusters = clusters
            break
    
    if not latest_clusters and state.clusters:
        # Fallback to last state if no complete set found
        latest_clusters = [state.clusters[-1]] if isinstance(state.clusters[-1], dict) else state.clusters[-1]
    
    if not latest_clusters:
        raise ValueError("No valid clusters found in state")
        
    
    # Process documents in batches
    labeled_docs = []
    for i in range(0, len(state.documents), batch_size):
        batch = state.documents[i : i + batch_size]
        batch_results = [
            await labeling_chain.ainvoke(
                {
                    "content": doc["content"] if isinstance(doc, dict) else doc.content,
                    "taxonomy": _format_taxonomy(latest_clusters),
                }
            )
            for doc in batch
        ]
        labeled_docs.extend(batch_results)

    # Update documents with labels
    updated_docs = [
        Doc(
            id=doc["id"] if isinstance(doc, dict) else doc.id,
            content=doc["content"] if isinstance(doc, dict) else doc.content,
            summary=doc.get("summary", "") if isinstance(doc, dict) else (doc.summary or ""),
            explanation=doc.get("explanation", "") if isinstance(doc, dict) else (doc.explanation or ""),
            category=category["category"]
        )
        for doc, category in zip(state.documents, labeled_docs)
    ]

    # Format results for display
    results_display = _format_results(updated_docs)
    message = AIMessage(content=f"✅ Documents have been labeled!\n\n{results_display}")

    return {
        "documents": updated_docs,
        "messages": [message],
        "status": ["Documents labeled successfully"],
    } 