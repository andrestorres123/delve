import re
import random
from typing import List, Optional, Dict, Union
from langchain_core.runnables import Runnable, RunnableConfig

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langsmith.schemas import Run

from taxonomy_generator.state import Doc, State
from taxonomy_generator.configuration import Configuration

def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


def to_xml(
    data: Union[Dict, List],
    tag_name: str,
    *,
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    nested: Optional[List[str]] = None,
    body_key: Optional[str] = None,
    list_item_tag: str = "item",
    max_body_length: Optional[int] = None,
) -> str:
    """Convert data structure to XML format.
    
    Args:
        data: The data to convert
        tag_name: The name of the root tag
        exclude: Keys to exclude from the output
        include: Keys to include in the output (if None, include all)
        nested: Keys that should be processed as nested structures
        body_key: Key whose value should be used as the tag body
        list_item_tag: Tag name to use for list items
        max_body_length: Maximum length for body text before truncating
    
    Returns:
        str: The XML representation of the data
    """
    skip = exclude or []
    nested = nested or []

    def process_dict(d: Dict) -> tuple[str, str]:
        attr_str = ""
        body = ""

        for key, value in d.items():
            if key == body_key:
                body += str(value)
                continue
            if value is None or key in skip:
                continue
            if include and key not in include:
                continue
            if key in nested:
                body += process_value(value, key)
            elif isinstance(value, (dict, list)):
                body += f"<{key}>{process_value(value, key)}</{key}>"
            else:
                attr_str += f' {key}="{value}"'

        return attr_str, body

    def process_value(value: Union[Dict, List, str, int, float], key: str) -> str:
        if isinstance(value, dict):
            attr, body = process_dict(value)

            if max_body_length and len(body) > max_body_length:
                body = body[:max_body_length] + "..."
            return f"<{key}{attr}>{body}</{key}>"
        elif isinstance(value, list):
            res = "".join(
                f"<{list_item_tag}>{process_value(item, list_item_tag)}</{list_item_tag}>"
                for item in value
            )
            if max_body_length and len(res) > max_body_length:
                res = res[:max_body_length] + "..."
            return res
        else:
            val = str(value)
            if max_body_length and len(val) > max_body_length:
                val = val[:max_body_length] + "..."
            return val

    if isinstance(data, dict):
        attr_str, body = process_dict(data)
        return f"<{tag_name}{attr_str}>{body}</{tag_name}>"
    elif isinstance(data, (list, tuple)):
        body = "".join(
            f"<{list_item_tag}>{process_value(item, list_item_tag)}</{list_item_tag}>"
            for item in data
        )
        return f"<{tag_name}>{body}</{tag_name}>"


def run_to_doc(
    run: Run,
    max_length: int = 500,
) -> Doc:
    """Convert a LangSmith run to a document.
    
    Args:
        run: The LangSmith run to convert
        max_length: Maximum length for content fields
    
    Returns:
        Doc: A document containing the run's content
    """
    inputs_str = to_xml(
        run.inputs,
        "inputs",
        include=["messages", "content", "type", "chat_history"],
        exclude=["__end__", "id"],
        max_body_length=max_length,
        body_key="content",
    )
    outputs_str = ""
    if run.outputs:
        outputs_str = "\n" + to_xml(
            run.outputs,
            "outputs",
            include=["answer"],
            exclude=["__end__", "documents"],
            max_body_length=max_length,
            body_key="answer",
        )
    return Doc(
        id=str(run.id),
        content=f"{inputs_str}{outputs_str}",
    )


def process_runs(left: List[Doc], right: List[Union[Doc, Run]], sample: Optional[int] = None) -> List[Doc]:
    """Process a list of runs, optionally sampling them.
    
    Args:
        left: Existing list of documents
        right: New runs or documents to process
        sample: Number of items to sample from right (if None, use all)
    
    Returns:
        List[Doc]: Combined list of documents
    """
    converted = [r if isinstance(r, dict) else run_to_doc(r) for r in right if right]

    if sample is not None and sample < len(converted):
        converted = random.sample(converted, sample)
    else:
        print(f"Processing all {len(converted)} documents.")

    return left + converted


def parse_taxa(output_text: str) -> Dict[str, List[Dict[str, str]]]:
    """Extract the taxonomy from the generated output."""
    
    cluster_matches = re.findall(
        r"\s*<id>(.*?)</id>\s*<name>(.*?)</name>\s*<description>(.*?)</description>\s*",
        output_text,
        re.DOTALL,
    )
    
    clusters = [
        {"id": id.strip(), "name": name.strip(), "description": description.strip()}
        for id, name, description in cluster_matches
    ]
    
    return {"clusters": clusters}


def format_docs(docs: List[Doc]) -> str:
    """Format documents as XML for taxonomy generation.
    
    Args:
        docs: List of documents to format
        
    Returns:
        str: XML formatted document summaries
    """
    xml_table = "<conversations>\n"
    for doc in docs:
        doc_id = doc["id"] if isinstance(doc, dict) else doc.id
        doc_summary = doc.get("summary", "") if isinstance(doc, dict) else (doc.summary or "")
        xml_table += f'<conv_summ id={doc_id}>{doc_summary}</conv_summ>\n'
    xml_table += "</conversations>"
    return xml_table


def format_taxonomy(clusters: List[Dict[str, str]]) -> str:
    """Format taxonomy clusters as XML.
    
    Args:
        clusters: List of cluster dictionaries
        
    Returns:
        str: XML formatted taxonomy
    """

    print("Clusters: ", clusters)

    xml = "<cluster_table>\n"
    for label in clusters:
        xml += "  <cluster>\n"
        xml += f'    <id>{label["id"]}</id>\n'
        xml += f'    <name>{label["name"]}</name>\n'
        xml += f'    <description>{label["description"]}</description>\n'
        xml += "  </cluster>\n"
    xml += "</cluster_table>"
    return xml


async def invoke_taxonomy_chain(
    chain: Runnable,
    state: State,
    config: RunnableConfig,
    mb_indices: List[int],
) -> Dict[str, List[List[Dict[str, str]]]]:
    """Invoke the taxonomy generation chain."""
    try:
        configuration = Configuration.from_runnable_config(config)
        minibatch = [state.documents[idx] for idx in mb_indices]
        data_table_xml = format_docs(minibatch)
        
        previous_taxonomy = state.clusters[-1] if state.clusters else []
        cluster_table_xml = format_taxonomy(previous_taxonomy)

        print("Previous Feedback: ", state.user_feedback)

        # Format feedback if it exists
        feedback = "No previous feedback provided."
        if state.user_feedback:
            feedback = f"Previous user feedback: {state.user_feedback.feedback}"
            if state.user_feedback.explanation:
                feedback += f"\nReason for modification: {state.user_feedback.explanation}"

        updated_taxonomy = await chain.ainvoke(
            {
                "data_xml": data_table_xml,
                "use_case": state.use_case,
                "cluster_table_xml": cluster_table_xml,
                "feedback": feedback,
                "suggestion_length": configuration.suggestion_length,
                "cluster_name_length": configuration.cluster_name_length,
                "cluster_description_length": configuration.cluster_description_length,
                "explanation_length": configuration.explanation_length,
                "max_num_clusters": configuration.max_num_clusters,
            }
        )
        return {
            "clusters": [updated_taxonomy["clusters"]],
            "status": ["Taxonomy generated.."],
        }
    except Exception as e:
        print("Taxonomy generation error: ", e)
        raise
