"""Node for getting user approval on generated taxonomy clusters."""

from typing import List, Dict
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from taxonomy_generator.state import State


def _format_clusters_for_display(clusters: List[Dict[str, str]]) -> str:
    """Format clusters in a readable way for user review.
    
    Args:
        clusters: List of cluster dictionaries
        
    Returns:
        str: Formatted string representation of clusters
    """
    output = "Generated Taxonomy Clusters:\n\n"
    for cluster in clusters:
        output += f"## {cluster['name'].upper()}\n"
        output += f"  ID: {cluster['id']}\n"
        output += f"  Description: {cluster['description']}\n\n"
    return output


async def request_taxonomy_approval(state: State, config: RunnableConfig) -> dict:
    """Request user approval for the generated taxonomy clusters.
    
    Args:
        state: Current application state
        config: Configuration for the run
        
    Returns:
        dict: Updated state with approval status
    """
    if not state.clusters:
        return {"status": ["No clusters to review."]}

    # Get the latest clusters
    latest_clusters = state.clusters[-1]
    
    # Format clusters for display
    formatted_clusters = _format_clusters_for_display(latest_clusters)
    
    # Create an AI message for the user
    message = AIMessage(content=(
        "Please review the generated taxonomy clusters and approve or reject them.\n\n"
        f"{formatted_clusters}\n"
    ))

    # Return the message and clusters for review
    return {
        "messages": [message],
        "status": ["Awaiting approval..."]
    } 