"""Node for retrieving runs from LangSmith."""

from datetime import datetime, timedelta
from langsmith import Client, traceable
from langchain_core.runnables import RunnableConfig

from taxonomy_generator.state import State
from taxonomy_generator.configuration import Configuration
from taxonomy_generator.utils import process_runs

@traceable
async def retrieve_runs(state: State, config: RunnableConfig) -> dict:
    """Retrieve and process runs from LangSmith.
    
    Args:
        state: Current application state
        config: Configuration for the run
        
    Returns:
        dict: Updated state fields with retrieved documents
        
    Raises:
        ValueError: If project_name is not set
    """

    if not state.project_name:
        raise ValueError("project_name not set in state")
    
    configuration = Configuration.from_runnable_config(config)

    client = Client(api_key=state.org_id)

    delta_days = datetime.now() - timedelta(days=state.days)

    runs = list(
        client.list_runs(
            project_name=state.project_name,
            filter="eq(is_root, true)",
            start_time=delta_days,
            select=["inputs", "outputs"],
            limit=configuration.max_runs,
        )
    )

    if len(runs) == configuration.max_runs:
        status_message = f"Fetched runs were capped at {configuration.max_runs} due to the set limit."
    else:
        status_message = f"Fetched {len(runs)} runs successfully..."

    return {
        "all_documents": process_runs(left=[], right=runs),
        "documents": process_runs(left=[], right=runs, sample=configuration.sample_size),
        "status": [status_message],
    }
