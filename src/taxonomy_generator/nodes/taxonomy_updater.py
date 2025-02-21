"""Node for updating taxonomies based on new document batches."""

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from taxonomy_generator.state import State
from taxonomy_generator.utils import load_chat_model, parse_taxa, invoke_taxonomy_chain
from taxonomy_generator.configuration import Configuration


def _setup_update_chain(configuration: Configuration):
    """Set up the chain for taxonomy updates.
    
    Args:
        model_name: Name of the model to use
        max_tokens: Maximum tokens for model response
        
    Returns:
        Chain for updating and parsing taxonomies
    """
    # Initialize the prompt
    update_prompt = hub.pull("wfh/tnt-llm-taxonomy-update")

    # Create the chain
    model = load_chat_model(configuration.fast_llm)

    return (
        update_prompt
        | model
        | StrOutputParser()
        | parse_taxa
    ).with_config(run_name="UpdateTaxonomy")


async def update_taxonomy(
    state: State,
    config: RunnableConfig
) -> dict:
    """Update taxonomy using the next batch of documents.
    
    Args:
        state: Current application state
        config: Configuration for the run
        model_name: Name of the model to use
        max_tokens: Maximum tokens for model response
        
    Returns:
        dict: Updated state fields with revised taxonomy
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Set up the chain
    update_chain = _setup_update_chain(configuration)

    # Determine which minibatch to use
    which_mb = len(state.clusters) % len(state.minibatches)

    # Update taxonomy using the next batch
    return await invoke_taxonomy_chain(
        update_chain,
        state,
        config,
        state.minibatches[which_mb]
    )
