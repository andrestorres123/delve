"""Node for reviewing and finalizing taxonomies."""

import random
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from taxonomy_generator.state import State
from taxonomy_generator.utils import load_chat_model, parse_taxa, invoke_taxonomy_chain
from taxonomy_generator.configuration import Configuration


def _setup_review_chain(configuration: Configuration):
    """Set up the chain for taxonomy review.
    
    Args:
        model_name: Name of the model to use
        max_tokens: Maximum tokens for model response
        
    Returns:
        Chain for reviewing and parsing taxonomies
    """
    # Initialize the prompt
    review_prompt = hub.pull("wfh/tnt-llm-taxonomy-review")

    # Create the chain
    model = load_chat_model(configuration.fast_llm)

    return (
        review_prompt
        | model
        | StrOutputParser()
        | parse_taxa
    ).with_config(run_name="ReviewTaxonomy")


async def review_taxonomy(
    state: State,
    config: RunnableConfig
) -> dict:
    """Review and finalize taxonomy using a random sample of documents.
    
    Args:
        state: Current application state
        config: Configuration for the run
        model_name: Name of the model to use
        max_tokens: Maximum tokens for model response
        
    Returns:
        dict: Updated state fields with reviewed taxonomy
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Set up the chain
    review_chain = _setup_review_chain(configuration)

    # Create random sample of documents
    batch_size = configuration.batch_size
    indices = list(range(len(state.documents)))
    random.shuffle(indices)
    sample_indices = indices[:batch_size]

    # Review taxonomy using sampled documents
    return await invoke_taxonomy_chain(
        review_chain,
        state,
        config,
        sample_indices
    )
