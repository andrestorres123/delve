"""Node for generating minibatches from documents."""

import random
from typing import List, Dict
from langchain_core.runnables import RunnableConfig

from taxonomy_generator.state import State
from taxonomy_generator.configuration import Configuration


def _create_batches(indices: List[int], batch_size: int) -> List[List[int]]:
    """Create batches of document indices.
    
    Args:
        indices: List of document indices to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches, where each batch is a list of document indices
    """
    if len(indices) < batch_size:
        return [indices]

    num_full_batches = len(indices) // batch_size
    batches = [
        indices[i * batch_size : (i + 1) * batch_size]
        for i in range(num_full_batches)
    ]

    leftovers = len(indices) % batch_size
    if leftovers:
        last_batch = indices[num_full_batches * batch_size :]
        elements_to_add = batch_size - leftovers
        last_batch += random.sample(indices, elements_to_add)
        batches.append(last_batch)

    return batches


async def generate_minibatches(state: State, config: RunnableConfig) -> dict:
    """Generate minibatches from documents for processing.
    
    Args:
        state: Current application state
        config: Configuration for the run
        
    Returns:
        dict: Updated state fields with minibatches
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Create and shuffle document indices
    indices = list(range(len(state.documents)))
    random.shuffle(indices)

    # Generate batches
    batches = _create_batches(indices, configuration.batch_size)

    return {
        "minibatches": batches,
        "status": ["Minibatches generated successfully.."],
    }
