"""Route model output to the next node in the graph."""

from typing import Literal

from taxonomy_generator.state import State

def should_review(state: State) -> Literal["update_taxonomy", "review_taxonomy"]:
    """Determine whether to continue updating or move to review."""
    num_minibatches = len(state.minibatches)
    num_revisions = len(state.clusters)
    if num_revisions < num_minibatches:
        return "update_taxonomy"
    return "review_taxonomy"