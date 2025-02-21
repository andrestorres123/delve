"""Route model output to the next node in the graph."""

from typing import Literal
from langchain_core.runnables import RunnableConfig

from taxonomy_generator.state import State, UserFeedback
from taxonomy_generator.configuration import Configuration
from taxonomy_generator.utils import load_chat_model
from taxonomy_generator.prompts import FEEDBACK_PROMPT


async def route_feedback(state: State, config: RunnableConfig) -> Literal["continue", "modify"]:
    return state.user_feedback.decision