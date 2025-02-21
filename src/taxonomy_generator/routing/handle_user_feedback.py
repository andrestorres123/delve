"""Route model output to the next node in the graph."""

from typing import Literal
from langchain_core.runnables import RunnableConfig

from taxonomy_generator.state import State, UserFeedback
from taxonomy_generator.configuration import Configuration
from taxonomy_generator.utils import load_chat_model
from taxonomy_generator.prompts import FEEDBACK_PROMPT


async def handle_user_feedback(state: State, config: RunnableConfig) -> Literal["continue", "modify"]:
    """Handle user feedback for taxonomy approval."""
    configuration = Configuration.from_runnable_config(config)

    model = load_chat_model(configuration.model)

    last_user_message = next(
        (msg for msg in reversed(state.messages) if msg.type == "human"),
        None,
    )
    if not last_user_message:
        raise ValueError("No user message found in state")

    chain = FEEDBACK_PROMPT | model.with_structured_output(UserFeedback)

    result = await chain.ainvoke({"input": last_user_message.content}, config)

    print("Feedback result: ", result)

    return {
        "user_feedback": result,
        "status": ["User feedback received..."],
    }