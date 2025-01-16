"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.nodes.call_model import call_model
from react_agent.routing import route_model_output
from react_agent.state import InputState, State
from react_agent.tools import TOOLS

builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "call_model")

builder.add_conditional_edges(
    "call_model",
    route_model_output,
)

builder.add_edge("tools", "call_model")

graph = builder.compile(
    interrupt_before=[], 
    interrupt_after=[],  
)
graph.name = "ReAct Agent"  
