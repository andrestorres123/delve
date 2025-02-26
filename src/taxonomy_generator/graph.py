"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from langgraph.graph import StateGraph, START, END

from taxonomy_generator.configuration import Configuration
from taxonomy_generator.routing.should_review import should_review
from taxonomy_generator.state import InputState, OutputState, State
from taxonomy_generator.nodes.runs_retriever import retrieve_runs
from taxonomy_generator.nodes.taxonomy_generator import generate_taxonomy
from taxonomy_generator.nodes.minibatches_generator import generate_minibatches
from taxonomy_generator.nodes.taxonomy_updater import update_taxonomy
from taxonomy_generator.nodes.taxonomy_reviewer import review_taxonomy
from taxonomy_generator.nodes.summary_generator import generate_summaries
from taxonomy_generator.nodes.doc_labeler import label_documents


builder = StateGraph(State, input=InputState, output=OutputState, config_schema=Configuration)

# Add nodes
builder.add_node("get_runs", retrieve_runs)
builder.add_node("summarize", generate_summaries)
builder.add_node("get_minibatches", generate_minibatches)
builder.add_node("generate_taxonomy", generate_taxonomy)
builder.add_node("update_taxonomy", update_taxonomy)
builder.add_node("review_taxonomy", review_taxonomy)

builder.add_node("label_documents", label_documents)

# Add edges
builder.add_edge(START, "get_runs")
builder.add_edge("get_runs", "summarize")
builder.add_edge("summarize", "get_minibatches")
builder.add_edge("get_minibatches", "generate_taxonomy")
builder.add_edge("generate_taxonomy", "update_taxonomy")
builder.add_edge("review_taxonomy", "label_documents")
builder.add_edge("label_documents", END)

builder.add_conditional_edges(
    "update_taxonomy",
    should_review,
    {
        "update_taxonomy": "update_taxonomy",
        "review_taxonomy": "review_taxonomy"
    }
)

graph = builder.compile()
graph.name = "Taxonomy Generation"
