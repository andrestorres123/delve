"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    fast_llm: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-haiku-20240307",
        metadata={
            "description": "A faster, lighter model for tasks like summarization. "
            "Should be in the form: provider/model-name."
        },
    )

    max_runs: int = field(
        default=500,
        metadata={
            "description": "Maximum number of runs to retrieve from LangSmith."
        },
    )

    sample_size: int = field(
        default=50,
        metadata={
            "description": "Number of runs to sample for processing."
        },
    )

    batch_size: int = field(
        default=200,
        metadata={
            "description": "Size of minibatches for document processing."
        },
    )

    suggestion_length: int = field(
        default=30,
        metadata={"description": "Maximum length for taxonomy suggestions"}
    )
    cluster_name_length: int = field(
        default=10,
        metadata={"description": "Maximum length for cluster names"}
    )
    cluster_description_length: int = field(
        default=30,
        metadata={"description": "Maximum length for cluster descriptions"}
    )
    explanation_length: int = field(
        default=20,
        metadata={"description": "Maximum length for explanations"}
    )
    max_num_clusters: int = field(
        default=25,
        metadata={"description": "Maximum number of clusters allowed"}
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
