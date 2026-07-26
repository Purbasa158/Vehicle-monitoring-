"""Reusable components for the ANPR application."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import ANPRPipeline, PipelineConfig

__all__ = ["ANPRPipeline", "PipelineConfig"]


def __getattr__(name: str):
    """Keep lightweight helpers importable without loading ML dependencies."""
    if name in __all__:
        from .pipeline import ANPRPipeline, PipelineConfig

        return {"ANPRPipeline": ANPRPipeline, "PipelineConfig": PipelineConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
