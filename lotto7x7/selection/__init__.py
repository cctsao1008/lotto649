"""Public API for statistics-first Lotto649 selection."""

from .pipeline import SelectionConfig, load_history, select

__all__ = ["SelectionConfig", "load_history", "select"]
