"""Public API for statistics-first Lotto649 selection."""

__all__ = ["SelectionConfig", "load_history", "select"]


def __getattr__(name):
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
