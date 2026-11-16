"""Compatibility namespace for the archived Lotto7x7 research package."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[1] / "research/lotto7x7"))

from ._compat import install as _install_module_aliases

__version__ = "0.1.0"

_install_module_aliases()
