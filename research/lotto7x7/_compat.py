"""Preserve former top-level imports and module entry points after relocation."""
from importlib import abc, import_module, util
import sys

from ._module_aliases import MODULE_GROUPS

_ALIASES = {
    f"lotto7x7.{name}": f"lotto7x7.{group}.{name}"
    for group, names in MODULE_GROUPS.items()
    for name in names
}
_ALIASES.update({
    "lotto7x7.selection": "lotto649",
    **{
        f"lotto7x7.selection.{name}": f"lotto649.{name}"
        for name in ("pipeline", "statistics", "hexagram", "cli")
    },
})



class _AliasLoader(abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        module = import_module(self.target)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Import machinery sets __spec__ from the alias on the shared object.
        # Keep reload, inspection and pickling anchored to the canonical file.
        module.__spec__ = self.canonical_spec

    def get_code(self, fullname):
        # runpy uses get_code for `python -m old_name`, not create_module.
        return compile(
            f"from runpy import run_module\n"
            f"run_module({self.target!r}, run_name='__main__', alter_sys=True)\n",
            f"<compatibility entry point {fullname}>", "exec",
        )


class _AliasFinder(abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        canonical = _ALIASES.get(fullname)
        if canonical is not None:
            return util.spec_from_loader(fullname, _AliasLoader(canonical))
        return None


def install():
    """Register only the historical aliases, after normal file-based finders."""
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.append(_AliasFinder())
