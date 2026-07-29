"""Core module import smoke test.

Every module in terradev_cli/core/ should be importable in a clean CI
environment. Modules that require heavy extras are skipped, not failed.
"""

import importlib
import pkgutil
from types import ModuleType

import pytest

import terradev_cli.core as core_pkg


def _discover_modules(package: ModuleType):
    return [m.name for m in pkgutil.walk_packages(package.__path__, package.__name__ + ".")]


MODULES = _discover_modules(core_pkg)


@pytest.mark.parametrize("module_name", MODULES)
def test_core_module_imports(module_name):
    """Import each core module, skipping those blocked by optional dependencies."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional dependency missing for {module_name}: {exc}")
