"""Smoke test: every terradev_cli module can be imported.

Importing a module executes its top-level statements, which raises coverage for
modules that have zero or low coverage and do not yet have dedicated tests.
Modules that require optional extras (torch, transformers, cloud SDKs, etc.)
are skipped instead of failing the build.
"""

import importlib
import pkgutil
from types import ModuleType

import pytest

import terradev_cli

# Optional extras that may not be installed in the default CI environment.
# Modules that fail to import because one of these is missing are skipped.
OPTIONAL_DEPS = {
    "torch",
    "transformers",
    "sentence_transformers",
    "gradio",
    "streamlit",
    "boto3",
    "google",
    "azure",
    "redis",
    "lmcache",
    "sglang",
    "wandb",
    "mlflow",
    "langchain",
    "langgraph",
    "sklearn",
    "nvidia",
    "pynvml",
    "triton",
    "ray",
    "kserve",
    "litellm",
    "opentelemetry",
    "phoenix",
    "qdrant_client",
    "vllm",
    "ollama",
    "huggingface_hub",  # installed, but some ml modules may fail earlier
    "datasets",
    "accelerate",
    "peft",
}


def _is_optional_missing(exc: BaseException, module_name: str) -> bool:
    """Return True if the import failed because of an optional dependency."""
    if not isinstance(exc, ModuleNotFoundError):
        return False
    missing = getattr(exc, "name", None) or module_name
    if missing in OPTIONAL_DEPS:
        return True
    return any(part in OPTIONAL_DEPS for part in missing.split("."))


def _discover_modules(package: ModuleType) -> list[str]:
    """Return all importable module names under the package."""
    return [m.name for m in pkgutil.walk_packages(package.__path__, package.__name__ + ".")]


MODULES = _discover_modules(terradev_cli)


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str):
    """Import each terradev_cli module, skipping those blocked by optional deps."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if _is_optional_missing(exc, module_name):
            pytest.skip(f"optional dependency missing for {module_name}: {exc}")
        raise
