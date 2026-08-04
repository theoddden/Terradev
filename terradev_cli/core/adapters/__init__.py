#!/usr/bin/env python3
"""Universal adapter layer for plug-and-play execution primitives."""

from .base import (
    Adapter,
    AdapterHealth,
    AdapterSpec,
    ServingEngineAdapter,
    ComputeModuleAdapter,
    ModelRegistryAdapter,
    DatasetRegistryAdapter,
    DatabaseBackendAdapter,
    VectorStoreBackendAdapter,
)
from .exceptions import AdapterError, AdapterNotFoundError, AdapterConfigError
from .registry import AdapterRegistry, REGISTRY
from .capabilities import Capability, Capabilities

# Import built-in adapters so they self-register on the global registry.
from . import builtins  # noqa: F401

# Ensure built-ins are registered even when individual modules are not imported.
REGISTRY.load_builtins()

__all__ = [
    "Adapter",
    "AdapterHealth",
    "AdapterSpec",
    "ServingEngineAdapter",
    "ComputeModuleAdapter",
    "ModelRegistryAdapter",
    "DatasetRegistryAdapter",
    "DatabaseBackendAdapter",
    "VectorStoreBackendAdapter",
    "AdapterError",
    "AdapterNotFoundError",
    "AdapterConfigError",
    "AdapterRegistry",
    "REGISTRY",
    "Capability",
    "Capabilities",
]
