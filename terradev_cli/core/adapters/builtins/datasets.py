#!/usr/bin/env python3
"""Built-in dataset registry adapter stubs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import AdapterHealth, AdapterSpec, DatasetRegistryAdapter
from ..capabilities import Capability, Capabilities
from ..registry import REGISTRY


class LocalDatasetRegistry(DatasetRegistryAdapter):
    """Local dataset registry."""

    KIND = "dataset"
    NAME = "local"
    VERSION = "0.1.0"
    DESCRIPTION = "Local dataset registry"
    CAPABILITIES = Capabilities([
        Capability.LOCAL_FS,
        Capability.CACHING,
    ])
    CONFIG_SCHEMA = {"required": ["base_path"]}

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="local dataset stub")

    async def resolve(self, dataset_uri: str) -> Dict[str, Any]:
        base = self.config.get("base_path", "")
        return {
            "uri": dataset_uri,
            "local_path": f"{base}/{dataset_uri}",
            "status": "resolved",
        }

    async def list_datasets(self, **filters) -> List[Dict[str, Any]]:
        return []


class HuggingFaceDatasetRegistry(DatasetRegistryAdapter):
    """Hugging Face datasets stub."""

    KIND = "dataset"
    NAME = "huggingface"
    VERSION = "0.1.0"
    DESCRIPTION = "Hugging Face dataset registry"
    CAPABILITIES = Capabilities([
        Capability.REMOTE_URI,
        Capability.CACHING,
    ])
    CONFIG_SCHEMA = {"required": ["name"]}

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="hf dataset stub")

    async def resolve(self, dataset_uri: str) -> Dict[str, Any]:
        return {"uri": dataset_uri, "name": self.config.get("name"), "status": "resolved"}

    async def list_datasets(self, **filters) -> List[Dict[str, Any]]:
        return []


REGISTRY.register(LocalDatasetRegistry.KIND, LocalDatasetRegistry.NAME, LocalDatasetRegistry)
REGISTRY.register(HuggingFaceDatasetRegistry.KIND, HuggingFaceDatasetRegistry.NAME, HuggingFaceDatasetRegistry)
