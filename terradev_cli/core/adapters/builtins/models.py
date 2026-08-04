#!/usr/bin/env python3
"""Built-in model registry adapter stubs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import AdapterHealth, AdapterSpec, ModelRegistryAdapter
from ..capabilities import Capability, Capabilities
from ..registry import REGISTRY


class LocalModelRegistry(ModelRegistryAdapter):
    """Local filesystem model registry."""

    KIND = "model"
    NAME = "local"
    VERSION = "0.1.0"
    DESCRIPTION = "Local model weight registry"
    CAPABILITIES = Capabilities([
        Capability.LOCAL_FS,
        Capability.VERSIONED,
        Capability.CACHING,
    ])
    CONFIG_SCHEMA = {"required": ["base_path"]}

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="local model stub")

    async def resolve(self, model_uri: str) -> Dict[str, Any]:
        base = self.config.get("base_path", "")
        return {
            "uri": model_uri,
            "local_path": f"{base}/{model_uri}",
            "status": "resolved",
        }

    async def list_models(self, **filters) -> List[Dict[str, Any]]:
        return []


class HuggingFaceModelRegistry(ModelRegistryAdapter):
    """Hugging Face Hub model registry stub."""

    KIND = "model"
    NAME = "huggingface"
    VERSION = "0.1.0"
    DESCRIPTION = "Hugging Face Hub registry"
    CAPABILITIES = Capabilities([
        Capability.REMOTE_URI,
        Capability.VERSIONED,
    ])
    CONFIG_SCHEMA = {"required": ["repo_id"]}

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="hf model stub")

    async def resolve(self, model_uri: str) -> Dict[str, Any]:
        return {"uri": model_uri, "repo_id": self.config.get("repo_id"), "status": "resolved"}

    async def list_models(self, **filters) -> List[Dict[str, Any]]:
        return []


REGISTRY.register(LocalModelRegistry.KIND, LocalModelRegistry.NAME, LocalModelRegistry)
REGISTRY.register(HuggingFaceModelRegistry.KIND, HuggingFaceModelRegistry.NAME, HuggingFaceModelRegistry)
