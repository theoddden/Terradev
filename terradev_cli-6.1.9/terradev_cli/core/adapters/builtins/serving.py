#!/usr/bin/env python3
"""Built-in serving engine adapter stubs."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from ..base import AdapterHealth, AdapterSpec, ServingEngineAdapter
from ..capabilities import Capability, Capabilities
from ..registry import REGISTRY


class VllmServingAdapter(ServingEngineAdapter):
    """vLLM serving engine placeholder."""

    KIND = "serving"
    NAME = "vllm"
    VERSION = "0.1.0"
    DESCRIPTION = "vLLM inference backend"
    CAPABILITIES = Capabilities([
        Capability.INFERENCE,
        Capability.STREAMING,
        Capability.BATCH,
    ])
    CONFIG_SCHEMA = {
        "required": ["model"],
        "properties": {
            "model": {"type": "string"},
            "max_model_len": {"type": "integer"},
        },
    }

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="vllm stub")

    async def load(self, model_uri: str, **options) -> Dict[str, Any]:
        return {"model": model_uri, "status": "loaded"}

    async def predict(self, inputs: Any, **options) -> Dict[str, Any]:
        return {"output": "ok", "model": self.config.get("model")}

    async def stream(self, inputs: Any, **options) -> AsyncIterator[Dict[str, Any]]:
        yield {"token": "ok"}

    async def unload(self, model_uri: str) -> bool:
        return True


class OllamaServingAdapter(ServingEngineAdapter):
    """Ollama serving engine placeholder."""

    KIND = "serving"
    NAME = "ollama"
    VERSION = "0.1.0"
    DESCRIPTION = "Ollama local inference backend"
    CAPABILITIES = Capabilities([
        Capability.INFERENCE,
        Capability.EMBEDDINGS,
    ])
    CONFIG_SCHEMA = {"required": ["model"]}

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="ollama stub")

    async def load(self, model_uri: str, **options) -> Dict[str, Any]:
        return {"model": model_uri, "status": "loaded"}

    async def predict(self, inputs: Any, **options) -> Dict[str, Any]:
        return {"output": "ok", "model": self.config.get("model")}

    async def unload(self, model_uri: str) -> bool:
        return True


REGISTRY.register(VllmServingAdapter.KIND, VllmServingAdapter.NAME, VllmServingAdapter)
REGISTRY.register(OllamaServingAdapter.KIND, OllamaServingAdapter.NAME, OllamaServingAdapter)
