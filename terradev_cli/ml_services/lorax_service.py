#!/usr/bin/env python3
"""
LoRAX Service - Client for Predibase LoRAX multi-LoRA inference server

LoRAX (LoRA eXchange) serves thousands of fine-tuned models on a single GPU
with dynamic adapter loading, heterogeneous continuous batching, and
production-ready deployment.

API Reference: https://loraexchange.ai/
"""

import logging
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class LoRAXQuantization(str, Enum):
    """Supported quantization methods in LoRAX"""
    NONE = "none"
    BITANDBYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"


@dataclass
class LoRAXConfig:
    """Configuration for LoRAX server connection"""
    host: str = "localhost"
    port: int = 8080
    base_model: Optional[str] = None  # e.g., "mistralai/Mistral-7B-Instruct-v0.1"
    api_key: Optional[str] = None
    timeout: int = 30
    quantization: LoRAXQuantization = LoRAXQuantization.NONE

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class LoRAXAdapter:
    """LoRA adapter metadata"""
    adapter_id: str  # HuggingFace repo ID or local path
    adapter_name: Optional[str] = None
    base_model: Optional[str] = None
    rank: Optional[int] = None
    loaded: bool = False


@dataclass
class LoRAXGenerateRequest:
    """Request to LoRAX generate endpoint"""
    inputs: str
    adapter_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class LoRAXGenerateResponse:
    """Response from LoRAX generate endpoint"""
    generated_text: str
    adapter_id: Optional[str] = None
    finish_reason: Optional[str] = None
    tokens_generated: int = 0


class LoRAXService:
    """
    Client for LoRAX multi-LoRA inference server.

    Provides:
    - Adapter management (list, load, unload)
    - Text generation with optional adapters
    - Server health checks
    - Model information
    """

    def __init__(self, config: LoRAXConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._session

    async def close(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Health & Info ──

    async def health_check(self) -> Dict[str, Any]:
        """Check if LoRAX server is healthy"""
        session = await self._get_session()
        try:
            async with session.get(f"{self.config.base_url}/health") as resp:
                if resp.status == 200:
                    return {"status": "healthy", "details": await resp.json()}
                return {"status": "unhealthy", "status_code": resp.status}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": str(e)}

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded base model"""
        session = await self._get_session()
        try:
            async with session.get(f"{self.config.base_url}/model") as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"error": f"HTTP {resp.status}"}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    async def list_loaded_adapters(self) -> List[LoRAXAdapter]:
        """List all currently loaded adapters on the server"""
        session = await self._get_session()
        try:
            async with session.get(f"{self.config.base_url}/adapters") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    adapters = []
                    for adapter_data in data.get("adapters", []):
                        adapters.append(LoRAXAdapter(
                            adapter_id=adapter_data.get("id", ""),
                            adapter_name=adapter_data.get("name"),
                            base_model=adapter_data.get("base_model"),
                            rank=adapter_data.get("rank"),
                            loaded=True
                        ))
                    return adapters
                return []
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to list adapters: {e}")
            return []

    # ── Adapter Management ──

    async def load_adapter(
        self,
        adapter_id: str,
        adapter_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a LoRA adapter onto the server.

        Args:
            adapter_id: HuggingFace repo ID or local path
            adapter_name: Optional custom name for the adapter

        Returns:
            Dict with status and details
        """
        session = await self._get_session()
        payload = {"adapter_id": adapter_id}
        if adapter_name:
            payload["adapter_name"] = adapter_name

        try:
            async with session.post(
                f"{self.config.base_url}/adapters/load",
                json=payload
            ) as resp:
                data = await resp.json()
                return {
                    "status": "loaded" if resp.status == 200 else "error",
                    "adapter_id": adapter_id,
                    "response": data
                }
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "adapter_id": adapter_id, "error": str(e)}

    async def unload_adapter(self, adapter_id: str) -> Dict[str, Any]:
        """
        Unload a LoRA adapter from the server.

        Args:
            adapter_id: Adapter ID to unload

        Returns:
            Dict with status and details
        """
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.config.base_url}/adapters/unload",
                json={"adapter_id": adapter_id}
            ) as resp:
                data = await resp.json()
                return {
                    "status": "unloaded" if resp.status == 200 else "error",
                    "adapter_id": adapter_id,
                    "response": data
                }
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "adapter_id": adapter_id, "error": str(e)}

    # ── Generation ──

    async def generate(
        self,
        prompt: str,
        adapter_id: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> LoRAXGenerateResponse:
        """
        Generate text using the base model or a specific adapter.

        Args:
            prompt: Input text prompt
            adapter_id: Optional adapter ID to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            LoRAXGenerateResponse with generated text
        """
        session = await self._get_session()
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                **kwargs
            }
        }
        if adapter_id:
            payload["parameters"]["adapter_id"] = adapter_id

        try:
            async with session.post(
                f"{self.config.base_url}/generate",
                json=payload
            ) as resp:
                data = await resp.json()
                return LoRAXGenerateResponse(
                    generated_text=data.get("generated_text", ""),
                    adapter_id=adapter_id,
                    finish_reason=data.get("finish_reason"),
                    tokens_generated=data.get("tokens_generated", 0)
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        adapter_id: Optional[str] = None,
        **kwargs
    ):
        """
        Generate text with streaming.

        Yields chunks of generated text as they arrive.
        """
        session = await self._get_session()
        payload = {
            "inputs": prompt,
            "parameters": {**kwargs}
        }
        if adapter_id:
            payload["parameters"]["adapter_id"] = adapter_id

        try:
            async with session.post(
                f"{self.config.base_url}/generate_stream",
                json=payload
            ) as resp:
                async for line in resp.content:
                    if line:
                        yield line.decode("utf-8")
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    # ── Server Management ──

    async def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics (if available)"""
        session = await self._get_session()
        try:
            async with session.get(f"{self.config.base_url}/stats") as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get stats: {e}")
            return {}


def get_lorax_service(
    host: str = "localhost",
    port: int = 8080,
    base_model: Optional[str] = None,
    api_key: Optional[str] = None
) -> LoRAXService:
    """Factory function to create a LoRAX service instance"""
    config = LoRAXConfig(
        host=host,
        port=port,
        base_model=base_model,
        api_key=api_key
    )
    return LoRAXService(config)
