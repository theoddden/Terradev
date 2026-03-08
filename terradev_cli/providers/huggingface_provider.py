#!/usr/bin/env python3
"""
HuggingFace Provider - Inference Endpoints GPU cloud integration
BYOAPI: Uses the end-client's HuggingFace API token
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseProvider):
    """HuggingFace provider for Inference Endpoints (dedicated GPU instances)"""

    API_BASE = "https://api.endpoints.huggingface.cloud/v2"
    INFERENCE_API = "https://api-inference.huggingface.co"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "huggingface"
        self.api_key = credentials.get("api_key", credentials.get("api_token", ""))
        self.namespace = credentials.get("namespace", "")

    # HuggingFace Inference Endpoints GPU instance types and pricing
    GPU_PRICING = {
        "A100": {"instance_type": "gpu-xlarge-a100", "price": 4.50, "mem": 80, "vcpus": 8},
        "A10G": {"instance_type": "gpu-medium-a10g", "price": 1.30, "mem": 24, "vcpus": 4},
        "T4": {"instance_type": "gpu-small-t4", "price": 0.60, "mem": 16, "vcpus": 4},
    }

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            return []

        # Try live endpoint listing to get real pricing
        if self.api_key and self.namespace:
            try:
                data = await self._make_request(
                    "GET", f"{self.API_BASE}/endpoint/{self.namespace}",
                )
                # If we have live endpoints, extract pricing info
                if isinstance(data, list) and data:
                    logger.info(f"Retrieved {len(data)} live HF endpoints")
            except Exception:
                pass

        return [{
            "instance_type": info["instance_type"],
            "gpu_type": gpu_type,
            "price_per_hour": info["price"],
            "region": region or "us-east-1",
            "available": True,
            "provider": "huggingface",
            "vcpus": info["vcpus"],
            "memory_gb": info["mem"],
            "gpu_count": 1,
            "spot": False,
        }]

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("HuggingFace API token not configured")
        if not self.namespace:
            raise Exception("HuggingFace namespace not configured (set 'namespace' in credentials)")

        endpoint_name = f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Real HuggingFace Inference Endpoints API call
        try:
            data = await self._make_request(
                "POST", f"{self.API_BASE}/endpoint/{self.namespace}",
                json={
                    "name": endpoint_name,
                    "type": "protected",
                    "compute": {
                        "accelerator": "gpu",
                        "instanceType": instance_type,
                        "instanceSize": "x1",
                        "scaling": {"minReplica": 1, "maxReplica": 1},
                    },
                    "model": {
                        "framework": "pytorch",
                        "image": {"huggingface": {}},
                        "repository": "placeholder/model",
                    },
                    "provider": {"region": region or "us-east-1", "vendor": "aws"},
                },
            )
            return {
                "instance_id": endpoint_name,
                "instance_type": instance_type,
                "region": region or "us-east-1",
                "gpu_type": gpu_type,
                "status": "provisioning",
                "provider": "huggingface",
                "endpoint_url": data.get("status", {}).get("url", ""),
            }
        except Exception as e:
            raise Exception(f"HuggingFace provision failed: {e}")

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key or not self.namespace:
            raise Exception("HuggingFace credentials not configured")
        try:
            data = await self._make_request(
                "GET", f"{self.API_BASE}/endpoint/{self.namespace}/{instance_id}",
            )
            hf_status = data.get("status", {}).get("state", "unknown")
            status_map = {
                "pending": "provisioning", "initializing": "provisioning",
                "running": "running", "failed": "error",
                "scaledToZero": "stopped",
            }
            return {
                "instance_id": instance_id,
                "status": status_map.get(hf_status, hf_status),
                "provider": "huggingface",
                "endpoint_url": data.get("status", {}).get("url", ""),
            }
        except Exception as e:
            raise Exception(f"HuggingFace status failed: {e}")

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key or not self.namespace:
            raise Exception("HuggingFace credentials not configured")
        # HF Inference Endpoints: scale to zero to "stop"
        await self._make_request(
            "PUT", f"{self.API_BASE}/endpoint/{self.namespace}/{instance_id}",
            json={"compute": {"scaling": {"minReplica": 0, "maxReplica": 0}}},
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key or not self.namespace:
            raise Exception("HuggingFace credentials not configured")
        await self._make_request(
            "PUT", f"{self.API_BASE}/endpoint/{self.namespace}/{instance_id}",
            json={"compute": {"scaling": {"minReplica": 1, "maxReplica": 1}}},
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key or not self.namespace:
            raise Exception("HuggingFace credentials not configured")
        await self._make_request(
            "DELETE", f"{self.API_BASE}/endpoint/{self.namespace}/{instance_id}",
        )
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key or not self.namespace:
            return []
        try:
            data = await self._make_request(
                "GET", f"{self.API_BASE}/endpoint/{self.namespace}",
            )
            endpoints = data if isinstance(data, list) else []
            return [
                {
                    "instance_id": ep.get("name", "unknown"),
                    "status": ep.get("status", {}).get("state", "unknown"),
                    "provider": "huggingface",
                    "endpoint_url": ep.get("status", {}).get("url", ""),
                }
                for ep in endpoints
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """HuggingFace Inference Endpoints don't support arbitrary command execution.
        Instead, send inference requests to the endpoint URL."""
        if not self.api_key or not self.namespace:
            raise Exception("HuggingFace credentials not configured")
        try:
            # Get endpoint URL
            status = await self.get_instance_status(instance_id)
            endpoint_url = status.get("endpoint_url", "")
            if not endpoint_url:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 1,
                    "output": "Endpoint URL not available — endpoint may still be initializing",
                    "async": async_exec,
                }
            # Send inference request with the command as input text
            data = await self._make_request(
                "POST", endpoint_url,
                json={"inputs": command},
            )
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 0,
                "output": str(data),
                "async": async_exec,
            }
        except Exception as e:
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"HuggingFace exec error: {e}",
                "async": async_exec,
            }

    async def run_inference(
        self, model_id: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run inference on HuggingFace Inference API (serverless)"""
        if not self.api_key:
            raise Exception("HuggingFace API token not configured")
        try:
            import time
            t0 = time.monotonic()
            data = await self._make_request(
                "POST", f"{self.INFERENCE_API}/models/{model_id}",
                json=inputs,
            )
            elapsed = time.monotonic() - t0
            return {
                "model_id": model_id,
                "status": "success",
                "result": data,
                "inference_time": round(elapsed, 3),
            }
        except Exception as e:
            return {"model_id": model_id, "status": "error", "error": str(e)}

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
