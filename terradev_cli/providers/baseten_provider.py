#!/usr/bin/env python3
"""
Baseten Provider - Baseten inference-focused GPU cloud integration
BYOAPI: Uses the end-client's Baseten API key
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider


class BasetenProvider(BaseProvider):
    """Baseten provider for inference-optimized GPU instances"""

    API_BASE = "https://api.baseten.co/v1"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "baseten"
        self.api_key = credentials.get("api_key", "")

    GPU_PRICING = {
        "A100": {"model_id": "a100-80gb", "price": 2.12, "mem": 80, "vcpus": 12},
        "A100-40": {"model_id": "a100-40gb", "price": 1.65, "mem": 40, "vcpus": 12},
        "H100": {"model_id": "h100-80gb", "price": 4.50, "mem": 80, "vcpus": 16},
        "T4": {"model_id": "t4-16gb", "price": 0.22, "mem": 16, "vcpus": 4},
        "A10G": {"model_id": "a10g-24gb", "price": 0.60, "mem": 24, "vcpus": 4},
    }

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Try live API first
        if self.api_key:
            try:
                live = await self._get_live_pricing(gpu_type)
                if live:
                    return live
            except Exception:
                pass

        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            return []

        return [{
            "instance_type": f"baseten-{info['model_id']}",
            "gpu_type": gpu_type,
            "price_per_hour": info["price"],
            "region": region or "us-east",
            "available": True,
            "provider": "baseten",
            "vcpus": info["vcpus"],
            "memory_gb": info["mem"],
            "gpu_count": 1,
            "spot": False,
        }]

    async def _get_live_pricing(self, gpu_type: str) -> List[Dict[str, Any]]:
        """Query Baseten API for available models/deployments"""
        data = await self._make_request(
            "GET", f"{self.API_BASE}/models",
        )
        quotes = []
        for model in data.get("models", []):
            deployment = model.get("primary_deployment", {})
            gpu_info = deployment.get("gpu", {})
            gpu_name = gpu_info.get("type", "")
            if gpu_type.lower() in gpu_name.lower():
                price = gpu_info.get("price_per_hour", 0)
                quotes.append({
                    "instance_type": f"baseten-{model.get('id', 'unknown')}",
                    "gpu_type": gpu_type,
                    "price_per_hour": price,
                    "region": deployment.get("region", "us-east"),
                    "available": deployment.get("status") == "ACTIVE",
                    "provider": "baseten",
                    "vcpus": gpu_info.get("vcpus", 0),
                    "memory_gb": gpu_info.get("memory_gb", 0),
                    "gpu_count": gpu_info.get("count", 1),
                    "spot": False,
                })
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Baseten API key not configured")

        gpu_info = self.GPU_PRICING.get(gpu_type)
        if not gpu_info:
            raise Exception(f"Unsupported GPU type for Baseten: {gpu_type}")

        # Deploy a model via Baseten API
        data = await self._make_request(
            "POST", f"{self.API_BASE}/models",
            json={
                "model_name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
                "gpu": gpu_info["model_id"],
                "min_replicas": 1,
                "max_replicas": 1,
            },
        )
        model_id = data.get("model", {}).get("id", f"baseten-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        return {
            "instance_id": model_id,
            "instance_type": instance_type,
            "region": region or "us-east",
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "baseten",
        }

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Baseten API key not configured")
        data = await self._make_request("GET", f"{self.API_BASE}/models/{instance_id}")
        model = data.get("model", {})
        deployment = model.get("primary_deployment", {})
        status_map = {"ACTIVE": "running", "BUILDING": "provisioning", "FAILED": "failed", "SCALED_TO_ZERO": "stopped"}
        return {
            "instance_id": instance_id,
            "status": status_map.get(deployment.get("status", ""), "unknown"),
            "provider": "baseten",
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Baseten API key not configured")
        # Scale to zero
        await self._make_request(
            "PATCH", f"{self.API_BASE}/models/{instance_id}/deployments/production",
            json={"min_replicas": 0, "max_replicas": 0},
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Baseten API key not configured")
        await self._make_request(
            "PATCH", f"{self.API_BASE}/models/{instance_id}/deployments/production",
            json={"min_replicas": 1, "max_replicas": 1},
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Baseten API key not configured")
        await self._make_request("DELETE", f"{self.API_BASE}/models/{instance_id}")
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request("GET", f"{self.API_BASE}/models")
            instances = []
            for model in data.get("models", []):
                deployment = model.get("primary_deployment", {})
                instances.append({
                    "instance_id": model.get("id"),
                    "status": deployment.get("status", "unknown").lower(),
                    "instance_type": deployment.get("gpu", {}).get("type", "unknown"),
                    "region": deployment.get("region", "unknown"),
                    "provider": "baseten",
                })
            return instances
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Baseten API key not configured")
        # Baseten supports running predictions — map command to prediction call
        data = await self._make_request(
            "POST", f"{self.API_BASE}/models/{instance_id}/predict",
            json={"command": command},
        )
        return {
            "instance_id": instance_id,
            "command": command,
            "exit_code": 0,
            "output": str(data.get("output", data)),
            "async": async_exec,
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else {}
