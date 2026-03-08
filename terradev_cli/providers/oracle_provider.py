#!/usr/bin/env python3
"""
Oracle Cloud Infrastructure (OCI) Provider - GPU instance integration
BYOAPI: Uses the end-client's OCI API key/tenancy credentials
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider


class OracleProvider(BaseProvider):
    """Oracle Cloud Infrastructure provider for GPU instances"""

    API_BASE = "https://iaas.{region}.oraclecloud.com/20160918"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "oracle"
        self.api_key = credentials.get("api_key", "")
        self.tenancy_id = credentials.get("secret_key", "")  # OCI tenancy OCID
        self.region = credentials.get("region", "us-ashburn-1")

    GPU_SHAPES = {
        "A100": {
            "shape": "BM.GPU.A100-v2.8",
            "price": 3.50,
            "mem": 80,
            "vcpus": 128,
            "gpus": 8,
            "gpu_mem_total": 640,
        },
        "A10": {
            "shape": "BM.GPU.A10.4",
            "price": 2.00,
            "mem": 24,
            "vcpus": 64,
            "gpus": 4,
            "gpu_mem_total": 96,
        },
        "V100": {
            "shape": "BM.GPU3.8",
            "price": 2.95,
            "mem": 16,
            "vcpus": 52,
            "gpus": 8,
            "gpu_mem_total": 128,
        },
        "P100": {
            "shape": "BM.GPU2.2",
            "price": 1.28,
            "mem": 16,
            "vcpus": 28,
            "gpus": 2,
            "gpu_mem_total": 32,
        },
    }

    def _api_url(self, path: str) -> str:
        return f"https://iaas.{self.region}.oraclecloud.com/20160918{path}"

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Try live API first
        if self.api_key and self.tenancy_id:
            try:
                live = await self._get_live_shapes(gpu_type, region)
                if live:
                    return live
            except Exception:
                pass

        info = self.GPU_SHAPES.get(gpu_type)
        if not info:
            return []

        return [{
            "instance_type": info["shape"],
            "gpu_type": gpu_type,
            "price_per_hour": info["price"],
            "region": region or self.region,
            "available": True,
            "provider": "oracle",
            "vcpus": info["vcpus"],
            "memory_gb": info["mem"],
            "gpu_count": info["gpus"],
            "spot": False,
        }]

    async def _get_live_shapes(self, gpu_type: str, region: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query OCI ListShapes API for GPU availability"""
        r = region or self.region
        url = f"https://iaas.{r}.oraclecloud.com/20160918/shapes?compartmentId={self.tenancy_id}"
        data = await self._make_request("GET", url)
        quotes = []
        for shape in data if isinstance(data, list) else data.get("items", data.get("shapes", [])):
            shape_name = shape.get("shape", "")
            gpus = shape.get("gpus", 0)
            if gpus > 0 and gpu_type.lower() in shape_name.lower():
                price = shape.get("price", self.GPU_SHAPES.get(gpu_type, {}).get("price", 0))
                quotes.append({
                    "instance_type": shape_name,
                    "gpu_type": gpu_type,
                    "price_per_hour": price,
                    "region": r,
                    "available": True,
                    "provider": "oracle",
                    "vcpus": shape.get("ocpus", 0) * 2,
                    "memory_gb": shape.get("memoryInGBs", 0),
                    "gpu_count": gpus,
                    "spot": False,
                })
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key or not self.tenancy_id:
            raise Exception("OCI credentials not configured (api_key + tenancy OCID required)")

        r = region or self.region
        url = f"https://iaas.{r}.oraclecloud.com/20160918/instances"
        data = await self._make_request(
            "POST", url,
            json={
                "compartmentId": self.tenancy_id,
                "shape": instance_type,
                "displayName": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
                "sourceDetails": {
                    "sourceType": "image",
                    "imageId": "ocid1.image.oc1..gpu-ubuntu-2204",
                },
                "createVnicDetails": {
                    "assignPublicIp": True,
                },
            },
        )
        instance_id = data.get("id", f"oracle-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        return {
            "instance_id": instance_id,
            "instance_type": instance_type,
            "region": r,
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "oracle",
        }

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("OCI credentials not configured")
        url = f"https://iaas.{self.region}.oraclecloud.com/20160918/instances/{instance_id}"
        data = await self._make_request("GET", url)
        status_map = {"RUNNING": "running", "STOPPED": "stopped", "TERMINATED": "terminated",
                       "PROVISIONING": "provisioning", "STARTING": "starting", "STOPPING": "stopping"}
        return {
            "instance_id": instance_id,
            "status": status_map.get(data.get("lifecycleState", ""), "unknown"),
            "provider": "oracle",
            "public_ip": data.get("publicIp"),
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("OCI credentials not configured")
        url = f"https://iaas.{self.region}.oraclecloud.com/20160918/instances/{instance_id}?action=STOP"
        await self._make_request("POST", url)
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("OCI credentials not configured")
        url = f"https://iaas.{self.region}.oraclecloud.com/20160918/instances/{instance_id}?action=START"
        await self._make_request("POST", url)
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("OCI credentials not configured")
        url = f"https://iaas.{self.region}.oraclecloud.com/20160918/instances/{instance_id}"
        await self._make_request("DELETE", url)
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key or not self.tenancy_id:
            return []
        try:
            url = f"https://iaas.{self.region}.oraclecloud.com/20160918/instances?compartmentId={self.tenancy_id}"
            data = await self._make_request("GET", url)
            items = data if isinstance(data, list) else data.get("items", [])
            return [
                {
                    "instance_id": i.get("id"),
                    "status": i.get("lifecycleState", "unknown").lower(),
                    "instance_type": i.get("shape", "unknown"),
                    "region": i.get("region", self.region),
                    "provider": "oracle",
                }
                for i in items
                if i.get("lifecycleState") != "TERMINATED"
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("OCI credentials not configured")
        # OCI supports RunCommand via the Instance Agent
        url = f"https://iaas.{self.region}.oraclecloud.com/20160918/instanceConsoleConnections"
        try:
            data = await self._make_request(
                "POST",
                f"https://computeinstanceagent.{self.region}.oci.oraclecloud.com/20180530/instanceAgentCommands",
                json={
                    "compartmentId": self.tenancy_id,
                    "executionTimeOutInSeconds": 3600,
                    "target": {"instanceId": instance_id},
                    "content": {
                        "source": {
                            "sourceType": "TEXT",
                            "text": command,
                        }
                    },
                },
            )
            cmd_id = data.get("id", "unknown")
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 0,
                "output": f"Command submitted via OCI Instance Agent. Command ID: {cmd_id}",
                "job_id": cmd_id,
                "async": async_exec,
            }
        except Exception as e:
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"OCI RunCommand error: {e}",
                "async": async_exec,
            }

    def _get_auth_headers(self) -> Dict[str, str]:
        # OCI uses request signing, not simple bearer tokens
        # For full implementation, use oci-python-sdk's Signer
        # This provides basic API key auth for compatible endpoints
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}
