#!/usr/bin/env python3
"""
TensorDock Provider - TensorDock GPU marketplace integration
BYOAPI: Uses the end-client's TensorDock API credentials
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider


class TensorDockProvider(BaseProvider):
    """TensorDock provider for GPU instances"""

    API_BASE = "https://dashboard.tensordock.com/api/v2"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "tensordock"
        self.api_key = credentials.get("api_key", "")
        self.api_token = credentials.get("api_token", "")

    GPU_PRICING = {
        "A100": {"model": "a100_pcie_80g", "price": 1.50, "mem": 80, "vcpus": 10},
        "V100": {"model": "v100_pcie_16g", "price": 0.39, "mem": 16, "vcpus": 6},
        "RTX4090": {"model": "geforcertx4090_pcie_24g", "price": 0.49, "mem": 24, "vcpus": 8},
        "RTX3090": {"model": "geforcertx3090_pcie_24g", "price": 0.20, "mem": 24, "vcpus": 6},
        "RTX3080": {"model": "geforcertx3080_pcie_10g", "price": 0.10, "mem": 10, "vcpus": 4},
        "T4": {"model": "t4_pcie_16g", "price": 0.10, "mem": 16, "vcpus": 4},
    }

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # CRITICAL FIX: Don't return quotes without API credentials (BYOAPI requirement)
        if not self.api_key:
            return []
            
        # Try live API first
        try:
            live = await self._get_live_availability(gpu_type)
            if live:
                return live
        except Exception:
            pass

        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            return []

        return [{
            "instance_type": info["model"],
            "gpu_type": gpu_type,
            "price_per_hour": info["price"],
            "region": region or "us-east",
            "available": True,
            "provider": "tensordock",
            "vcpus": info["vcpus"],
            "memory_gb": info["mem"],
            "gpu_count": 1,
            "spot": False,
        }]

    async def _get_live_availability(self, gpu_type: str) -> List[Dict[str, Any]]:
        # TensorDock v2 API: GET /api/v2/locations
        data = await self._make_request("GET", f"{self.API_BASE}/locations")
        quotes = []
        locations = data.get("data", {}).get("attributes", {}).get("locations", [])
        if isinstance(locations, dict):
            locations = list(locations.values())
        for loc in locations:
            gpus = loc.get("gpus", loc.get("gpu", {}))
            if isinstance(gpus, dict):
                gpus = [gpus]
            for gpu_info in (gpus if isinstance(gpus, list) else []):
                gpu_name = gpu_info.get("name", gpu_info.get("gpu_model", ""))
                if gpu_type.lower() in gpu_name.lower():
                    price = gpu_info.get("price", gpu_info.get("price_per_hour", 0))
                    quotes.append({
                        "instance_type": gpu_name,
                        "gpu_type": gpu_type,
                        "price_per_hour": price,
                        "region": loc.get("region", loc.get("location", "unknown")),
                        "available": gpu_info.get("available", gpu_info.get("amount", 0)) > 0,
                        "provider": "tensordock",
                        "vcpus": 8,
                        "memory_gb": gpu_info.get("vram", gpu_info.get("memory_gb", 0)),
                        "gpu_count": 1,
                        "spot": False,
                    })
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key or not self.api_token:
            raise Exception("TensorDock credentials not configured")

        info = self.GPU_PRICING.get(gpu_type, {})
        data = await self._make_request(
            "POST", f"{self.API_BASE}/instances",
            json={
                "gpu_model": info.get("model", instance_type),
                "gpu_count": 1,
                "vcpus": 8,
                "ram": 32,
                "storage": 100,
                "os": "Ubuntu 22.04 LTS",
                "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
            },
        )
        return {
            "instance_id": data.get("server", {}).get("id", f"td-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            "instance_type": instance_type,
            "region": region,
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "tensordock",
        }

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key or not self.api_token:
            raise Exception("TensorDock credentials not configured")
        data = await self._make_request(
            "GET", f"{self.API_BASE}/instances/{instance_id}",
        )
        inst = data.get("data", {}).get("attributes", data)
        return {
            "instance_id": instance_id,
            "status": inst.get("status", "unknown"),
            "provider": "tensordock",
            "public_ip": inst.get("ip", inst.get("public_ip")),
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key or not self.api_token:
            raise Exception("TensorDock credentials not configured")
        await self._make_request(
            "POST", f"{self.API_BASE}/instances/{instance_id}/stop",
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key or not self.api_token:
            raise Exception("TensorDock credentials not configured")
        await self._make_request(
            "POST", f"{self.API_BASE}/instances/{instance_id}/start",
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key or not self.api_token:
            raise Exception("TensorDock credentials not configured")
        await self._make_request(
            "DELETE", f"{self.API_BASE}/instances/{instance_id}",
        )
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request(
                "GET", f"{self.API_BASE}/instances",
            )
            instances = data.get("data", {}).get("attributes", {}).get("instances", [])
            return [
                {
                    "instance_id": str(vm.get("id")),
                    "status": vm.get("status", "unknown"),
                    "instance_type": vm.get("gpu_model", "unknown"),
                    "region": vm.get("region", vm.get("location", "unknown")),
                    "provider": "tensordock",
                    "public_ip": vm.get("ip"),
                }
                for vm in (instances if isinstance(instances, list) else [])
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.api_key or not self.api_token:
            raise Exception("TensorDock credentials not configured")
        # TensorDock VMs have SSH access — get IP first
        try:
            status = await self.get_instance_status(instance_id)
            public_ip = status.get("public_ip")
            if not public_ip:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 1,
                    "output": "No public IP available — instance may still be provisioning",
                    "async": async_exec,
                }
            import subprocess
            ssh_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                "-o", "ConnectTimeout=10",
                f"user@{public_ip}", command,
            ]
            if async_exec:
                proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "job_id": str(proc.pid),
                    "output": f"Async SSH process started (PID: {proc.pid})",
                    "async": True,
                }
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "async": False,
            }
        except Exception as e:
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"TensorDock exec error: {e}",
                "async": async_exec,
            }

    def _get_auth_headers(self) -> Dict[str, str]:
        # TensorDock v2 API requires Bearer token auth on all endpoints
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return {}
