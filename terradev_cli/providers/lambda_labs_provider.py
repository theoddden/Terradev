#!/usr/bin/env python3
"""
Lambda Labs Provider - Lambda Cloud GPU integration
BYOAPI: Uses the end-client's Lambda API key
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider


class LambdaLabsProvider(BaseProvider):
    """Lambda Labs provider for GPU instances"""

    API_BASE = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "lambda_labs"
        self.api_key = credentials.get("api_key", "")

    GPU_PRICING = {
        "A100": {"type": "gpu_1x_a100_sxm4", "price": 1.29, "mem": 40, "vcpus": 30, "ram": 200},
        "A100-80": {"type": "gpu_1x_a100_sxm4_80gb", "price": 1.49, "mem": 80, "vcpus": 30, "ram": 200},
        "H100": {"type": "gpu_1x_h100_sxm5", "price": 2.49, "mem": 80, "vcpus": 26, "ram": 200},
        "A10": {"type": "gpu_1x_a10", "price": 0.60, "mem": 24, "vcpus": 30, "ram": 200},
        "V100": {"type": "gpu_1x_v100_sxm2", "price": 0.50, "mem": 16, "vcpus": 6, "ram": 46},
    }

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # CRITICAL FIX: Don't return quotes without API credentials (BYOAPI requirement)
        if not self.api_key:
            return []
            
        # Try live API first
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
            "instance_type": info["type"],
            "gpu_type": gpu_type,
            "price_per_hour": info["price"],
            "region": region or "us-east-1",
            "available": True,
            "provider": "lambda_labs",
            "vcpus": info["vcpus"],
            "memory_gb": info["mem"],
            "gpu_count": 1,
            "spot": False,
        }]

    async def _get_live_availability(self, gpu_type: str) -> List[Dict[str, Any]]:
        data = await self._make_request("GET", f"{self.API_BASE}/instance-types")
        quotes = []
        for type_name, type_data in data.get("data", {}).items():
            spec = type_data.get("instance_type", {}).get("specs", {})
            gpu_desc = spec.get("gpus", [{}])[0].get("description", "")
            if gpu_type.lower() in gpu_desc.lower():
                price = type_data.get("instance_type", {}).get("price_cents_per_hour", 0) / 100
                for region_info in type_data.get("regions_with_capacity_available", []):
                    quotes.append({
                        "instance_type": type_name,
                        "gpu_type": gpu_type,
                        "price_per_hour": price,
                        "region": region_info.get("name", "unknown"),
                        "available": True,
                        "provider": "lambda_labs",
                        "vcpus": spec.get("vcpus", 0),
                        "memory_gb": spec.get("gpus", [{}])[0].get("ram_gb", 0),
                        "gpu_count": len(spec.get("gpus", [])),
                        "spot": False,
                    })
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Lambda Labs API key not configured")
        data = await self._make_request(
            "POST", f"{self.API_BASE}/instance-operations/launch",
            json={
                "region_name": region or "us-east-1",
                "instance_type_name": instance_type,
                "ssh_key_names": [],
                "quantity": 1,
                "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
            },
        )
        ids = data.get("data", {}).get("instance_ids", [])
        return {
            "instance_id": ids[0] if ids else f"lambda-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "instance_type": instance_type,
            "region": region,
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "lambda_labs",
        }

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Lambda Labs API key not configured")
        data = await self._make_request("GET", f"{self.API_BASE}/instances/{instance_id}")
        inst = data.get("data", {})
        return {
            "instance_id": instance_id,
            "status": inst.get("status", "unknown"),
            "instance_type": inst.get("instance_type", {}).get("name", "unknown"),
            "region": inst.get("region", {}).get("name", "unknown"),
            "provider": "lambda_labs",
            "public_ip": inst.get("ip"),
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        # Lambda doesn't support stop — only terminate
        return await self.terminate_instance(instance_id)

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        raise Exception("Lambda Labs does not support restart — launch a new instance instead")

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Lambda Labs API key not configured")
        await self._make_request(
            "POST", f"{self.API_BASE}/instance-operations/terminate",
            json={"instance_ids": [instance_id]},
        )
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request("GET", f"{self.API_BASE}/instances")
            return [
                {
                    "instance_id": i.get("id"),
                    "status": i.get("status", "unknown"),
                    "instance_type": i.get("instance_type", {}).get("name", "unknown"),
                    "region": i.get("region", {}).get("name", "unknown"),
                    "provider": "lambda_labs",
                    "public_ip": i.get("ip"),
                }
                for i in data.get("data", [])
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Lambda Labs API key not configured")
        # Lambda Labs instances have SSH access — get IP first
        try:
            status = await self.get_instance_status(instance_id)
            public_ip = status.get("public_ip")
            if not public_ip:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 1,
                    "output": "No public IP available for SSH — instance may still be provisioning",
                    "async": async_exec,
                }
            import subprocess
            ssh_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                "-o", "ConnectTimeout=10",
                f"ubuntu@{public_ip}", command,
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
                "output": f"Lambda exec error: {e}",
                "async": async_exec,
            }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
