#!/usr/bin/env python3
"""
Hyperstack Provider Integration for Terradev
GPU VM provisioning and management via NexGen Cloud

CRITICAL FIX v4.0.5:
- Converted from standalone class to BaseProvider subclass for factory compatibility
- Auth header: apiKey (Hyperstack-specific, NOT Authorization: Bearer)
- API base: https://infrahub-api.nexgencloud.com/v1
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class HyperstackProvider(BaseProvider):
    """Hyperstack GPU provider for Terradev (NexGen Cloud)"""

    API_BASE = "https://infrahub-api.nexgencloud.com/v1"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "hyperstack"
        self.api_key = credentials.get("api_key", "")
        self.environment = credentials.get("environment", "default-CANADA-1")
        self.ssh_key_name = credentials.get("ssh_key_name", "")

    GPU_PRICING = {
        "H100": {"flavor": "n3-H100x1", "price": 3.35, "mem": 80, "vcpus": 26, "gpus": 1},
        "H100-8x": {"flavor": "n3-H100x8", "price": 26.80, "mem": 80, "vcpus": 208, "gpus": 8},
        "A100": {"flavor": "n2-A100x1", "price": 2.10, "mem": 80, "vcpus": 12, "gpus": 1},
        "A100-8x": {"flavor": "n2-A100x8", "price": 16.80, "mem": 80, "vcpus": 96, "gpus": 8},
        "L40": {"flavor": "n2-L40x1", "price": 1.15, "mem": 48, "vcpus": 12, "gpus": 1},
    }

    # ── Authentication ────────────────────────────────────────────────
    # Hyperstack uses a custom 'apiKey' header, NOT 'Authorization: Bearer'

    def _get_auth_headers(self) -> Dict[str, str]:
        if self.api_key:
            return {"apiKey": self.api_key, "Content-Type": "application/json"}
        return {}

    # ── BaseProvider implementation ───────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        # Try live flavors API first
        try:
            live = await self._get_live_flavors(gpu_type)
            if live:
                return live
        except Exception:
            pass

        # Static fallback
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            return []

        return [{
            "instance_type": info["flavor"],
            "gpu_type": gpu_type,
            "price_per_hour": info["price"],
            "region": region or "CANADA-1",
            "available": True,
            "provider": "hyperstack",
            "vcpus": info["vcpus"],
            "memory_gb": info["mem"],
            "gpu_count": info["gpus"],
            "spot": False,
        }]

    async def _get_live_flavors(self, gpu_type: str) -> List[Dict[str, Any]]:
        data = await self._make_request("GET", f"{self.API_BASE}/core/flavors")
        quotes = []
        for flavor in data.get("flavors", []):
            name = flavor.get("name", "")
            if gpu_type.lower() in name.lower():
                quotes.append({
                    "instance_type": name,
                    "gpu_type": gpu_type,
                    "price_per_hour": flavor.get("price_per_hour", 0),
                    "region": flavor.get("region_name", "CANADA-1"),
                    "available": flavor.get("stock_available", True),
                    "provider": "hyperstack",
                    "vcpus": flavor.get("cpu", 0),
                    "memory_gb": (flavor.get("ram", 0) or 0) // 1024 if flavor.get("ram", 0) > 100 else flavor.get("ram", 0),
                    "gpu_count": flavor.get("gpu", 1),
                    "spot": False,
                })
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Hyperstack API key not configured")

        instance_name = f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}"
        data = await self._make_request(
            "POST", f"{self.API_BASE}/core/virtual-machines",
            json={
                "name": instance_name,
                "environment_name": self.environment,
                "key_name": self.ssh_key_name or None,
                "image_name": "Ubuntu Server 22.04 LTS (Jammy Jellyfish)",
                "flavor_name": instance_type,
                "count": 1,
                "assign_floating_ip": True,
            },
        )
        vm = data.get("virtual_machine", data.get("instances", [{}]))
        if isinstance(vm, list):
            vm = vm[0] if vm else {}
        vm_id = str(vm.get("id", f"hs-{datetime.now().strftime('%Y%m%d%H%M%S')}"))

        return {
            "instance_id": vm_id,
            "instance_type": instance_type,
            "region": region or "CANADA-1",
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "hyperstack",
        }

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Hyperstack API key not configured")
        data = await self._make_request(
            "GET", f"{self.API_BASE}/core/virtual-machines/{instance_id}",
        )
        vm = data.get("virtual_machine", data)
        return {
            "instance_id": instance_id,
            "status": vm.get("status", "unknown"),
            "provider": "hyperstack",
            "public_ip": vm.get("floating_ip", vm.get("floating_ip_address")),
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Hyperstack API key not configured")
        await self._make_request(
            "GET", f"{self.API_BASE}/core/virtual-machines/{instance_id}/hibernate",
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Hyperstack API key not configured")
        await self._make_request(
            "GET", f"{self.API_BASE}/core/virtual-machines/{instance_id}/restore",
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Hyperstack API key not configured")
        await self._make_request(
            "DELETE", f"{self.API_BASE}/core/virtual-machines/{instance_id}",
        )
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request(
                "GET", f"{self.API_BASE}/core/virtual-machines",
            )
            vms = data.get("virtual_machines", data.get("instances", []))
            return [
                {
                    "instance_id": str(vm.get("id", "unknown")),
                    "status": vm.get("status", "unknown"),
                    "instance_type": vm.get("flavor", {}).get("name", "unknown") if isinstance(vm.get("flavor"), dict) else vm.get("flavor_name", "unknown"),
                    "region": vm.get("environment", {}).get("name", "unknown") if isinstance(vm.get("environment"), dict) else vm.get("environment_name", "unknown"),
                    "provider": "hyperstack",
                    "public_ip": vm.get("floating_ip", vm.get("floating_ip_address")),
                }
                for vm in vms
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Hyperstack API key not configured")
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
                "output": f"Hyperstack exec error: {e}",
                "async": async_exec,
            }
