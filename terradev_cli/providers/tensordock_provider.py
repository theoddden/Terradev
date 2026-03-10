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

    # v0Name identifiers used by the v2 API
    GPU_PRICING = {
        "A100": {"v0Name": "a100-pcie-80gb", "price": 1.50, "mem": 80, "vcpus": 10},
        "H100": {"v0Name": "h100-sxm5-80gb", "price": 2.49, "mem": 80, "vcpus": 16},
        "V100": {"v0Name": "v100-pcie-16gb", "price": 0.39, "mem": 16, "vcpus": 6},
        "RTX4090": {"v0Name": "geforcertx4090-pcie-24gb", "price": 0.49, "mem": 24, "vcpus": 8},
        "RTX3090": {"v0Name": "geforcertx3090-pcie-24gb", "price": 0.20, "mem": 24, "vcpus": 6},
        "RTX3080": {"v0Name": "geforcertx3080-pcie-10gb", "price": 0.10, "mem": 10, "vcpus": 4},
        "T4": {"v0Name": "t4-pcie-16gb", "price": 0.10, "mem": 16, "vcpus": 4},
        "L40S": {"v0Name": "l40s-pcie-48gb", "price": 1.14, "mem": 48, "vcpus": 10},
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
        locations = data.get("data", {}).get("locations", [])
        for loc in locations:
            loc_id = loc.get("id", "")
            city = loc.get("city", "unknown")
            state = loc.get("stateprovince", "")
            region_label = f"{city}, {state}" if state else city
            for gpu_info in loc.get("gpus", []):
                v0name = gpu_info.get("v0Name", "")
                display = gpu_info.get("displayName", v0name)
                if gpu_type.lower() in display.lower() or gpu_type.lower() in v0name.lower():
                    price = gpu_info.get("price_per_hr", 0)
                    max_count = gpu_info.get("max_count", 0)
                    res = gpu_info.get("resources", {})
                    quotes.append({
                        "instance_type": v0name,
                        "gpu_type": gpu_type,
                        "price_per_hour": price,
                        "region": region_label,
                        "available": max_count > 0,
                        "provider": "tensordock",
                        "vcpus": res.get("max_vcpus", 8),
                        "memory_gb": res.get("max_ram_gb", 0),
                        "gpu_count": 1,
                        "spot": False,
                        "_location_id": loc_id,
                        "_v0name": v0name,
                    })
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("TensorDock credentials not configured")

        info = self.GPU_PRICING.get(gpu_type, {})
        v0name = info.get("v0Name", instance_type)

        # Resolve location_id: prefer _location_id stashed in a previous quote,
        # otherwise query the locations API to find a matching location.
        location_id = ""
        try:
            locs = await self._make_request("GET", f"{self.API_BASE}/locations")
            for loc in locs.get("data", {}).get("locations", []):
                for g in loc.get("gpus", []):
                    if g.get("v0Name", "") == v0name and g.get("max_count", 0) > 0:
                        location_id = loc.get("id", "")
                        break
                if location_id:
                    break
        except Exception:
            pass

        # Read SSH public key from terradev keystore
        ssh_pub = ""
        ssh_pub_path = os.path.expanduser("~/.terradev/keys")
        if os.path.isdir(ssh_pub_path):
            for f in os.listdir(ssh_pub_path):
                if f.endswith(".pub"):
                    with open(os.path.join(ssh_pub_path, f)) as fh:
                        ssh_pub = fh.read().strip()
                    break

        # TensorDock v2: JSON:API-style envelope
        body = {
            "data": {
                "type": "virtualmachine",
                "attributes": {
                    "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
                    "image": "ubuntu2404",
                    "resources": {
                        "vcpu_count": info.get("vcpus", 8),
                        "ram_gb": 32,
                        "storage_gb": 200,
                        "gpus": {
                            v0name: {"count": 1}
                        },
                    },
                    "location_id": location_id,
                    "useDedicatedIp": True,
                    "ssh_key": ssh_pub or "ssh-ed25519 placeholder",
                },
            }
        }

        resp = await self._make_request(
            "POST", f"{self.API_BASE}/instances",
            json=body,
        )
        inst = resp.get("data", resp)
        return {
            "instance_id": inst.get("id", f"td-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            "instance_type": v0name,
            "region": region,
            "gpu_type": gpu_type,
            "status": inst.get("status", "provisioning"),
            "provider": "tensordock",
        }

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("TensorDock credentials not configured")
        data = await self._make_request(
            "GET", f"{self.API_BASE}/instances/{instance_id}",
        )
        # v2 response: top-level keys (type, id, name, status, ipAddress, ...)
        inst = data if "status" in data else data.get("data", data)
        return {
            "instance_id": instance_id,
            "status": inst.get("status", "unknown"),
            "provider": "tensordock",
            "public_ip": inst.get("ipAddress", inst.get("ip")),
            "rate_hourly": inst.get("rateHourly"),
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("TensorDock credentials not configured")
        await self._make_request(
            "POST", f"{self.API_BASE}/instances/{instance_id}/stop",
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("TensorDock credentials not configured")
        await self._make_request(
            "POST", f"{self.API_BASE}/instances/{instance_id}/start",
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
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
            # v2 response: {"data": {"instances": [{"type", "id", "attributes": {...}}]}}
            instances = data.get("data", {}).get("instances", [])
            return [
                {
                    "instance_id": str(vm.get("id")),
                    "status": vm.get("attributes", {}).get("status", vm.get("status", "unknown")),
                    "instance_type": vm.get("attributes", {}).get("name", "unknown"),
                    "region": vm.get("attributes", {}).get("region", "unknown"),
                    "provider": "tensordock",
                    "public_ip": vm.get("attributes", {}).get("ipAddress"),
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
