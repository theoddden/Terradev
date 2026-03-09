#!/usr/bin/env python3
"""
DigitalOcean Provider Integration for Terradev
GPU Droplet provisioning and management

CRITICAL FIX v4.0.5:
- Converted from standalone class to BaseProvider subclass for factory compatibility
- Auth: Authorization: Bearer (standard DO API)
- API base: https://api.digitalocean.com/v2
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class DigitalOceanProvider(BaseProvider):
    """DigitalOcean GPU provider for Terradev"""

    API_BASE = "https://api.digitalocean.com/v2"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "digitalocean"
        self.api_key = credentials.get("api_key", credentials.get("api_token", ""))
        self.default_region = credentials.get("region", "tor1")

    GPU_PRICING = {
        "H100": {"slug": "gpu-h100x1-80gb", "price": 3.52, "mem": 240, "vcpus": 20, "gpus": 1},
        "H100-8x": {"slug": "gpu-h100x8-640gb", "price": 28.16, "mem": 1920, "vcpus": 160, "gpus": 8},
    }

    def _get_auth_headers(self) -> Dict[str, str]:
        if self.api_key:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        return {}

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        # Try live sizes API
        try:
            live = await self._get_live_sizes(gpu_type, region)
            if live:
                return live
        except Exception:
            pass

        # Static fallback
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            return []
        return [{
            "instance_type": info["slug"],
            "gpu_type": gpu_type,
            "price_per_hour": info["price"],
            "region": region or self.default_region,
            "available": True,
            "provider": "digitalocean",
            "vcpus": info["vcpus"],
            "memory_gb": info["mem"],
            "gpu_count": info["gpus"],
            "spot": False,
        }]

    async def _get_live_sizes(self, gpu_type: str, region: Optional[str] = None) -> List[Dict[str, Any]]:
        data = await self._make_request("GET", f"{self.API_BASE}/sizes")
        quotes = []
        for size in data.get("sizes", []):
            slug = size.get("slug", "")
            if "gpu" not in slug.lower():
                continue
            if gpu_type.lower() not in slug.lower():
                continue
            r = region or self.default_region
            regions = size.get("regions", [])
            if r and regions and r not in regions:
                continue
            quotes.append({
                "instance_type": slug,
                "gpu_type": gpu_type,
                "price_per_hour": size.get("price_hourly", 0),
                "region": r,
                "available": size.get("available", True),
                "provider": "digitalocean",
                "vcpus": size.get("vcpus", 0),
                "memory_gb": (size.get("memory", 0) or 0) // 1024 if (size.get("memory", 0) or 0) > 1024 else size.get("memory", 0),
                "gpu_count": 1,
                "spot": False,
            })
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("DigitalOcean API token not configured")

        instance_name = f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}"
        data = await self._make_request(
            "POST", f"{self.API_BASE}/droplets",
            json={
                "name": instance_name,
                "region": region or self.default_region,
                "size": instance_type,
                "image": "gpu-h100-base",
                "tags": ["terradev", "gpu"],
                "monitoring": True,
            },
        )
        droplet = data.get("droplet", data)
        droplet_id = str(droplet.get("id", f"do-{datetime.now().strftime('%Y%m%d%H%M%S')}"))
        return {
            "instance_id": droplet_id,
            "instance_type": instance_type,
            "region": region or self.default_region,
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "digitalocean",
        }

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("DigitalOcean API token not configured")
        data = await self._make_request("GET", f"{self.API_BASE}/droplets/{instance_id}")
        droplet = data.get("droplet", data)
        public_ip = None
        for net in droplet.get("networks", {}).get("v4", []):
            if net.get("type") == "public":
                public_ip = net.get("ip_address")
                break
        status_map = {"active": "running", "new": "provisioning", "off": "stopped", "archive": "terminated"}
        return {
            "instance_id": instance_id,
            "status": status_map.get(droplet.get("status", ""), droplet.get("status", "unknown")),
            "provider": "digitalocean",
            "public_ip": public_ip,
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("DigitalOcean API token not configured")
        await self._make_request(
            "POST", f"{self.API_BASE}/droplets/{instance_id}/actions",
            json={"type": "shutdown"},
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("DigitalOcean API token not configured")
        await self._make_request(
            "POST", f"{self.API_BASE}/droplets/{instance_id}/actions",
            json={"type": "power_on"},
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("DigitalOcean API token not configured")
        await self._make_request("DELETE", f"{self.API_BASE}/droplets/{instance_id}")
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request(
                "GET", f"{self.API_BASE}/droplets?tag_name=terradev",
            )
            droplets = data.get("droplets", [])
            instances = []
            for d in droplets:
                public_ip = None
                for net in d.get("networks", {}).get("v4", []):
                    if net.get("type") == "public":
                        public_ip = net.get("ip_address")
                        break
                instances.append({
                    "instance_id": str(d.get("id", "unknown")),
                    "status": d.get("status", "unknown"),
                    "instance_type": d.get("size", {}).get("slug", "unknown") if isinstance(d.get("size"), dict) else d.get("size_slug", "unknown"),
                    "region": d.get("region", {}).get("slug", "unknown") if isinstance(d.get("region"), dict) else "unknown",
                    "provider": "digitalocean",
                    "public_ip": public_ip,
                })
            return instances
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("DigitalOcean API token not configured")
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
                f"root@{public_ip}", command,
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
                "output": f"DigitalOcean exec error: {e}",
                "async": async_exec,
            }
