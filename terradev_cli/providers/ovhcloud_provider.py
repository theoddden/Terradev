#!/usr/bin/env python3
"""
OVHcloud Provider - OVHcloud Public Cloud GPU integration
BYOAPI: Uses the end-client's OVH Application Key + Secret + Consumer Key
API: https://api.ovh.com/v1/cloud/project/{project_id}

Strengths: 15+ EU regions, GDPR compliance, renewable energy, ISO27001/SOC
Auth: OVH signature — $1$ + SHA1(AS+CK+METHOD+URL+BODY+TIMESTAMP)
"""

import hashlib
import json as _json
import logging
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OVHcloudProvider(BaseProvider):
    """OVHcloud Public Cloud provider for GPU instances"""

    # Regional API endpoints
    API_ENDPOINTS = {
        "ovh-eu": "https://eu.api.ovh.com/v1",
        "ovh-us": "https://api.us.ovhcloud.com/v1",
        "ovh-ca": "https://ca.api.ovh.com/v1",
    }

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "ovhcloud"
        self.application_key = credentials.get("application_key", "")
        self.application_secret = credentials.get("application_secret", "")
        self.consumer_key = credentials.get("consumer_key", "")
        self.project_id = credentials.get("project_id", "")
        endpoint_name = credentials.get("endpoint", "ovh-eu")
        self.api_base = self.API_ENDPOINTS.get(endpoint_name, self.API_ENDPOINTS["ovh-eu"])
        self._time_delta: Optional[int] = None

    # ── GPU flavor mapping ────────────────────────────────────────────
    # OVHcloud GPU flavors: t2-*, g3-*, g4-*
    # L40S (48GB GDDR6), V100S (32GB HBM2), L4 (24GB GDDR6)
    GPU_PRICING = {
        "L40S": {
            "flavor_prefix": "g3-",
            "gpu_model": "NVIDIA L40S",
            "price": 1.70,
            "mem": 48,
            "vcpus": 16,
            "gpu_count": 1,
        },
        "L40S-4x": {
            "flavor_prefix": "g3-",
            "gpu_model": "NVIDIA L40S",
            "price": 6.80,
            "mem": 48,
            "vcpus": 64,
            "gpu_count": 4,
        },
        "V100S": {
            "flavor_prefix": "t2-",
            "gpu_model": "NVIDIA V100S",
            "price": 0.88,
            "mem": 32,
            "vcpus": 8,
            "gpu_count": 1,
        },
        "V100S-4x": {
            "flavor_prefix": "t2-",
            "gpu_model": "NVIDIA V100S",
            "price": 3.52,
            "mem": 32,
            "vcpus": 32,
            "gpu_count": 4,
        },
        "L4": {
            "flavor_prefix": "g4-",
            "gpu_model": "NVIDIA L4",
            "price": 1.00,
            "mem": 24,
            "vcpus": 8,
            "gpu_count": 1,
        },
    }

    REGIONS = [
        "GRA7", "GRA9", "GRA11", "SBG5", "BHS5",
        "WAW1", "UK1", "DE1", "RBX-HPC",
    ]

    # ── OVH Signature Authentication ──────────────────────────────────

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    async def _get_time_delta(self) -> int:
        """Get server time delta for signature.

        /auth/time returns a plain integer (Unix timestamp), NOT JSON.
        Must use raw aiohttp to avoid response.json() parsing failure.
        """
        if self._time_delta is not None:
            return self._time_delta
        import aiohttp
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            async with self.session.get(
                f"{self.api_base}/auth/time",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.text()
                server_time = int(body.strip())
            self._time_delta = server_time - int(time.time())
        except Exception:
            self._time_delta = 0
        return self._time_delta

    def _sign(self, method: str, url: str, body: str, timestamp: str) -> str:
        """Compute OVH API signature: $1$SHA1(AS+CK+METHOD+URL+BODY+TS)."""
        to_sign = "+".join([
            self.application_secret,
            self.consumer_key,
            method.upper(),
            url,
            body,
            timestamp,
        ])
        return "$1$" + hashlib.sha1(to_sign.encode("utf-8")).hexdigest()

    async def _ovh_request(
        self, method: str, path: str, body: Optional[Dict] = None
    ) -> Any:
        """Make an OVH-signed API request.

        Uses raw aiohttp session — NOT _make_request — because
        _make_request merges _get_auth_headers() into headers which
        would clobber the required OVH X-Ovh-Signature headers.
        """
        import aiohttp

        if not self.application_key or not self.application_secret or not self.consumer_key:
            raise Exception("OVHcloud credentials not configured")

        delta = await self._get_time_delta()
        timestamp = str(int(time.time()) + delta)
        url = f"{self.api_base}{path}"
        body_str = _json.dumps(body) if body else ""

        signature = self._sign(method, url, body_str, timestamp)

        headers = {
            "X-Ovh-Application": self.application_key,
            "X-Ovh-Consumer": self.consumer_key,
            "X-Ovh-Timestamp": timestamp,
            "X-Ovh-Signature": signature,
            "Content-Type": "application/json",
        }

        if not self.session:
            self.session = aiohttp.ClientSession()

        kwargs: Dict[str, Any] = {"headers": headers}
        if body:
            kwargs["data"] = body_str  # pre-serialised JSON string

        async with self.session.request(
            method.upper(), url, timeout=aiohttp.ClientTimeout(total=30), **kwargs
        ) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"OVHcloud HTTP {response.status}: {error_text}")
            text = await response.text()
            if not text:
                return {}
            return _json.loads(text)

    # ── Capacity / Quotes ─────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if self.application_key and self.application_secret and self.consumer_key and self.project_id:
            try:
                live = await self._get_live_flavors(gpu_type, region)
                if live:
                    return live
            except Exception as e:
                logger.debug(f"OVHcloud live flavors error: {e}")

        # Static fallback
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            for key, val in self.GPU_PRICING.items():
                if gpu_type.upper().startswith(key.split("-")[0]):
                    info = val
                    break
        if not info:
            return []

        return [
            {
                "instance_type": f"{info['flavor_prefix']}gpu-{gpu_type.lower()}",
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": region or "GRA7",
                "available": True,
                "provider": "ovhcloud",
                "vcpus": info["vcpus"],
                "memory_gb": info["mem"],
                "gpu_count": info["gpu_count"],
                "spot": False,
            }
        ]

    async def _get_live_flavors(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query /cloud/project/{id}/flavor for live GPU flavors."""
        path = f"/cloud/project/{self.project_id}/flavor"
        if region:
            path += f"?region={region}"
        data = await self._ovh_request("GET", path)

        quotes = []
        gpu_lower = gpu_type.lower()
        for flavor in data if isinstance(data, list) else []:
            fname = flavor.get("name", "").lower()
            ftype = flavor.get("type", "").lower()
            if "gpu" not in ftype and "gpu" not in fname:
                continue
            if gpu_lower not in fname and gpu_lower not in flavor.get("planCodes", "").lower():
                continue
            quotes.append(
                {
                    "instance_type": flavor.get("id", flavor.get("name", "")),
                    "gpu_type": gpu_type,
                    "price_per_hour": flavor.get("pricingsHourly", {}).get("price", 0),
                    "region": region or "GRA7",
                    "available": flavor.get("available", True),
                    "provider": "ovhcloud",
                    "vcpus": flavor.get("vcpus", 0),
                    "memory_gb": flavor.get("ram", 0) / 1024 if flavor.get("ram", 0) > 1024 else flavor.get("ram", 0),
                    "gpu_count": flavor.get("gpus", 1),
                    "spot": False,
                    "flavor_name": flavor.get("name", ""),
                }
            )
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    # ── Provisioning ──────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.project_id:
            raise Exception("OVHcloud project_id not configured — run `terradev configure ovhcloud`")

        # Get default image (Ubuntu 22.04)
        image_id = self.credentials.get("image_id", "")
        ssh_key_id = self.credentials.get("ssh_key_id", "")

        if not image_id:
            # Try to find Ubuntu 22.04 image
            images = await self._ovh_request(
                "GET", f"/cloud/project/{self.project_id}/image?osType=linux&region={region}"
            )
            for img in (images if isinstance(images, list) else []):
                if "ubuntu" in img.get("name", "").lower() and "22.04" in img.get("name", ""):
                    image_id = img.get("id", "")
                    break

        body = {
            "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
            "flavorId": instance_type,
            "imageId": image_id,
            "region": region or "GRA7",
        }
        if ssh_key_id:
            body["sshKeyId"] = ssh_key_id

        data = await self._ovh_request(
            "POST", f"/cloud/project/{self.project_id}/instance", body=body
        )

        return {
            "instance_id": data.get("id", f"ovh-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            "instance_type": instance_type,
            "region": region,
            "gpu_type": gpu_type,
            "status": data.get("status", "provisioning"),
            "provider": "ovhcloud",
            "metadata": {"name": body["name"]},
        }

    # ── Instance management ───────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        data = await self._ovh_request(
            "GET", f"/cloud/project/{self.project_id}/instance/{instance_id}"
        )
        ip_addresses = data.get("ipAddresses", [])
        public_ip = None
        for addr in ip_addresses:
            if addr.get("type") == "public" and addr.get("version") == 4:
                public_ip = addr.get("ip")
                break

        return {
            "instance_id": instance_id,
            "status": data.get("status", "unknown").lower(),
            "instance_type": data.get("flavor", {}).get("id", "unknown"),
            "region": data.get("region", "unknown"),
            "provider": "ovhcloud",
            "public_ip": public_ip,
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        await self._ovh_request(
            "POST",
            f"/cloud/project/{self.project_id}/instance/{instance_id}/stop",
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        await self._ovh_request(
            "POST",
            f"/cloud/project/{self.project_id}/instance/{instance_id}/start",
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        await self._ovh_request(
            "DELETE",
            f"/cloud/project/{self.project_id}/instance/{instance_id}",
        )
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.project_id:
            return []
        try:
            data = await self._ovh_request(
                "GET", f"/cloud/project/{self.project_id}/instance"
            )
            return [
                {
                    "instance_id": inst.get("id", "unknown"),
                    "status": inst.get("status", "unknown").lower(),
                    "instance_type": inst.get("flavor", {}).get("id", "unknown"),
                    "region": inst.get("region", "unknown"),
                    "provider": "ovhcloud",
                }
                for inst in (data if isinstance(data, list) else [])
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute command via SSH (OVH has no remote-exec API)."""
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
                "output": f"OVHcloud exec error: {e}",
                "async": async_exec,
            }

    # ── OVH-specific helpers ──────────────────────────────────────────

    async def get_project_regions(self) -> List[Dict[str, Any]]:
        """List regions available in the project."""
        return await self._ovh_request(
            "GET", f"/cloud/project/{self.project_id}/region"
        )

    async def get_ssh_keys(self) -> List[Dict[str, Any]]:
        """List SSH keys in the project."""
        return await self._ovh_request(
            "GET", f"/cloud/project/{self.project_id}/sshkey"
        )
