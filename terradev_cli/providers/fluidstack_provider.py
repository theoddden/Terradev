#!/usr/bin/env python3
"""
FluidStack Provider - FluidStack GPU cloud integration
BYOAPI: Uses the end-client's FluidStack API key
API: https://platform.fluidstack.io/v1

Strengths: 20+ regions incl. Asia/LatAm, spot arbitrage king, <5min launch
Auth: Custom 'api-key' header (NOT Bearer token — this is FluidStack-specific)
"""

import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class FluidStackProvider(BaseProvider):
    """FluidStack provider for GPU instances"""

    API_BASE = "https://platform.fluidstack.io/v1"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "fluidstack"
        self.api_key = credentials.get("api_key", "")

    # ── GPU configuration mapping ─────────────────────────────────────
    # FluidStack gpu_type values from /configurations endpoint
    GPU_PRICING = {
        "H100_SXM5_80GB": {
            "gpu_type": "H100_SXM5_80GB",
            "price": 2.50,
            "mem": 80,
            "vcpus": 16,
            "gpu_count": 1,
        },
        "H100-8x": {
            "gpu_type": "H100_SXM5_80GB",
            "price": 20.00,
            "mem": 80,
            "vcpus": 96,
            "gpu_count": 8,
        },
        "A100_SXM4_80GB": {
            "gpu_type": "A100_SXM4_80GB",
            "price": 1.50,
            "mem": 80,
            "vcpus": 16,
            "gpu_count": 1,
        },
        "A100": {
            "gpu_type": "A100_SXM4_80GB",
            "price": 1.50,
            "mem": 80,
            "vcpus": 16,
            "gpu_count": 1,
        },
        "H100": {
            "gpu_type": "H100_SXM5_80GB",
            "price": 2.50,
            "mem": 80,
            "vcpus": 16,
            "gpu_count": 1,
        },
        "RTX4090": {
            "gpu_type": "RTX_4090_24GB",
            "price": 0.40,
            "mem": 24,
            "vcpus": 8,
            "gpu_count": 1,
        },
        "RTX3090": {
            "gpu_type": "RTX_3090_24GB",
            "price": 0.15,
            "mem": 24,
            "vcpus": 8,
            "gpu_count": 1,
        },
        "L40S": {
            "gpu_type": "L40S_48GB",
            "price": 1.10,
            "mem": 48,
            "vcpus": 16,
            "gpu_count": 1,
        },
        "A6000": {
            "gpu_type": "RTX_A6000_48GB",
            "price": 0.80,
            "mem": 48,
            "vcpus": 12,
            "gpu_count": 1,
        },
    }

    # ── Authentication ────────────────────────────────────────────────

    def _get_auth_headers(self) -> Dict[str, str]:
        # FluidStack uses a custom 'api-key' header, NOT 'Authorization: Bearer'
        # Per: docs.fluidstack.io/get-started/api-overview
        if self.api_key:
            return {"api-key": self.api_key}
        return {}

    # ── Capacity / Quotes ─────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        try:
            live = await self._get_live_configurations(gpu_type)
            if live:
                return live
        except Exception as e:
            logger.debug(f"FluidStack API error: {e}")

        # Static fallback
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            for key, val in self.GPU_PRICING.items():
                if gpu_type.upper().startswith(key.split("-")[0].split("_")[0]):
                    info = val
                    break
        if not info:
            return []

        return [
            {
                "instance_type": info["gpu_type"],
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": region or "us-east-1",
                "available": True,
                "provider": "fluidstack",
                "vcpus": info["vcpus"],
                "memory_gb": info["mem"],
                "gpu_count": info["gpu_count"],
                "spot": True,
            }
        ]

    async def _get_live_configurations(self, gpu_type: str) -> List[Dict[str, Any]]:
        """Query GET /configurations for real-time GPU availability and pricing."""
        data = await self._make_request("GET", f"{self.API_BASE}/configurations")

        quotes = []
        gpu_upper = gpu_type.upper()
        for config in (data if isinstance(data, list) else []):
            config_gpu = config.get("gpu_type", "")
            if gpu_upper not in config_gpu.upper() and config_gpu.upper() not in gpu_upper:
                # Also try matching common names
                if not any(
                    alias in config_gpu.upper()
                    for alias in [gpu_upper.replace("-", "_"), gpu_upper.split("_")[0]]
                ):
                    continue

            for plan in config.get("plans", [config]):
                price = plan.get("price_per_hour", plan.get("price", 0))
                if not price:
                    continue
                quotes.append(
                    {
                        "instance_type": config_gpu,
                        "gpu_type": gpu_type,
                        "price_per_hour": price,
                        "region": plan.get("region", "us-east-1"),
                        "available": plan.get("available", True),
                        "provider": "fluidstack",
                        "vcpus": config.get("vcpus", plan.get("vcpus", 16)),
                        "memory_gb": config.get("ram_gb", plan.get("ram_gb", 0)),
                        "gpu_count": config.get("gpu_count", 1),
                        "spot": plan.get("spot", True),
                        "estimated_provisioning_time_s": config.get(
                            "estimated_provisioning_time_s"
                        ),
                    }
                )

        return sorted(quotes, key=lambda q: q["price_per_hour"])

    # ── Provisioning ──────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("FluidStack API key not configured")

        ssh_key_name = self.credentials.get("ssh_key_name", "")
        if not ssh_key_name:
            # Try to get first available SSH key
            try:
                keys = await self._make_request("GET", f"{self.API_BASE}/ssh_keys")
                if isinstance(keys, list) and keys:
                    ssh_key_name = keys[0].get("name", "")
            except Exception:
                pass

        body: Dict[str, Any] = {
            "gpu_type": instance_type,
            "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
        }
        if ssh_key_name:
            body["ssh_keys"] = [ssh_key_name]

        data = await self._make_request(
            "POST", f"{self.API_BASE}/instances", json=body
        )

        return {
            "instance_id": data.get("id", f"fs-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            "instance_type": instance_type,
            "region": region,
            "gpu_type": gpu_type,
            "status": data.get("status", "provisioning"),
            "provider": "fluidstack",
            "metadata": {
                "name": data.get("name", body["name"]),
                "ssh_key": ssh_key_name,
            },
        }

    # ── Instance management ───────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("FluidStack API key not configured")

        data = await self._make_request(
            "GET", f"{self.API_BASE}/instances/{instance_id}"
        )
        return {
            "instance_id": instance_id,
            "status": data.get("status", "unknown"),
            "instance_type": data.get("gpu_type", "unknown"),
            "region": data.get("region", "unknown"),
            "provider": "fluidstack",
            "public_ip": data.get("ip_address"),
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("FluidStack API key not configured")
        await self._make_request(
            "POST", f"{self.API_BASE}/instances/{instance_id}/stop"
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("FluidStack API key not configured")
        await self._make_request(
            "POST", f"{self.API_BASE}/instances/{instance_id}/start"
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("FluidStack API key not configured")
        await self._make_request(
            "DELETE", f"{self.API_BASE}/instances/{instance_id}"
        )
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request("GET", f"{self.API_BASE}/instances")
            return [
                {
                    "instance_id": inst.get("id", "unknown"),
                    "status": inst.get("status", "unknown"),
                    "instance_type": inst.get("gpu_type", "unknown"),
                    "region": inst.get("region", "unknown"),
                    "provider": "fluidstack",
                    "public_ip": inst.get("ip_address"),
                }
                for inst in (data if isinstance(data, list) else [])
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute command via SSH (FluidStack provides SSH access)."""
        if not self.api_key:
            raise Exception("FluidStack API key not configured")

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
                "output": f"FluidStack exec error: {e}",
                "async": async_exec,
            }

    # ── FluidStack-specific helpers ───────────────────────────────────

    async def get_configurations(self) -> List[Dict[str, Any]]:
        """List all available GPU configurations."""
        return await self._make_request("GET", f"{self.API_BASE}/configurations")

    async def get_os_templates(self) -> List[Dict[str, Any]]:
        """List available OS templates."""
        return await self._make_request("GET", f"{self.API_BASE}/templates")

    async def get_ssh_keys(self) -> List[Dict[str, Any]]:
        """List SSH keys on account."""
        return await self._make_request("GET", f"{self.API_BASE}/ssh_keys")
