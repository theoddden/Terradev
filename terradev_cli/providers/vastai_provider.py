#!/usr/bin/env python3
"""
Vast.ai Provider - Vast.ai GPU marketplace integration
BYOAPI: Uses the end-client's Vast.ai API key
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class VastAIProvider(BaseProvider):
    """Vast.ai provider for GPU instances - BYOAPI only, no static fallback data"""

    API_BASE = "https://console.vast.ai/api/v0"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "vastai"
        self.api_key = credentials.get("api_key", "")

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # BYOAPI REQUIREMENT: No static fallback data - must have API key
        if not self.api_key:
            return []
            
        # Try live API only - no static fallback
        try:
            live = await self._get_live_offers(gpu_type)
            if live:
                return live
        except Exception as e:
            logger.debug(f"Vast.ai API error: {e}")
            return []

    async def _get_live_offers(self, gpu_type: str) -> List[Dict[str, Any]]:
        data = await self._make_request(
            "GET",
            f"{self.API_BASE}/bundles?q={{\"gpu_name\":\"{gpu_type}\",\"order\":[[\"dph_total\",\"asc\"]],\"type\":\"on-demand\"}}",
        )
        quotes = []
        for offer in data.get("offers", [])[:5]:
            quotes.append({
                "instance_type": f"vastai-{offer.get('id', 'unknown')}",
                "gpu_type": gpu_type,
                "price_per_hour": offer.get("dph_total", 0),
                "region": offer.get("geolocation", "unknown"),
                "available": True,
                "provider": "vastai",
                "vcpus": offer.get("cpu_cores_effective", 0),
                "memory_gb": offer.get("gpu_ram", 0) / 1024,
                "gpu_count": offer.get("num_gpus", 1),
                "spot": True,
            })
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Vast.ai API key not configured")
        offer_id = instance_type.replace("vastai-", "")
        data = await self._make_request(
            "PUT",
            f"{self.API_BASE}/asks/{offer_id}/",
            json={
                "client_id": "me",
                "image": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
                "disk": 50,
                "label": f"terradev-{gpu_type.lower()}",
            },
        )
        return {
            "instance_id": str(data.get("new_contract", offer_id)),
            "instance_type": instance_type,
            "region": region,
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "vastai",
        }

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Vast.ai API key not configured")
        data = await self._make_request("GET", f"{self.API_BASE}/instances/{instance_id}")
        inst = data.get("instances", [{}])[0] if isinstance(data.get("instances"), list) else data
        return {
            "instance_id": instance_id,
            "status": inst.get("actual_status", "unknown"),
            "provider": "vastai",
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Vast.ai API key not configured")
        await self._make_request("PUT", f"{self.API_BASE}/instances/{instance_id}/", json={"state": "stopped"})
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Vast.ai API key not configured")
        await self._make_request("PUT", f"{self.API_BASE}/instances/{instance_id}/", json={"state": "running"})
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Vast.ai API key not configured")
        await self._make_request("DELETE", f"{self.API_BASE}/instances/{instance_id}/")
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request("GET", f"{self.API_BASE}/instances?owner=me")
            return [
                {
                    "instance_id": str(i.get("id")),
                    "status": i.get("actual_status", "unknown"),
                    "instance_type": i.get("gpu_name", "unknown"),
                    "region": i.get("geolocation", "unknown"),
                    "provider": "vastai",
                }
                for i in data.get("instances", [])
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Vast.ai API key not configured")
        # Vast.ai supports direct SSH — get instance SSH info first
        try:
            data = await self._make_request("GET", f"{self.API_BASE}/instances/{instance_id}?owner=me")
            inst = data.get("instances", [{}])[0] if isinstance(data.get("instances"), list) else data
            ssh_host = inst.get("ssh_host", "")
            ssh_port = inst.get("ssh_port", 22)
            if ssh_host:
                import subprocess
                ssh_cmd = [
                    "ssh", "-o", "StrictHostKeyChecking=accept-new",
                    "-o", f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                    "-o", "ConnectTimeout=10",
                    "-p", str(ssh_port), f"root@{ssh_host}", command,
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
            # Fallback: Vast.ai execute endpoint
            exec_data = await self._make_request(
                "PUT", f"{self.API_BASE}/instances/{instance_id}/",
                json={"command": command},
            )
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 0,
                "output": str(exec_data),
                "async": async_exec,
            }
        except Exception as e:
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"Vast.ai exec error: {e}",
                "async": async_exec,
            }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
