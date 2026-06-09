#!/usr/bin/env python3
"""
E2E Networks Provider - E2E Networks GPU cloud integration (India)
BYOAPI: Uses the end-client's E2E Networks Bearer token
API: https://api.e2enetworks.com/myaccount/api/v1

Auth: Authorization: Bearer {token}
Compute model: Traditional VM-style GPU nodes
Strengths: NSE-listed India hyperscaler, H200/H100/A100/B200, MeitY empanelled
Docs: https://docs.e2enetworks.com/api/myaccount/
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class E2ENetworksProvider(BaseProvider):
    """E2E Networks provider — VM-based GPU nodes, India-first hyperscaler"""

    API_BASE = "https://api.e2enetworks.com/myaccount/api/v1"

    # Map Terradev canonical GPU names → E2E Networks node plan prefixes
    GPU_PLAN_MAP = {
        "B200":        "b200",
        "H200":        "h200",
        "H100":        "h100",
        "H100-SXM":    "h100-sxm",
        "A100":        "a100-80gb",
        "A100-80GB":   "a100-80gb",
        "A100-40GB":   "a100-40gb",
        "L40S":        "l40s",
        "L4":          "l4",
        "T4":          "t4",
        "V100":        "v100",
        "RTX3090":     "rtx3090",
    }

    # Reference pricing (USD/hr per GPU) — live API takes precedence
    # Pricing sourced from e2enetworks.com/pricing (USD region)
    GPU_PRICING = {
        "B200":      {"price": 6.50, "mem_gb": 192, "vcpus": 32, "ram_gb": 256, "gpu_count": 1},
        "H200":      {"price": 4.50, "mem_gb": 141, "vcpus": 24, "ram_gb": 192, "gpu_count": 1},
        "H100":      {"price": 3.20, "mem_gb": 80,  "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "H100-SXM":  {"price": 3.50, "mem_gb": 80,  "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "A100":      {"price": 1.80, "mem_gb": 80,  "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "A100-80GB": {"price": 1.80, "mem_gb": 80,  "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "A100-40GB": {"price": 1.20, "mem_gb": 40,  "vcpus": 12, "ram_gb": 96,  "gpu_count": 1},
        "L40S":      {"price": 1.10, "mem_gb": 48,  "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "L4":        {"price": 0.55, "mem_gb": 24,  "vcpus": 8,  "ram_gb": 64,  "gpu_count": 1},
        "T4":        {"price": 0.40, "mem_gb": 16,  "vcpus": 8,  "ram_gb": 32,  "gpu_count": 1},
        "V100":      {"price": 0.90, "mem_gb": 32,  "vcpus": 12, "ram_gb": 96,  "gpu_count": 1},
        "RTX3090":   {"price": 0.35, "mem_gb": 24,  "vcpus": 8,  "ram_gb": 32,  "gpu_count": 1},
    }

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "e2enetworks"
        self.api_key = credentials.get("api_key", "")
        # E2E Networks may require a project ID for multi-project accounts
        self.project_id = credentials.get("project_id", "")

    # ── Authentication ────────────────────────────────────────────────

    def _get_auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ── URL helpers ───────────────────────────────────────────────────

    def _nodes_url(self, path: str = "") -> str:
        """Build a nodes endpoint URL. Project scoping is done via ?project_id= query param."""
        return f"{self.API_BASE}/nodes{path}"

    def _project_params(self) -> Dict[str, str]:
        """Return project_id as a query-param dict if configured."""
        return {"project_id": self.project_id} if self.project_id else {}

    # ── Capacity / Quotes ─────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        try:
            live = await self._get_live_plans(gpu_type, region)
            if live:
                return live
        except Exception as e:
            logger.debug(f"E2E Networks API error: {e}")

        # Static fallback
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            # Try case-insensitive partial match
            upper = gpu_type.upper()
            for key, val in self.GPU_PRICING.items():
                if upper in key.upper() or key.upper() in upper:
                    info = val
                    break
        if not info:
            return []

        return [
            {
                "instance_type": self.GPU_PLAN_MAP.get(gpu_type, gpu_type.lower()),
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": region or "in-mumbai-1",
                "available": True,
                "provider": "e2enetworks",
                "vcpus": info["vcpus"],
                "memory_gb": info["ram_gb"],
                "gpu_memory_gb": info["mem_gb"],
                "gpu_count": info["gpu_count"],
                "spot": False,
                "india_based": True,
            }
        ]

    async def _get_live_plans(
        self, gpu_type: str, region: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Query node plans/types from the API."""
        try:
            data = await self._make_request("GET", f"{self.API_BASE}/plans/")
        except Exception:
            # Fall back to node-types endpoint
            data = await self._make_request("GET", f"{self.API_BASE}/node-types/")

        plans = data if isinstance(data, list) else data.get("data", data.get("results", []))
        gpu_upper = gpu_type.upper()
        quotes = []

        for plan in plans:
            plan_name = plan.get("name", plan.get("plan_name", "")).upper()
            plan_gpu = plan.get("gpu_type", plan.get("gpu", "")).upper()

            if (
                gpu_upper not in plan_name
                and gpu_upper not in plan_gpu
                and not any(part in plan_name for part in gpu_upper.split("-") if len(part) > 2)
            ):
                continue

            price_inr = float(plan.get("price_per_hour", plan.get("price", 0)) or 0)
            # Convert INR to USD if price looks like INR (>10 per hour for GPU)
            price_usd = price_inr / 83.5 if price_inr > 10 else price_inr

            plan_region = plan.get("region", region or "in-mumbai-1")
            if region and plan_region != region:
                continue

            quotes.append(
                {
                    "instance_type": plan.get("name", plan.get("plan_name", gpu_type)),
                    "gpu_type": gpu_type,
                    "price_per_hour": round(price_usd, 4),
                    "price_inr_per_hour": price_inr,
                    "region": plan_region,
                    "available": plan.get("available", True),
                    "provider": "e2enetworks",
                    "vcpus": plan.get("vcpus", plan.get("cpu", 16)),
                    "memory_gb": plan.get("ram_gb", plan.get("ram", 128)),
                    "gpu_memory_gb": plan.get("gpu_memory_gb", 0),
                    "gpu_count": plan.get("gpu_count", plan.get("gpu_num", 1)),
                    "spot": False,
                    "india_based": True,
                }
            )

        return sorted(quotes, key=lambda q: q["price_per_hour"])

    # ── Provisioning ──────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("E2E Networks API key not configured")

        ssh_key = self.credentials.get("ssh_key_name", "")
        image = self.credentials.get("image", "ubuntu-22.04-x64")
        gpu_count = int(self.credentials.get("gpu_count", 1))

        body: Dict[str, Any] = {
            "name": f"terradev-{gpu_type.lower().replace('_', '-')}-{datetime.now().strftime('%H%M%S')}",
            "plan": instance_type,
            "location": region or "in-mumbai-1",
            "image": image,
            "gpu_count": gpu_count,
        }
        if ssh_key:
            body["ssh_keys"] = [ssh_key]

        data = await self._make_request("POST", self._nodes_url("/"), json=body)
        node = data if isinstance(data, dict) else data.get("data", {})

        return {
            "instance_id": str(node.get("id", f"e2e-{datetime.now().strftime('%Y%m%d%H%M%S')}")),
            "instance_type": instance_type,
            "region": region or "in-mumbai-1",
            "gpu_type": gpu_type,
            "status": node.get("status", "provisioning"),
            "provider": "e2enetworks",
            "metadata": {
                "name": node.get("name", body["name"]),
                "image": image,
                "gpu_count": gpu_count,
                "ssh_key": ssh_key,
                "india_based": True,
            },
        }

    # ── Instance management ───────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("E2E Networks API key not configured")

        data = await self._make_request("GET", self._nodes_url(f"/{instance_id}/"))
        node = data if isinstance(data, dict) else data.get("data", {})

        return {
            "instance_id": instance_id,
            "status": node.get("status", "unknown"),
            "instance_type": node.get("plan", node.get("node_type", "unknown")),
            "region": node.get("location", node.get("region", "unknown")),
            "provider": "e2enetworks",
            "public_ip": node.get("ip", node.get("public_ip")),
            "private_ip": node.get("private_ip"),
            "gpu_count": node.get("gpu_count", 1),
            "india_based": True,
        }

    async def _node_action(self, instance_id: str, action_type: str) -> Dict[str, Any]:
        """Send a lifecycle action to a node.

        Endpoint: PUT /api/v1/nodes/{node_id}/actions/
        Body: {"type": <action_type>}
        Supported types: power_off, power_on, reboot, reinstall, rename, ...
        See: https://docs.e2enetworks.com/api/myaccount/compute/nodes/actions/node-action/
        """
        body = {"type": action_type}
        data = await self._make_request(
            "PUT", self._nodes_url(f"/{instance_id}/actions/"),
            json=body, params=self._project_params()
        )
        return data if isinstance(data, dict) else {}

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("E2E Networks API key not configured")
        await self._node_action(instance_id, "power_off")
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("E2E Networks API key not configured")
        await self._node_action(instance_id, "power_on")
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("E2E Networks API key not configured")
        await self._make_request("DELETE", self._nodes_url(f"/{instance_id}/"))
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request("GET", self._nodes_url("/"))
            nodes = data if isinstance(data, list) else data.get("data", data.get("results", []))
            return [
                {
                    "instance_id": str(node.get("id", "unknown")),
                    "status": node.get("status", "unknown"),
                    "instance_type": node.get("plan", node.get("node_type", "unknown")),
                    "region": node.get("location", node.get("region", "unknown")),
                    "provider": "e2enetworks",
                    "public_ip": node.get("ip", node.get("public_ip")),
                    "gpu_count": node.get("gpu_count", 1),
                    "india_based": True,
                }
                for node in (nodes if isinstance(nodes, list) else [])
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute command via SSH on the node's public IP."""
        if not self.api_key:
            raise Exception("E2E Networks API key not configured")

        try:
            status = await self.get_instance_status(instance_id)
            public_ip = status.get("public_ip")
            if not public_ip:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 1,
                    "output": "No public IP — node may still be provisioning",
                    "async": async_exec,
                }

            import subprocess

            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                "-o", "ConnectTimeout=10",
                f"root@{public_ip}",
                command,
            ]

            if async_exec:
                proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "job_id": str(proc.pid),
                    "output": f"Async SSH started (PID: {proc.pid})",
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
                "output": f"E2E Networks exec error: {e}",
                "async": async_exec,
            }

    # ── E2E Networks-specific helpers ─────────────────────────────────

    async def get_plans(self) -> List[Dict[str, Any]]:
        """List all available node plans/configurations."""
        try:
            data = await self._make_request("GET", f"{self.API_BASE}/plans/")
        except Exception:
            data = await self._make_request("GET", f"{self.API_BASE}/node-types/")
        return data if isinstance(data, list) else data.get("data", data.get("results", []))

    async def get_images(self) -> List[Dict[str, Any]]:
        """List available OS images for node provisioning."""
        data = await self._make_request("GET", f"{self.API_BASE}/images/")
        return data if isinstance(data, list) else data.get("data", data.get("results", []))

    async def reboot_instance(self, instance_id: str) -> Dict[str, Any]:
        """Reboot a running node."""
        if not self.api_key:
            raise Exception("E2E Networks API key not configured")
        await self._node_action(instance_id, "reboot")
        return {"instance_id": instance_id, "action": "reboot", "status": "rebooting"}

