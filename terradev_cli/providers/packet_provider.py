#!/usr/bin/env python3
"""
Packet.ai Provider - Packet.ai Dynamic / Dedicated GPU Cloud

API: https://packet.ai/dynamic-gpu-cloud (REST, inferred from CLI and SkyPilot)
CLI: https://packet.ai/cli
SkyPilot integration: sky launch --cloud packet --gpus H100:1
Auth: Bearer token in Authorization header

Packet.ai offers Dynamic (spot / preemptible) and Dedicated (on-demand) GPU
instances.  SSH is the primary interaction model; preinstalled images include
CUDA, vLLM, PyTorch and Jupyter.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class PacketProvider(BaseProvider):
    """Packet.ai GPU cloud provider — dynamic and dedicated instances."""

    # Packet.ai REST API base URL (inferred from dashboard/CLI/SkyPilot docs).
    # Change via credentials `api_endpoint` if Packet exposes a different host.
    API_BASE = "https://api.packet.ai/v1"

    # Supported Packet.ai regions per SkyPilot config docs.
    REGIONS = ["us-east", "us-west", "us-central", "eu-central"]

    # Map Terradev canonical GPU names → Packet `gpu_type` strings.
    # Sources: packet.ai CLI samples, pricing page, SkyPilot integration notes.
    GPU_TYPE_MAP = {
        "H100-80GB": "h100",
        "H100": "h100",
        "H100-SXM": "h100",
        "H100-PCIe": "h100",
        "H200-141GB": "h200",
        "H200": "h200",
        "B200-192GB": "b200",
        "B200": "b200",
        "B300-262GB": "b300",
        "B300": "b300",
        "A100-80GB": "a100",
        "A100-40GB": "a100-40gb",
        "A100": "a100",
        "A100-PCIe": "a100",
        "A100-PCIe-80G": "a100",
        "A100-PCIe-40G": "a100-40gb",
        "L40S-48GB": "l40s",
        "L40S": "l40s",
        "L40-48GB": "l40",
        "L40": "l40",
        "L4-24GB": "l4",
        "L4": "l4",
        "A10G-24GB": "a10g",
        "A10G": "a10g",
        "A10-24GB": "a10",
        "A10": "a10",
        "T4-16GB": "t4",
        "T4": "t4",
        "RTX-Pro-6000-96GB": "rtx-pro-6000",
        "RTXPro6000": "rtx-pro-6000",
        "RTX-4090": "rtx-4090",
        "RTX4090": "rtx-4090",
        "RTX-3090": "rtx-3090",
        "RTX3090": "rtx-3090",
    }

    # Reference pricing (USD/hr) — live GET /v1/gpus takes precedence.
    # CLI sample prices: RTX PRO 6000 $1.29, H100 $1.95, B200 $3.75.
    GPU_PRICING = {
        "rtx-pro-6000": {"price": 1.29, "mem_gb": 96, "vcpus": 16, "ram_gb": 128},
        "h100": {"price": 1.95, "mem_gb": 80, "vcpus": 24, "ram_gb": 128},
        "h200": {"price": 2.95, "mem_gb": 141, "vcpus": 24, "ram_gb": 256},
        "b200": {"price": 3.75, "mem_gb": 192, "vcpus": 32, "ram_gb": 512},
        "b300": {"price": 5.49, "mem_gb": 262, "vcpus": 48, "ram_gb": 512},
        "a100": {"price": 1.09, "mem_gb": 80, "vcpus": 16, "ram_gb": 128},
        "a100-40gb": {"price": 0.79, "mem_gb": 40, "vcpus": 12, "ram_gb": 64},
        "l40s": {"price": 0.99, "mem_gb": 48, "vcpus": 12, "ram_gb": 64},
        "l40": {"price": 0.89, "mem_gb": 48, "vcpus": 12, "ram_gb": 64},
        "l4": {"price": 0.49, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
        "a10g": {"price": 0.69, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
        "a10": {"price": 0.59, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
        "t4": {"price": 0.35, "mem_gb": 16, "vcpus": 4, "ram_gb": 16},
        "rtx-4090": {"price": 0.69, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
        "rtx-3090": {"price": 0.29, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
    }

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "packet"
        self.api_key = credentials.get("api_key", "")
        self.api_base = credentials.get("api_endpoint", self.API_BASE)

    # ── Authentication ─────────────────────────────────────────────────

    def _get_auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ── GPU type resolution ─────────────────────────────────────────────

    def _resolve_gpu_type(self, gpu_type: str) -> str:
        """Map a Terradev canonical GPU name to a Packet gpu_type string."""
        if gpu_type in self.GPU_TYPE_MAP:
            return self.GPU_TYPE_MAP[gpu_type]
        upper = gpu_type.upper()
        for key, val in self.GPU_TYPE_MAP.items():
            if key.upper() == upper:
                return val
        for key, val in self.GPU_TYPE_MAP.items():
            if upper in key.upper() or key.upper() in upper:
                return val
        return gpu_type.lower().replace(" ", "-").replace("_", "-")

    def _resolve_region(self, region: Optional[str]) -> str:
        if region in self.REGIONS:
            return region
        if region:
            for r in self.REGIONS:
                if r.startswith(region.lower().replace("_", "-").lower()):
                    return r
        return "us-east"

    # ── Capacity / Quotes ───────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        packet_gpu = self._resolve_gpu_type(gpu_type)
        target_region = self._resolve_region(region)

        try:
            live = await self._get_live_gpu_types(packet_gpu, target_region)
            if live:
                return live
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Packet API error: {e}")

        info = self.GPU_PRICING.get(packet_gpu)
        if not info:
            return []

        quotes = []
        for spot in (False, True):
            spot_factor = 0.5 if spot else 1.0
            quotes.append(
                {
                    "instance_type": f"packet-{packet_gpu}",
                    "gpu_type": gpu_type,
                    "price_per_hour": round(info["price"] * spot_factor, 4),
                    "region": target_region,
                    "available": True,
                    "provider": "packet",
                    "vcpus": info["vcpus"],
                    "memory_gb": info["ram_gb"],
                    "gpu_memory_gb": info["mem_gb"],
                    "gpu_count": 1,
                    "spot": spot,
                }
            )
        return quotes

    async def _get_live_gpu_types(self, packet_gpu: str, region: str) -> List[Dict[str, Any]]:
        """Query GET /v1/gpus for live availability."""
        data = await self._make_request("GET", f"{self.api_base}/gpus")
        gpu_lower = packet_gpu.lower()
        quotes = []

        for entry in data if isinstance(data, list) else data.get("data", []):
            entry_type = (
                str(entry.get("type", entry.get("gpu_type", ""))).lower().replace(" ", "-")
            )
            if gpu_lower not in entry_type and entry_type not in gpu_lower:
                continue

            price = entry.get("price_per_hour", entry.get("price", 0))
            price = float(price) if price else 0.0

            for tier in ("dynamic", "dedicated"):
                tier_price = price
                if tier == "dynamic":
                    tier_price = price * 0.5

                quotes.append(
                    {
                        "instance_type": f"packet-{packet_gpu}-{tier}",
                        "gpu_type": packet_gpu,
                        "price_per_hour": round(tier_price, 4),
                        "region": region,
                        "available": entry.get("status", "available") == "available",
                        "provider": "packet",
                        "vcpus": entry.get("vcpus", 0) or 8,
                        "memory_gb": entry.get("ram_gb", 0) or 64,
                        "gpu_memory_gb": entry.get("vram_gb", entry.get("memory_gb", 0)) or 0,
                        "gpu_count": 1,
                        "spot": tier == "dynamic",
                        "tier": tier,
                    }
                )

        return sorted(quotes, key=lambda q: q["price_per_hour"])

    # ── Provisioning ────────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str, ssh_public_key: str = ""
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Packet.ai API key not configured")

        packet_gpu = self._resolve_gpu_type(gpu_type)
        target_region = self._resolve_region(region)

        body: Dict[str, Any] = {
            "gpu_type": packet_gpu,
            "region": target_region,
            "tier": "dynamic" if "dynamic" in (instance_type or "").lower() else "dedicated",
            "image": "ubuntu-22.04-cuda",
        }
        if ssh_public_key:
            body["ssh_public_key"] = ssh_public_key

        data = await self._make_request("POST", f"{self.api_base}/instances", json=body)
        instance = data.get("data", data) if isinstance(data, dict) else data

        return {
            "instance_id": str(instance.get("id", instance.get("instance_id", "unknown"))),
            "instance_type": instance_type,
            "region": target_region,
            "gpu_type": gpu_type,
            "status": self._normalize_status(instance.get("status", "provisioning")),
            "provider": "packet",
            "spot": body["tier"] == "dynamic",
            "metadata": {
                "gpu_type": packet_gpu,
                "tier": body["tier"],
                "ssh": instance.get("ssh", ""),
                "ip": instance.get("public_ip", instance.get("ip", "")),
            },
        }

    # ── Instance management ─────────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Packet.ai API key not configured")

        data = await self._make_request("GET", f"{self.api_base}/instances/{instance_id}")
        instance = data.get("data", data) if isinstance(data, dict) else data

        return {
            "instance_id": instance_id,
            "status": self._normalize_status(instance.get("status", "unknown")),
            "raw_status": instance.get("status", "unknown"),
            "instance_type": instance.get("gpu_type", "unknown"),
            "region": instance.get("region", "unknown"),
            "provider": "packet",
            "public_ip": instance.get("public_ip", instance.get("ip", "")),
            "spot": instance.get("tier", "") == "dynamic",
            "metadata": {
                "ssh": instance.get("ssh", ""),
            },
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Packet.ai API key not configured")
        await self._make_request("POST", f"{self.api_base}/instances/{instance_id}/stop")
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Packet.ai API key not configured")
        await self._make_request("POST", f"{self.api_base}/instances/{instance_id}/start")
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Packet.ai API key not configured")
        await self._make_request("DELETE", f"{self.api_base}/instances/{instance_id}")
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request("GET", f"{self.api_base}/instances")
            instances = data if isinstance(data, list) else data.get("data", [])

            return [
                {
                    "instance_id": str(inst.get("id", inst.get("instance_id", "unknown"))),
                    "status": self._normalize_status(inst.get("status", "unknown")),
                    "instance_type": inst.get("gpu_type", "unknown"),
                    "region": inst.get("region", "unknown"),
                    "provider": "packet",
                    "public_ip": inst.get("public_ip", inst.get("ip", "")),
                    "spot": inst.get("tier", "") == "dynamic",
                }
                for inst in (instances if isinstance(instances, list) else [])
            ]
        except Exception:  # noqa: BLE001
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Packet instances are SSH-managed; raw command execution is not exposed."""
        return {
            "instance_id": instance_id,
            "command": command,
            "exit_code": 1,
            "output": "Use packet ssh or the instance SSH endpoint for remote commands",
            "async": async_exec,
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_status(status: str) -> str:
        mapping = {
            "creating": "provisioning",
            "provisioning": "provisioning",
            "starting": "starting",
            "running": "running",
            "active": "running",
            "ready": "running",
            "stopping": "stopping",
            "stopped": "stopped",
            "terminating": "terminating",
            "terminated": "terminated",
            "deleted": "terminated",
            "failed": "error",
            "error": "error",
        }
        return mapping.get(status.lower(), status.lower())

    async def get_gpu_types(self) -> List[Dict[str, Any]]:
        """List Packet GPU types and pricing."""
        return await self._make_request("GET", f"{self.api_base}/gpus")

    async def get_instance_logs(self, instance_id: str) -> Dict[str, Any]:
        """Retrieve cloud-init / console logs for an instance."""
        return await self._make_request("GET", f"{self.api_base}/instances/{instance_id}/logs")
