#!/usr/bin/env python3
"""
Modal Provider - Modal serverless GPU integration

API: https://modal.com/docs/sdk/py/latest (Python SDK) / https://modal.com/docs/guide/gpu
Auth: Modal token ID + token secret, or combined API key formatted as `token_id.token_secret`
      Tokens may be passed as `Authorization: Bearer {token_id}.{token_secret}`
      or as separate `Modal-Key` and `Modal-Secret` headers.

Modal is a serverless GPU platform.  In Terradev the "instance" abstraction maps to
a deployed Modal App or Function, and "provisioning" deploys a GPU-backed App.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class ModalProvider(BaseProvider):
    """Modal serverless GPU provider — maps App/Function lifecycle to BaseProvider."""

    # Modal uses a gRPC/REST hybrid backend.  The public HTTP surface used here is
    # inferred from the Modal web-function auth scheme and CLI/SKDs.  Real apps may
    # also be deployed through the `modal` Python SDK; this provider keeps to HTTP
    # so the provider architecture and drift tests remain uniform.
    API_BASE = "https://api.modal.com/v1"

    # Default container image for GPU workloads when none is supplied.
    DEFAULT_IMAGE = "modalai/pytorch:2.6.0-py3.11-cuda12.8.1-cudnn9-devel-ubuntu22.04"

    # Map Terradev canonical GPU names → Modal `gpu` strings.
    # Sources: https://modal.com/docs/guide/gpu and Modal SDK reference.
    GPU_TYPE_MAP = {
        "H100-80GB": "H100",
        "H100": "H100",
        "H100-SXM": "H100",
        "H100-PCIe": "H100",
        "H200-141GB": "H200",
        "H200": "H200",
        "B200-192GB": "B200",
        "B200": "B200",
        "B300-262GB": "B300",
        "B300": "B300",
        "A100-80GB": "A100-80GB",
        "A100-40GB": "A100-40GB",
        "A100": "A100-80GB",
        "A100-PCIe": "A100-80GB",
        "A100-PCIe-80G": "A100-80GB",
        "A100-PCIe-40G": "A100-40GB",
        "L40S-48GB": "L40S",
        "L40S": "L40S",
        "L40-48GB": "L40",
        "L40": "L40",
        "L4-24GB": "L4",
        "L4": "L4",
        "A10G-24GB": "A10G",
        "A10G": "A10G",
        "A10-24GB": "A10",
        "A10": "A10",
        "T4-16GB": "T4",
        "T4": "T4",
        "RTX-Pro-6000-96GB": "RTX-Pro-6000",
        "RTXPro6000": "RTX-Pro-6000",
        "RTX-4090": "RTX-4090",
        "RTX4090": "RTX-4090",
        "RTX-3090": "RTX-3090",
        "RTX3090": "RTX-3090",
    }

    # Reference serverless GPU pricing (USD/hr) — live API takes precedence.
    # These are rough market estimates; update as Modal publishes price lists.
    GPU_PRICING = {
        "H100": {"price": 2.99, "mem_gb": 80, "vcpus": 16, "ram_gb": 64},
        "H200": {"price": 4.49, "mem_gb": 141, "vcpus": 16, "ram_gb": 64},
        "B200": {"price": 6.99, "mem_gb": 192, "vcpus": 24, "ram_gb": 128},
        "B300": {"price": 8.99, "mem_gb": 262, "vcpus": 32, "ram_gb": 192},
        "A100-80GB": {"price": 1.89, "mem_gb": 80, "vcpus": 16, "ram_gb": 64},
        "A100-40GB": {"price": 1.29, "mem_gb": 40, "vcpus": 12, "ram_gb": 48},
        "L40S": {"price": 1.19, "mem_gb": 48, "vcpus": 16, "ram_gb": 64},
        "L40": {"price": 0.99, "mem_gb": 48, "vcpus": 12, "ram_gb": 48},
        "L4": {"price": 0.45, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
        "A10G": {"price": 0.75, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
        "A10": {"price": 0.70, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
        "T4": {"price": 0.35, "mem_gb": 16, "vcpus": 4, "ram_gb": 16},
        "RTX-Pro-6000": {"price": 1.49, "mem_gb": 96, "vcpus": 16, "ram_gb": 64},
        "RTX-4090": {"price": 0.69, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
        "RTX-3090": {"price": 0.29, "mem_gb": 24, "vcpus": 8, "ram_gb": 32},
    }

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "modal"
        self.token_id = credentials.get("token_id", "")
        self.token_secret = credentials.get("token_secret", "")
        # Allow passing a single combined key
        combined = credentials.get("api_key", "")
        if combined and not (self.token_id and self.token_secret):
            if "." in combined:
                self.token_id, self.token_secret = combined.split(".", 1)
            else:
                self.token_id = combined

    # ── Authentication ─────────────────────────────────────────────────

    def _get_auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.token_id and self.token_secret:
            headers["Authorization"] = f"Bearer {self.token_id}.{self.token_secret}"
        elif self.token_id:
            headers["Authorization"] = f"Bearer {self.token_id}"
        return headers

    # ── GPU type resolution ─────────────────────────────────────────────

    def _resolve_gpu_type(self, gpu_type: str) -> str:
        """Map a Terradev canonical GPU name to a Modal gpu string."""
        if gpu_type in self.GPU_TYPE_MAP:
            return self.GPU_TYPE_MAP[gpu_type]
        upper = gpu_type.upper()
        for key, val in self.GPU_TYPE_MAP.items():
            if key.upper() == upper:
                return val
        for key, val in self.GPU_TYPE_MAP.items():
            if upper in key.upper() or key.upper() in upper:
                return val
        return gpu_type

    # ── Capacity / Quotes ───────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not (self.token_id and self.token_secret):
            return []

        modal_gpu = self._resolve_gpu_type(gpu_type)

        try:
            live = await self._get_live_gpu_types(gpu_type, modal_gpu, region)
            if live:
                return live
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Modal API error: {e}")

        info = self.GPU_PRICING.get(modal_gpu)
        if not info:
            return []

        return [
            {
                "instance_type": f"modal-{modal_gpu}",
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": region or "us-east",
                "available": True,
                "provider": "modal",
                "vcpus": info["vcpus"],
                "memory_gb": info["ram_gb"],
                "gpu_memory_gb": info["mem_gb"],
                "gpu_count": 1,
                "spot": False,
                "serverless": True,
            }
        ]

    async def _get_live_gpu_types(
        self, gpu_type: str, modal_gpu: str, region: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Query GET /v1/gpu-types for live availability."""
        data = await self._make_request("GET", f"{self.API_BASE}/gpu-types")
        gpu_upper = modal_gpu.upper()
        quotes = []

        for entry in data if isinstance(data, list) else data.get("data", []):
            entry_type = entry.get("gpu", entry.get("gpu_type", "")).upper()
            if gpu_upper not in entry_type and entry_type not in gpu_upper:
                continue

            for r in entry.get("regions", [region or "us-east"]):
                target_region = r if isinstance(r, str) else r.get("region", "us-east")
                if region and target_region != region:
                    continue
                price = entry.get("price_per_hour", entry.get("price", 0))
                quotes.append(
                    {
                        "instance_type": f"modal-{modal_gpu}",
                        "gpu_type": gpu_type,
                        "price_per_hour": float(price) if price else 0.0,
                        "region": target_region,
                        "available": entry.get("available", True),
                        "provider": "modal",
                        "vcpus": entry.get("vcpus", 8),
                        "memory_gb": entry.get("ram_gb", 32),
                        "gpu_memory_gb": entry.get("gpu_memory_gb", 0),
                        "gpu_count": 1,
                        "spot": False,
                        "serverless": True,
                    }
                )

        return sorted(quotes, key=lambda q: q["price_per_hour"])

    # ── Provisioning ────────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str, ssh_public_key: str = ""
    ) -> Dict[str, Any]:
        if not (self.token_id and self.token_secret):
            raise Exception("Modal token id and secret not configured")

        modal_gpu = self._resolve_gpu_type(gpu_type)
        image = self.credentials.get("image", self.DEFAULT_IMAGE)
        app_name = f"terradev-{modal_gpu.lower().replace('_', '-')}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        body: Dict[str, Any] = {
            "name": app_name,
            "image": image,
            "gpu": modal_gpu,
            "region": region or "us-east",
            "environment": {
                "PROVIDER": "modal",
                "PYTHONUNBUFFERED": "1",
            },
        }
        if ssh_public_key:
            body["ssh_public_key"] = ssh_public_key

        data = await self._make_request("POST", f"{self.API_BASE}/apps", json=body)
        app = data.get("data", data) if isinstance(data, dict) else data

        return {
            "instance_id": str(app.get("id", app.get("app_id", app_name))),
            "instance_type": instance_type,
            "region": region or "us-east",
            "gpu_type": gpu_type,
            "status": app.get("status", "provisioning"),
            "provider": "modal",
            "serverless": True,
            "metadata": {
                "name": app.get("name", app_name),
                "image": image,
                "gpu": modal_gpu,
                "endpoint": app.get("endpoint", ""),
            },
        }

    # ── Instance management ─────────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not (self.token_id and self.token_secret):
            raise Exception("Modal token id and secret not configured")

        data = await self._make_request("GET", f"{self.API_BASE}/apps/{instance_id}")
        app = data.get("data", data) if isinstance(data, dict) else data

        status_map = {
            "provisioning": "provisioning",
            "running": "running",
            "deploying": "provisioning",
            "stopping": "stopping",
            "stopped": "stopped",
            "terminating": "terminating",
            "terminated": "terminated",
            "failed": "error",
        }
        raw_status = app.get("status", "unknown")

        return {
            "instance_id": instance_id,
            "status": status_map.get(raw_status, raw_status.lower()),
            "raw_status": raw_status,
            "instance_type": app.get("gpu", "unknown"),
            "region": app.get("region", "unknown"),
            "provider": "modal",
            "endpoint": app.get("endpoint", ""),
            "serverless": True,
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not (self.token_id and self.token_secret):
            raise Exception("Modal token id and secret not configured")
        try:
            await self._make_request("POST", f"{self.API_BASE}/apps/{instance_id}/stop")
        except Exception:  # noqa: BLE001
            pass
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not (self.token_id and self.token_secret):
            raise Exception("Modal token id and secret not configured")
        try:
            await self._make_request("POST", f"{self.API_BASE}/apps/{instance_id}/start")
        except Exception:  # noqa: BLE001
            pass
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not (self.token_id and self.token_secret):
            raise Exception("Modal token id and secret not configured")
        await self._make_request("DELETE", f"{self.API_BASE}/apps/{instance_id}")
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not (self.token_id and self.token_secret):
            return []
        try:
            data = await self._make_request("GET", f"{self.API_BASE}/apps")
            apps = data if isinstance(data, list) else data.get("data", [])

            status_map = {
                "provisioning": "provisioning",
                "running": "running",
                "deploying": "provisioning",
                "stopping": "stopping",
                "stopped": "stopped",
                "terminating": "terminating",
                "terminated": "terminated",
                "failed": "error",
            }
            return [
                {
                    "instance_id": str(app.get("id", app.get("app_id", "unknown"))),
                    "status": status_map.get(app.get("status", ""), app.get("status", "unknown").lower()),
                    "instance_type": app.get("gpu", "unknown"),
                    "region": app.get("region", "unknown"),
                    "provider": "modal",
                    "endpoint": app.get("endpoint", ""),
                    "serverless": True,
                }
                for app in (apps if isinstance(apps, list) else [])
            ]
        except Exception:  # noqa: BLE001
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Modal is serverless — commands are not supported."""
        return {
            "instance_id": instance_id,
            "command": command,
            "exit_code": 1,
            "output": "Modal serverless Apps do not support raw SSH command execution",
            "async": async_exec,
        }

    # ── Modal-specific helpers ──────────────────────────────────────────

    async def list_gpu_types(self) -> List[Dict[str, Any]]:
        """List available Modal GPU types and pricing."""
        return await self._make_request("GET", f"{self.API_BASE}/gpu-types")

    async def get_app_logs(self, instance_id: str) -> Dict[str, Any]:
        """Retrieve logs for an App."""
        return await self._make_request("GET", f"{self.API_BASE}/apps/{instance_id}/logs")
