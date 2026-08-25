#!/usr/bin/env python3
"""
Yotta Labs Provider - Yotta Labs / Shakti Cloud GPU integration
BYOAPI: Uses the end-client's Yotta Labs API key
API: https://api.yottalabs.ai/v2

Auth: X-Api-Key header (NOT Bearer token — Yotta Labs specific)
Compute model: Pod-based containers (similar to RunPod)
Strengths: India-region GPU cloud, H100/A100/RTX4090, pay-per-second billing
Docs: https://docs.yottalabs.ai/api-and-sdk/api-keys
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class YottaLabsProvider(BaseProvider):
    """Yotta Labs (Shakti Cloud) provider — pod-based GPU compute"""

    API_BASE = "https://api.yottalabs.ai/v2"

    # Default container image for GPU workloads
    DEFAULT_IMAGE = "yottalabsai/pytorch:2.9.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"

    # Map Terradev canonical GPU names → Yotta Labs gpuType strings
    # Source: https://docs.yottalabs.ai/api-and-sdk/api-guides (GPU Types Reference)
    GPU_TYPE_MAP = {
        "H100":        "NVIDIA_H100_80G",
        "H100-80GB":   "NVIDIA_H100_80G",
        "H100-SXM":    "NVIDIA_H100_80G",
        "H100-PCIe":   "NVIDIA_H100_PCIe_80G",
        "H200":        "NVIDIA_H200_141G",
        "B200":        "NVIDIA_B200_180G",
        "B300":        "NVIDIA_B300_262G",
        "A100":        "NVIDIA_A100_80G",
        "A100-80GB":   "NVIDIA_A100_80G",
        "A100-PCIe":   "NVIDIA_A100_PCIe_80G",
        "A100-40GB":   "NVIDIA_A100_PCIe_40G",
        "RTX4090":     "NVIDIA_RTX_4090_24G",
        "RTX5090":     "NVIDIA_RTX_5090_32G",
        "RTX3090":     "NVIDIA_RTX_3090_24G",
        "L40S":        "NVIDIA_L40S_48G",
        "L40":         "NVIDIA_L40_48G",
        "A6000":       "NVIDIA_RTX_A6000_48G",
        "RTX6000Ada":  "NVIDIA_RTX_6000_Ada_48G",
        "RTXPro6000":  "NVIDIA_RTX_PRO_6000_96G",
        "MI300X":      "AMD_MI300X_192G",
        "T4":          "NVIDIA_T4_16G",
        "V100":        "NVIDIA_V100_16G",
    }

    # Reference pricing (USD/hr per GPU) — live API takes precedence
    # Keys must exactly match the gpuType strings from GPU_TYPE_MAP / Yotta API reference
    GPU_PRICING = {
        "NVIDIA_H100_80G":          {"price": 2.99, "mem_gb": 80,  "vcpus": 16, "ram_gb": 64},
        "NVIDIA_H100_PCIe_80G":     {"price": 2.49, "mem_gb": 80,  "vcpus": 16, "ram_gb": 64},
        "NVIDIA_H200_141G":         {"price": 4.49, "mem_gb": 141, "vcpus": 16, "ram_gb": 64},
        "NVIDIA_B200_180G":         {"price": 6.99, "mem_gb": 180, "vcpus": 24, "ram_gb": 128},
        "NVIDIA_B300_262G":         {"price": 8.99, "mem_gb": 262, "vcpus": 32, "ram_gb": 192},
        "NVIDIA_A100_80G":          {"price": 1.89, "mem_gb": 80,  "vcpus": 16, "ram_gb": 64},
        "NVIDIA_A100_PCIe_80G":     {"price": 1.79, "mem_gb": 80,  "vcpus": 16, "ram_gb": 64},
        "NVIDIA_A100_PCIe_40G":     {"price": 1.29, "mem_gb": 40,  "vcpus": 12, "ram_gb": 48},
        "NVIDIA_A100_40G":          {"price": 1.29, "mem_gb": 40,  "vcpus": 12, "ram_gb": 48},
        "NVIDIA_RTX_4090_24G":      {"price": 0.69, "mem_gb": 24,  "vcpus": 8,  "ram_gb": 32},
        "NVIDIA_RTX_5090_32G":      {"price": 0.99, "mem_gb": 32,  "vcpus": 8,  "ram_gb": 32},
        "NVIDIA_RTX_3090_24G":      {"price": 0.29, "mem_gb": 24,  "vcpus": 8,  "ram_gb": 32},
        "NVIDIA_L40S_48G":          {"price": 1.19, "mem_gb": 48,  "vcpus": 16, "ram_gb": 64},
        "NVIDIA_L40_48G":           {"price": 0.99, "mem_gb": 48,  "vcpus": 12, "ram_gb": 48},
        "NVIDIA_RTX_A6000_48G":     {"price": 0.79, "mem_gb": 48,  "vcpus": 12, "ram_gb": 48},
        "NVIDIA_RTX_6000_Ada_48G":  {"price": 0.89, "mem_gb": 48,  "vcpus": 12, "ram_gb": 48},
        "NVIDIA_RTX_PRO_6000_96G":  {"price": 1.49, "mem_gb": 96,  "vcpus": 16, "ram_gb": 64},
        "AMD_MI300X_192G":          {"price": 3.49, "mem_gb": 192, "vcpus": 24, "ram_gb": 128},
        "NVIDIA_T4_16G":            {"price": 0.35, "mem_gb": 16,  "vcpus": 4,  "ram_gb": 16},
        "NVIDIA_V100_16G":          {"price": 0.55, "mem_gb": 16,  "vcpus": 8,  "ram_gb": 32},
    }

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "yottalabs"
        self.api_key = credentials.get("api_key", "")

    # ── Authentication ────────────────────────────────────────────────

    def _get_auth_headers(self) -> Dict[str, str]:
        # Yotta Labs uses X-Api-Key header per docs.yottalabs.ai/api-and-sdk/api-keys
        if self.api_key:
            return {
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json",
            }
        return {"Content-Type": "application/json"}

    # ── GPU type resolution ───────────────────────────────────────────

    def _resolve_gpu_type(self, gpu_type: str) -> str:
        """Map a Terradev canonical GPU name to a Yotta Labs gpuType string."""
        # Exact match first
        if gpu_type in self.GPU_TYPE_MAP:
            return self.GPU_TYPE_MAP[gpu_type]
        # Case-insensitive
        upper = gpu_type.upper()
        for key, val in self.GPU_TYPE_MAP.items():
            if key.upper() == upper:
                return val
        # Partial match (e.g. "H100" matches "H100-80GB")
        for key, val in self.GPU_TYPE_MAP.items():
            if upper in key.upper() or key.upper() in upper:
                return val
        # If already looks like a Yotta type, pass through
        return gpu_type

    # ── Capacity / Quotes ─────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        yotta_gpu_type = self._resolve_gpu_type(gpu_type)

        try:
            live = await self._get_live_gpu_types(gpu_type, yotta_gpu_type, region)
            if live:
                return live
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Yotta Labs API error: {e}")

        # Static fallback
        info = self.GPU_PRICING.get(yotta_gpu_type)
        if not info:
            return []

        return [
            {
                "instance_type": yotta_gpu_type,
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": region or "ap-south-1",
                "available": True,
                "provider": "yottalabs",
                "vcpus": info["vcpus"],
                "memory_gb": info["ram_gb"],
                "gpu_memory_gb": info["mem_gb"],
                "gpu_count": 1,
                "spot": False,
                "container_based": True,
            }
        ]

    async def _get_live_gpu_types(
        self, gpu_type: str, yotta_gpu_type: str, region: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Query GET /v2/gpu-types for live availability."""
        data = await self._make_request("GET", f"{self.API_BASE}/gpu-types")
        gpu_upper = yotta_gpu_type.upper()
        quotes = []

        for entry in data if isinstance(data, list) else data.get("data", []):
            entry_type = entry.get("gpuType", entry.get("gpu_type", "")).upper()
            if gpu_upper not in entry_type and entry_type not in gpu_upper:
                if not any(
                    part in entry_type for part in gpu_upper.split("_") if len(part) > 2
                ):
                    continue

            for r in entry.get("regions", [region or "ap-south-1"]):
                target_region = r if isinstance(r, str) else r.get("region", "ap-south-1")
                if region and target_region != region:
                    continue
                price = entry.get("pricePerHour", entry.get("price_per_hour", 0))
                quotes.append(
                    {
                        "instance_type": entry.get("gpuType", yotta_gpu_type),
                        "gpu_type": gpu_type,
                        "price_per_hour": float(price) if price else 0.0,
                        "region": target_region,
                        "available": entry.get("available", True),
                        "provider": "yottalabs",
                        "vcpus": entry.get("minSingleCardVcpu", 8),
                        "memory_gb": entry.get("minSingleCardRamInGb", 32),
                        "gpu_memory_gb": entry.get("gpuMemoryInGb", 0),
                        "gpu_count": 1,
                        "spot": False,
                        "container_based": True,
                    }
                )

        return sorted(quotes, key=lambda q: q["price_per_hour"])

    # ── Provisioning ──────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str, ssh_public_key: str = ""
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Yotta Labs API key not configured")

        yotta_gpu_type = self._resolve_gpu_type(gpu_type)
        image = self.credentials.get("image", self.DEFAULT_IMAGE)
        gpu_count = int(self.credentials.get("gpu_count", 1))
        volume_gb = int(self.credentials.get("volume_gb", 100))

        body: Dict[str, Any] = {
            "name": f"terradev-{gpu_type.lower().replace('_', '-')}-{datetime.now().strftime('%H%M%S')}",
            "image": image,
            "gpuType": yotta_gpu_type,
            "gpuCount": gpu_count,
            "containerVolumeInGb": volume_gb,
            "regions": [region] if region else ["ap-south-1"],
            "minSingleCardVcpu": 8,
            "minSingleCardRamInGb": 32,
            "environmentVars": [
                {"key": "PYTHONUNBUFFERED", "value": "1"},
                {"key": "PROVIDER", "value": "yottalabs"},
            ],
            "expose": [{"port": 8888, "protocol": "http"}],
        }

        data = await self._make_request("POST", f"{self.API_BASE}/pods", json=body)

        pod = data if isinstance(data, dict) else data.get("data", {})
        return {
            "instance_id": str(pod.get("id", f"yl-{datetime.now().strftime('%Y%m%d%H%M%S')}")),
            "instance_type": yotta_gpu_type,
            "region": region or "ap-south-1",
            "gpu_type": gpu_type,
            "status": pod.get("status", "Initialize"),
            "provider": "yottalabs",
            "metadata": {
                "name": pod.get("name", body["name"]),
                "image": image,
                "gpu_count": gpu_count,
                "volume_gb": volume_gb,
                "container_based": True,
            },
        }

    # ── Instance management ───────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Yotta Labs API key not configured")

        data = await self._make_request("GET", f"{self.API_BASE}/pods/{instance_id}")
        pod = data if isinstance(data, dict) else data.get("data", {})

        # Normalize Yotta status → Terradev status
        status_map = {
            "Initialize": "provisioning",
            "Running":    "running",
            "Stopping":   "stopping",
            "Stopped":    "stopped",
            "Terminating": "terminating",
            "Terminated": "terminated",
            "Failed":     "error",
        }
        raw_status = pod.get("status", "unknown")

        return {
            "instance_id": instance_id,
            "status": status_map.get(raw_status, raw_status.lower()),
            "raw_status": raw_status,
            "instance_type": pod.get("gpuType", "unknown"),
            "region": pod.get("region", "unknown"),
            "provider": "yottalabs",
            "public_ip": pod.get("publicIp"),
            "gpu_count": pod.get("gpuCount", 1),
            "container_based": True,
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Yotta Labs API key not configured")
        # Yotta Labs: pause = soft stop, volume storage persists
        await self._make_request("POST", f"{self.API_BASE}/pods/{instance_id}/pause")
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Yotta Labs API key not configured")
        await self._make_request("POST", f"{self.API_BASE}/pods/{instance_id}/run")
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Yotta Labs API key not configured")
        await self._make_request("DELETE", f"{self.API_BASE}/pods/{instance_id}")
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request("GET", f"{self.API_BASE}/pods")
            pods = data if isinstance(data, list) else data.get("data", [])

            status_map = {
                "Initialize": "provisioning",
                "Running":    "running",
                "Stopping":   "stopping",
                "Stopped":    "stopped",
                "Terminating": "terminating",
                "Terminated": "terminated",
                "Failed":     "error",
            }
            return [
                {
                    "instance_id": str(pod.get("id", "unknown")),
                    "status": status_map.get(pod.get("status", ""), pod.get("status", "unknown").lower()),
                    "instance_type": pod.get("gpuType", "unknown"),
                    "region": pod.get("region", "unknown"),
                    "provider": "yottalabs",
                    "public_ip": pod.get("publicIp"),
                    "gpu_count": pod.get("gpuCount", 1),
                    "container_based": True,
                }
                for pod in (pods if isinstance(pods, list) else [])
            ]
        except Exception:  # noqa: BLE001
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute command on pod via SSH (requires public IP and SSH key)."""
        if not self.api_key:
            raise Exception("Yotta Labs API key not configured")

        try:
            status = await self.get_instance_status(instance_id)
            public_ip = status.get("public_ip")
            if not public_ip:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 1,
                    "output": "No public IP — pod may still be initializing or have no SSH exposure",
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
        except Exception as e:  # noqa: BLE001
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"Yotta Labs exec error: {e}",
                "async": async_exec,
            }

    # ── Yotta Labs-specific helpers ───────────────────────────────────

    async def get_gpu_types(self) -> List[Dict[str, Any]]:
        """List all available GPU types and their availability."""
        return await self._make_request("GET", f"{self.API_BASE}/gpu-types")

    async def get_pod_logs(self, instance_id: str) -> Dict[str, Any]:
        """Retrieve logs for a pod (system + container)."""
        return await self._make_request("GET", f"{self.API_BASE}/pods/{instance_id}/logs")
