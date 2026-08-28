#!/usr/bin/env python3
"""
Gcore Provider — Gcore Cloud GPU integration
BYOAPI: Uses the end-client's Gcore permanent API token
API: https://api.gcore.com

Auth: Authorization: APIKey {token}
Compute model: VM instances in project/region scope
Strengths: Clean single-key REST API, global edge network,
           strong EU/MENA presence, OpenAPI spec available
Docs: https://docs.gcore.com/api-reference/cloud
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class GcoreProvider(BaseProvider):
    """Gcore Cloud provider — VM-based GPU and general compute instances."""

    API_BASE = "https://api.gcore.com"
    CLOUD_BASE = "https://api.gcore.com/cloud"

    # Map Terradev canonical GPU names → Gcore flavor substring hints.
    # Gcore flavor names contain GPU info like "g1-gpu-2-48-rtx3090" etc.
    GPU_PLAN_MAP = {
        "H100": "h100",
        "H100-80GB": "h100",
        "A100": "a100",
        "A100-80GB": "a100-80",
        "A100-40GB": "a100-40",
        "L40S": "l40s",
        "L4": "l4",
        "RTX3090": "rtx3090",
        "RTX4090": "rtx4090",
        "V100": "v100",
        "T4": "t4",
    }

    # Reference pricing (USD/hr per GPU) — live Gcore flavor prices take precedence.
    # These are conservative placeholders until the API returns real prices.
    GPU_PRICING = {
        "H100": {"price": 3.50, "mem_gb": 80, "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "H100-80GB": {"price": 3.50, "mem_gb": 80, "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "A100": {"price": 2.20, "mem_gb": 80, "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "A100-80GB": {"price": 2.20, "mem_gb": 80, "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "A100-40GB": {"price": 1.50, "mem_gb": 40, "vcpus": 12, "ram_gb": 96, "gpu_count": 1},
        "L40S": {"price": 1.40, "mem_gb": 48, "vcpus": 16, "ram_gb": 128, "gpu_count": 1},
        "L4": {"price": 0.60, "mem_gb": 24, "vcpus": 8, "ram_gb": 64, "gpu_count": 1},
        "RTX3090": {"price": 0.40, "mem_gb": 24, "vcpus": 8, "ram_gb": 32, "gpu_count": 1},
        "RTX4090": {"price": 0.70, "mem_gb": 24, "vcpus": 8, "ram_gb": 32, "gpu_count": 1},
        "V100": {"price": 1.00, "mem_gb": 32, "vcpus": 12, "ram_gb": 96, "gpu_count": 1},
        "T4": {"price": 0.45, "mem_gb": 16, "vcpus": 8, "ram_gb": 32, "gpu_count": 1},
    }

    # Regions with known Gcore GPU capacity. Region IDs are discovered at runtime.
    DEFAULT_REGIONS = {
        "luxembourg-1",
        "luxembourg-2",
        "amsterdam-1",
        "frankfurt-1",
        "istanbul-1",
        "dubai-1",
        "singapore-1",
        "hongkong-1",
        "tokyo-1",
        "sydney-1",
        "santiago-1",
        "washington-1",
        "chicago-1",
        "santa-clara-1",
    }

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "gcore"
        self.api_key = credentials.get("api_key", "")
        self.project_id = credentials.get("project_id", "")
        self.region_id = credentials.get("region_id", "")
        self.region_name = credentials.get("region", "")
        # Gcore creation is asynchronous; poll tasks if needed.
        self._last_task_id: Optional[str] = None

    # ── Authentication ────────────────────────────────────────────────

    def _get_auth_headers(self) -> Dict[str, str]:
        """Gcore uses a non-Bearer APIKey prefix."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"APIKey {self.api_key}"
        return headers

    # ── Scope discovery ───────────────────────────────────────────────

    async def _resolve_project_id(self) -> Optional[str]:
        """Discover project ID from /cloud/v1/projects if not configured."""
        if self.project_id:
            return str(self.project_id)
        if not self.api_key:
            return None
        try:
            data = await self._make_request(
                "GET", f"{self.CLOUD_BASE}/v1/projects"
            )
            results = data.get("results", []) if isinstance(data, dict) else data
            if results:
                first = results[0]
                self.project_id = str(first.get("id", first.get("project_id", "")))
                return self.project_id
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Gcore project discovery failed: {e}")
        return None

    async def _resolve_region_id(self) -> Optional[str]:
        """Discover a GPU-capable region from /cloud/v1/regions if not configured."""
        if self.region_id:
            return str(self.region_id)
        if not self.api_key:
            return None
        try:
            data = await self._make_request(
                "GET", f"{self.CLOUD_BASE}/v1/regions", params={"limit": "50"}
            )
            results = data.get("results", []) if isinstance(data, dict) else data
            for region in results or []:
                if not region.get("has_kvm"):
                    continue
                region_name = str(region.get("name", region.get("region", ""))).lower()
                if self.region_name and self.region_name.lower() not in region_name:
                    continue
                self.region_id = str(region.get("id", region.get("region_id", "")))
                self.region_name = str(region.get("name", region_name))
                return self.region_id
            # Fallback to first available region
            if results:
                first = results[0]
                self.region_id = str(first.get("id", first.get("region_id", "")))
                self.region_name = str(first.get("name", ""))
                return self.region_id
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Gcore region discovery failed: {e}")
        return None

    async def _get_scope(self) -> tuple[Optional[str], Optional[str]]:
        """Return resolved (project_id, region_id)."""
        project_id = await self._resolve_project_id()
        region_id = await self._resolve_region_id()
        return project_id, region_id

    # ── URL helpers ───────────────────────────────────────────────────

    def _url(self, *parts: str, **params: Any) -> str:
        """Build a cloud URL from path segments, ignoring None params."""
        path = "/".join(str(p).strip("/") for p in parts if p is not None)
        return f"{self.CLOUD_BASE}/v1/{path}"

    # ── Capacity / Quotes ─────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        project_id, region_id = await self._get_scope()
        if not project_id or not region_id:
            return []

        try:
            live = await self._get_live_flavors(gpu_type, project_id, region_id)
            if live:
                return live
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Gcore live quote error: {e}")

        # Static fallback
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            for key, val in self.GPU_PRICING.items():
                if gpu_type.upper() in key.upper() or key.upper() in gpu_type.upper():
                    info = val
                    break
        if not info:
            return []

        return [
            {
                "instance_type": self.GPU_PLAN_MAP.get(gpu_type, gpu_type.lower()),
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": region or self.region_name or f"region-{region_id}",
                "available": True,
                "provider": "gcore",
                "vcpus": info["vcpus"],
                "memory_gb": info["ram_gb"],
                "gpu_memory_gb": info["mem_gb"],
                "gpu_count": info["gpu_count"],
                "spot": False,
            }
        ]

    async def _get_live_flavors(
        self, gpu_type: str, project_id: str, region_id: str
    ) -> List[Dict[str, Any]]:
        """Query Gcore flavors and filter by GPU type, including prices."""
        data = await self._make_request(
            "GET",
            self._url("flavors", project_id, region_id),
            params={"include_prices": "true"},
        )
        flavors = data.get("results", []) if isinstance(data, dict) else data
        if not flavors:
            return []

        gpu_hint = self.GPU_PLAN_MAP.get(gpu_type, gpu_type.lower())
        gpu_upper = gpu_type.upper()
        quotes: List[Dict[str, Any]] = []

        for flavor in flavors or []:
            flavor_name = str(flavor.get("flavor_name", flavor.get("name", ""))).lower()
            gpu_in_flavor = any(
                h in flavor_name
                for h in (gpu_hint.lower(), gpu_type.lower())
            ) or gpu_upper.replace("-", " ") in flavor_name.upper()
            if not gpu_in_flavor:
                continue

            price = flavor.get("price_per_hour")
            if price is None:
                price = flavor.get("price_per_month", 0) / 720
            price = float(price or 0)

            ram_mb = flavor.get("ram", 0) or 0
            ram_gb = ram_mb / 1024 if ram_mb else 0
            vcpus = flavor.get("vcpus", 0) or flavor.get("cpu", 0) or 0

            # Gcore flavor names encode GPU count sometimes.
            gpu_count = 1
            if "x2" in flavor_name or " 2x" in flavor_name:
                gpu_count = 2
            elif "x4" in flavor_name or " 4x" in flavor_name:
                gpu_count = 4
            elif "x8" in flavor_name or " 8x" in flavor_name:
                gpu_count = 8

            quote_region = region or self.region_name or f"region-{region_id}"
            quotes.append(
                {
                    "instance_type": flavor.get("flavor_name", flavor.get("name", gpu_type)),
                    "gpu_type": gpu_type,
                    "price_per_hour": round(price, 4),
                    "region": quote_region,
                    "available": flavor.get("available", True),
                    "provider": "gcore",
                    "vcpus": vcpus,
                    "memory_gb": round(ram_gb, 1),
                    "gpu_count": gpu_count,
                    "spot": False,
                    "flavor_id": flavor.get("flavor_id", flavor.get("id", "")),
                }
            )

        return sorted(quotes, key=lambda q: q["price_per_hour"])

    # ── Provisioning ──────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str, ssh_public_key: str = ""
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Gcore API key not configured")

        project_id, region_id = await self._get_scope()
        if not project_id or not region_id:
            raise Exception("Gcore project_id and region_id could not be resolved")

        # Discover an image if the user did not supply one.
        image_id = self.credentials.get("image_id", "")
        if not image_id:
            image_id = await self._discover_gpu_image(project_id, region_id)

        name = f"terradev-{gpu_type.lower().replace('_', '-')}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        body: Dict[str, Any] = {
            "names": [name],
            "flavor": instance_type,
            "image": image_id,
        }

        keypair = self.credentials.get("ssh_key_name", "")
        if keypair:
            body["keypair_name"] = keypair
        if ssh_public_key:
            # Gcore uses named keypairs; public key itself is not accepted inline.
            body.setdefault("keypair_name", "terradev-key")

        volumes = self.credentials.get("volumes")
        if volumes:
            body["volumes"] = volumes

        data = await self._make_request(
            "POST",
            self._url("instances", project_id, region_id),
            json=body,
        )

        task_id = None
        if isinstance(data, dict):
            tasks = data.get("tasks", [])
            task_id = tasks[0] if tasks else None

        # Gcore returns 200 with a task, not the instance object directly.
        instance_id = f"gcore-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if task_id:
            self._last_task_id = str(task_id)
            instance_id = str(task_id)

        return {
            "instance_id": instance_id,
            "instance_type": instance_type,
            "region": region or self.region_name or f"region-{region_id}",
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "gcore",
            "task_id": self._last_task_id,
            "metadata": {
                "name": name,
                "image_id": image_id,
                "project_id": project_id,
                "region_id": region_id,
            },
        }

    async def _discover_gpu_image(self, project_id: str, region_id: str) -> str:
        """Pick a recent Ubuntu image for GPU workloads."""
        try:
            data = await self._make_request(
                "GET",
                self._url("images", project_id, region_id),
                params={"os_distro": "ubuntu", "limit": "50"},
            )
            images = data.get("results", []) if isinstance(data, dict) else data
            for image in images or []:
                name = str(image.get("name", "")).lower()
                if "ubuntu" in name and ("22.04" in name or "24.04" in name or "20.04" in name):
                    return str(image.get("id", image.get("image_id", "")))
            if images:
                return str(images[0].get("id", images[0].get("image_id", "")))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Gcore image discovery failed: {e}")
        return ""

    # ── Instance management ───────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Gcore API key not configured")

        project_id, region_id = await self._get_scope()
        if not project_id or not region_id:
            raise Exception("Gcore project_id and region_id could not be resolved")

        # If the instance_id looks like a task, poll the task endpoint first.
        if instance_id.startswith("gcore-"):
            # Task or synthetic ID — fall back to list search.
            instances = await self.list_instances()
            for inst in instances:
                if inst.get("instance_id") == instance_id:
                    return inst
            return {
                "instance_id": instance_id,
                "status": "provisioning",
                "provider": "gcore",
            }

        try:
            data = await self._make_request(
                "GET",
                self._url("instances", project_id, region_id, instance_id),
            )
            return self._normalize_instance(data, instance_id)
        except Exception as e:  # noqa: BLE001
            return {
                "instance_id": instance_id,
                "status": "unknown",
                "provider": "gcore",
                "error": str(e),
            }

    def _normalize_instance(self, data: Dict[str, Any], instance_id: str) -> Dict[str, Any]:
        """Normalize a Gcore instance object into the Terradev contract."""
        if not isinstance(data, dict):
            data = {}

        status = str(data.get("status", "unknown")).lower()
        if status in ("active", "running"):
            status = "running"
        elif status in ("stopped", "shutoff"):
            status = "stopped"

        addresses = data.get("addresses", {}) or {}
        public_ip = ""
        for _net, addrs in addresses.items():
            for addr in addrs or []:
                if addr.get("type") == "fixed":
                    public_ip = addr.get("addr", "")
                if addr.get("type") == "floating" and not public_ip:
                    public_ip = addr.get("addr", "")

        flavor = data.get("flavor", {}) or {}
        flavor_name = flavor.get("flavor_name", flavor.get("name", "unknown"))

        return {
            "instance_id": instance_id,
            "status": status,
            "instance_type": flavor_name,
            "region": data.get("region", self.region_name or "unknown"),
            "provider": "gcore",
            "public_ip": public_ip,
            "private_ip": public_ip,
            "gpu_count": 1,
            "metadata": data,
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Gcore API key not configured")
        project_id, region_id = await self._get_scope()
        await self._make_request(
            "POST",
            self._url("instances", project_id, region_id, instance_id, "stop"),
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Gcore API key not configured")
        project_id, region_id = await self._get_scope()
        await self._make_request(
            "POST",
            self._url("instances", project_id, region_id, instance_id, "start"),
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("Gcore API key not configured")
        project_id, region_id = await self._get_scope()
        await self._make_request(
            "DELETE",
            self._url("instances", project_id, region_id, instance_id),
        )
        return {
            "instance_id": instance_id,
            "action": "terminate",
            "status": "terminating",
        }

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        project_id, region_id = await self._get_scope()
        if not project_id or not region_id:
            return []

        try:
            data = await self._make_request(
                "GET",
                self._url("instances", project_id, region_id),
            )
            instances = data.get("results", []) if isinstance(data, dict) else data
            return [
                self._normalize_instance(inst, str(inst.get("id", inst.get("instance_id", "unknown"))))
                for inst in (instances if isinstance(instances, list) else [])
            ]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Gcore list_instances error: {e}")
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute a command via SSH on the instance public IP."""
        if not self.api_key:
            raise Exception("Gcore API key not configured")

        status = await self.get_instance_status(instance_id)
        public_ip = status.get("public_ip")
        if not public_ip:
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": "No public IP — instance may still be provisioning",
                "async": async_exec,
            }

        try:
            import subprocess

            user = self.credentials.get("ssh_user", "ubuntu")
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={__import__('os').path.expanduser('~/.terradev/known_hosts')}",
                "-o", "ConnectTimeout=10",
                f"{user}@{public_ip}",
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
                "output": f"Gcore exec error: {e}",
                "async": async_exec,
            }

    # ── Gcore-specific helpers ────────────────────────────────────────

    async def list_projects(self) -> List[Dict[str, Any]]:
        """List Gcore projects."""
        if not self.api_key:
            return []
        data = await self._make_request("GET", f"{self.CLOUD_BASE}/v1/projects")
        return data.get("results", []) if isinstance(data, dict) else data

    async def list_regions(self) -> List[Dict[str, Any]]:
        """List Gcore regions."""
        if not self.api_key:
            return []
        data = await self._make_request(
            "GET", f"{self.CLOUD_BASE}/v1/regions", params={"limit": "50"}
        )
        return data.get("results", []) if isinstance(data, dict) else data
