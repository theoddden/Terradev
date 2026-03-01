#!/usr/bin/env python3
"""
Hetzner Provider - Hetzner Cloud API + Robot API (dedicated GPU) integration
BYOAPI: Uses the end-client's Hetzner Cloud API token (and optional Robot credentials)
API: https://api.hetzner.cloud/v1 (Cloud) / https://robot-ws.your-server.de (Dedicated)

Strengths: Extreme value EU pricing, 100% renewable energy, GDPR, dedicated GPU (GEX44)
Auth: Cloud = Bearer token, Robot = HTTP Basic (user + password)
"""

import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class HetznerProvider(BaseProvider):
    """Hetzner Cloud + Dedicated GPU provider"""

    CLOUD_API = "https://api.hetzner.cloud/v1"
    ROBOT_API = "https://robot-ws.your-server.de"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "hetzner"
        self.api_token = credentials.get("api_token", "")
        # Robot API credentials for dedicated servers (GEX44)
        self.robot_user = credentials.get("robot_user", "")
        self.robot_password = credentials.get("robot_password", "")

    # ── Server type / pricing mapping ─────────────────────────────────
    # Cloud GPU types (if/when available) + Dedicated GEX lineup
    GPU_PRICING = {
        "RTX4000-SFF": {
            "server_type": "gex44",
            "api": "robot",
            "price_monthly": 69.90,
            "price": 0.10,  # approximate hourly
            "mem": 20,
            "vcpus": 14,
            "gpu_count": 1,
            "gpu_model": "NVIDIA RTX 4000 SFF Ada",
            "cpu": "Intel Core i5-13500",
            "ram_gb": 64,
            "storage": "2x 1.92TB NVMe SSD",
        },
    }

    # Hetzner Cloud server types that could be GPU-capable
    # (Hetzner may add cloud GPU types in the future)
    CLOUD_GPU_PREFIXES = ["gex", "gpu", "ccx"]

    LOCATIONS = [
        {"name": "fsn1", "city": "Falkenstein", "country": "DE"},
        {"name": "nbg1", "city": "Nuremberg", "country": "DE"},
        {"name": "hel1", "city": "Helsinki", "country": "FI"},
        {"name": "ash", "city": "Ashburn", "country": "US"},
        {"name": "hil", "city": "Hillsboro", "country": "US"},
    ]

    # ── Authentication ────────────────────────────────────────────────

    def _get_auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _get_robot_auth(self) -> Optional[Any]:
        """Return aiohttp.BasicAuth for Robot API."""
        if self.robot_user and self.robot_password:
            import aiohttp
            return aiohttp.BasicAuth(self.robot_user, self.robot_password)
        return None

    # ── Capacity / Quotes ─────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        quotes = []

        # Try Cloud API for GPU server types
        if self.api_token:
            try:
                cloud_quotes = await self._get_cloud_gpu_types(gpu_type, region)
                quotes.extend(cloud_quotes)
            except Exception as e:
                logger.debug(f"Hetzner Cloud API error: {e}")

        # Try Robot API for dedicated GPU servers
        if self.robot_user and self.robot_password:
            try:
                robot_quotes = await self._get_robot_gpu_products(gpu_type)
                quotes.extend(robot_quotes)
            except Exception as e:
                logger.debug(f"Hetzner Robot API error: {e}")

        # Static fallback
        if not quotes:
            info = self.GPU_PRICING.get(gpu_type)
            if not info:
                for key, val in self.GPU_PRICING.items():
                    if gpu_type.upper() in key.upper():
                        info = val
                        break
            if info:
                quotes.append(
                    {
                        "instance_type": info["server_type"],
                        "gpu_type": gpu_type,
                        "price_per_hour": info["price"],
                        "price_monthly": info.get("price_monthly"),
                        "region": region or "fsn1",
                        "available": True,
                        "provider": "hetzner",
                        "vcpus": info["vcpus"],
                        "memory_gb": info["mem"],
                        "gpu_count": info["gpu_count"],
                        "spot": False,
                        "dedicated": info.get("api") == "robot",
                    }
                )

        return sorted(quotes, key=lambda q: q.get("price_per_hour", 999))

    async def _get_cloud_gpu_types(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query Hetzner Cloud /server_types for GPU-capable types."""
        data = await self._make_request("GET", f"{self.CLOUD_API}/server_types")

        quotes = []
        gpu_lower = gpu_type.lower()
        for st in data.get("server_types", []):
            name = st.get("name", "").lower()
            desc = st.get("description", "").lower()
            # Filter for GPU types
            if not any(prefix in name for prefix in self.CLOUD_GPU_PREFIXES):
                if "gpu" not in desc:
                    continue
            if gpu_lower not in name and gpu_lower not in desc:
                continue

            # Check location availability
            available_locations = [
                p.get("location", {}).get("name", "")
                for p in st.get("prices", [])
            ]
            if region and region not in available_locations:
                continue

            for price_info in st.get("prices", []):
                loc = price_info.get("location", {}).get("name", "")
                if region and loc != region:
                    continue
                hourly = float(price_info.get("price_hourly", {}).get("gross", 0))
                monthly = float(price_info.get("price_monthly", {}).get("gross", 0))
                quotes.append(
                    {
                        "instance_type": st["name"],
                        "gpu_type": gpu_type,
                        "price_per_hour": hourly,
                        "price_monthly": monthly,
                        "region": loc,
                        "available": True,
                        "provider": "hetzner",
                        "vcpus": st.get("cores", 0),
                        "memory_gb": st.get("memory", 0),
                        "gpu_count": 1,
                        "spot": False,
                        "dedicated": False,
                    }
                )
        return quotes

    async def _get_robot_gpu_products(self, gpu_type: str) -> List[Dict[str, Any]]:
        """Query Robot API /order/server/product for dedicated GPU servers."""
        import aiohttp

        auth = self._get_robot_auth()
        if not auth:
            return []

        if not self.session:
            self.session = aiohttp.ClientSession()

        async with self.session.get(
            f"{self.ROBOT_API}/order/server/product",
            auth=auth,
            headers={"Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if response.status != 200:
                return []
            try:
                data = await response.json(content_type=None)
            except Exception:
                return []

        quotes = []
        gpu_upper = gpu_type.upper()
        for product in (data if isinstance(data, list) else data.get("product", [])):
            p = product.get("product", product)
            name = p.get("name", "").upper()
            desc = ", ".join(p.get("description", [])) if isinstance(p.get("description"), list) else str(p.get("description", ""))
            if "GPU" not in name and "GEX" not in name and "GPU" not in desc.upper():
                continue
            if gpu_upper not in name and gpu_upper not in desc.upper():
                # Accept any GPU product if searching generically
                if gpu_upper not in ("GPU", "RTX4000", "RTX4000-SFF", "GEX44"):
                    continue

            monthly = 0
            for price in p.get("prices", []):
                if price.get("currency") == "EUR":
                    monthly = float(price.get("value", 0))
                    break

            quotes.append(
                {
                    "instance_type": p.get("id", name),
                    "gpu_type": gpu_type,
                    "price_per_hour": monthly / 730 if monthly else 0.10,
                    "price_monthly": monthly,
                    "region": "fsn1",
                    "available": True,
                    "provider": "hetzner",
                    "vcpus": 14,
                    "memory_gb": 20,
                    "gpu_count": 1,
                    "spot": False,
                    "dedicated": True,
                }
            )
        return quotes

    # ── Provisioning ──────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        # Determine if this is a Cloud or Robot provisioning
        info = self.GPU_PRICING.get(gpu_type, {})
        if info.get("api") == "robot" or instance_type.startswith("gex"):
            return await self._provision_robot(instance_type, region, gpu_type)
        return await self._provision_cloud(instance_type, region, gpu_type)

    async def _provision_cloud(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_token:
            raise Exception("Hetzner Cloud API token not configured")

        # Get SSH keys
        ssh_keys = []
        try:
            key_data = await self._make_request("GET", f"{self.CLOUD_API}/ssh_keys")
            ssh_keys = [k["id"] for k in key_data.get("ssh_keys", [])]
        except Exception:
            pass

        body = {
            "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
            "server_type": instance_type,
            "image": "ubuntu-22.04",
            "location": region or "fsn1",
            "ssh_keys": ssh_keys,
            "labels": {"terradev": "true", "gpu_type": gpu_type.lower()},
        }

        data = await self._make_request(
            "POST", f"{self.CLOUD_API}/servers", json=body
        )

        server = data.get("server", {})
        return {
            "instance_id": str(server.get("id", f"hetzner-{datetime.now().strftime('%Y%m%d%H%M%S')}")),
            "instance_type": instance_type,
            "region": region,
            "gpu_type": gpu_type,
            "status": server.get("status", "provisioning"),
            "provider": "hetzner",
            "public_ip": server.get("public_net", {}).get("ipv4", {}).get("ip"),
            "metadata": {"root_password": data.get("root_password")},
        }

    async def _provision_robot(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        """Order a dedicated GPU server via Robot API."""
        import aiohttp

        auth = self._get_robot_auth()
        if not auth:
            raise Exception("Hetzner Robot credentials not configured for dedicated GPU")

        if not self.session:
            self.session = aiohttp.ClientSession()

        body = {
            "product_id": instance_type,
            "location": region or "FSN",
        }

        async with self.session.post(
            f"{self.ROBOT_API}/order/server/transaction",
            auth=auth,
            json=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"Hetzner Robot HTTP {response.status}: {error_text}")
            data = await response.json(content_type=None)

        transaction = data.get("transaction", data)
        return {
            "instance_id": str(transaction.get("id", f"hetzner-ded-{datetime.now().strftime('%Y%m%d%H%M%S')}")),
            "instance_type": instance_type,
            "region": region,
            "gpu_type": gpu_type,
            "status": "ordering",
            "provider": "hetzner",
            "dedicated": True,
            "metadata": {"transaction_id": transaction.get("id")},
        }

    # ── Instance management ───────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_token:
            raise Exception("Hetzner Cloud API token not configured")

        data = await self._make_request(
            "GET", f"{self.CLOUD_API}/servers/{instance_id}"
        )
        server = data.get("server", {})
        return {
            "instance_id": instance_id,
            "status": server.get("status", "unknown"),
            "instance_type": server.get("server_type", {}).get("name", "unknown"),
            "region": server.get("datacenter", {}).get("location", {}).get("name", "unknown"),
            "provider": "hetzner",
            "public_ip": server.get("public_net", {}).get("ipv4", {}).get("ip"),
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_token:
            raise Exception("Hetzner Cloud API token not configured")
        await self._make_request(
            "POST", f"{self.CLOUD_API}/servers/{instance_id}/actions/poweroff"
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_token:
            raise Exception("Hetzner Cloud API token not configured")
        await self._make_request(
            "POST", f"{self.CLOUD_API}/servers/{instance_id}/actions/poweron"
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_token:
            raise Exception("Hetzner Cloud API token not configured")
        await self._make_request(
            "DELETE", f"{self.CLOUD_API}/servers/{instance_id}"
        )
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_token:
            return []
        try:
            data = await self._make_request(
                "GET", f"{self.CLOUD_API}/servers?label_selector=terradev%3Dtrue"
            )
            return [
                {
                    "instance_id": str(s.get("id", "unknown")),
                    "status": s.get("status", "unknown"),
                    "instance_type": s.get("server_type", {}).get("name", "unknown"),
                    "region": s.get("datacenter", {}).get("location", {}).get("name", "unknown"),
                    "provider": "hetzner",
                    "public_ip": s.get("public_net", {}).get("ipv4", {}).get("ip"),
                }
                for s in data.get("servers", [])
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute command via SSH."""
        if not self.api_token:
            raise Exception("Hetzner Cloud API token not configured")

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
                "output": f"Hetzner exec error: {e}",
                "async": async_exec,
            }

    # ── Hetzner-specific helpers ──────────────────────────────────────

    async def get_server_types(self) -> List[Dict[str, Any]]:
        """List all available Hetzner Cloud server types."""
        data = await self._make_request("GET", f"{self.CLOUD_API}/server_types")
        return data.get("server_types", [])

    async def get_locations(self) -> List[Dict[str, Any]]:
        """List available Hetzner Cloud locations."""
        data = await self._make_request("GET", f"{self.CLOUD_API}/locations")
        return data.get("locations", [])

    async def get_ssh_keys(self) -> List[Dict[str, Any]]:
        """List SSH keys."""
        data = await self._make_request("GET", f"{self.CLOUD_API}/ssh_keys")
        return data.get("ssh_keys", [])
