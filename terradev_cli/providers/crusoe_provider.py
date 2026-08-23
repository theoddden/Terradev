#!/usr/bin/env python3
"""
Crusoe Cloud Provider - Crusoe Cloud GPU integration
BYOAPI: Uses the end-client's Crusoe API access key + secret key
API: https://api.crusoecloud.com/v1alpha5
"""

import base64
import hashlib
import hmac
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

import aiohttp

from .base_provider import BaseProvider


class CrusoeProvider(BaseProvider):
    """Crusoe Cloud provider for GPU instances"""

    API_BASE = "https://api.crusoecloud.com/v1alpha5"
    REQUIRES_AUTH = False  # Static GPU_PRICING table is used when no Crusoe keys

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "crusoe"
        self.access_key = credentials.get("access_key", "")
        self.secret_key = credentials.get("secret_key", "")
        self.project_id = credentials.get("project_id", "")
        self._bearer_token: Optional[str] = None

    # ── GPU product name mapping ──────────────────────────────────────
    # Crusoe product names follow the pattern: <gpu>.<count>x
    # Pricing is approximate on-demand $/hr per GPU (as of early 2026).
    GPU_PRICING = {
        "A100": {
            "product_name": "a100.1x",
            "price": 2.20,
            "mem": 80,
            "vcpus": 12,
            "gpu_count": 1,
        },
        "A100-8x": {
            "product_name": "a100.8x",
            "price": 17.60,
            "mem": 80,
            "vcpus": 96,
            "gpu_count": 8,
        },
        "H100": {
            "product_name": "h100.1x",
            "price": 2.85,
            "mem": 80,
            "vcpus": 26,
            "gpu_count": 1,
        },
        "H100-8x": {
            "product_name": "h100.8x",
            "price": 22.80,
            "mem": 80,
            "vcpus": 208,
            "gpu_count": 8,
        },
        "A40": {
            "product_name": "a40.1x",
            "price": 0.80,
            "mem": 48,
            "vcpus": 8,
            "gpu_count": 1,
        },
        "L40S": {
            "product_name": "l40s.1x",
            "price": 1.58,
            "mem": 48,
            "vcpus": 16,
            "gpu_count": 1,
        },
    }

    # ── Authentication ────────────────────────────────────────────────

    async def _ensure_auth(self):
        """No-op: Crusoe uses per-request HMAC-SHA256 signing (see _make_request)."""
        pass

    def _get_auth_headers(self) -> Dict[str, str]:
        # Auth is injected per-request via _make_request override (HMAC-SHA256 signed).
        return {}

    # ── Per-request HMAC-SHA256 signing ──────────────────────────────

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Override base _make_request to inject Crusoe HMAC-SHA256 signed headers.

        Signature scheme (v1.0):
          payload = path + "\n" + canonicalized_query + "\n" + METHOD + "\n" + timestamp + "\n"
          sig     = base64url(hex(hmac_sha256(payload, urlsafe_b64decode(secret_key))))
          header  = "Bearer 1.0:{access_key_id}:{sig}"
        See: https://docs.crusoecloud.com/reference/api/
        """
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
            self._owns_session = True

        headers = kwargs.pop("headers", {})
        headers.update(self._build_signed_headers(method, url))
        headers.setdefault("Content-Type", "application/json")

        async with self.session.request(method, url, headers=headers, **kwargs) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")
            return await response.json()

    def _build_signed_headers(self, method: str, url: str) -> Dict[str, str]:
        """Build X-Crusoe-Timestamp and Authorization headers for a single request."""
        if not self.access_key or not self.secret_key:
            return {}

        parsed = urlparse(url)
        path = parsed.path
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        # Canonicalize query params: sort lexicographically, join with &
        if parsed.query:
            pairs = sorted(
                part for part in parsed.query.split("&") if part
            )
            query_str = "&".join(pairs)
        else:
            query_str = ""

        payload = f"{path}\n{query_str}\n{method.upper()}\n{timestamp}\n"

        # Decode URL-safe base64 secret key (add padding if needed)
        sk = self.secret_key
        padding = (4 - len(sk) % 4) % 4
        secret_bytes = base64.urlsafe_b64decode(sk + "=" * padding)

        # HMAC-SHA256 → hex-encode → URL-safe base64 (no padding)
        mac = hmac.new(secret_bytes, payload.encode("utf-8"), hashlib.sha256)
        hex_digest = mac.hexdigest()
        sig_b64 = base64.urlsafe_b64encode(hex_digest.encode("utf-8")).rstrip(b"=").decode()

        return {
            "X-Crusoe-Timestamp": timestamp,
            "Authorization": f"Bearer 1.0:{self.access_key}:{sig_b64}",
        }

    # ── Capacity / Quotes ─────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get instance quotes for a GPU type from Crusoe Cloud."""

        # Try live capacity API first
        if self.access_key and self.secret_key:
            try:
                live = await self._get_live_capacity(gpu_type, region)
                if live:
                    return live
            except Exception:  # noqa: BLE001
                pass

        # Fallback to static pricing
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            # Try matching by lowercase prefix
            for key, val in self.GPU_PRICING.items():
                if gpu_type.upper().startswith(key.split("-")[0]):
                    info = val
                    break
        if not info:
            return []

        return [
            {
                "instance_type": info["product_name"],
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": region or "us-northcentral1-a",
                "available": True,
                "provider": "crusoe",
                "vcpus": info["vcpus"],
                "memory_gb": info["mem"],
                "gpu_count": info["gpu_count"],
                "spot": False,
            }
        ]

    async def _get_live_capacity(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query Crusoe /capacities endpoint for real-time availability."""
        await self._ensure_auth()

        # Map our GPU type to Crusoe product_name prefix
        gpu_lower = gpu_type.lower().replace("-", ".")
        if not gpu_lower.endswith("x"):
            gpu_lower += ".1x"

        url = f"{self.API_BASE}/capacities"
        # Build query string
        qs_parts = [f"product_name={gpu_lower}"]
        if region:
            qs_parts.append(f"location={region}")
        if qs_parts:
            url += "?" + "&".join(qs_parts)

        data = await self._make_request("GET", url)

        quotes = []
        for cap in data.get("capacities", data.get("items", [])):
            product = cap.get("product_name", "")
            location = cap.get("location", "unknown")
            amount = cap.get("amount", 0)
            if amount <= 0:
                continue

            # Look up pricing from our static table
            price = self._price_for_product(product)
            gpu_count = self._gpu_count_from_product(product)

            quotes.append(
                {
                    "instance_type": product,
                    "gpu_type": gpu_type,
                    "price_per_hour": price,
                    "region": location,
                    "available": True,
                    "provider": "crusoe",
                    "vcpus": 12 * gpu_count,
                    "memory_gb": 80,
                    "gpu_count": gpu_count,
                    "spot": False,
                }
            )

        return sorted(quotes, key=lambda q: q["price_per_hour"])

    # ── Provisioning ──────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str, ssh_public_key: str = ""
    ) -> Dict[str, Any]:
        if not self.access_key or not self.secret_key:
            raise Exception("Crusoe API credentials not configured")
        if not self.project_id:
            raise Exception(
                "Crusoe project_id not configured — run `terradev configure crusoe`"
            )

        await self._ensure_auth()

        body = {
            "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
            "type": instance_type,
            "location": region or "us-northcentral1-a",
            "image": "ubuntu22.04-nvidia-pcie-docker:latest",
            "ssh_public_key": self._get_ssh_key(),
            "startup_script": "",
        }

        data = await self._make_request(
            "POST",
            f"{self.API_BASE}/compute/vms?project_id={self.project_id}",
            json=body,
        )

        # The response wraps the VM in an async operation
        vm = data.get("instance", data.get("vm", data))
        vm_id = vm.get("id", f"crusoe-{datetime.now().strftime('%Y%m%d%H%M%S')}")

        return {
            "instance_id": vm_id,
            "instance_type": instance_type,
            "region": region,
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "crusoe",
            "metadata": {
                "operation_id": data.get("operation", {}).get("operation_id", ""),
            },
        }

    # ── Instance management ───────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.access_key or not self.secret_key:
            raise Exception("Crusoe API credentials not configured")
        if not self.project_id:
            raise Exception("Crusoe project_id not configured")

        await self._ensure_auth()
        data = await self._make_request(
            "GET",
            f"{self.API_BASE}/compute/vms/{instance_id}?project_id={self.project_id}",
        )
        vm = data.get("instance", data)
        return {
            "instance_id": instance_id,
            "status": vm.get("state", "unknown"),
            "instance_type": vm.get("type", "unknown"),
            "region": vm.get("location", "unknown"),
            "provider": "crusoe",
            "public_ip": self._extract_public_ip(vm),
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.access_key or not self.secret_key:
            raise Exception("Crusoe API credentials not configured")
        await self._ensure_auth()
        await self._make_request(
            "POST",
            f"{self.API_BASE}/compute/vms/{instance_id}/action?project_id={self.project_id}",
            json={"action": "STOP"},
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.access_key or not self.secret_key:
            raise Exception("Crusoe API credentials not configured")
        await self._ensure_auth()
        await self._make_request(
            "POST",
            f"{self.API_BASE}/compute/vms/{instance_id}/action?project_id={self.project_id}",
            json={"action": "START"},
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.access_key or not self.secret_key:
            raise Exception("Crusoe API credentials not configured")
        await self._ensure_auth()
        await self._make_request(
            "DELETE",
            f"{self.API_BASE}/compute/vms/{instance_id}?project_id={self.project_id}",
        )
        return {
            "instance_id": instance_id,
            "action": "terminate",
            "status": "terminating",
        }

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.access_key or not self.secret_key or not self.project_id:
            return []
        try:
            await self._ensure_auth()
            data = await self._make_request(
                "GET",
                f"{self.API_BASE}/compute/vms?project_id={self.project_id}",
            )
            instances = data.get("items", data.get("instances", []))
            return [
                {
                    "instance_id": vm.get("id", "unknown"),
                    "status": vm.get("state", "unknown"),
                    "instance_type": vm.get("type", "unknown"),
                    "region": vm.get("location", "unknown"),
                    "provider": "crusoe",
                    "public_ip": self._extract_public_ip(vm),
                }
                for vm in instances
            ]
        except Exception:  # noqa: BLE001
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute a command on a Crusoe VM via SSH."""
        if not self.access_key or not self.secret_key:
            raise Exception("Crusoe API credentials not configured")

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
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                "-o",
                "ConnectTimeout=10",
                f"root@{public_ip}",
                command,
            ]

            if async_exec:
                proc = subprocess.Popen(
                    ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "job_id": str(proc.pid),
                    "output": f"Async SSH process started (PID: {proc.pid})",
                    "async": True,
                }

            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=300
            )
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
                "output": f"Crusoe exec error: {e}",
                "async": async_exec,
            }

    # ── Crusoe-specific helpers ───────────────────────────────────────

    async def get_locations(self) -> List[Dict[str, Any]]:
        """List all Crusoe Cloud locations."""
        await self._ensure_auth()
        data = await self._make_request("GET", f"{self.API_BASE}/locations")
        return data.get("items", data.get("locations", []))

    async def get_vm_types(self) -> List[Dict[str, Any]]:
        """List all available VM types with GPU specs."""
        await self._ensure_auth()
        data = await self._make_request("GET", f"{self.API_BASE}/compute/vms/types")
        return data.get("items", data.get("types", []))

    async def get_billing_costs(self, org_id: str) -> Dict[str, Any]:
        """Get billing costs for an organization."""
        await self._ensure_auth()
        return await self._make_request(
            "GET", f"{self.API_BASE}/organizations/{org_id}/billing/costs"
        )

    async def get_reservations(self, org_id: str) -> List[Dict[str, Any]]:
        """Get GPU reservations for an organization."""
        await self._ensure_auth()
        data = await self._make_request(
            "GET", f"{self.API_BASE}/organizations/{org_id}/reservations"
        )
        return data.get("items", data.get("reservations", []))

    # ── Internal helpers ──────────────────────────────────────────────

    def _price_for_product(self, product_name: str) -> float:
        """Look up approximate price for a Crusoe product name."""
        for _key, info in self.GPU_PRICING.items():
            if info["product_name"] == product_name:
                return info["price"]
        # Heuristic: parse GPU count from product name
        parts = product_name.split(".")
        if len(parts) == 2:
            base = parts[0]
            count_str = parts[1].replace("x", "")
            try:
                count = int(count_str)
            except ValueError:
                count = 1
            # Find base price
            for _key, info in self.GPU_PRICING.items():
                if info["product_name"].startswith(base + ".1x"):
                    return info["price"] * count
        return 2.50  # conservative default

    def _gpu_count_from_product(self, product_name: str) -> int:
        """Extract GPU count from Crusoe product name like 'a100.8x'."""
        try:
            return int(product_name.split(".")[1].replace("x", ""))
        except (IndexError, ValueError):
            return 1

    def _extract_public_ip(self, vm: Dict[str, Any]) -> Optional[str]:
        """Extract public IP from Crusoe VM network interfaces."""
        for nic in vm.get("network_interfaces", []):
            ips = nic.get("public_ipv4", {})
            if isinstance(ips, dict):
                addr = ips.get("address")
                if addr:
                    return addr
            elif isinstance(ips, str) and ips:
                return ips
        return None

    def _get_ssh_key(self) -> str:
        """Read the user's default SSH public key."""
        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
        ed25519_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
        for path in [ed25519_path, ssh_key_path]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read().strip()
        return ""
