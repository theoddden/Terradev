#!/usr/bin/env python3
"""
CoreWeave Provider - CoreWeave Kubernetes-native GPU cloud
BYOAPI: Uses the end-client's CoreWeave credentials
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider


class CoreWeaveProvider(BaseProvider):
    """CoreWeave provider for GPU instances"""

    API_BASE = "https://api.coreweave.com"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "coreweave"
        self.api_key = credentials.get("api_key", "")
        self.namespace = credentials.get("namespace", "default")

    GPU_PRICING = {
        "A100": {"type": "a100-80gb", "price": 2.21, "mem": 80, "vcpus": 15},
        "A100-40": {"type": "a100-40gb", "price": 2.06, "mem": 40, "vcpus": 15},
        "H100": {"type": "h100-80gb", "price": 4.76, "mem": 80, "vcpus": 18},
        "A40": {"type": "a40-48gb", "price": 1.28, "mem": 48, "vcpus": 12},
        "RTX4090": {"type": "rtx4090-24gb", "price": 0.74, "mem": 24, "vcpus": 8},
        "V100": {"type": "v100-16gb", "price": 0.80, "mem": 16, "vcpus": 4},
    }

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # CRITICAL FIX: Don't return quotes without API credentials (BYOAPI requirement)
        if not self.api_key:
            return []
            
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            return []

        return [
            {
                "instance_type": info["type"],
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": region or "us-east-04e",
                "available": True,
                "provider": "coreweave",
                "vcpus": info["vcpus"],
                "memory_gb": info["mem"],
                "gpu_count": 1,
                "spot": False,
            },
            {
                "instance_type": info["type"],
                "gpu_type": gpu_type,
                "price_per_hour": round(info["price"] * 0.5, 2),
                "region": region or "us-east-04e",
                "available": True,
                "provider": "coreweave",
                "vcpus": info["vcpus"],
                "memory_gb": info["mem"],
                "gpu_count": 1,
                "spot": True,
            },
        ]

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("CoreWeave API key not configured")

        instance_name = f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        try:
            data = await self._make_request(
                "POST", f"{self.API_BASE}/v1/namespaces/{self.namespace}/virtualservers",
                json={
                    "apiVersion": "virtualservers.coreweave.com/v1alpha1",
                    "kind": "VirtualServer",
                    "metadata": {
                        "name": instance_name,
                        "labels": {"managed-by": "terradev", "gpu-type": gpu_type.lower()},
                    },
                    "spec": {
                        "region": region or "us-east-04e",
                        "os": {"type": "linux"},
                        "resources": {
                            "gpu": {"type": instance_type, "count": 1},
                            "cpu": {"count": 8},
                            "memory": "32Gi",
                        },
                        "storage": {"root": {"size": "200Gi", "storageClassName": "block-nvme-us-east-04e"}},
                    },
                },
            )
            return {
                "instance_id": instance_name,
                "instance_type": instance_type,
                "region": region or "us-east-04e",
                "gpu_type": gpu_type,
                "status": "provisioning",
                "provider": "coreweave",
            }
        except Exception as e:
            raise Exception(f"CoreWeave provision failed: {e}")

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("CoreWeave API key not configured")
        try:
            data = await self._make_request(
                "GET", f"{self.API_BASE}/v1/namespaces/{self.namespace}/virtualservers/{instance_id}"
            )
            status = data.get("status", {}).get("phase", "unknown")
            return {"instance_id": instance_id, "status": status.lower(), "provider": "coreweave"}
        except Exception as e:
            raise Exception(f"CoreWeave status failed: {e}")

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("CoreWeave API key not configured")
        await self._make_request(
            "PATCH", f"{self.API_BASE}/v1/namespaces/{self.namespace}/virtualservers/{instance_id}",
            json={"spec": {"running": False}},
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("CoreWeave API key not configured")
        await self._make_request(
            "PATCH", f"{self.API_BASE}/v1/namespaces/{self.namespace}/virtualservers/{instance_id}",
            json={"spec": {"running": True}},
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("CoreWeave API key not configured")
        await self._make_request(
            "DELETE", f"{self.API_BASE}/v1/namespaces/{self.namespace}/virtualservers/{instance_id}"
        )
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            data = await self._make_request(
                "GET", f"{self.API_BASE}/v1/namespaces/{self.namespace}/virtualservers?labelSelector=managed-by=terradev"
            )
            return [
                {
                    "instance_id": vs.get("metadata", {}).get("name"),
                    "status": vs.get("status", {}).get("phase", "unknown").lower(),
                    "provider": "coreweave",
                }
                for vs in data.get("items", [])
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("CoreWeave API key not configured")
        # CoreWeave is Kubernetes-native — use kubectl exec or their exec API
        try:
            # Try kubectl exec first (requires kubeconfig configured for CoreWeave)
            import subprocess
            kubectl_cmd = [
                "kubectl", "--namespace", self.namespace,
                "exec", instance_id, "--", "sh", "-c", command,
            ]
            if async_exec:
                proc = subprocess.Popen(kubectl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "job_id": str(proc.pid),
                    "output": f"Async kubectl exec started (PID: {proc.pid})",
                    "async": True,
                }
            result = subprocess.run(kubectl_cmd, capture_output=True, text=True, timeout=300)
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "async": False,
            }
        except Exception as e:
            # Fallback: CoreWeave exec API
            try:
                data = await self._make_request(
                    "POST",
                    f"{self.API_BASE}/v1/namespaces/{self.namespace}/virtualservers/{instance_id}/exec",
                    json={"command": ["sh", "-c", command]},
                )
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "output": str(data.get("output", data)),
                    "async": async_exec,
                }
            except Exception as api_err:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 1,
                    "output": f"CoreWeave exec error: kubectl={e}; api={api_err}",
                    "async": async_exec,
                }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
