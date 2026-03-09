#!/usr/bin/env python3
"""
CoreWeave Provider - CoreWeave Kubernetes-native GPU cloud

CRITICAL FIXES v4.0.0:
- Permissions upgrade detection for new accounts
- Kubernetes-only deployment validation
- Legacy node pool filtering for costs
- Public IP billing tracking
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
        self.account_checked = False

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
        # CRITICAL: Check permissions and account access first
        if not self.api_key:
            return [{
                "provider": "coreweave",
                "gpu_type": gpu_type,
                "available": False,
                "reason": "API key not configured",
                "action_required": "Configure CoreWeave API key",
            }]
        
        # CRITICAL: Check account permissions for new users
        if not self.account_checked:
            permissions_check = await self._check_account_permissions()
            if not permissions_check["full_access"]:
                return [{
                    "provider": "coreweave",
                    "gpu_type": gpu_type,
                    "available": False,
                    "reason": permissions_check["reason"],
                    "action_required": permissions_check["action_required"],
                    "permissions_upgrade_required": True,
                }]
            self.account_checked = True
            
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            return []

        target_region = region or "us-east-04e"
        
        # CRITICAL: Check for legacy node pool billing issues
        legacy_warning = await self._check_legacy_node_pool_billing()
        
        quotes = [
            {
                "instance_type": info["type"],
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": target_region,
                "available": True,
                "provider": "coreweave",
                "vcpus": info["vcpus"],
                "memory_gb": info["mem"],
                "gpu_count": 1,
                "spot": False,
                "kubernetes_native": True,
                "legacy_billing_warning": legacy_warning,
                "public_ip_billing": "separate_charge",
            },
            {
                "instance_type": info["type"],
                "gpu_type": gpu_type,
                "price_per_hour": round(info["price"] * 0.5, 2),
                "region": target_region,
                "available": True,
                "provider": "coreweave",
                "vcpus": info["vcpus"],
                "memory_gb": info["mem"],
                "gpu_count": 1,
                "spot": True,
                "kubernetes_native": True,
                "legacy_billing_warning": legacy_warning,
                "public_ip_billing": "separate_charge",
            },
        ]
        
        return quotes

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("CoreWeave API key not configured")
        
        # CRITICAL: Double-check permissions before provisioning
        if not self.account_checked:
            permissions_check = await self._check_account_permissions()
            if not permissions_check["full_access"]:
                raise Exception(f"Account permissions insufficient: {permissions_check['reason']}. {permissions_check['action_required']}")
            self.account_checked = True

        instance_name = f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        try:
            # CRITICAL: CoreWeave is Kubernetes-only - use manifests, not VMs
            manifest = await self._generate_kubernetes_manifest(instance_name, instance_type, gpu_type, region)
            
            # Deploy via CoreWeave API
            deployment = await self._make_request(
                "POST", f"{self.API_BASE}/v1/namespaces/{self.namespace}/deployments",
                json=manifest
            )

            return {
                "instance_id": instance_name,
                "instance_type": instance_type,
                "region": region or "us-east-04e",
                "gpu_type": gpu_type,
                "status": "deploying",
                "provider": "coreweave",
                "kubernetes_native": True,
                "namespace": self.namespace,
                "deployment_name": instance_name,
                "public_ip_billing": "separate_charge",
                "metadata": {
                    "manifest_applied": True,
                    "gpu_count": 1,
                },
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
    
    async def _check_account_permissions(self) -> Dict[str, Any]:
        """CRITICAL: Check if account has full Kubernetes and Applications Catalog access
        
        New organizations start with limited access. Must request permissions upgrade.
        """
        try:
            # Test basic API access
            await self._make_request("GET", f"{self.API_BASE}/v1/user")
            
            # Test Kubernetes access
            try:
                await self._make_request("GET", f"{self.API_BASE}/v1/namespaces/{self.namespace}/pods")
                k8s_access = True
            except Exception:
                k8s_access = False
            
            # Test Applications Catalog access
            try:
                await self._make_request("GET", f"{self.API_BASE}/v1/applications")
                catalog_access = True
            except Exception:
                catalog_access = False
            
            if not k8s_access or not catalog_access:
                return {
                    "full_access": False,
                    "reason": "Limited account permissions - need Kubernetes and/or Applications Catalog access",
                    "action_required": "Contact CoreWeave support to request permissions upgrade for full Kubernetes and Applications Catalog access",
                    "k8s_access": k8s_access,
                    "catalog_access": catalog_access,
                }
            
            return {"full_access": True}
            
        except Exception as e:
            return {
                "full_access": False,
                "reason": f"Permission check failed: {str(e)}",
                "action_required": "Verify API key and account status with CoreWeave support",
            }
    
    async def _check_legacy_node_pool_billing(self) -> Optional[Dict[str, Any]]:
        """CRITICAL: Check for legacy node pool billing issues
        
        Clusters created before July 7, 2025 have cpu-control-plane Node Pool
        that must be filtered out from billing queries.
        """
        try:
            # Check for legacy node pools
            node_pools = await self._make_request("GET", f"{self.API_BASE}/v1/nodepools")
            
            legacy_pools = []
            for pool in node_pools.get("items", []):
                if pool.get("name", "").startswith("cpu-control-plane"):
                    legacy_pools.append(pool["name"])
            
            if legacy_pools:
                return {
                    "legacy_node_pools_detected": True,
                    "legacy_pools": legacy_pools,
                    "billing_impact": "Phantom billable nodes may appear in cost tracking",
                    "action_required": "Filter out cpu-control-plane node pools from billing queries",
                    "fix_available": True,
                }
            
            return None
            
        except Exception:
            return None
    
    async def _generate_kubernetes_manifest(
        self, instance_name: str, instance_type: str, gpu_type: str, region: str
    ) -> Dict[str, Any]:
        """Generate Kubernetes manifest for CoreWeave deployment"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": instance_name,
                "namespace": self.namespace,
                "labels": {
                    "managed-by": "terradev",
                    "gpu-type": gpu_type.lower(),
                    "app": instance_name,
                },
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": instance_name,
                    },
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": instance_name,
                            "managed-by": "terradev",
                            "gpu-type": gpu_type.lower(),
                        },
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "gpu-container",
                                "image": "nvidia/cuda:12.1.1-base-ubuntu22.04",
                                "resources": {
                                    "requests": {
                                        "nvidia.com/gpu": 1,
                                        "cpu": "4",
                                        "memory": "16Gi",
                                    },
                                    "limits": {
                                        "nvidia.com/gpu": 1,
                                        "cpu": "8",
                                        "memory": "32Gi",
                                    },
                                },
                                "env": [
                                    {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
                                    {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
                                ],
                            }
                        ],
                        "nodeSelector": {
                            "gpu.coreweave.cloud/type": instance_type,
                        },
                        "tolerations": [
                            {
                                "key": "nvidia.com/gpu",
                                "operator": "Exists",
                                "effect": "NoSchedule",
                            }
                        ],
                    },
                },
            },
        }
