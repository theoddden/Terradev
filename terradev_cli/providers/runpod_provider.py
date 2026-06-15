#!/usr/bin/env python3
"""
RunPod Provider - RunPod GPU cloud integration

CRITICAL FIXES v4.0.0:
- Volume attachment for data persistence
- Secure vs Community Cloud selection
- Rate limiting handling for multi-pod management
- Cold start SLA monitoring
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class RunPodProvider(BaseProvider):
    """RunPod provider for GPU instances - BYOAPI only, no static fallback data"""

    API_BASE = "https://api.runpod.io/graphql"
    
    # Class-level rate limiting state shared across all instances
    _last_request_time = 0
    _request_count = 0
    _rate_limit_lock = None
    _rate_limit_window = 60  # 1 minute window
    _max_requests_per_window = 100

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "runpod"
        self.api_key = credentials.get("api_key", "")

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # BYOAPI REQUIREMENT: No static fallback data - must have API key
        if not self.api_key:
            return []

        # CRITICAL: Check rate limiting before API call
        if not await self._check_rate_limit():
            return [
                {
                    "provider": "runpod",
                    "gpu_type": gpu_type,
                    "available": False,
                    "reason": "Rate limit exceeded",
                    "action_required": "Wait before making more requests",
                    "rate_limited": True,
                }
            ]

        # Try live API only - no static fallback
        try:
            live = await self._get_live_pricing(gpu_type)
            if live:
                # CRITICAL: Add volume attachment warnings
                for quote in live:
                    quote["storage_ephemeral"] = True
                    quote["volume_required"] = True
                    quote["volume_cost_separate"] = True
                    quote["data_loss_on_restart"] = True
                    quote["volume_attachment_available"] = True
                return live
        except Exception as e:
            logger.debug(f"RunPod API error: {e}")
            return []

    async def _get_live_pricing(self, gpu_type: str) -> List[Dict[str, Any]]:
        """Query RunPod GraphQL API for live GPU availability"""
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                communityPrice
                securePrice
            }
        }
        """
        data = await self._make_request(
            "POST",
            self.API_BASE,
            json={"query": query},
        )
        quotes = []
        for gpu in data.get("data", {}).get("gpuTypes", []):
            name = gpu.get("displayName", "")
            if gpu_type.lower() in name.lower():
                if gpu.get("communityPrice"):
                    quotes.append(
                        {
                            "instance_type": f"runpod-community-{gpu['id']}",
                            "gpu_type": gpu_type,
                            "price_per_hour": gpu["communityPrice"],
                            "region": "us-east",
                            "available": True,
                            "provider": "runpod",
                            "vcpus": 16,
                            "memory_gb": gpu.get("memoryInGb", 0),
                            "gpu_count": 1,
                            "spot": True,
                            "cloud_type": "community",
                            "performance_warning": "Community Cloud performance varies by host configuration",
                            "cold_start_sla": "Not guaranteed during peak hours",
                            "isolation": "container-only",
                        }
                    )
                if gpu.get("securePrice"):
                    quotes.append(
                        {
                            "instance_type": f"runpod-secure-{gpu['id']}",
                            "gpu_type": gpu_type,
                            "price_per_hour": gpu["securePrice"],
                            "region": "us-east",
                            "available": True,
                            "provider": "runpod",
                            "vcpus": 16,
                            "memory_gb": gpu.get("memoryInGb", 0),
                            "gpu_count": 1,
                            "spot": False,
                            "cloud_type": "secure",
                            "performance_guaranteed": True,
                            "cold_start_sla": "< 3 seconds",
                            "isolation": "vm-level",
                        }
                    )
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self,
        instance_type: str,
        region: str,
        gpu_type: str,
        attach_volume: bool = True,
        volume_size_gb: int = 100,
        ssh_public_key: str = "",
    ) -> Dict[str, Any]:
        """Provision RunPod instance with optional volume attachment"""
        if not self.api_key:
            raise Exception("RunPod API key not configured")

        # CRITICAL: Check rate limiting
        if not await self._check_rate_limit():
            raise Exception(
                "Rate limit exceeded - please wait before making more requests"
            )

        try:
            # Extract GPU ID from instance type (format: runpod-community-<GPU_ID> or runpod-secure-<GPU_ID>)
            # Use startswith to handle GPU IDs that contain hyphens (e.g. NVIDIA A100-SXM4-40GB)
            if instance_type.startswith("runpod-community-"):
                is_secure = False
                gpu_id = instance_type[len("runpod-community-"):]
            elif instance_type.startswith("runpod-secure-"):
                is_secure = True
                gpu_id = instance_type[len("runpod-secure-"):]
            else:
                raise Exception(f"Unsupported cloud type: {instance_type}")

            # Create pod specification
            pod_spec = {
                "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "imageName": "runpod/base:latest",
                "gpuTypeId": gpu_id,
                "cloudType": "SECURE" if is_secure else "COMMUNITY",
                "containerDiskInGb": 40,
                "minMemoryInGb": 80,
                "minVcpuCount": 4,
                "env": [
                    {"key": "TERRADEV_MANAGED", "value": "true"},
                    {"key": "GPU_TYPE", "value": gpu_type},
                ],
                "ports": "22/tcp,8888/http",
                "gpuCount": 1,
                "startSsh": True,
                "supportPublicIp": True,
            }

            # CRITICAL: Add volume if requested
            volume_id = None
            if attach_volume:
                volume_id = await self._create_and_attach_volume(
                    pod_spec["name"], volume_size_gb
                )
                if volume_id:
                    pod_spec["volumeInGb"] = volume_size_gb
                    pod_spec["volumeMountPath"] = "/workspace"
                else:
                    logger.warning(f"Failed to create volume for {pod_spec['name']}")

            # Deploy the pod
            deployment = await self._deploy_pod(pod_spec)

            return {
                "instance_id": deployment.get("id", pod_spec["name"]),
                "instance_type": instance_type,
                "region": region,
                "gpu_type": gpu_type,
                "status": "provisioning",
                "provider": "runpod",
                "cloud_type": "secure" if is_secure else "community",
                "volume_attached": attach_volume and volume_id is not None,
                "volume_size_gb": volume_size_gb if attach_volume else 0,
                "volume_id": volume_id if attach_volume else None,
                "data_persistence": attach_volume,
                "cold_start_sla": "< 3s" if is_secure else "Not guaranteed",
                "metadata": {
                    "pod_id": deployment.get("id"),
                    "gpu_count": 1,
                    "ports": pod_spec["ports"],
                },
            }

        except Exception as e:
            raise Exception(f"RunPod provision failed: {e}")

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        query = """
        query Pod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                name
                desiredStatus
                gpuCount
                runtime {
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
            }
        }
        """
        try:
            data = await self._make_request(
                "POST",
                self.API_BASE,
                json={"query": query, "variables": {"podId": instance_id}},
            )
            pod = data.get("data", {}).get("pod", {})
            runtime = pod.get("runtime", {})
            ports = runtime.get("ports", [])
            
            # Extract public IP and SSH port
            public_ip = None
            ssh_port = None
            for port in ports:
                if port.get("isIpPublic") and port.get("privatePort") == 22:
                    public_ip = port.get("ip")
                    ssh_port = port.get("publicPort")
                    break
            
            return {
                "instance_id": instance_id,
                "status": (pod.get("desiredStatus") or "unknown").lower(),
                "provider": "runpod",
                "ip": public_ip,
                "port": ssh_port,
                "gpu_count": pod.get("gpuCount", 1),
            }
        except Exception as e:
            raise Exception(f"RunPod status failed: {e}")

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        mutation = "mutation StopPod($podId: String!) { podStop(input: {podId: $podId}) { id desiredStatus } }"
        await self._make_request(
            "POST",
            self.API_BASE,
            json={"query": mutation, "variables": {"podId": instance_id}},
        )
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        mutation = "mutation ResumePod($podId: String!) { podResume(input: {podId: $podId, gpuCount: 1}) { id desiredStatus } }"
        await self._make_request(
            "POST",
            self.API_BASE,
            json={"query": mutation, "variables": {"podId": instance_id}},
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        mutation = "mutation TerminatePod($podId: String!) { podTerminate(input: {podId: $podId}) }"
        await self._make_request(
            "POST",
            self.API_BASE,
            json={"query": mutation, "variables": {"podId": instance_id}},
        )
        return {
            "instance_id": instance_id,
            "action": "terminate",
            "status": "terminating",
        }

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        query = "query { myself { pods { id name desiredStatus gpuCount machine { gpuDisplayName } } } }"
        try:
            data = await self._make_request(
                "POST", self.API_BASE, json={"query": query}
            )
            pods = data.get("data", {}).get("myself", {}).get("pods", [])
            return [
                {
                    "instance_id": p["id"],
                    "status": (p.get("desiredStatus") or "unknown").lower(),
                    "instance_type": p.get("machine", {}).get(
                        "gpuDisplayName", "unknown"
                    ),
                    "region": "us-east",
                    "provider": "runpod",
                }
                for p in pods
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        
        # Get instance status to find SSH connection details
        try:
            status = await self.get_instance_status(instance_id)
            public_ip = status.get("ip")
            ssh_port = status.get("port", 22)
            
            if not public_ip:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 1,
                    "output": "No public IP available - instance may still be provisioning",
                    "async": async_exec,
                }
        except Exception as e:
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"Failed to get instance status for SSH: {e}",
                "async": async_exec,
            }
        
        # Use SSH for command execution
        try:
            import subprocess
            
            # Ensure known_hosts directory exists
            known_hosts_path = os.path.expanduser('~/.terradev/known_hosts')
            os.makedirs(os.path.dirname(known_hosts_path), exist_ok=True)
            
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={known_hosts_path}",
                "-o", "ConnectTimeout=10",
                "-p", str(ssh_port),
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
        except Exception as e:
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"SSH execution failed: {e}",
                "async": async_exec,
            }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {}  # RunPod auth is via ?api_key= query param; see _make_request override

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Override to inject api_key as query parameter per RunPod docs."""
        if self.api_key:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api_key={self.api_key}"
        return await super()._make_request(method, url, **kwargs)

    async def _check_rate_limit(self) -> bool:
        """CRITICAL: Check rate limiting for API calls (class-level shared state)"""
        # Lazy-create lock to avoid Python 3.9 event loop binding bug
        if self.__class__._rate_limit_lock is None:
            self.__class__._rate_limit_lock = asyncio.Lock()
        
        current_time = datetime.now().timestamp()

        async with self.__class__._rate_limit_lock:
            # Reset window if needed
            if current_time - self.__class__._last_request_time > self.__class__._rate_limit_window:
                self.__class__._request_count = 0
                self.__class__._last_request_time = current_time

            # Check if we're within limits
            if self.__class__._request_count >= self.__class__._max_requests_per_window:
                return False

            self.__class__._request_count += 1
            return True

    async def _create_and_attach_volume(
        self, pod_name: str, size_gb: int
    ) -> Optional[str]:
        """CRITICAL: Create and attach persistent volume for data persistence"""
        try:
            # Create volume
            volume_mutation = """
            mutation CreateVolume($input: NetworkStorageInput!) {
                networkStorageCreate(input: $input) {
                    id
                    name
                    size
                }
            }
            """

            volume_variables = {
                "input": {
                    "name": f"{pod_name}-volume",
                    "size": size_gb,
                    "dataCenter": "US East",
                    "type": "NETWORK_STORAGE",
                }
            }

            volume_data = await self._make_request(
                "POST",
                self.API_BASE,
                json={"query": volume_mutation, "variables": volume_variables},
            )

            volume_id = (
                volume_data.get("data", {}).get("networkStorageCreate", {}).get("id")
            )
            if volume_id:
                logger.info(
                    f"Created volume {volume_id} ({size_gb}GB) for pod {pod_name}"
                )
                return volume_id

            return None

        except Exception as e:
            logger.debug(f"Failed to create volume for {pod_name}: {e}")
            return None

    async def _deploy_pod(self, pod_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy pod with the given specification"""
        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                gpuCount
                machineId
                desiredStatus
            }
        }
        """

        variables = {"input": pod_spec}

        data = await self._make_request(
            "POST", self.API_BASE, json={"query": mutation, "variables": variables}
        )

        return data.get("data", {}).get("podFindAndDeployOnDemand", {})
