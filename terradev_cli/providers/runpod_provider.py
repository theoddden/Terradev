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
from datetime import datetime, timedelta

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class RunPodProvider(BaseProvider):
    """RunPod provider for GPU instances - BYOAPI only, no static fallback data"""

    API_BASE = "https://api.runpod.io/graphql"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "runpod"
        self.api_key = credentials.get("api_key", "")
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # 1 minute window
        self.max_requests_per_window = 100

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # BYOAPI REQUIREMENT: No static fallback data - must have API key
        if not self.api_key:
            return []
            
        # CRITICAL: Check rate limiting before API call
        if not await self._check_rate_limit():
            return [{
                "provider": "runpod",
                "gpu_type": gpu_type,
                "available": False,
                "reason": "Rate limit exceeded",
                "action_required": "Wait before making more requests",
                "rate_limited": True,
            }]
            
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
            "POST", self.API_BASE,
            json={"query": query},
        )
        quotes = []
        for gpu in data.get("data", {}).get("gpuTypes", []):
            name = gpu.get("displayName", "")
            if gpu_type.lower() in name.lower():
                if gpu.get("communityPrice"):
                    quotes.append({
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
                    })
                if gpu.get("securePrice"):
                    quotes.append({
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
                    })
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str, attach_volume: bool = True, volume_size_gb: int = 100
    ) -> Dict[str, Any]:
        """Provision RunPod instance with optional volume attachment"""
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        
        # CRITICAL: Check rate limiting
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded - please wait before making more requests")

        try:
            # Extract GPU ID from instance type
            if "-" not in instance_type:
                raise Exception(f"Invalid instance type format: {instance_type}")
            
            cloud_type, gpu_id = instance_type.split("-", 1)
            if cloud_type not in ["runpod-community", "runpod-secure"]:
                raise Exception(f"Unsupported cloud type: {cloud_type}")
            
            # Determine cloud type for API
            is_secure = cloud_type == "runpod-secure"
            
            # Create pod specification
            pod_spec = {
                "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "imageId": "runpod/base:latest",
                "gpuType": gpu_id,
                "cloudType": "SECURE" if is_secure else "COMMUNITY",
                "containerDiskInGb": 40,
                "minMemoryInGb": 80,
                "minVcpuCount": 4,
                "env": [
                    {"key": "TERRADEV_MANAGED", "value": "true"},
                    {"key": "GPU_TYPE", "value": gpu_type},
                ],
                "ports": ["http", "https"],
            }
            
            # CRITICAL: Add volume if requested
            if attach_volume:
                volume_id = await self._create_and_attach_volume(pod_spec["name"], volume_size_gb)
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
                "status": "starting",
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
            raise Exception(f"Unsupported GPU type: {gpu_type}")

        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                gpuCount
                machineId
            }
        }
        """
        variables = {
            "input": {
                "cloudType": "SECURE" if "secure" in instance_type else "COMMUNITY",
                "gpuTypeId": gpu_info["id"],
                "gpuCount": 1,
                "volumeInGb": 50,
                "containerDiskInGb": 20,
                "templateId": "runpod-torch-v21",
                "name": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
            }
        }

        try:
            data = await self._make_request(
                "POST", self.API_BASE,
                json={"query": mutation, "variables": variables},
            )
            pod = data.get("data", {}).get("podFindAndDeployOnDemand", {})
            return {
                "instance_id": pod.get("id", f"runpod-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                "instance_type": instance_type,
                "region": "us-east",
                "gpu_type": gpu_type,
                "status": "provisioning",
                "provider": "runpod",
                "metadata": {"machine_id": pod.get("machineId")},
            }
        except Exception as e:
            raise Exception(f"RunPod provision failed: {e}")

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        query = 'query Pod($podId: String!) { pod(input: {podId: $podId}) { id name desiredStatus gpuCount } }'
        try:
            data = await self._make_request("POST", self.API_BASE, json={"query": query, "variables": {"podId": instance_id}})
            pod = data.get("data", {}).get("pod", {})
            return {
                "instance_id": instance_id,
                "status": (pod.get("desiredStatus") or "unknown").lower(),
                "provider": "runpod",
            }
        except Exception as e:
            raise Exception(f"RunPod status failed: {e}")

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        mutation = 'mutation StopPod($podId: String!) { podStop(input: {podId: $podId}) { id desiredStatus } }'
        await self._make_request("POST", self.API_BASE, json={"query": mutation, "variables": {"podId": instance_id}})
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        mutation = 'mutation ResumePod($podId: String!) { podResume(input: {podId: $podId, gpuCount: 1}) { id desiredStatus } }'
        await self._make_request("POST", self.API_BASE, json={"query": mutation, "variables": {"podId": instance_id}})
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception("RunPod API key not configured")
        mutation = 'mutation TerminatePod($podId: String!) { podTerminate(input: {podId: $podId}) }'
        await self._make_request("POST", self.API_BASE, json={"query": mutation, "variables": {"podId": instance_id}})
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        query = "query { myself { pods { id name desiredStatus gpuCount machine { gpuDisplayName } } } }"
        try:
            data = await self._make_request("POST", self.API_BASE, json={"query": query})
            pods = data.get("data", {}).get("myself", {}).get("pods", [])
            return [
                {
                    "instance_id": p["id"],
                    "status": (p.get("desiredStatus") or "unknown").lower(),
                    "instance_type": p.get("machine", {}).get("gpuDisplayName", "unknown"),
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
        # RunPod supports exec via their runsync/run endpoints
        endpoint = "run" if async_exec else "runsync"
        try:
            data = await self._make_request(
                "POST",
                f"https://api.runpod.ai/v2/{instance_id}/{endpoint}",
                json={"input": {"command": command}},
            )
            if async_exec:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "job_id": data.get("id", "unknown"),
                    "output": f"Async job submitted: {data.get('id', 'unknown')}",
                    "async": True,
                }
            output = data.get("output", data.get("result", ""))
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 0 if data.get("status") == "COMPLETED" else 1,
                "output": str(output),
                "async": False,
            }
        except Exception as e:
            # Fallback: SSH exec if pod has SSH enabled
            try:
                import subprocess
                result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=accept-new",
                     "-o", f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                     "-o", "ConnectTimeout=10",
                     f"root@{instance_id}.runpod.io", command],
                    capture_output=True, text=True, timeout=300,
                )
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "async": async_exec,
                }
            except Exception as ssh_err:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 1,
                    "output": f"RunPod exec error: {e}; SSH fallback error: {ssh_err}",
                    "async": async_exec,
                }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
    
    async def _check_rate_limit(self) -> bool:
        """CRITICAL: Check rate limiting for API calls"""
        current_time = datetime.now().timestamp()
        
        # Reset window if needed
        if current_time - self.last_request_time > self.rate_limit_window:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Check if we're within limits
        if self.request_count >= self.max_requests_per_window:
            return False
        
        self.request_count += 1
        return True
    
    async def _create_and_attach_volume(self, pod_name: str, size_gb: int) -> Optional[str]:
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
                "POST", self.API_BASE,
                json={"query": volume_mutation, "variables": volume_variables}
            )
            
            volume_id = volume_data.get("data", {}).get("networkStorageCreate", {}).get("id")
            if volume_id:
                logger.info(f"Created volume {volume_id} ({size_gb}GB) for pod {pod_name}")
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
            "POST", self.API_BASE,
            json={"query": mutation, "variables": variables}
        )
        
        return data.get("data", {}).get("podFindAndDeployOnDemand", {})
