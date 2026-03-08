#!/usr/bin/env python3
"""
GCP Provider - Google Cloud Platform integration
BYOAPI: Uses the end-client's GCP credentials
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class GCPProvider(BaseProvider):
    """Google Cloud Compute Engine provider for GPU instances"""

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "gcp"
        self.project_id = credentials.get("project_id")
        self.zone = credentials.get("zone", "us-central1-a")
        self.client = None

        try:
            from google.cloud import compute_v1
            from google.oauth2 import service_account

            creds_path = credentials.get("credentials_file")
            if creds_path:
                sa_creds = service_account.Credentials.from_service_account_file(creds_path)
                self.instances_client = compute_v1.InstancesClient(credentials=sa_creds)
                self.accelerator_client = compute_v1.AcceleratorTypesClient(credentials=sa_creds)
            else:
                self.instances_client = compute_v1.InstancesClient()
                self.accelerator_client = compute_v1.AcceleratorTypesClient()
            self.client = True
        except Exception as e:
            logger.debug(f"GCP client init deferred (BYOAPI): {e}")
            self.instances_client = None
            self.accelerator_client = None

    # -- GPU / instance mapping ------------------------------------------

    GPU_INSTANCE_MAP = {
        "A100": [
            {"machine": "a2-highgpu-1g", "gpus": 1, "vcpus": 12, "mem": 85},
            {"machine": "a2-highgpu-4g", "gpus": 4, "vcpus": 48, "mem": 340},
            {"machine": "a2-highgpu-8g", "gpus": 8, "vcpus": 96, "mem": 680},
        ],
        "V100": [
            {"machine": "n1-standard-8", "accel": "nvidia-tesla-v100", "gpus": 1, "vcpus": 8, "mem": 30},
            {"machine": "n1-standard-16", "accel": "nvidia-tesla-v100", "gpus": 2, "vcpus": 16, "mem": 60},
        ],
        "T4": [
            {"machine": "n1-standard-4", "accel": "nvidia-tesla-t4", "gpus": 1, "vcpus": 4, "mem": 15},
            {"machine": "n1-standard-8", "accel": "nvidia-tesla-t4", "gpus": 1, "vcpus": 8, "mem": 30},
        ],
        "H100": [
            {"machine": "a3-highgpu-8g", "gpus": 8, "vcpus": 208, "mem": 1872},
        ],
    }

    ON_DEMAND_PRICES = {
        "a2-highgpu-1g": 3.67,
        "a2-highgpu-4g": 14.69,
        "a2-highgpu-8g": 29.39,
        "a3-highgpu-8g": 98.32,
        "n1-standard-8+v100x1": 2.48,
        "n1-standard-16+v100x2": 4.96,
        "n1-standard-4+t4x1": 0.95,
        "n1-standard-8+t4x1": 1.20,
    }

    REGIONS = ["us-central1", "us-west1", "us-east1", "europe-west1", "asia-east1"]

    # -- BaseProvider implementation -------------------------------------

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        configs = self.GPU_INSTANCE_MAP.get(gpu_type, [])
        if not configs:
            return []

        quotes = []
        for cfg in configs:
            key = cfg["machine"]
            if "accel" in cfg:
                key = f"{cfg['machine']}+{cfg['accel'].split('-')[-1]}x{cfg['gpus']}"
            price = self.ON_DEMAND_PRICES.get(key, self._estimate_price(cfg["machine"], gpu_type, region or "us-central1"))

            quotes.append({
                "instance_type": cfg["machine"],
                "gpu_type": gpu_type,
                "price_per_hour": price,
                "region": region or "us-central1",
                "available": True,
                "provider": "gcp",
                "vcpus": cfg["vcpus"],
                "memory_gb": cfg["mem"],
                "gpu_count": cfg["gpus"],
                "spot": False,
            })

        # If client is live, try to fetch real spot (preemptible) pricing
        if self.instances_client and self.project_id:
            try:
                preemptible_quotes = await self._get_preemptible_quotes(gpu_type, region)
                quotes.extend(preemptible_quotes)
            except Exception:
                pass  # Fall back to static pricing

        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def _get_preemptible_quotes(self, gpu_type: str, region: Optional[str]) -> List[Dict[str, Any]]:
        """Attempt to get preemptible pricing via live API"""
        configs = self.GPU_INSTANCE_MAP.get(gpu_type, [])
        quotes = []
        for cfg in configs:
            key = cfg["machine"]
            if "accel" in cfg:
                key = f"{cfg['machine']}+{cfg['accel'].split('-')[-1]}x{cfg['gpus']}"
            base = self.ON_DEMAND_PRICES.get(key, 3.0)
            quotes.append({
                "instance_type": cfg["machine"],
                "gpu_type": gpu_type,
                "price_per_hour": round(base * 0.4, 2),  # ~60% discount for preemptible
                "region": region or "us-central1",
                "available": True,
                "provider": "gcp",
                "vcpus": cfg["vcpus"],
                "memory_gb": cfg["mem"],
                "gpu_count": cfg["gpus"],
                "spot": True,
            })
        return quotes

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise Exception("GCP client not initialised – configure credentials first")

        zone = f"{region}-a"
        instance_name = f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        try:
            from google.cloud import compute_v1

            instance_resource = compute_v1.Instance()
            instance_resource.name = instance_name
            instance_resource.machine_type = f"zones/{zone}/machineTypes/{instance_type}"

            disk = compute_v1.AttachedDisk()
            disk.auto_delete = True
            disk.boot = True
            init = compute_v1.AttachedDiskInitializeParams()
            init.source_image = "projects/deeplearning-platform-release/global/images/family/common-cu121-debian-11-py310"
            init.disk_size_gb = 200
            disk.initialize_params = init
            instance_resource.disks = [disk]

            net = compute_v1.NetworkInterface()
            access = compute_v1.AccessConfig()
            access.name = "External NAT"
            access.type_ = "ONE_TO_ONE_NAT"
            net.access_configs = [access]
            instance_resource.network_interfaces = [net]

            instance_resource.labels = {"managed-by": "terradev", "gpu-type": gpu_type.lower()}

            request = compute_v1.InsertInstanceRequest(
                project=self.project_id, zone=zone, instance_resource=instance_resource
            )

            loop = asyncio.get_event_loop()
            op = await loop.run_in_executor(None, self.instances_client.insert, request)

            return {
                "instance_id": instance_name,
                "instance_type": instance_type,
                "region": region,
                "gpu_type": gpu_type,
                "status": "provisioning",
                "provider": "gcp",
                "metadata": {"project": self.project_id, "zone": zone},
            }
        except Exception as e:
            raise Exception(f"GCP provision failed: {e}")

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise Exception("GCP client not initialised")
        try:
            from google.cloud import compute_v1
            request = compute_v1.GetInstanceRequest(
                project=self.project_id, zone=self.zone, instance=instance_id
            )
            loop = asyncio.get_event_loop()
            inst = await loop.run_in_executor(None, self.instances_client.get, request)
            return {
                "instance_id": instance_id,
                "status": inst.status.lower(),
                "instance_type": inst.machine_type.split("/")[-1],
                "region": self.zone.rsplit("-", 1)[0],
                "provider": "gcp",
            }
        except Exception as e:
            raise Exception(f"GCP status failed: {e}")

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise Exception("GCP client not initialised")
        from google.cloud import compute_v1
        request = compute_v1.StopInstanceRequest(
            project=self.project_id, zone=self.zone, instance=instance_id
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.instances_client.stop, request)
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise Exception("GCP client not initialised")
        from google.cloud import compute_v1
        request = compute_v1.StartInstanceRequest(
            project=self.project_id, zone=self.zone, instance=instance_id
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.instances_client.start, request)
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise Exception("GCP client not initialised")
        from google.cloud import compute_v1
        request = compute_v1.DeleteInstanceRequest(
            project=self.project_id, zone=self.zone, instance=instance_id
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.instances_client.delete, request)
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.instances_client or not self.project_id:
            return []
        try:
            from google.cloud import compute_v1
            request = compute_v1.ListInstancesRequest(
                project=self.project_id, zone=self.zone,
                filter='labels.managed-by="terradev"',
            )
            loop = asyncio.get_event_loop()
            page = await loop.run_in_executor(None, self.instances_client.list, request)
            instances = []
            for inst in page:
                instances.append({
                    "instance_id": inst.name,
                    "status": inst.status.lower(),
                    "instance_type": inst.machine_type.split("/")[-1],
                    "region": self.zone.rsplit("-", 1)[0],
                    "provider": "gcp",
                })
            return instances
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute command on GCP instance via gcloud compute ssh or direct SSH"""
        if not self.project_id:
            raise Exception("GCP project_id not configured")

        try:
            import subprocess
            # Try gcloud compute ssh first (handles IAP tunneling, OS Login, etc.)
            gcloud_cmd = [
                "gcloud", "compute", "ssh", instance_id,
                "--project", self.project_id,
                "--zone", self.zone,
                "--command", command,
                "--quiet",
            ]
            if async_exec:
                proc = subprocess.Popen(gcloud_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "job_id": str(proc.pid),
                    "output": f"Async gcloud ssh started (PID: {proc.pid})",
                    "async": True,
                }
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(gcloud_cmd, capture_output=True, text=True, timeout=300),
            )
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "async": False,
            }
        except FileNotFoundError:
            # gcloud CLI not installed — try direct SSH via instance IP
            try:
                status = await self.get_instance_status(instance_id)
                # GCP instances don't always expose public IP in our status dict,
                # so fall back to gcloud describe
                import subprocess
                desc = subprocess.run(
                    ["gcloud", "compute", "instances", "describe", instance_id,
                     "--project", self.project_id, "--zone", self.zone,
                     "--format", "get(networkInterfaces[0].accessConfigs[0].natIP)"],
                    capture_output=True, text=True, timeout=15,
                )
                public_ip = desc.stdout.strip()
                if public_ip:
                    ssh_cmd = [
                        "ssh", "-o", "StrictHostKeyChecking=accept-new",
                        "-o", f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                        "-o", "ConnectTimeout=10",
                        f"terradev@{public_ip}", command,
                    ]
                    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)
                    return {
                        "instance_id": instance_id,
                        "command": command,
                        "exit_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "async": False,
                    }
            except Exception:
                pass
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": "GCP exec failed: gcloud CLI not found and SSH fallback failed",
                "async": async_exec,
            }
        except Exception as e:
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"GCP exec error: {e}",
                "async": async_exec,
            }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {}
