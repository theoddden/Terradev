#!/usr/bin/env python3
"""
GCP Provider - Google Cloud Platform integration

"""

import asyncio
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
        self.reservation_client = None

        try:
            from google.cloud import compute_v1
            from google.oauth2 import service_account

            creds_path = credentials.get("credentials_file")
            if creds_path:
                sa_creds = service_account.Credentials.from_service_account_file(
                    creds_path
                )
                self.instances_client = compute_v1.InstancesClient(credentials=sa_creds)
                self.accelerator_client = compute_v1.AcceleratorTypesClient(
                    credentials=sa_creds
                )
                self.reservation_client = compute_v1.ReservationsClient(
                    credentials=sa_creds
                )
            else:
                self.instances_client = compute_v1.InstancesClient()
                self.accelerator_client = compute_v1.AcceleratorTypesClient()
                self.reservation_client = compute_v1.ReservationsClient()
            self.client = True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"GCP client init deferred (BYOAPI): {e}")
            self.instances_client = None
            self.accelerator_client = None
            self.reservation_client = None

    # -- GPU / instance mapping ------------------------------------------

    GPU_INSTANCE_MAP = {
        "A100": [
            {"machine": "a2-highgpu-1g", "gpus": 1, "vcpus": 12, "mem": 85},
            {"machine": "a2-highgpu-4g", "gpus": 4, "vcpus": 48, "mem": 340},
            {"machine": "a2-highgpu-8g", "gpus": 8, "vcpus": 96, "mem": 680},
        ],
        "V100": [
            {
                "machine": "n1-standard-8",
                "accel": "nvidia-tesla-v100",
                "gpus": 1,
                "vcpus": 8,
                "mem": 30,
            },
            {
                "machine": "n1-standard-16",
                "accel": "nvidia-tesla-v100",
                "gpus": 2,
                "vcpus": 16,
                "mem": 60,
            },
        ],
        "T4": [
            {
                "machine": "n1-standard-4",
                "accel": "nvidia-tesla-t4",
                "gpus": 1,
                "vcpus": 4,
                "mem": 15,
            },
            {
                "machine": "n1-standard-8",
                "accel": "nvidia-tesla-t4",
                "gpus": 1,
                "vcpus": 8,
                "mem": 30,
            },
        ],
        "H100": [
            {
                "machine": "a3-highgpu-8g",
                "gpus": 8,
                "vcpus": 208,
                "mem": 1872,
                "requires_reservation": True,
            },
            {
                "machine": "a4-highgpu-1g",
                "gpus": 1,
                "vcpus": 72,
                "mem": 416,
                "requires_reservation": True,
            },
            {
                "machine": "a4-highgpu-8g",
                "gpus": 8,
                "vcpus": 176,
                "mem": 3328,
                "requires_reservation": True,
            },
            {
                "machine": "a4x-highgpu-1g",
                "gpus": 1,
                "vcpus": 96,
                "mem": 832,
                "requires_reservation": True,
            },
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

    # -- TPU / accelerator-optimized machine family -----------------------
    # Compute Engine is the supported path for v5p, v6e, and TPU7x.
    # Prices are on-demand USD per chip-hour. VM price = price_per_chip_hour * chips.
    # Zones are sourced from https://cloud.google.com/tpu/docs/regions-zones
    TPU_V6E_ZONES = [
        "asia-northeast1-b",
        "europe-west4-a",
        "southamerica-west1-a",
        "us-central1-b",
        "us-east1-d",
        "us-east5-a",
        "us-east5-b",
        "us-south1-ai1b",
    ]
    TPU_V5P_ZONES = [
        "europe-west4-b",
        "us-central1-a",
        "us-east5-a",
    ]
    TPU_V7X_ZONES = [
        "us-central1-ai1a",
        "us-central1-c",
    ]

    TPU_MACHINE_MAP = {
        "TPU-V6E-1T": {
            "machine_type": "ct6e-standard-1t",
            "tpu_chips": 1,
            "vcpus": 44,
            "mem": 176,
            "hbm_gib": 32,
            "price_per_chip_hour": 2.70,
            "image_project": "ubuntu-os-accelerator-images",
            "image_family": "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
            "preferred_zone": "us-east5-a",
            "zones": TPU_V6E_ZONES,
            "regions": [
                "asia-northeast1",
                "europe-west4",
                "southamerica-west1",
                "us-central1",
                "us-east1",
                "us-east5",
                "us-south1",
            ],
        },
        "TPU-V6E-4T": {
            "machine_type": "ct6e-standard-4t",
            "tpu_chips": 4,
            "vcpus": 180,
            "mem": 720,
            "hbm_gib": 128,
            "price_per_chip_hour": 2.70,
            "image_project": "ubuntu-os-accelerator-images",
            "image_family": "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
            "preferred_zone": "us-east5-a",
            "zones": TPU_V6E_ZONES,
            "regions": [
                "asia-northeast1",
                "europe-west4",
                "southamerica-west1",
                "us-central1",
                "us-east1",
                "us-east5",
                "us-south1",
            ],
        },
        "TPU-V6E-8T": {
            "machine_type": "ct6e-standard-8t",
            "tpu_chips": 8,
            "vcpus": 360,
            "mem": 1440,
            "hbm_gib": 256,
            "price_per_chip_hour": 2.70,
            "image_project": "ubuntu-os-accelerator-images",
            "image_family": "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
            "preferred_zone": "us-east5-a",
            "zones": TPU_V6E_ZONES,
            "regions": [
                "asia-northeast1",
                "europe-west4",
                "southamerica-west1",
                "us-central1",
                "us-east1",
                "us-east5",
                "us-south1",
            ],
        },
        "TPU-V5P-4T": {
            "machine_type": "ct5p-hightpu-4t",
            "tpu_chips": 4,
            "vcpus": 208,
            "mem": 448,
            "hbm_gib": 380,
            "price_per_chip_hour": 4.20,
            "image_project": "ubuntu-os-accelerator-images",
            "image_family": "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
            "preferred_zone": "us-east5-a",
            "zones": TPU_V5P_ZONES,
            "regions": ["europe-west4", "us-central1", "us-east5"],
        },
        "TPU-V7X-4T": {
            "machine_type": "tpu7x-standard-4t",
            "tpu_chips": 4,
            "vcpus": 224,
            "mem": 960,
            "hbm_gib": 768,
            "price_per_chip_hour": 12.00,
            "image_project": "ubuntu-os-accelerator-images",
            "image_family": "ubuntu-accel-2404-amd64-tpu-tpu7x",
            "preferred_zone": "us-central1-c",
            "zones": TPU_V7X_ZONES,
            "regions": ["us-central1"],
        },
    }

    # -- BaseProvider implementation -------------------------------------

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get instance quotes with capacity and reservation checks"""
        # Return empty list if no valid credentials
        if not self.credentials:
            return []

        target_region = region or self.zone.rsplit("-", 1)[0]

        # TPU requests are routed through Compute Engine machine types, not GPU maps
        if self._is_tpu_type(gpu_type):
            return await self._get_tpu_quotes(gpu_type, target_region)

        configs = self.GPU_INSTANCE_MAP.get(gpu_type, [])
        if not configs:
            return []

        # CRITICAL: Check zone availability first
        zone_availability = await self._check_zone_availability(gpu_type, target_region)

        quotes = []
        for cfg in configs:
            # Check if this instance requires capacity reservation
            requires_reservation = cfg.get("requires_reservation", False)

            key = cfg["machine"]
            if "accel" in cfg:
                key = f"{cfg['machine']}+{cfg['accel'].split('-')[-1]}x{cfg['gpus']}"
            price = self.ON_DEMAND_PRICES.get(
                key, self._estimate_price(cfg["machine"], gpu_type, target_region)
            )

            quote = {
                "instance_type": cfg["machine"],
                "gpu_type": gpu_type,
                "price_per_hour": price,
                "region": target_region,
                "available": True,
                "provider": "gcp",
                "vcpus": cfg["vcpus"],
                "memory_gb": cfg["mem"],
                "gpu_count": cfg["gpus"],
                "requires_reservation": requires_reservation,
                "zone_availability": zone_availability,
            }

            # CRITICAL: Add reservation workflow for A3/A4/A4X
            if requires_reservation:
                quote["reservation_info"] = {
                    "required": True,
                    "workflow": "capacity_reservation",
                    "lead_time_hours": 24,
                    "auto_reserve_available": True,
                }

            # CRITICAL: Add TPU vs GPU guidance
            if gpu_type in ["A100", "H100"]:
                quote["tpu_alternative"] = await self._get_tpu_alternative(
                    gpu_type, target_region
                )

            # CRITICAL: Add Flex-start VM option for short workloads
            if not requires_reservation:
                quote["flex_start_available"] = (
                    await self._check_flex_start_availability(
                        cfg["machine"], target_region
                    )
                )

            quotes.append(quote)

        # If client is live, try to fetch real spot (preemptible) pricing
        if self.instances_client and self.project_id:
            try:
                preemptible_quotes = await self._get_preemptible_quotes(
                    gpu_type, region
                )
                quotes.extend(preemptible_quotes)
            except Exception:  # noqa: BLE001
                pass  # Fall back to static pricing

        return sorted(quotes, key=lambda q: q["price_per_hour"])

    async def _get_preemptible_quotes(
        self, gpu_type: str, region: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Attempt to get preemptible pricing via live API"""
        configs = self.GPU_INSTANCE_MAP.get(gpu_type, [])
        quotes = []
        for cfg in configs:
            key = cfg["machine"]
            if "accel" in cfg:
                key = f"{cfg['machine']}+{cfg['accel'].split('-')[-1]}x{cfg['gpus']}"
            base = self.ON_DEMAND_PRICES.get(key, 3.0)
            quotes.append(
                {
                    "instance_type": cfg["machine"],
                    "gpu_type": gpu_type,
                    "price_per_hour": round(
                        base * 0.4, 2
                    ),  # ~60% discount for preemptible
                    "region": region or "us-central1",
                    "available": True,
                    "provider": "gcp",
                    "vcpus": cfg["vcpus"],
                    "memory_gb": cfg["mem"],
                    "gpu_count": cfg["gpus"],
                    "spot": True,
                }
            )
        return quotes

    def _is_tpu_type(self, gpu_type: str) -> bool:
        """Return True if the requested accelerator is a TPU."""
        return self._normalize_tpu_key(gpu_type) in self.TPU_MACHINE_MAP

    def _normalize_tpu_key(self, gpu_type: str) -> str:
        """Normalize a TPU request key (e.g. 'tpu-v6e-8t' -> 'TPU-V6E-8T')."""
        if not gpu_type:
            return ""
        key = gpu_type.strip().upper()
        # Accept an optional trailing digit without a 'T' suffix
        if key.startswith("TPU-") and not key.endswith("T"):
            # Map TPU-V6E-8 -> TPU-V6E-8T for convenience
            candidate = f"{key}T"
            if candidate in self.TPU_MACHINE_MAP:
                return candidate
        return key

    def _tpu_zone_for_region(self, cfg: Dict[str, Any], region: str) -> Optional[str]:
        """Return a supported TPU zone in the requested region, or None."""
        candidates = [z for z in cfg.get("zones", []) if z.startswith(region)]
        if not candidates:
            return None
        if self.zone in candidates:
            return self.zone
        return candidates[0]

    async def _get_tpu_quotes(
        self, tpu_key: str, region: str
    ) -> List[Dict[str, Any]]:
        """Return a Compute Engine TPU quote for the requested accelerator."""
        key = self._normalize_tpu_key(tpu_key)
        cfg = self.TPU_MACHINE_MAP.get(key)
        if not cfg:
            return []

        target_region = region or self.zone.rsplit("-", 1)[0]
        region_supported = target_region in cfg["regions"]
        if not region_supported:
            # Do not advertise TPU quotes in regions where this machine type
            # has no known zones; avoids provisioning in the wrong location.
            return []

        target_zone = self._tpu_zone_for_region(cfg, target_region)
        if not target_zone:
            return []

        # The quote region must be the zone's region for consistency.
        target_region = target_zone.rsplit("-", 1)[0]

        price_per_hour = round(cfg["price_per_chip_hour"] * cfg["tpu_chips"], 2)

        return [
            {
                "instance_type": cfg["machine_type"],
                "gpu_type": key,
                "price_per_hour": price_per_hour,
                "region": target_region,
                "available": True,
                "provider": "gcp",
                "vcpus": cfg["vcpus"],
                "memory_gb": cfg["mem"],
                "gpu_count": cfg["tpu_chips"],
                "tpu_chips": cfg["tpu_chips"],
                "tpu_hbm_gib": cfg["hbm_gib"],
                "tpu_type": cfg["machine_type"],
                "requires_reservation": False,
                "zone_availability": {
                    "status": "available",
                    "available_zones": [target_zone],
                    "recommended_zone": target_zone,
                    "recommended_region": target_region,
                },
                "tpu_image": f"projects/{cfg['image_project']}/global/images/family/{cfg['image_family']}",
                "tpu_non_cuda_warning": (
                    "TPU requires JAX, PyTorch XLA, or TensorFlow; "
                    "CUDA-dependent code will not run."
                ),
            }
        ]

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str, ssh_public_key: str = ""
    ) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise RuntimeError("GCP client not initialised – configure credentials first")

        try:
            from google.cloud import compute_v1

            instance_name = (
                f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )

            is_tpu = self._is_tpu_type(gpu_type)

            if is_tpu:
                tpu_key = self._normalize_tpu_key(gpu_type)
                cfg = self.TPU_MACHINE_MAP[tpu_key]
                zone = self._tpu_zone_for_region(cfg, region) or cfg.get("preferred_zone") or f"{region}-a"

                disk_image = (
                    f"projects/{cfg['image_project']}"
                    f"/global/images/family/{cfg['image_family']}"
                )

                labels = {
                    "managed-by": "terradev",
                    "accelerator": "tpu",
                    "tpu-type": tpu_key.lower(),
                }

                scheduling = compute_v1.Scheduling(
                    on_host_maintenance="TERMINATE",
                    provisioning_model="STANDARD",
                )

                tpu_metadata = {
                    "tpu_chips": str(cfg["tpu_chips"]),
                    "tpu_machine_type": cfg["machine_type"],
                    "tpu_software_stack": cfg["image_family"],
                    "tpu_non_cuda_warning": (
                        "TPU requires JAX, PyTorch XLA, or TensorFlow; "
                        "CUDA-dependent code will not run."
                    ),
                }
            else:
                zone = f"{region}-a"
                disk_image = "projects/deeplearning-platform-release/global/images/family/common-cu121-debian-11-py310"
                labels = {"managed-by": "terradev", "gpu-type": gpu_type.lower()}
                scheduling = compute_v1.Scheduling()
                tpu_metadata = {}

            instance_resource = compute_v1.Instance()
            instance_resource.name = instance_name
            instance_resource.machine_type = (
                f"zones/{zone}/machineTypes/{instance_type}"
            )

            disk = compute_v1.AttachedDisk()
            disk.auto_delete = True
            disk.boot = True
            init = compute_v1.AttachedDiskInitializeParams()
            init.source_image = disk_image
            init.disk_size_gb = 200
            disk.initialize_params = init
            instance_resource.disks = [disk]

            net = compute_v1.NetworkInterface()
            access = compute_v1.AccessConfig()
            access.name = "External NAT"
            access.type_ = "ONE_TO_ONE_NAT"
            net.access_configs = [access]
            instance_resource.network_interfaces = [net]

            instance_resource.labels = labels
            instance_resource.scheduling = scheduling

            if tpu_metadata:
                items = [
                    compute_v1.Items(key=k, value=v)
                    for k, v in tpu_metadata.items()
                ]
                instance_resource.metadata = compute_v1.Metadata(items=items)

            request = compute_v1.InsertInstanceRequest(
                project=self.project_id, zone=zone, instance_resource=instance_resource
            )

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.instances_client.insert, request)

            result: Dict[str, Any] = {
                "instance_id": instance_name,
                "instance_type": instance_type,
                "region": region,
                "gpu_type": gpu_type,
                "status": "provisioning",
                "provider": "gcp",
                "metadata": {"project": self.project_id, "zone": zone},
            }
            if is_tpu:
                result["tpu_chips"] = cfg["tpu_chips"]
                result["tpu_type"] = cfg["machine_type"]
                result["tpu_image"] = disk_image
            return result
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"GCP provision failed: {e}") from e

    async def _resolve_zone(self, instance_id: str) -> Optional[str]:
        """Return the zone for an instance, falling back to a project search.

        The zone stored in credentials may not be the zone where a TPU VM was
        actually created (e.g. the TPU type only exists in us-east5-a). We try
        self.zone first, then ask gcloud to locate the instance.
        """
        if not self.project_id:
            return self.zone
        if not hasattr(self, "_zone_cache"):
            self._zone_cache: Dict[str, str] = {}
        if instance_id in self._zone_cache:
            return self._zone_cache[instance_id]

        try:
            from google.cloud import compute_v1

            request = compute_v1.GetInstanceRequest(
                project=self.project_id, zone=self.zone, instance=instance_id
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.instances_client.get, request)
            self._zone_cache[instance_id] = self.zone
            return self.zone
        except Exception:
            pass

        try:
            import subprocess

            result = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "list",
                    "--project",
                    self.project_id,
                    "--filter",
                    f"name={instance_id}",
                    "--format",
                    "value(zone)",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                zones = [z.strip() for z in result.stdout.strip().splitlines() if z.strip()]
                if zones:
                    zone = zones[0]
                    self._zone_cache[instance_id] = zone
                    return zone
        except Exception:
            pass

        return self.zone

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise Exception("GCP client not initialised")
        try:
            from google.cloud import compute_v1

            zone = await self._resolve_zone(instance_id)
            request = compute_v1.GetInstanceRequest(
                project=self.project_id, zone=zone, instance=instance_id
            )
            loop = asyncio.get_running_loop()
            inst = await loop.run_in_executor(None, self.instances_client.get, request)
            return {
                "instance_id": instance_id,
                "status": inst.status.lower(),
                "instance_type": inst.machine_type.split("/")[-1],
                "region": zone.rsplit("-", 1)[0],
                "provider": "gcp",
            }
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"GCP status failed: {e}") from e

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise Exception("GCP client not initialised")
        from google.cloud import compute_v1

        zone = await self._resolve_zone(instance_id)
        request = compute_v1.StopInstanceRequest(
            project=self.project_id, zone=zone, instance=instance_id
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.instances_client.stop, request)
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise Exception("GCP client not initialised")
        from google.cloud import compute_v1

        zone = await self._resolve_zone(instance_id)
        request = compute_v1.StartInstanceRequest(
            project=self.project_id, zone=zone, instance=instance_id
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.instances_client.start, request)
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.instances_client or not self.project_id:
            raise Exception("GCP client not initialised")
        from google.cloud import compute_v1

        zone = await self._resolve_zone(instance_id)
        request = compute_v1.DeleteInstanceRequest(
            project=self.project_id, zone=zone, instance=instance_id
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.instances_client.delete, request)
        return {
            "instance_id": instance_id,
            "action": "terminate",
            "status": "terminating",
        }

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.instances_client or not self.project_id:
            return []
        try:
            from google.cloud import compute_v1

            request = compute_v1.ListInstancesRequest(
                project=self.project_id,
                zone=self.zone,
                filter='labels.managed-by="terradev"',
            )
            loop = asyncio.get_running_loop()
            page = await loop.run_in_executor(None, self.instances_client.list, request)
            instances = []
            for inst in page:
                instances.append(
                    {
                        "instance_id": inst.name,
                        "status": inst.status.lower(),
                        "instance_type": inst.machine_type.split("/")[-1],
                        "region": self.zone.rsplit("-", 1)[0],
                        "provider": "gcp",
                    }
                )
            return instances
        except Exception:  # noqa: BLE001
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute command on GCP instance via gcloud compute ssh or direct SSH"""
        if not self.project_id:
            raise Exception("GCP project_id not configured")

        try:
            import subprocess

            zone = await self._resolve_zone(instance_id)

            # Try gcloud compute ssh first (handles IAP tunneling, OS Login, etc.)
            gcloud_cmd = [
                "gcloud",
                "compute",
                "ssh",
                instance_id,
                "--project",
                self.project_id,
                "--zone",
                zone,
                "--command",
                command,
                "--quiet",
            ]
            if async_exec:
                proc = subprocess.Popen(
                    gcloud_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "job_id": str(proc.pid),
                    "output": f"Async gcloud ssh started (PID: {proc.pid})",
                    "async": True,
                }
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    gcloud_cmd, capture_output=True, text=True, timeout=300
                ),
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
                await self.get_instance_status(instance_id)
                # GCP instances don't always expose public IP in our status dict,
                # so fall back to gcloud describe
                import subprocess

                desc = subprocess.run(
                    [
                        "gcloud",
                        "compute",
                        "instances",
                        "describe",
                        instance_id,
                        "--project",
                        self.project_id,
                        "--zone",
                        self.zone,
                        "--format",
                        "get(networkInterfaces[0].accessConfigs[0].natIP)",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                public_ip = desc.stdout.strip()
                if public_ip:
                    ssh_cmd = [
                        "ssh",
                        "-o",
                        "StrictHostKeyChecking=accept-new",
                        "-o",
                        f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                        "-o",
                        "ConnectTimeout=10",
                        f"terradev@{public_ip}",
                        command,
                    ]
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
            except Exception:  # noqa: BLE001
                pass
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": "GCP exec failed: gcloud CLI not found and SSH fallback failed",
                "async": async_exec,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"GCP exec error: {e}",
                "async": async_exec,
            }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {}

    async def _check_zone_availability(
        self, gpu_type: str, region: str
    ) -> Dict[str, Any]:
        """CRITICAL: Check zone availability for GPU types

        H100s on GCP are often only in us-central1-a and us-east4-b.
        Probing avoids silent failures.
        """
        if not self.accelerator_client:
            return {"status": "unknown", "reason": "client not initialized"}

        try:
            # Map GPU types to accelerator names
            accelerator_map = {
                "A100": "nvidia-tesla-a100",
                "H100": "nvidia-tesla-h100",
                "V100": "nvidia-tesla-v100",
                "T4": "nvidia-tesla-t4",
            }

            accelerator_name = accelerator_map.get(gpu_type)
            if not accelerator_name:
                return {"status": "unknown", "reason": f"unknown GPU type: {gpu_type}"}

            # Check common zones in the region
            zones = [f"{region}-a", f"{region}-b", f"{region}-c", f"{region}-d"]
            available_zones = []

            for zone in zones:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        lambda z=zone: self.accelerator_client.get(
                            project=self.project_id,
                            zone=z,
                            accelerator_type=accelerator_name,
                        ),
                    )
                    available_zones.append(zone)
                except Exception:  # noqa: BLE001
                    continue  # Zone doesn't have this GPU type

            if not available_zones:
                return {
                    "status": "unavailable",
                    "reason": f"{gpu_type} not available in any zone of {region}",
                    "available_zones": [],
                    "recommended_regions": [
                        "us-central1",
                        "us-east4",
                    ],  # Known H100 regions
                }

            return {
                "status": "available",
                "available_zones": available_zones,
                "recommended_zone": available_zones[0],
                "zone_count": len(available_zones),
            }

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Zone availability check failed: {e}")
            return {"status": "error", "reason": str(e)}

    async def _get_tpu_alternative(
        self, gpu_type: str, region: str
    ) -> Optional[Dict[str, Any]]:
        """CRITICAL: Provide TPU vs GPU guidance

        GCP actively pushes TPUs for inference. Compare costs and performance.
        """
        # TPU mapping for GPU alternatives (Compute Engine machine types)
        tpu_alternatives = {
            "A100": {
                "tpu_type": "TPU-V6E-4T",
                "performance_ratio": 0.85,
                "cost_ratio": 0.55,
                "use_case": "training",
            },
            "H100": {
                "tpu_type": "TPU-V6E-8T",
                "performance_ratio": 0.95,
                "cost_ratio": 0.50,
                "use_case": "training",
            },
        }

        alternative = tpu_alternatives.get(gpu_type)
        if not alternative:
            return None

        return {
            "recommended": True,
            "reason": f"TPU {alternative['tpu_type']} offers {alternative['cost_ratio']*100:.0f}% cost with {alternative['performance_ratio']*100:.0f}% performance",
            "tpu_type": alternative["tpu_type"],
            "cost_savings_percent": int((1 - alternative["cost_ratio"]) * 100),
            "performance_impact_percent": int(
                (1 - alternative["performance_ratio"]) * 100
            ),
            "use_case": alternative["use_case"],
            "egress_warning": "Moving data to TPU may incur egress costs if data is in another cloud",
        }

    async def _check_flex_start_availability(
        self, machine_type: str, region: str
    ) -> bool:
        """Check if Flex-start VMs are available for short workloads"""
        # Flex-start is available for most N1-series machines
        if machine_type.startswith("n1-"):
            return True
        # Not available for A2/A3/A4 series
        if machine_type.startswith(("a2-", "a3-", "a4-")):
            return False
        return False

    async def create_capacity_reservation(
        self, gpu_type: str, instance_type: str, region: str, count: int = 1
    ) -> Dict[str, Any]:
        """CRITICAL: Create capacity reservation for A3/A4/A4X instances"""
        if not self.reservation_client:
            raise Exception("Reservation client not initialized")

        reservation_name = (
            f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        try:
            from google.cloud import compute_v1

            # Map GPU types to accelerator configurations
            accelerator_configs = {
                "H100": {
                    "accelerator_type": "nvidia-tesla-h100",
                    "accelerator_count": 8 if "8g" in instance_type else 1,
                }
            }

            config = accelerator_configs.get(gpu_type)
            if not config:
                raise RuntimeError(f"No accelerator config for {gpu_type}")

            # Create reservation
            reservation = compute_v1.Reservation(
                name=reservation_name,
                specific_reservation_required=True,
                specific_reservation=compute_v1.ReservationSpecificReservation(
                    count=count,
                    instance_properties=compute_v1.ReservationSpecificReservationInstanceProperties(
                        machine_type=instance_type,
                        guest_accelerators=[
                            compute_v1.AcceleratorConfig(
                                accelerator_type=config["accelerator_type"],
                                accelerator_count=config["accelerator_count"],
                            )
                        ],
                    ),
                ),
            )

            loop = asyncio.get_running_loop()
            operation = await loop.run_in_executor(
                None,
                lambda: self.reservation_client.insert(
                    project=self.project_id,
                    region=region,
                    reservation_resource=reservation,
                ),
            )

            return {
                "reservation_id": reservation_name,
                "status": "creating",
                "region": region,
                "instance_type": instance_type,
                "gpu_type": gpu_type,
                "count": count,
                "operation": operation.name,
                "estimated_ready_minutes": 15,
            }

        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to create capacity reservation: {e}") from e
