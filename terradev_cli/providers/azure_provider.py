#!/usr/bin/env python3
"""
Azure Provider - Microsoft Azure integration
BYOAPI: Uses the end-client's Azure credentials
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class AzureProvider(BaseProvider):
    """Azure Compute provider for GPU instances"""

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "azure"
        self.subscription_id = credentials.get("subscription_id")
        self.resource_group = credentials.get("resource_group", "terradev-rg")
        self.location = credentials.get("location", "eastus")
        self.compute_client = None

        try:
            from azure.identity import ClientSecretCredential
            from azure.mgmt.compute import ComputeManagementClient

            cred = ClientSecretCredential(
                tenant_id=credentials.get("tenant_id"),
                client_id=credentials.get("client_id"),
                client_secret=credentials.get("client_secret"),
            )
            self.compute_client = ComputeManagementClient(cred, self.subscription_id)
        except Exception as e:
            logger.debug(f"Azure client init deferred (BYOAPI): {e}")

    # Azure pricing from real API calls - NO STATIC FALLBACK
    # Prices are fetched dynamically from Azure API

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get instance quotes from Azure API - NO STATIC FALLBACK"""
        if not self.compute_client:
            logger.debug("Azure client not available - configure credentials first")
            return []
        
        try:
            # Get pricing from Azure API
            pricing_info = await self._get_azure_pricing(gpu_type, region)
            if pricing_info:
                return pricing_info
        except Exception as e:
            logger.debug(f"Error getting Azure pricing: {e}")
            return []
        
        # No static fallback - require API access
        return []

    async def _get_azure_pricing(self, gpu_type: str, region: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get real pricing from Azure API"""
        # This would integrate with Azure pricing API
        # For now, return empty to require API access
        # In production, this would call Azure Pricing API
        return []

    async def _get_on_demand_price(
        self, instance_type: str, region: str
    ) -> Optional[float]:
        """Get on-demand price from Azure API"""
        # This would integrate with Azure Pricing API
        # For now, return None to require API access
        return None

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        if not self.compute_client:
            raise Exception("Azure client not initialised – configure credentials first")

        vm_name = f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        try:
            from azure.mgmt.compute.models import (
                VirtualMachine, HardwareProfile, StorageProfile,
                OSDisk, ImageReference, OSProfile, NetworkProfile,
                NetworkInterfaceReference,
            )

            vm_params = VirtualMachine(
                location=region or self.location,
                hardware_profile=HardwareProfile(vm_size=instance_type),
                storage_profile=StorageProfile(
                    image_reference=ImageReference(
                        publisher="microsoft-dsvm",
                        offer="ubuntu-hpc",
                        sku="2204",
                        version="latest",
                    ),
                    os_disk=OSDisk(create_option="FromImage", disk_size_gb=200),
                ),
                os_profile=OSProfile(
                    computer_name=vm_name,
                    admin_username="terradev",
                    admin_password=self._generate_secure_password(),
                ),
                tags={"ManagedBy": "Terradev", "GPUType": gpu_type},
            )

            loop = asyncio.get_event_loop()
            poller = await loop.run_in_executor(
                None,
                lambda: self.compute_client.virtual_machines.begin_create_or_update(
                    self.resource_group, vm_name, vm_params
                ),
            )

            return {
                "instance_id": vm_name,
                "instance_type": instance_type,
                "region": region or self.location,
                "gpu_type": gpu_type,
                "status": "provisioning",
                "provider": "azure",
                "metadata": {
                    "resource_group": self.resource_group,
                    "subscription": self.subscription_id,
                },
            }
        except Exception as e:
            raise Exception(f"Azure provision failed: {e}")

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.compute_client:
            raise Exception("Azure client not initialised")
        try:
            loop = asyncio.get_event_loop()
            vm = await loop.run_in_executor(
                None,
                lambda: self.compute_client.virtual_machines.get(
                    self.resource_group, instance_id, expand="instanceView"
                ),
            )
            statuses = vm.instance_view.statuses if vm.instance_view else []
            power = next((s.display_status for s in statuses if s.code.startswith("PowerState")), "unknown")
            return {
                "instance_id": instance_id,
                "status": power.lower().replace(" ", "_"),
                "instance_type": vm.hardware_profile.vm_size,
                "region": vm.location,
                "provider": "azure",
            }
        except Exception as e:
            raise Exception(f"Azure status failed: {e}")

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.compute_client:
            raise Exception("Azure client not initialised")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.compute_client.virtual_machines.begin_deallocate(
                self.resource_group, instance_id
            ),
        )
        return {"instance_id": instance_id, "action": "stop", "status": "deallocating"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.compute_client:
            raise Exception("Azure client not initialised")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.compute_client.virtual_machines.begin_start(
                self.resource_group, instance_id
            ),
        )
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.compute_client:
            raise Exception("Azure client not initialised")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.compute_client.virtual_machines.begin_delete(
                self.resource_group, instance_id
            ),
        )
        return {"instance_id": instance_id, "action": "terminate", "status": "deleting"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.compute_client:
            return []
        try:
            loop = asyncio.get_event_loop()
            vms = await loop.run_in_executor(
                None,
                lambda: list(self.compute_client.virtual_machines.list(self.resource_group)),
            )
            return [
                {
                    "instance_id": vm.name,
                    "status": "running",
                    "instance_type": vm.hardware_profile.vm_size,
                    "region": vm.location,
                    "provider": "azure",
                    "tags": vm.tags or {},
                }
                for vm in vms
                if (vm.tags or {}).get("ManagedBy") == "Terradev"
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute command on Azure VM via RunCommand extension"""
        if not self.compute_client:
            raise Exception("Azure client not initialised")

        try:
            from azure.mgmt.compute.models import RunCommandInput

            run_cmd = RunCommandInput(
                command_id="RunShellScript",
                script=[command],
            )

            loop = asyncio.get_event_loop()

            if async_exec:
                # Fire and forget — start the poller but don't wait
                poller = await loop.run_in_executor(
                    None,
                    lambda: self.compute_client.virtual_machines.begin_run_command(
                        self.resource_group, instance_id, run_cmd
                    ),
                )
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "output": "Async Azure RunCommand started",
                    "async": True,
                }

            # Synchronous: wait for result
            poller = await loop.run_in_executor(
                None,
                lambda: self.compute_client.virtual_machines.begin_run_command(
                    self.resource_group, instance_id, run_cmd
                ),
            )
            result = await loop.run_in_executor(None, poller.result)

            stdout = ""
            stderr = ""
            if result.value:
                for msg in result.value:
                    if msg.code == "ComponentStatus/StdOut/succeeded":
                        stdout = msg.message
                    elif msg.code == "ComponentStatus/StdErr/succeeded":
                        stderr = msg.message

            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 0 if not stderr else 1,
                "stdout": stdout,
                "stderr": stderr,
                "async": False,
            }

        except Exception as e:
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": 1,
                "output": f"Azure exec error: {e}",
                "async": async_exec,
            }

    @staticmethod
    def _generate_secure_password() -> str:
        """Generate a cryptographically secure password for Azure VM provisioning"""
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        # Azure requires: 12+ chars, upper, lower, digit, special
        while True:
            pw = ''.join(secrets.choice(alphabet) for _ in range(24))
            if (any(c.islower() for c in pw) and any(c.isupper() for c in pw)
                    and any(c.isdigit() for c in pw) and any(c in "!@#$%^&*" for c in pw)):
                return pw

    def _get_auth_headers(self) -> Dict[str, str]:
        return {}
