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
        self.quota_client = None

        try:
            from azure.identity import ClientSecretCredential
            from azure.mgmt.compute import ComputeManagementClient
            from azure.mgmt.resource import ResourceManagementClient
            from azure.mgmt.quota import QuotaManagementClient

            cred = ClientSecretCredential(
                tenant_id=credentials.get("tenant_id"),
                client_id=credentials.get("client_id"),
                client_secret=credentials.get("client_secret"),
            )
            self.compute_client = ComputeManagementClient(cred, self.subscription_id)
            self.quota_client = QuotaManagementClient(cred, self.subscription_id)
        except Exception as e:
            logger.debug(f"Azure client init deferred (BYOAPI): {e}")

    # Azure pricing from real API calls - NO STATIC FALLBACK
    # Prices are fetched dynamically from Azure API

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get instance quotes from Azure API with quota validation"""
        if not self.compute_client:
            logger.debug("Azure client not available - configure credentials first")
            return []
        
        target_region = region or self.location
        
        # CRITICAL: Check GPU quota first - biggest operational landmine
        quota_check = await self._check_gpu_quota(gpu_type, target_region)
        if not quota_check["available"]:
            logger.error(f"Azure GPU quota BLOCKED: {quota_check['reason']}")
            return [{
                "provider": "azure",
                "gpu_type": gpu_type,
                "region": target_region,
                "available": False,
                "quota_block": True,
                "reason": quota_check["reason"],
                "action_required": quota_check["action_required"],
            }]
        
        try:
            # Get pricing from Azure API
            pricing_info = await self._get_azure_pricing(gpu_type, target_region)
            if pricing_info:
                # Add quota status to all quotes
                for quote in pricing_info:
                    quote.update({
                        "quota_available": True,
                        "quota_remaining": quota_check["remaining"],
                    })
                return pricing_info
        except Exception as e:
            logger.debug(f"Error getting Azure pricing: {e}")
            return []
        
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

        # CRITICAL: Double-check quota before provisioning
        quota_check = await self._check_gpu_quota(gpu_type, region)
        if not quota_check["available"]:
            raise Exception(f"GPU quota blocked: {quota_check['reason']}. {quota_check['action_required']}")

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
                "quota_remaining": quota_check["remaining"],
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

    async def _check_gpu_quota(self, gpu_type: str, region: str) -> Dict[str, Any]:
        """CRITICAL: Check GPU quota availability - Azure's biggest operational landmine
        
        GPU quota starts at 0 for specialized families. Must be manually approved.
        Quota calculation uses instance_count=1, not raw core count.
        Separate quota pools for subscription vs ML workspace.
        """
        if not self.quota_client:
            return {
                "available": False,
                "reason": "Quota client not initialized - check Azure credentials",
                "action_required": "Configure Azure service principal with quota reader permissions",
                "remaining": 0,
            }
        
        try:
            # Map GPU types to Azure quota resource names
            gpu_quota_mapping = {
                "A100": "Standard_ND_A100_v4_Series",
                "H100": "Standard_ND_H100_v5_Series", 
                "V100": "Standard_NC_v3_Series",
                "T4": "Standard_NCasT4_v3_Series",
                "A10": "Standard_NCA_A10_v4_Series",
                "L40S": "Standard_NCL_L40S_v3_Series",
            }
            
            quota_resource = gpu_quota_mapping.get(gpu_type)
            if not quota_resource:
                return {
                    "available": False,
                    "reason": f"Unknown GPU type: {gpu_type}",
                    "action_required": "Use supported GPU type (A100, H100, V100, T4, A10, L40S)",
                    "remaining": 0,
                }
            
            loop = asyncio.get_event_loop()
            
            # Check subscription-level quota first
            sub_quota = await loop.run_in_executor(
                None,
                lambda: self.quota_client.quota.get(
                    scope=f"/subscriptions/{self.subscription_id}",
                    resource_name=quota_resource,
                )
            )
            
            # CRITICAL: Azure uses instance_count=1 for quota calculation
            # Not the raw core count that users assume
            available_quota = sub_quota.properties.limit.value - sub_quota.properties.current_value.value
            
            if available_quota <= 0:
                return {
                    "available": False,
                    "reason": f"GPU quota exhausted (0/{sub_quota.properties.limit.value})",
                    "action_required": f"Request quota increase for {quota_resource} in Azure portal",
                    "remaining": 0,
                    "limit": sub_quota.properties.limit.value,
                    "current": sub_quota.properties.current_value.value,
                }
            
            # Check if this is ML workspace region (separate quota pool)
            ml_workspace_quotas = await self._check_ml_workspace_quota(gpu_type, region)
            if ml_workspace_quotas and ml_workspace_quotas["available"] <= 0:
                return {
                    "available": False,
                    "reason": f"ML workspace quota exhausted (0/{ml_workspace_quotas['limit']})",
                    "action_required": f"Request quota increase for ML workspace in Azure ML Studio",
                    "remaining": 0,
                    "subscription_quota": available_quota,
                    "ml_workspace_quota": 0,
                }
            
            return {
                "available": True,
                "remaining": available_quota,
                "limit": sub_quota.properties.limit.value,
                "current": sub_quota.properties.current_value.value,
                "ml_workspace_quota": ml_workspace_quotas.get("available") if ml_workspace_quotas else None,
            }
            
        except Exception as e:
            logger.debug(f"Quota check failed: {e}")
            return {
                "available": False,
                "reason": f"Quota check failed: {str(e)}",
                "action_required": "Verify Azure permissions and subscription access",
                "remaining": 0,
            }
    
    async def _check_ml_workspace_quota(self, gpu_type: str, region: str) -> Optional[Dict[str, Any]]:
        """Check ML workspace quota (separate from subscription quota)"""
        try:
            # This would require ML workspace client - for now return None
            # In production, integrate with Azure ML quota APIs
            return None
        except Exception:
            return None
