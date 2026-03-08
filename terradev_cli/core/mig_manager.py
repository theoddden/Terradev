#!/usr/bin/env python3
"""
MIG Manager - Multi-Instance GPU enablement and management

CRITICAL FIXES v4.0.0:
- MIG enablement for GPU fractionation on AWS/GCP/Azure
- Cost optimization for smaller workloads
- Automatic MIG configuration detection
- Instance compatibility validation
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MIGManager:
    """Manages Multi-Instance GPU (MIG) configuration and enablement"""

    # MIG-enabled GPU models and their partition profiles
    MIG_ENABLED_GPUS = {
        "A100": {
            "profiles": {
                "1g.5gb": {"memory_gb": 5, "compute": 1/7, "instances": 7},
                "1g.10gb": {"memory_gb": 10, "compute": 1/7, "instances": 7},
                "2g.10gb": {"memory_gb": 10, "compute": 2/7, "instances": 3},
                "3g.20gb": {"memory_gb": 20, "compute": 3/7, "instances": 2},
                "4g.20gb": {"memory_gb": 20, "compute": 4/7, "instances": 1},
                "7g.40gb": {"memory_gb": 40, "compute": 7/7, "instances": 1},
            },
            "total_memory_gb": 40,
            "max_instances": 7,
        },
        "A30": {
            "profiles": {
                "1g.6gb": {"memory_gb": 6, "compute": 1/4, "instances": 4},
                "2g.12gb": {"memory_gb": 12, "compute": 2/4, "instances": 2},
                "4g.24gb": {"memory_gb": 24, "compute": 4/4, "instances": 1},
            },
            "total_memory_gb": 24,
            "max_instances": 4,
        },
        "H100": {
            "profiles": {
                "1g.5gb": {"memory_gb": 5, "compute": 1/7, "instances": 7},
                "1g.10gb": {"memory_gb": 10, "compute": 1/7, "instances": 7},
                "2g.10gb": {"memory_gb": 10, "compute": 2/7, "instances": 3},
                "3g.20gb": {"memory_gb": 20, "compute": 3/7, "instances": 2},
                "4g.20gb": {"memory_gb": 20, "compute": 4/7, "instances": 1},
                "7g.40gb": {"memory_gb": 40, "compute": 7/7, "instances": 1},
            },
            "total_memory_gb": 80,
            "max_instances": 7,
        },
    }

    # Provider-specific MIG enablement commands
    MIG_ENABLE_COMMANDS = {
        "aws": {
            "A100": "sudo nvidia-smi -i 0 -mig 0 && sudo nvidia-smi -i 0 -mig 1",
            "H100": "sudo nvidia-smi -i 0 -mig 0 && sudo nvidia-smi -i 0 -mig 1",
            "A30": "sudo nvidia-smi -i 0 -mig 0 && sudo nvidia-smi -i 0 -mig 1",
        },
        "gcp": {
            "A100": "sudo nvidia-smi -i 0 -mig 0 && sudo nvidia-smi -i 0 -mig 1",
            "H100": "sudo nvidia-smi -i 0 -mig 0 && sudo nvidia-smi -i 0 -mig 1",
            "A30": "sudo nvidia-smi -i 0 -mig 0 && sudo nvidia-smi -i 0 -mig 1",
        },
        "azure": {
            "A100": "sudo nvidia-smi -i 0 -mig 0 && sudo nvidia-smi -i 0 -mig 1",
            "H100": "sudo nvidia-smi -i 0 -mig 0 && sudo nvidia-smi -i 0 -mig 1",
            "A30": "sudo nvidia-smi -i 0 -mig 0 && sudo nvidia-smi -i 0 -mig 1",
        },
    }

    def __init__(self, provider: str):
        self.provider = provider.lower()

    async def check_mig_support(self, gpu_type: str, instance_type: str) -> Dict[str, Any]:
        """Check if MIG is supported for the GPU/instance combination"""
        gpu_info = self.MIG_ENABLED_GPUS.get(gpu_type)
        if not gpu_info:
            return {
                "supported": False,
                "reason": f"GPU type {gpu_type} does not support MIG",
                "mig_enabled_gpus": list(self.MIG_ENABLED_GPUS.keys()),
            }

        # Check provider-specific compatibility
        provider_compatible = await self._check_provider_compatibility(gpu_type, instance_type)
        if not provider_compatible["compatible"]:
            return {
                "supported": False,
                "reason": provider_compatible["reason"],
                "action_required": provider_compatible.get("action_required"),
            }

        return {
            "supported": True,
            "gpu_type": gpu_type,
            "instance_type": instance_type,
            "total_memory_gb": gpu_info["total_memory_gb"],
            "max_instances": gpu_info["max_instances"],
            "available_profiles": gpu_info["profiles"],
            "provider": self.provider,
        }

    async def _check_provider_compatibility(self, gpu_type: str, instance_type: str) -> Dict[str, Any]:
        """Check provider-specific MIG compatibility"""
        if self.provider == "aws":
            return await self._check_aws_mig_compatibility(gpu_type, instance_type)
        elif self.provider == "gcp":
            return await self._check_gcp_mig_compatibility(gpu_type, instance_type)
        elif self.provider == "azure":
            return await self._check_azure_mig_compatibility(gpu_type, instance_type)
        else:
            return {"compatible": True, "reason": "Provider not specifically checked"}

    async def _check_aws_mig_compatibility(self, gpu_type: str, instance_type: str) -> Dict[str, Any]:
        """Check AWS MIG compatibility"""
        # AWS MIG support by instance type
        mig_aws_instances = {
            "A100": ["p4d.24xlarge", "p4de.24xlarge"],
            "H100": ["p5.48xlarge"],
            "A30": ["g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g5.8xlarge", "g5.12xlarge", "g5.16xlarge", "g5.24xlarge"],
        }

        compatible_instances = mig_aws_instances.get(gpu_type, [])
        if instance_type not in compatible_instances:
            return {
                "compatible": False,
                "reason": f"AWS instance {instance_type} does not support MIG for {gpu_type}",
                "compatible_instances": compatible_instances,
                "action_required": f"Use one of: {', '.join(compatible_instances)} or choose non-MIG instance",
            }

        return {"compatible": True}

    async def _check_gcp_mig_compatibility(self, gpu_type: str, instance_type: str) -> Dict[str, Any]:
        """Check GCP MIG compatibility"""
        # GCP MIG support by instance type
        mig_gcp_instances = {
            "A100": ["a2-highgpu-1g", "a2-highgpu-4g", "a2-highgpu-8g"],
            "H100": ["a3-highgpu-8g", "a4-highgpu-1g", "a4-highgpu-8g"],
            "A30": ["a2-megagpu-4g"],
        }

        compatible_instances = mig_gcp_instances.get(gpu_type, [])
        if instance_type not in compatible_instances:
            return {
                "compatible": False,
                "reason": f"GCP instance {instance_type} does not support MIG for {gpu_type}",
                "compatible_instances": compatible_instances,
                "action_required": f"Use one of: {', '.join(compatible_instances)} or choose non-MIG instance",
            }

        return {"compatible": True}

    async def _check_azure_mig_compatibility(self, gpu_type: str, instance_type: str) -> Dict[str, Any]:
        """Check Azure MIG compatibility"""
        # Azure MIG support by instance series
        mig_azure_series = {
            "A100": ["Standard_ND96asr_v4", "Standard_ND96amsr_A100_v4"],
            "H100": ["Standard_ND96isr_H100_v5"],
            "A30": ["Standard_NC96ads_A100_v4"],  # A30 not directly available, using A100 series as proxy
        }

        compatible_instances = [inst for inst in mig_azure_series.get(gpu_type, []) if instance_type.startswith(inst.split('_')[0])]
        if not compatible_instances:
            return {
                "compatible": False,
                "reason": f"Azure instance {instance_type} does not support MIG for {gpu_type}",
                "compatible_series": mig_azure_series.get(gpu_type, []),
                "action_required": f"Use one of: {', '.join(mig_azure_series.get(gpu_type, []))} or choose non-MIG instance",
            }

        return {"compatible": True}

    async def enable_mig(self, gpu_type: str, instance_id: str, provider_executor) -> Dict[str, Any]:
        """Enable MIG on the specified instance"""
        if gpu_type not in self.MIG_ENABLED_GPUS:
            return {
                "success": False,
                "reason": f"GPU type {gpu_type} does not support MIG",
            }

        enable_command = self.MIG_ENABLE_COMMANDS.get(self.provider, {}).get(gpu_type)
        if not enable_command:
            return {
                "success": False,
                "reason": f"No MIG enable command for {self.provider}/{gpu_type}",
            }

        try:
            # Execute MIG enable command on the instance
            result = await provider_executor.execute_command(instance_id, enable_command, async_exec=False)
            
            if result.get("exit_code", 1) == 0:
                # Verify MIG is enabled
                mig_status = await self._check_mig_status(instance_id, provider_executor)
                return {
                    "success": True,
                    "message": "MIG enabled successfully",
                    "mig_status": mig_status,
                    "command_executed": enable_command,
                }
            else:
                return {
                    "success": False,
                    "reason": f"MIG enable command failed: {result.get('stderr', 'Unknown error')}",
                    "exit_code": result.get("exit_code"),
                }

        except Exception as e:
            return {
                "success": False,
                "reason": f"Failed to enable MIG: {str(e)}",
            }

    async def _check_mig_status(self, instance_id: str, provider_executor) -> Dict[str, Any]:
        """Check current MIG status on the instance"""
        try:
            # Check MIG mode
            mig_mode_cmd = "nvidia-smi -q -d MIG | grep 'MIG mode' | awk '{print $4}'"
            result = await provider_executor.execute_command(instance_id, mig_mode_cmd, async_exec=False)
            
            mig_mode = result.get("stdout", "").strip()
            
            # List MIG devices if enabled
            mig_devices = []
            if mig_mode == "Enabled":
                mig_list_cmd = "nvidia-smi -L | grep 'MIG'"
                list_result = await provider_executor.execute_command(instance_id, mig_list_cmd, async_exec=False)
                mig_devices = [line.strip() for line in list_result.get("stdout", "").split('\n') if line.strip()]

            return {
                "mode": mig_mode,
                "devices": mig_devices,
                "enabled": mig_mode == "Enabled",
            }

        except Exception as e:
            return {
                "mode": "Unknown",
                "devices": [],
                "enabled": False,
                "error": str(e),
            }

    async def create_mig_instances(self, gpu_type: str, instance_id: str, profiles: List[str], provider_executor) -> Dict[str, Any]:
        """Create MIG instances with specified profiles"""
        if gpu_type not in self.MIG_ENABLED_GPUS:
            return {
                "success": False,
                "reason": f"GPU type {gpu_type} does not support MIG",
            }

        gpu_info = self.MIG_ENABLED_GPUS[gpu_type]
        created_instances = []
        
        try:
            for i, profile in enumerate(profiles):
                if profile not in gpu_info["profiles"]:
                    return {
                        "success": False,
                        "reason": f"Invalid MIG profile: {profile}",
                        "valid_profiles": list(gpu_info["profiles"].keys()),
                    }

                # Create MIG instance
                create_cmd = f"sudo nvidia-smi mig -cgi {profile} -i 0"
                result = await provider_executor.execute_command(instance_id, create_cmd, async_exec=False)
                
                if result.get("exit_code", 1) != 0:
                    return {
                        "success": False,
                        "reason": f"Failed to create MIG instance {profile}: {result.get('stderr', 'Unknown error')}",
                        "created_instances": created_instances,
                    }

                created_instances.append({
                    "profile": profile,
                    "instance_id": i,
                    "memory_gb": gpu_info["profiles"][profile]["memory_gb"],
                    "compute_fraction": gpu_info["profiles"][profile]["compute"],
                })

            # Verify all instances were created
            mig_status = await self._check_mig_status(instance_id, provider_executor)
            
            return {
                "success": True,
                "created_instances": created_instances,
                "mig_status": mig_status,
                "total_instances": len(created_instances),
            }

        except Exception as e:
            return {
                "success": False,
                "reason": f"Failed to create MIG instances: {str(e)}",
                "created_instances": created_instances,
            }

    async def get_mig_cost_analysis(self, gpu_type: str, instance_type: str, hourly_cost: float) -> Dict[str, Any]:
        """Analyze cost savings with MIG for different workloads"""
        if gpu_type not in self.MIG_ENABLED_GPUS:
            return {
                "mig_supported": False,
                "reason": f"GPU type {gpu_type} does not support MIG",
            }

        gpu_info = self.MIG_ENABLED_GPUS[gpu_type]
        
        analysis = {
            "mig_supported": True,
            "gpu_type": gpu_type,
            "instance_type": instance_type,
            "full_gpu_cost_per_hour": hourly_cost,
            "mig_options": [],
        }

        for profile_name, profile_config in gpu_info["profiles"].items():
            # Estimate cost per MIG instance (proportional to compute fraction)
            mig_cost_per_hour = hourly_cost * profile_config["compute"]
            
            # Calculate potential savings for fractional workloads
            utilization_scenarios = [0.25, 0.5, 0.75, 1.0]
            
            profile_analysis = {
                "profile": profile_name,
                "memory_gb": profile_config["memory_gb"],
                "compute_fraction": profile_config["compute"],
                "instances_per_gpu": profile_config["instances"],
                "cost_per_hour": mig_cost_per_hour,
                "utilization_scenarios": [],
            }

            for utilization in utilization_scenarios:
                # Cost when using MIG instances at this utilization
                mig_instances_needed = max(1, int(utilization * profile_config["instances"]))
                mig_total_cost = mig_instances_needed * mig_cost_per_hour
                
                # Cost when using full GPU at this utilization
                full_gpu_cost = hourly_cost * utilization
                
                savings = full_gpu_cost - mig_total_cost
                savings_percent = (savings / full_gpu_cost * 100) if full_gpu_cost > 0 else 0
                
                profile_analysis["utilization_scenarios"].append({
                    "utilization": utilization,
                    "full_gpu_cost": round(full_gpu_cost, 4),
                    "mig_total_cost": round(mig_total_cost, 4),
                    "savings_per_hour": round(savings, 4),
                    "savings_percent": round(savings_percent, 1),
                })

            analysis["mig_options"].append(profile_analysis)

        return analysis

    async def recommend_mig_configuration(self, gpu_type: str, workload_memory_gb: float, compute_requirement: float) -> Dict[str, Any]:
        """Recommend optimal MIG configuration based on workload requirements"""
        if gpu_type not in self.MIG_ENABLED_GPUS:
            return {
                "mig_supported": False,
                "reason": f"GPU type {gpu_type} does not support MIG",
            }

        gpu_info = self.MIG_ENABLED_GPUS[gpu_type]
        recommendations = []

        for profile_name, profile_config in gpu_info["profiles"].items():
            # Check if profile meets requirements
            memory_ok = profile_config["memory_gb"] >= workload_memory_gb
            compute_ok = profile_config["compute"] >= compute_requirement

            if memory_ok and compute_ok:
                # Calculate efficiency metrics
                memory_efficiency = workload_memory_gb / profile_config["memory_gb"]
                compute_efficiency = compute_requirement / profile_config["compute"]
                overall_efficiency = (memory_efficiency + compute_efficiency) / 2

                recommendations.append({
                    "profile": profile_name,
                    "memory_gb": profile_config["memory_gb"],
                    "compute_fraction": profile_config["compute"],
                    "memory_efficiency": round(memory_efficiency, 2),
                    "compute_efficiency": round(compute_efficiency, 2),
                    "overall_efficiency": round(overall_efficiency, 2),
                    "waste_memory_gb": round(profile_config["memory_gb"] - workload_memory_gb, 2),
                    "waste_compute": round(profile_config["compute"] - compute_requirement, 2),
                })

        # Sort by overall efficiency (best fit first)
        recommendations.sort(key=lambda x: x["overall_efficiency"], reverse=True)

        return {
            "mig_supported": True,
            "gpu_type": gpu_type,
            "workload_requirements": {
                "memory_gb": workload_memory_gb,
                "compute_fraction": compute_requirement,
            },
            "recommendations": recommendations,
            "best_fit": recommendations[0] if recommendations else None,
        }
