#!/usr/bin/env python3
"""
Migration Orchestrator - Lightweight cross-provider workload migration

Core functionality:
- Workload discovery from JobStateManager
- Cost projection using egress optimizer
- Dry-run analysis with detailed breakdown
- Provider compatibility checking
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .job_state_manager import JobStateManager, JobStatus
from .egress_optimizer import estimate_egress_cost, find_cheapest_multihop

logger = logging.getLogger(__name__)


@dataclass
class MigrationPlan:
    """Structured migration plan for dry-run visualization"""
    source: Dict[str, Any]
    target: Dict[str, Any]
    compatibility: Dict[str, Any]
    costs: Dict[str, Any]
    steps: List[str]
    total_downtime: str
    confidence_score: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class WorkloadState:
    """Serializable workload state for migration"""
    job_id: str
    name: str
    framework: str
    gpu_type: str
    gpu_count: int
    current_step: int
    total_steps: int
    checkpoint_size_gb: float
    data_size_gb: float
    env_vars: Dict[str, str]
    region: str
    provider: str


class MigrationOrchestrator:
    """Lightweight migration orchestration with dry-run support"""
    
    def __init__(self):
        self.job_manager = JobStateManager()
        
        # GPU compatibility matrix (performance deltas)
        self.gpu_compatibility = {
            "A100": {"A100": 1.0, "H100": 1.15, "A40": 0.7, "L40S": 0.6},
            "H100": {"H100": 1.0, "A100": 0.87, "A40": 0.6, "L40S": 0.5},
            "A40": {"A40": 1.0, "L40S": 1.1, "A100": 1.4, "H100": 1.7},
            "L40S": {"L40S": 1.0, "A40": 0.9, "A100": 1.7, "H100": 2.0},
        }
    
    def discover_workloads(self) -> List[WorkloadState]:
        """Discover active workloads from JobStateManager"""
        workloads = []
        
        try:
            # Get all jobs and filter manually since list_jobs doesn't support status_filter
            all_jobs = self.job_manager.list_jobs()
            jobs = [job for job in all_jobs if job.status in [JobStatus.RUNNING, JobStatus.PAUSED]]
            
            for job in jobs:
                # Parse job config to extract workload state
                config = json.loads(job.config_json) if job.config_json else {}
                topology = json.loads(job.topology_json) if job.topology_json else {}
                
                workload = WorkloadState(
                    job_id=job.id,
                    name=job.name,
                    framework=config.get("framework", "unknown"),
                    gpu_type=topology.get("gpu_type", "A100"),
                    gpu_count=topology.get("gpu_count", 1),
                    current_step=job.current_step or 0,
                    total_steps=job.total_steps or 1000,
                    checkpoint_size_gb=self._estimate_checkpoint_size(job),
                    data_size_gb=config.get("data_size_gb", 10.0),
                    env_vars=config.get("env_vars", {}),
                    region=topology.get("region", "us-east-1"),
                    provider=config.get("provider", "unknown")
                )
                workloads.append(workload)
                
        except Exception as e:
            logger.error(f"Failed to discover workloads: {e}")
            
        return workloads
    
    def plan_migration(
        self,
        source_provider: str,
        target_provider: str,
        instance_id: Optional[str] = None,
        workload_id: Optional[str] = None,
        dry_run: bool = True
    ) -> MigrationPlan:
        """Plan migration with detailed cost and compatibility analysis"""
        
        # Find source workload
        source_workload = self._find_workload(source_provider, instance_id, workload_id)
        if not source_workload:
            raise ValueError(f"Workload not found: provider={source_provider}, instance={instance_id}, workload={workload_id}")
        
        # Get target pricing (mock for lightweight version)
        target_pricing = self._get_target_pricing(target_provider, source_workload.gpu_type)
        
        # Calculate transfer costs
        transfer_cost = self._calculate_transfer_cost(
            source_workload.provider, source_workload.region,
            target_provider, source_workload.region,
            source_workload.checkpoint_size_gb + source_workload.data_size_gb
        )
        
        # Check GPU compatibility
        target_gpu = self._map_target_gpu(target_provider, source_workload.gpu_type)
        compatibility = self._check_gpu_compatibility(source_workload.gpu_type, target_gpu)
        
        # Build migration steps
        steps = self._build_migration_steps(source_workload, target_provider, target_gpu)
        
        # Calculate total downtime estimate
        downtime = self._estimate_downtime(source_workload, transfer_cost > 0)
        
        # Calculate costs
        total_cost = self._calculate_total_costs(
            source_workload, target_pricing, transfer_cost
        )
        
        # Generate warnings
        warnings = self._generate_warnings(source_workload, target_provider, compatibility)
        
        return MigrationPlan(
            source={
                "provider": source_workload.provider,
                "instance_id": instance_id or "auto-detected",
                "gpu_type": source_workload.gpu_type,
                "gpu_count": source_workload.gpu_count,
                "region": source_workload.region,
                "workload_id": source_workload.job_id,
                "progress": f"{source_workload.current_step}/{source_workload.total_steps}"
            },
            target={
                "provider": target_provider,
                "instance_type": f"{target_gpu.lower()}.1x",
                "gpu_type": target_gpu,
                "gpu_count": source_workload.gpu_count,
                "region": source_workload.region,  # Assume same region for simplicity
                "hourly_cost": target_pricing
            },
            compatibility=compatibility,
            costs=total_cost,
            steps=steps,
            total_downtime=downtime,
            confidence_score=self._calculate_confidence(source_workload, target_provider),
            warnings=warnings
        )
    
    def _find_workload(self, provider: str, instance_id: Optional[str], workload_id: Optional[str]) -> Optional[WorkloadState]:
        """Find workload by provider, instance ID, or workload ID"""
        workloads = self.discover_workloads()
        
        for workload in workloads:
            if workload_id and workload.job_id == workload_id:
                return workload
            if workload.provider.lower() == provider.lower():
                # For lightweight version, assume instance ID matches if provider matches
                return workload
                
        return None
    
    def _estimate_checkpoint_size(self, job) -> float:
        """Estimate checkpoint size based on job metadata"""
        # Lightweight estimation based on framework and GPU count
        base_size = 2.0  # GB base
        if job.framework and "megatron" in job.framework.lower():
            base_size *= 2
        return base_size
    
    def _get_target_pricing(self, provider: str, gpu_type: str) -> float:
        """Get target provider pricing (mock data for lightweight version)"""
        pricing_data = {
            "runpod": {"A100": 2.0, "H100": 3.0, "A40": 1.0, "L40S": 0.8},
            "crusoe": {"A100": 2.2, "H100": 2.85, "A40": 0.8, "L40S": 1.58},
            "coreweave": {"A100": 2.5, "H100": 3.2, "A40": 1.2, "L40S": 1.0},
            "aws": {"A100": 3.0, "H100": 4.0, "A40": 1.5, "L40S": 1.2},
        }
        return pricing_data.get(provider.lower(), {}).get(gpu_type, 2.0)
    
    def _calculate_transfer_cost(self, src_provider: str, src_region: str, 
                               dst_provider: str, dst_region: str, size_gb: float) -> float:
        """Calculate data transfer cost using egress optimizer"""
        try:
            # Use existing egress optimizer
            direct_cost = estimate_egress_cost(src_provider, src_region, dst_provider, dst_region, size_gb)
            
            # Check for cheaper multi-hop routes
            multihop = find_cheapest_multihop(src_provider, dst_provider, size_gb)
            if multihop["total_cost"] < direct_cost:
                return multihop["total_cost"]
            
            return direct_cost
        except Exception:
            # Fallback: assume zero cost for same-provider, $0.05/GB for cross-provider
            return 0.0 if src_provider == dst_provider else size_gb * 0.05
    
    def _map_target_gpu(self, provider: str, source_gpu: str) -> str:
        """Map source GPU to closest equivalent on target provider"""
        # For lightweight version, assume same GPU is available
        gpu_mapping = {
            "runpod": ["A100", "H100", "A40", "L40S"],
            "crusoe": ["A100", "H100", "A40", "L40S"],
            "coreweave": ["A100", "H100", "A40", "L40S"],
            "aws": ["A100", "H100", "A40", "L40S"],
        }
        
        available_gpus = gpu_mapping.get(provider.lower(), ["A100"])
        return source_gpu if source_gpu in available_gpus else available_gpus[0]
    
    def _check_gpu_compatibility(self, source_gpu: str, target_gpu: str) -> Dict[str, Any]:
        """Check GPU compatibility and performance delta"""
        performance_delta = self.gpu_compatibility.get(source_gpu, {}).get(target_gpu, 1.0)
        
        return {
            "gpu_match": source_gpu == target_gpu,
            "performance_delta": performance_delta,
            "performance_change": f"{(performance_delta - 1) * 100:+.1f}%",
            "memory_compatible": True,  # Simplified for lightweight version
            "compute_compatible": True,
        }
    
    def _build_migration_steps(self, workload: WorkloadState, target_provider: str, target_gpu: str) -> List[str]:
        """Build detailed migration steps"""
        total_data_gb = workload.checkpoint_size_gb + workload.data_size_gb
        
        steps = [
            f"1. Checkpoint current job (est. 2 min)",
            f"2. Transfer {total_data_gb:.1f}GB data",
            f"3. Provision {target_provider} {target_gpu} instance",
            f"4. Setup environment and dependencies",
            f"5. Restore checkpoint and resume training",
        ]
        
        return steps
    
    def _estimate_downtime(self, workload: WorkloadState, cross_provider: bool) -> str:
        """Estimate migration downtime"""
        base_time = 5  # minutes
        if cross_provider:
            base_time += 3
        if workload.checkpoint_size_gb > 10:
            base_time += 5
        
        return f"{base_time}-{base_time + 4} minutes"
    
    def _calculate_total_costs(self, workload: WorkloadState, target_hourly: float, transfer_cost: float) -> Dict[str, Any]:
        """Calculate comprehensive cost breakdown"""
        current_hourly = self._get_target_pricing(workload.provider, workload.gpu_type)
        
        return {
            "data_transfer": round(transfer_cost, 4),
            "target_hourly": target_hourly,
            "source_hourly": current_hourly,
            "hourly_savings": round(current_hourly - target_hourly, 4),
            "estimated_monthly_savings": round((current_hourly - target_hourly) * 24 * 30, 2),
        }
    
    def _generate_warnings(self, workload: WorkloadState, target_provider: str, compatibility: Dict[str, Any]) -> List[str]:
        """Generate migration warnings"""
        warnings = []
        
        if compatibility["performance_delta"] < 0.8:
            warnings.append(f"Performance degradation: {compatibility['performance_change']}")
        
        if workload.checkpoint_size_gb > 50:
            warnings.append("Large checkpoint size may increase migration time")
        
        if workload.provider == target_provider:
            warnings.append("Same-provider migration - consider instance upgrade instead")
        
        return warnings
    
    def _calculate_confidence(self, workload: WorkloadState, target_provider: str) -> float:
        """Calculate migration confidence score"""
        confidence = 0.9  # Base confidence
        
        # Boost confidence for same-provider migrations
        if workload.provider == target_provider:
            confidence += 0.05
        
        # Reduce confidence for large checkpoints
        if workload.checkpoint_size_gb > 100:
            confidence -= 0.1
        
        return min(confidence, 1.0)
