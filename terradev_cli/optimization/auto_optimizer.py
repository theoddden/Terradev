"""
Terradev Auto-Optimizer with CUCo Integration

This module provides the main auto-optimization engine that integrates
CUCo optimization with Terradev's existing optimization features.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path

from .cuco_optimizer import CUCoOptimizer, OptimizationResult, OptimizationDecision, WorkloadProfile
from ..core.config import TerradevConfig
from ..core.monitoring import MetricsCollector
from ..providers.base import BaseProvider
from ..core.warm_pool_manager import WarmPoolManager
from ..core.semantic_router import SemanticRouter

logger = logging.getLogger(__name__)

class OptimizationTrigger(Enum):
    """Optimization trigger events"""
    DEPLOYMENT = "deployment"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COST_THRESHOLD = "cost_threshold"
    SCHEDULED = "scheduled"
    MANUAL = "manual"

@dataclass
class OptimizationContext:
    """Context for optimization decisions"""
    deployment_id: str
    workload_spec: Dict[str, Any]
    current_metrics: Dict[str, float]
    trigger: OptimizationTrigger
    timestamp: float
    user_preferences: Dict[str, Any]

@dataclass
class OptimizationPlan:
    """Plan for applying optimizations"""
    optimizations: List[str]
    expected_performance_gain: float
    expected_cost_increase: float
    confidence_score: float
    reasoning: str
    estimated_duration: float

class AutoOptimizer:
    """
    Main auto-optimization engine with CUCo integration
    """
    
    def __init__(self, config: TerradevConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics = metrics_collector
        self.cuco_optimizer = CUCoOptimizer(config, metrics_collector)
        self.warm_pool_manager = WarmPoolManager(config)
        self.semantic_router = SemanticRouter(config)
        
        # Optimization state
        self.active_optimizations = {}
        self.optimization_history = {}
        self.performance_baseline = {}
        
        # Configuration
        self.optimization_enabled = config.get("optimization.auto_optimize", True)
        self.cuco_enabled = config.get("optimization.cuco.enabled", True)
        self.performance_threshold = config.get("optimization.performance_threshold", 0.8)
        self.cost_threshold = config.get("optimization.cost_threshold", 1.5)
        
    async def analyze_deployment(self, deployment_id: str, workload_spec: Dict[str, Any]) -> OptimizationPlan:
        """Analyze deployment and create optimization plan"""
        
        logger.info(f"Analyzing deployment {deployment_id} for optimization opportunities")
        
        # Create optimization context
        context = OptimizationContext(
            deployment_id=deployment_id,
            workload_spec=workload_spec,
            current_metrics=self.metrics.get_deployment_metrics(deployment_id),
            trigger=OptimizationTrigger.DEPLOYMENT,
            timestamp=time.time(),
            user_preferences=workload_spec.get("optimization_preferences", {})
        )
        
        # Analyze workload for CUCo optimization
        workload_profile = self.cuco_optimizer.analyze_workload(workload_spec)
        
        # Determine applicable optimizations
        optimizations = []
        expected_gain = 1.0
        expected_cost = 1.0
        reasoning_parts = []
        
        # CUCo optimization
        if self.cuco_enabled and self.cuco_optimizer.should_optimize(workload_profile)[0]:
            cuco_result = self.cuco_optimizer.optimize_workload(workload_profile, deployment_id)
            
            if cuco_result.decision == OptimizationDecision.APPLY:
                optimizations.append("cuco_kernel_optimization")
                expected_gain *= cuco_result.performance_gain
                expected_cost *= (1 + cuco_result.cost_increase)
                reasoning_parts.append(f"CUCo: {cuco_result.performance_gain:.2f}x speedup")
        
        # Warm pool optimization
        if self.should_apply_warm_pool(context):
            optimizations.append("warm_pool_optimization")
            expected_gain *= 1.1  # 10% improvement from warm pool
            expected_cost *= 1.05  # 5% cost increase
            reasoning_parts.append("Warm pool: 10% reduction in cold start latency")
        
        # Semantic routing optimization
        if self.should_apply_semantic_routing(context):
            optimizations.append("semantic_routing")
            expected_gain *= 1.05  # 5% improvement from routing
            expected_cost *= 1.02  # 2% cost increase
            reasoning_parts.append("Semantic routing: 5% improvement in request handling")
        
        # Auto-scaling optimization
        if self.should_apply_auto_scaling(context):
            optimizations.append("auto_scaling")
            expected_gain *= 1.15  # 15% improvement from scaling
            expected_cost *= 0.9   # 10% cost reduction from efficient scaling
            reasoning_parts.append("Auto-scaling: 15% performance improvement, 10% cost reduction")
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(context, optimizations)
        
        # Estimate duration
        estimated_duration = sum([
            self._estimate_optimization_duration(opt) for opt in optimizations
        ])
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No optimizations applicable"
        
        return OptimizationPlan(
            optimizations=optimizations,
            expected_performance_gain=expected_gain,
            expected_cost_increase=expected_cost,
            confidence_score=confidence,
            reasoning=reasoning,
            estimated_duration=estimated_duration
        )
    
    async def apply_optimizations(self, deployment_id: str, plan: OptimizationPlan) -> Dict[str, Any]:
        """Apply optimization plan to deployment"""
        
        logger.info(f"Applying {len(plan.optimizations)} optimizations to deployment {deployment_id}")
        
        results = {
            "deployment_id": deployment_id,
            "applied_optimizations": [],
            "failed_optimizations": [],
            "actual_performance_gain": 1.0,
            "actual_cost_increase": 1.0,
            "total_time": 0.0,
            "errors": []
        }
        
        start_time = time.time()
        
        # Apply optimizations in order
        for optimization in plan.optimizations:
            try:
                if optimization == "cuco_kernel_optimization":
                    result = await self._apply_cuco_optimization(deployment_id)
                elif optimization == "warm_pool_optimization":
                    result = await self._apply_warm_pool_optimization(deployment_id)
                elif optimization == "semantic_routing":
                    result = await self._apply_semantic_routing(deployment_id)
                elif optimization == "auto_scaling":
                    result = await self._apply_auto_scaling(deployment_id)
                else:
                    result = {"success": False, "error": f"Unknown optimization: {optimization}"}
                
                if result["success"]:
                    results["applied_optimizations"].append(optimization)
                    results["actual_performance_gain"] *= result.get("performance_gain", 1.0)
                    results["actual_cost_increase"] *= result.get("cost_multiplier", 1.0)
                else:
                    results["failed_optimizations"].append(optimization)
                    results["errors"].append(result.get("error", "Unknown error"))
                    
            except Exception as e:
                logger.error(f"Failed to apply optimization {optimization}: {str(e)}")
                results["failed_optimizations"].append(optimization)
                results["errors"].append(str(e))
        
        results["total_time"] = time.time() - start_time
        
        # Store optimization history
        self.optimization_history[deployment_id] = {
            "plan": plan,
            "results": results,
            "timestamp": time.time()
        }
        
        # Update performance baseline
        self._update_performance_baseline(deployment_id, results)
        
        logger.info(f"Optimization completed for deployment {deployment_id}: "
                   f"{len(results['applied_optimizations'])} applied, "
                   f"{results['actual_performance_gain']:.2f}x performance gain")
        
        return results
    
    async def monitor_and_optimize(self, deployment_id: str):
        """Continuously monitor deployment and trigger optimizations"""
        
        while deployment_id in self.active_optimizations:
            try:
                # Get current metrics
                current_metrics = self.metrics.get_deployment_metrics(deployment_id)
                
                # Check for performance degradation
                if self._detect_performance_degradation(deployment_id, current_metrics):
                    logger.info(f"Performance degradation detected for deployment {deployment_id}")
                    
                    # Trigger re-optimization
                    await self._trigger_reoptimization(deployment_id, OptimizationTrigger.PERFORMANCE_DEGRADATION)
                
                # Check cost thresholds
                if self._check_cost_threshold(deployment_id, current_metrics):
                    logger.info(f"Cost threshold exceeded for deployment {deployment_id}")
                    
                    # Trigger cost optimization
                    await self._trigger_cost_optimization(deployment_id, OptimizationTrigger.COST_THRESHOLD)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop for deployment {deployment_id}: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def should_apply_warm_pool(self, context: OptimizationContext) -> bool:
        """Determine if warm pool optimization should be applied"""
        
        workload_spec = context.workload_spec
        
        # Check if workload benefits from warm pool
        workload_types = ["llm_inference", "batch_inference", "serving"]
        if workload_spec.get("type", "").lower() not in workload_types:
            return False
        
        # Check request patterns
        if context.current_metrics.get("requests_per_minute", 0) < 10:
            return False
        
        # Check user preferences
        if context.user_preferences.get("disable_warm_pool", False):
            return False
        
        return True
    
    def should_apply_semantic_routing(self, context: OptimizationContext) -> bool:
        """Determine if semantic routing optimization should be applied"""
        
        workload_spec = context.workload_spec
        
        # Check if workload is inference/serving
        if workload_spec.get("type", "").lower() not in ["inference", "serving"]:
            return False
        
        # Check if multiple models/endpoints
        if len(workload_spec.get("endpoints", [])) < 2:
            return False
        
        # Check request volume
        if context.current_metrics.get("requests_per_minute", 0) < 100:
            return False
        
        return True
    
    def should_apply_auto_scaling(self, context: OptimizationContext) -> bool:
        """Determine if auto-scaling optimization should be applied"""
        
        workload_spec = context.workload_spec
        
        # Check if workload is variable
        if not workload_spec.get("variable_load", False):
            return False
        
        # Check if scaling limits are defined
        if not workload_spec.get("min_instances") or not workload_spec.get("max_instances"):
            return False
        
        return True
    
    async def _apply_cuco_optimization(self, deployment_id: str) -> Dict[str, Any]:
        """Apply CUCo kernel optimization"""
        
        try:
            # Get workload specification
            workload_spec = self._get_workload_spec(deployment_id)
            
            # Analyze workload
            profile = self.cuco_optimizer.analyze_workload(workload_spec)
            
            # Apply optimization
            result = self.cuco_optimizer.optimize_workload(profile, deployment_id)
            
            if result.decision == OptimizationDecision.APPLY:
                # Deploy optimized kernels
                success = await self._deploy_cuco_kernels(deployment_id, result.kernel_code)
                
                if success:
                    return {
                        "success": True,
                        "performance_gain": result.performance_gain,
                        "cost_multiplier": 1 + result.cost_increase,
                        "optimization_time": result.optimization_time
                    }
                else:
                    return {"success": False, "error": "Failed to deploy CUCo kernels"}
            else:
                return {"success": False, "error": f"CUCo optimization not applicable: {result.reasoning}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _apply_warm_pool_optimization(self, deployment_id: str) -> Dict[str, Any]:
        """Apply warm pool optimization"""
        
        try:
            # Configure warm pool
            success = await self.warm_pool_manager.configure_warm_pool(deployment_id)
            
            if success:
                return {
                    "success": True,
                    "performance_gain": 1.1,  # 10% improvement
                    "cost_multiplier": 1.05,  # 5% cost increase
                    "optimization_time": 30.0  # 30 seconds
                }
            else:
                return {"success": False, "error": "Failed to configure warm pool"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _apply_semantic_routing(self, deployment_id: str) -> Dict[str, Any]:
        """Apply semantic routing optimization"""
        
        try:
            # Configure semantic routing
            success = await self.semantic_router.configure_routing(deployment_id)
            
            if success:
                return {
                    "success": True,
                    "performance_gain": 1.05,  # 5% improvement
                    "cost_multiplier": 1.02,  # 2% cost increase
                    "optimization_time": 15.0  # 15 seconds
                }
            else:
                return {"success": False, "error": "Failed to configure semantic routing"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _apply_auto_scaling(self, deployment_id: str) -> Dict[str, Any]:
        """Apply auto-scaling optimization"""
        
        try:
            # Configure auto-scaling
            success = await self._configure_auto_scaling(deployment_id)
            
            if success:
                return {
                    "success": True,
                    "performance_gain": 1.15,  # 15% improvement
                    "cost_multiplier": 0.9,   # 10% cost reduction
                    "optimization_time": 45.0  # 45 seconds
                }
            else:
                return {"success": False, "error": "Failed to configure auto-scaling"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _calculate_confidence_score(self, context: OptimizationContext, optimizations: List[str]) -> float:
        """Calculate confidence score for optimization plan"""
        
        base_confidence = 0.8  # Base confidence
        
        # Adjust based on optimization types
        if "cuco_kernel_optimization" in optimizations:
            base_confidence += 0.1  # High confidence in CUCo
        
        if "warm_pool_optimization" in optimizations:
            base_confidence += 0.05  # Medium confidence in warm pool
        
        if "semantic_routing" in optimizations:
            base_confidence += 0.03  # Lower confidence in semantic routing
        
        if "auto_scaling" in optimizations:
            base_confidence += 0.02  # Lower confidence in auto-scaling
        
        # Adjust based on data availability
        if context.current_metrics:
            base_confidence += 0.05
        
        # Adjust based on historical performance
        if deployment_id in self.optimization_history:
            historical_success = self._calculate_historical_success_rate(context.deployment_id)
            base_confidence *= (0.5 + historical_success)
        
        return min(base_confidence, 1.0)
    
    def _estimate_optimization_duration(self, optimization: str) -> float:
        """Estimate duration for optimization"""
        
        durations = {
            "cuco_kernel_optimization": 300.0,  # 5 minutes
            "warm_pool_optimization": 30.0,    # 30 seconds
            "semantic_routing": 15.0,          # 15 seconds
            "auto_scaling": 45.0               # 45 seconds
        }
        
        return durations.get(optimization, 60.0)  # Default 1 minute
    
    def _detect_performance_degradation(self, deployment_id: str, current_metrics: Dict[str, float]) -> bool:
        """Detect if performance has degraded"""
        
        if deployment_id not in self.performance_baseline:
            return False
        
        baseline = self.performance_baseline[deployment_id]
        
        # Check key performance metrics
        current_latency = current_metrics.get("average_latency", float('inf'))
        baseline_latency = baseline.get("average_latency", float('inf'))
        
        if current_latency > baseline_latency * (1 / self.performance_threshold):
            return True
        
        # Check error rates
        current_error_rate = current_metrics.get("error_rate", 0)
        baseline_error_rate = baseline.get("error_rate", 0)
        
        if current_error_rate > baseline_error_rate * 2:
            return True
        
        return False
    
    def _check_cost_threshold(self, deployment_id: str, current_metrics: Dict[str, float]) -> bool:
        """Check if cost thresholds are exceeded"""
        
        current_cost = current_metrics.get("cost_per_hour", 0)
        
        if current_cost > self.cost_threshold:
            return True
        
        return False
    
    async def _trigger_reoptimization(self, deployment_id: str, trigger: OptimizationTrigger):
        """Trigger re-optimization for deployment"""
        
        logger.info(f"Triggering re-optimization for deployment {deployment_id} due to {trigger.value}")
        
        # Get current workload spec
        workload_spec = self._get_workload_spec(deployment_id)
        
        # Analyze and apply optimizations
        plan = await self.analyze_deployment(deployment_id, workload_spec)
        
        if plan.optimizations and plan.confidence_score > 0.7:
            await self.apply_optimizations(deployment_id, plan)
        else:
            logger.info(f"Skipping optimization for deployment {deployment_id}: "
                       f"confidence too low ({plan.confidence_score:.2f})")
    
    async def _trigger_cost_optimization(self, deployment_id: str, trigger: OptimizationTrigger):
        """Trigger cost optimization for deployment"""
        
        logger.info(f"Triggering cost optimization for deployment {deployment_id}")
        
        # Focus on cost-reducing optimizations
        workload_spec = self._get_workload_spec(deployment_id)
        
        # Prefer auto-scaling and warm pool for cost optimization
        context = OptimizationContext(
            deployment_id=deployment_id,
            workload_spec=workload_spec,
            current_metrics=self.metrics.get_deployment_metrics(deployment_id),
            trigger=trigger,
            timestamp=time.time(),
            user_preferences={"optimize_for_cost": True}
        )
        
        optimizations = []
        
        if self.should_apply_auto_scaling(context):
            optimizations.append("auto_scaling")
        
        if self.should_apply_warm_pool(context):
            optimizations.append("warm_pool_optimization")
        
        if optimizations:
            plan = OptimizationPlan(
                optimizations=optimizations,
                expected_performance_gain=1.0,
                expected_cost_increase=0.9,  # Expect cost reduction
                confidence_score=0.8,
                reasoning="Cost optimization focus",
                estimated_duration=60.0
            )
            
            await self.apply_optimizations(deployment_id, plan)
    
    def _get_workload_spec(self, deployment_id: str) -> Dict[str, Any]:
        """Get workload specification for deployment"""
        # This would typically come from deployment storage
        # For now, return a basic structure
        return {
            "deployment_id": deployment_id,
            "type": "llm_training",
            "gpu_count": 4,
            "framework": "pytorch",
            "model_size": 70000000000,  # 70B
            "batch_size": 32,
            "sequence_length": 2048
        }
    
    async def _deploy_cuco_kernels(self, deployment_id: str, kernel_code: str) -> bool:
        """Deploy CUCo optimized kernels"""
        
        try:
            # Save kernel code to deployment directory
            kernel_path = Path(f"/var/terradev/deployments/{deployment_id}/cuco_kernels.cu")
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(kernel_path, 'w') as f:
                f.write(kernel_code)
            
            # Compile kernels (simplified)
            compile_result = await self._compile_cuda_kernels(kernel_path)
            
            if compile_result["success"]:
                # Update deployment configuration
                await self._update_deployment_config(deployment_id, {
                    "cuco_optimized": True,
                    "kernel_path": str(kernel_path),
                    "compiled_kernel_path": compile_result["output_path"]
                })
                return True
            else:
                logger.error(f"Failed to compile CUCo kernels: {compile_result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying CUCo kernels: {str(e)}")
            return False
    
    async def _compile_cuda_kernels(self, kernel_path: Path) -> Dict[str, Any]:
        """Compile CUDA kernels"""
        
        try:
            # This is a simplified compilation process
            # In reality, would use nvcc with proper flags
            import subprocess
            
            output_path = kernel_path.with_suffix('.ptx')
            
            # Compile command (simplified)
            cmd = [
                "nvcc",
                "-ptx",
                "-arch=sm_80",
                str(kernel_path),
                "-o",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return {"success": True, "output_path": str(output_path)}
            else:
                return {"success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Compilation timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _update_deployment_config(self, deployment_id: str, config_update: Dict[str, Any]):
        """Update deployment configuration"""
        
        # This would typically update deployment storage
        config_path = Path(f"/var/terradev/deployments/{deployment_id}/config.json")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        config.update(config_update)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def _configure_auto_scaling(self, deployment_id: str) -> bool:
        """Configure auto-scaling for deployment"""
        
        # This would integrate with the auto-scaling system
        # For now, simulate success
        await asyncio.sleep(1)
        return True
    
    def _calculate_historical_success_rate(self, deployment_id: str) -> float:
        """Calculate historical optimization success rate"""
        
        if deployment_id not in self.optimization_history:
            return 0.5  # Default success rate
        
        history = self.optimization_history[deployment_id]
        results = history.get("results", {})
        
        applied = len(results.get("applied_optimizations", []))
        total = applied + len(results.get("failed_optimizations", []))
        
        if total == 0:
            return 0.5
        
        return applied / total
    
    def _update_performance_baseline(self, deployment_id: str, results: Dict[str, Any]):
        """Update performance baseline for deployment"""
        
        current_metrics = self.metrics.get_deployment_metrics(deployment_id)
        
        # Adjust baseline based on optimization results
        if results["actual_performance_gain"] > 1.0:
            # Performance improved, update baseline
            self.performance_baseline[deployment_id] = current_metrics
        else:
            # No improvement, keep existing baseline
            if deployment_id not in self.performance_baseline:
                self.performance_baseline[deployment_id] = current_metrics
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations"""
        
        total_deployments = len(self.optimization_history)
        total_optimizations = sum(
            len(hist.get("results", {}).get("applied_optimizations", []))
            for hist in self.optimization_history.values()
        )
        
        cuco_optimizations = sum(
            1 for hist in self.optimization_history.values()
            if "cuco_kernel_optimization" in hist.get("results", {}).get("applied_optimizations", [])
        )
        
        avg_performance_gain = 1.0
        if total_optimizations > 0:
            gains = [
                hist.get("results", {}).get("actual_performance_gain", 1.0)
                for hist in self.optimization_history.values()
            ]
            avg_performance_gain = sum(gains) / len(gains)
        
        return {
            "total_deployments": total_deployments,
            "total_optimizations": total_optimizations,
            "cuco_optimizations": cuco_optimizations,
            "average_performance_gain": avg_performance_gain,
            "optimization_success_rate": total_optimizations / max(total_deployments, 1),
            "active_monitoring": len(self.active_optimizations)
        }
