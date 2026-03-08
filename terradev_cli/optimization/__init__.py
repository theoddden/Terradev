"""
Terradev Optimization Module

This module provides automatic optimization capabilities including:
- CUCo compute-communication kernel optimization
- Warm pool management
- Semantic routing
- Auto-scaling
- Performance monitoring and adaptation
"""

from .cuco_optimizer import CUCoOptimizer, OptimizationResult, OptimizationDecision, WorkloadProfile, CUCoMetrics
from .auto_optimizer import AutoOptimizer, OptimizationPlan, OptimizationContext, OptimizationTrigger

__all__ = [
    "CUCoOptimizer",
    "OptimizationResult", 
    "OptimizationDecision",
    "WorkloadProfile",
    "CUCoMetrics",
    "AutoOptimizer",
    "OptimizationPlan",
    "OptimizationContext", 
    "OptimizationTrigger"
]
