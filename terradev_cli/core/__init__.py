# terradev_cli.core

# CUDA Graph optimization is automatically enabled when these modules are imported
# No user intervention required - everything runs passively in the background

from .semantic_router import NUMAEndpointScorer
from .warm_pool_manager import WarmPoolManager  
from .cuda_graph_integrator import CUDAGraphIntegrator, get_cuda_graph_integrator

# Default instances for easy access
from typing import Optional

_default_numa_scorer: Optional[NUMAEndpointScorer] = None
_default_warm_pool: Optional[WarmPoolManager] = None
_default_integrator: Optional[CUDAGraphIntegrator] = None


def get_default_numa_scorer() -> NUMAEndpointScorer:
    """Get the default NUMA endpoint scorer instance"""
    global _default_numa_scorer
    if _default_numa_scorer is None:
        _default_numa_scorer = NUMAEndpointScorer()
    return _default_numa_scorer


def get_default_warm_pool() -> WarmPoolManager:
    """Get the default warm pool manager instance"""
    global _default_warm_pool
    if _default_warm_pool is None:
        from .warm_pool_manager import WarmPoolConfig
        config = WarmPoolConfig()
        _default_warm_pool = WarmPoolManager(config)
    return _default_warm_pool


def get_default_cuda_graph_integrator() -> CUDAGraphIntegrator:
    """Get the default CUDA Graph integrator instance"""
    global _default_integrator
    if _default_integrator is None:
        _default_integrator = get_cuda_graph_integrator(
            get_default_numa_scorer(),
            get_default_warm_pool()
        )
    return _default_integrator


# Auto-enable CUDA Graph optimization when module is imported
def _enable_cuda_graph_optimization():
    """Automatically enable CUDA Graph optimization in the background"""
    try:
        import asyncio

        # Only schedule background tasks when an event loop is already running.
        # At module-import time the loop is typically *not* running, so
        # creating tasks here would produce "Task was destroyed but pending"
        # warnings.  Instead, defer the start to the first running-loop check.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — nothing to schedule yet.  The warm pool will
            # be started explicitly by commands that need it.
            return

        # Get default instances
        numa_scorer = get_default_numa_scorer()
        warm_pool = get_default_warm_pool()
        integrator = get_default_cuda_graph_integrator()

        # Start warm pool with CUDA Graph optimization
        if not warm_pool._running:
            loop.create_task(warm_pool.start())

    except Exception as e:
        # Fail silently - CUDA Graph optimization is optional
        pass


# Enable CUDA Graph optimization on module import
_enable_cuda_graph_optimization()
