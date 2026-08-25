"""Tests for terradev_cli.optimization.cuco_optimizer."""

import pytest

from terradev_cli.core.config import TerradevConfig
from terradev_cli.core.monitoring import MetricsCollector
from terradev_cli.optimization.cuco_optimizer import (
    CUCoOptimizer,
    OptimizationDecision,
    WorkloadProfile,
)


@pytest.fixture
def optimizer():
    config = TerradevConfig._create_default()
    metrics = MetricsCollector()
    return CUCoOptimizer(config, metrics)


def test_optimizer_instantiates(optimizer):
    assert optimizer.p95_boundaries
    assert optimizer.min_gpu_count == 2


def test_analyze_workload(optimizer):
    spec = {
        "name": "moe_training",
        "framework": "transformer",
        "gpu_count": 8,
        "distributed": True,
        "model_parallelism": True,
        "operations": ["allgather", "gemm", "attention"],
        "batch_size": 64,
        "sequence_length": 4096,
        "model_size": 70_000_000_000,
        "provider": "aws",
    }
    profile = optimizer.analyze_workload(spec)
    assert isinstance(profile, WorkloadProfile)
    assert profile.workload_type == "moe"
    assert profile.network_topology in ("nvlink", "infiniband", "roce")


def test_should_optimize_suitable(optimizer):
    profile = optimizer.analyze_workload(
        {
            "name": "llm_training",
            "framework": "transformer",
            "gpu_count": 8,
            "distributed": True,
            "operations": ["allreduce"],
            "network_topology": "infiniband",
        }
    )
    should, reason = optimizer.should_optimize(profile)
    assert should is True
    assert "suitable" in reason.lower()


def test_should_optimize_rejects_small_gpu_count(optimizer):
    profile = optimizer.analyze_workload({"name": "inference", "gpu_count": 1})
    should, reason = optimizer.should_optimize(profile)
    assert should is False
    assert "Insufficient GPUs" in reason


def test_optimize_workload_skip_low_gpu(optimizer):
    profile = optimizer.analyze_workload({"name": "inference", "gpu_count": 1})
    result = optimizer.optimize_workload(profile, "dep-1")
    assert result.decision == OptimizationDecision.SKIP


def test_optimize_workflow_happy_path(optimizer):
    profile = optimizer.analyze_workload(
        {
            "name": "moe_training",
            "framework": "transformer",
            "gpu_count": 8,
            "distributed": True,
            "model_parallelism": True,
            "operations": ["allgather", "gemm", "attention"],
            "batch_size": 128,
            "model_size": 70_000_000_000,
            "provider": "aws",
            "network_topology": "infiniband",
        }
    )
    result = optimizer.optimize_workload(profile, "dep-2")
    assert result.decision in (OptimizationDecision.APPLY, OptimizationDecision.SKIP, OptimizationDecision.RETRY)
    assert result.performance_gain >= 1.0
    assert isinstance(result.reasoning, str)
