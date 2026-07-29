"""Tests for terradev_cli.core.migration_orchestrator.

Workload migration is a core moat feature: moving a job to a cheaper provider
with a reliable cost and compatibility analysis before any real instances are
provisioned.
"""

from datetime import datetime

import pytest

from terradev_cli.core.job_state_manager import JobStateManager, JobStatus
from terradev_cli.core.migration_orchestrator import MigrationOrchestrator, WorkloadState


@pytest.fixture
def manager(tmp_path):
    db = tmp_path / "jobs.db"
    jm = JobStateManager(str(db))
    yield jm
    jm.close()


@pytest.fixture
def orchestrator(manager):
    return MigrationOrchestrator()


def test_discover_workloads_finds_running_jobs(manager, orchestrator):
    """Running and paused jobs are discovered as migration candidates."""
    config = {
        "provider": "runpod",
        "data_size_gb": 20.0,
        "env_vars": {"KEY": "VALUE"},
    }
    topology = {"gpu_type": "A100", "gpu_count": 2, "region": "us-east-1"}
    job = manager.create_job(
        name="train-1",
        framework="pytorch",
        config=config,
        nodes=["node-1"],
        topology=topology,
        total_steps=1000,
    )
    manager.update_job_status(job.id, JobStatus.RUNNING)
    manager.update_job_step(job.id, 500)

    workloads = orchestrator.discover_workloads()
    assert len(workloads) == 1
    assert workloads[0].provider == "runpod"
    assert workloads[0].gpu_type == "A100"
    assert workloads[0].gpu_count == 2


def test_plan_migration_success(manager, orchestrator):
    """A migration plan contains source, target, costs, and compatibility."""
    config = {"provider": "runpod", "data_size_gb": 20.0}
    topology = {"gpu_type": "A100", "gpu_count": 2, "region": "us-east-1"}
    job = manager.create_job(
        name="train-1",
        framework="pytorch",
        config=config,
        nodes=["node-1"],
        topology=topology,
        total_steps=1000,
    )
    manager.update_job_status(job.id, JobStatus.RUNNING)

    plan = orchestrator.plan_migration(
        source_provider="runpod",
        target_provider="coreweave",
        dry_run=True,
    )

    assert plan.source["provider"] == "runpod"
    assert plan.target["provider"] == "coreweave"
    assert plan.target["gpu_type"] == "A100"
    assert "data_transfer" in plan.costs
    assert "steps" in plan
    assert plan.confidence_score > 0


def test_plan_migration_unknown_provider_raises(orchestrator):
    """Planning migration without a matching workload raises ValueError."""
    with pytest.raises(ValueError, match="Workload not found"):
        orchestrator.plan_migration("nonexistent", "crusoe")


def test_gpu_compatibility_within_family(orchestrator):
    """Same GPU family is a perfect match."""
    compat = orchestrator._check_gpu_compatibility("A100", "A100")
    assert compat["gpu_match"] is True
    assert compat["performance_delta"] == 1.0


def test_gpu_compatibility_cross_family(orchestrator):
    """Cross-GPU migration reports a performance delta."""
    compat = orchestrator._check_gpu_compatibility("A100", "H100")
    assert compat["gpu_match"] is False
    assert compat["performance_delta"] == 1.15


def test_map_target_gpu_same_as_source(orchestrator):
    """Target provider keeps the source GPU if available."""
    assert orchestrator._map_target_gpu("crusoe", "A100") == "A100"


def test_map_target_gpu_falls_back_when_unavailable(orchestrator):
    """If the source GPU is unavailable, fall back to the provider default."""
    assert orchestrator._map_target_gpu("runpod", "T4") == "A100"


def test_calculate_transfer_cost_same_provider_zero(orchestrator):
    """Same-provider data transfer has zero cost in the fallback model."""
    cost = orchestrator._calculate_transfer_cost("aws", "us-east-1", "aws", "us-east-1", 100)
    assert cost == 0.0


def test_calculate_transfer_cost_cross_provider(orchestrator):
    """Cross-provider transfer falls back to a flat per-GB rate."""
    cost = orchestrator._calculate_transfer_cost("aws", "us-east-1", "gcp", "us-central1", 100)
    assert cost > 0


def test_build_migration_steps_match_workload(orchestrator):
    """Migration steps reference the workload data and target."""
    workload = WorkloadState(
        job_id="j1",
        name="train",
        framework="pytorch",
        gpu_type="A100",
        gpu_count=2,
        current_step=100,
        total_steps=1000,
        checkpoint_size_gb=4.0,
        data_size_gb=20.0,
        env_vars={},
        region="us-east-1",
        provider="runpod",
    )
    steps = orchestrator._build_migration_steps(workload, "coreweave", "A100")
    assert any("Transfer" in s for s in steps)
    assert any("coreweave" in s for s in steps)


def test_warnings_for_same_provider(orchestrator):
    """Same-provider migrations are flagged with a warning."""
    workload = WorkloadState(
        job_id="j1",
        name="train",
        framework="pytorch",
        gpu_type="A100",
        gpu_count=1,
        current_step=0,
        total_steps=1000,
        checkpoint_size_gb=4.0,
        data_size_gb=10.0,
        env_vars={},
        region="us-east-1",
        provider="aws",
    )
    compat = orchestrator._check_gpu_compatibility("A100", "A100")
    warnings = orchestrator._generate_warnings(workload, "aws", compat)
    assert any("Same-provider" in w for w in warnings)
