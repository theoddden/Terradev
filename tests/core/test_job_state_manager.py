"""Tests for terradev_cli.core.job_state_manager.

Job state is the source of truth for training runs, checkpoints, and cost.
These tests guard the SQLite-backed CRUD and aggregate metrics.
"""

from datetime import datetime

import pytest

from terradev_cli.core.job_state_manager import (
    CheckpointRecord,
    CheckpointStatus,
    JobRecord,
    JobStateManager,
    JobStatus,
)


@pytest.fixture
def manager(tmp_path):
    db = tmp_path / "jobs.db"
    m = JobStateManager(db_path=str(db))
    try:
        yield m
    finally:
        m.close()


def test_create_and_get_job(manager):
    """Jobs can be created and retrieved with the stored data."""
    job = manager.create_job(
        name="test-run",
        framework="pytorch",
        config={"gpus_per_node": 4},
        nodes=["node-1"],
        topology={"ring": ["node-1"]},
        total_steps=100,
    )
    assert job.id.startswith("job-")
    assert job.status == JobStatus.CREATED

    loaded = manager.get_job(job.id)
    assert loaded is not None
    assert loaded.name == "test-run"
    assert loaded.config == {"gpus_per_node": 4}
    assert loaded.topology == {"ring": ["node-1"]}
    assert loaded.nodes == ["node-1"]
    assert loaded.total_steps == 100


def test_list_jobs_by_status(manager):
    """list_jobs filters by status and respects limits."""
    created = manager.create_job("a", "pytorch", {}, ["n1"])
    manager.create_job("b", "pytorch", {}, ["n2"])
    manager.update_job_status(created.id, JobStatus.RUNNING)

    running = manager.list_jobs(status=JobStatus.RUNNING.value)
    assert len(running) == 1
    assert running[0].id == created.id

    all_jobs = manager.list_jobs(limit=10)
    assert len(all_jobs) == 2


def test_job_status_transitions_set_timestamps(manager):
    """Status transitions set started_at and finished_at when appropriate."""
    job = manager.create_job("t", "pytorch", {}, ["n1"])
    manager.update_job_status(job.id, JobStatus.RUNNING)
    loaded = manager.get_job(job.id)
    assert loaded.started_at is not None
    assert loaded.status == JobStatus.RUNNING

    manager.update_job_status(job.id, JobStatus.COMPLETED, error_message="")
    loaded = manager.get_job(job.id)
    assert loaded.finished_at is not None
    assert loaded.status == JobStatus.COMPLETED


def test_update_job_step_and_cost(manager):
    """Step and cost updates persist correctly."""
    job = manager.create_job("t", "pytorch", {}, ["n1"])
    manager.update_job_step(job.id, 25, cost_usd=1.50)
    loaded = manager.get_job(job.id)
    assert loaded.current_step == 25
    assert loaded.cost_usd == pytest.approx(1.50)


def test_set_cost_rate_and_checkpoint(manager):
    """Cost rate and last checkpoint can be set."""
    job = manager.create_job("t", "pytorch", {}, ["n1"])
    manager.set_cost_rate(job.id, 3.50)
    manager.set_job_checkpoint(job.id, "ckpt-123")
    loaded = manager.get_job(job.id)
    assert loaded.cost_per_gpu_hour == pytest.approx(3.50)
    assert loaded.last_checkpoint_id == "ckpt-123"


def test_checkpoint_lifecycle(manager):
    """Checkpoints are created, committed, and listed."""
    job = manager.create_job("t", "pytorch", {}, ["n1"])
    ckpt = manager.create_checkpoint(
        job_id=job.id,
        step=10,
        path="/tmp/ckpt",
        manifest={"layers": ["l1"]},
        size_bytes=1_000_000,
    )
    assert ckpt.id.startswith("ckpt-")
    assert ckpt.status == CheckpointStatus.WRITING

    manager.commit_checkpoint(ckpt.id)
    loaded = manager.get_latest_checkpoint(job.id)
    assert loaded is not None
    assert loaded.status == CheckpointStatus.COMMITTED

    assert len(manager.list_checkpoints(job.id)) == 1


def test_promote_and_delete_old_checkpoints(manager):
    """Promoted checkpoints are tracked and old ones can be deleted."""
    job = manager.create_job("t", "pytorch", {}, ["n1"])
    ckpts = [
        manager.create_checkpoint(job.id, i, f"/tmp/{i}")
        for i in range(5)
    ]
    for c in ckpts:
        manager.commit_checkpoint(c.id)
        manager.promote_checkpoint(c.id)

    promoted = [c for c in manager.list_checkpoints(job.id) if c.promoted]
    assert len(promoted) == 5

    deleted = manager.delete_old_checkpoints(job.id, keep=2)
    assert deleted == 3
    remaining = manager.list_checkpoints(job.id)
    assert len([c for c in remaining if c.status != CheckpointStatus.DELETED]) == 2


def test_job_record_properties_and_to_dict():
    """JobRecord computes derived metrics and serializes correctly."""
    now = datetime.now()
    job = JobRecord(
        id="job-1",
        name="x",
        framework="pytorch",
        status=JobStatus.RUNNING,
        config={"gpus_per_node": 4},
        nodes=["n1", "n2"],
        started_at=now,
        current_step=10,
        total_steps=100,
    )
    assert job.gpu_count == 8
    assert job.elapsed_hours >= 0
    assert job.efficiency == pytest.approx(10 / max(job.gpu_hours, 0.001), rel=0.01)

    d = job.to_dict()
    assert d["id"] == "job-1"
    assert d["status"] == "running"
    assert "progress_pct" in d
    assert d["gpu_hours"] >= 0


def test_running_jobs_summary_and_metrics(manager):
    """Aggregate queries return dicts for MCP consumption."""
    job = manager.create_job("t", "pytorch", {}, ["n1"], total_steps=10)
    manager.update_job_status(job.id, JobStatus.RUNNING)
    manager.update_job_step(job.id, 5)
    manager.create_checkpoint(job.id, 5, "/tmp/5")

    summary = manager.running_jobs_summary()
    assert len(summary) == 1
    assert summary[0]["status"] == "running"

    metrics = manager.job_metrics(job.id)
    assert metrics["current_step"] == 5
    assert metrics["checkpoint_count"] == 1


def test_total_cost_aggregation(manager):
    """total_cost returns per-job or overall cost."""
    job = manager.create_job("t", "pytorch", {}, ["n1"])
    manager.update_job_step(job.id, 10, cost_usd=2.50)

    assert manager.total_cost(job.id) == pytest.approx(2.50)
    assert manager.total_cost() == pytest.approx(2.50)
