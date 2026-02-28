#!/usr/bin/env python3
"""
Tests for JobStateManager — SQLite-backed job + checkpoint state.

Covers:
  1. Job CRUD (create, get, list, update status)
  2. Step/cost tracking
  3. Checkpoint recording
  4. Computed properties (GPU-hours, ETA, efficiency)
  5. Running jobs summary (MCP output)
  6. Edge cases (missing job, double-close, concurrent access)
"""

import os
import sys
import tempfile
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.core.job_state_manager import (
    JobStateManager,
    JobRecord,
    JobStatus,
    CheckpointStatus,
)


@pytest.fixture
def mgr(tmp_path):
    """Fresh in-memory-style JobStateManager per test."""
    db = str(tmp_path / "test_jobs.db")
    m = JobStateManager(db_path=db)
    yield m
    m.close()


# ── Job CRUD ─────────────────────────────────────────────────────────────────


class TestJobCRUD:
    def test_create_job(self, mgr):
        job = mgr.create_job(
            name="test-run",
            framework="torchrun",
            config={"lr": 1e-4},
            nodes=["10.0.0.1", "10.0.0.2"],
            total_steps=1000,
        )
        assert job.id.startswith("job-")
        assert job.name == "test-run"
        assert job.framework == "torchrun"
        assert job.status == JobStatus.CREATED
        assert job.total_steps == 1000
        assert len(job.nodes) == 2

    def test_get_job(self, mgr):
        created = mgr.create_job("j1", "deepspeed", {}, ["localhost"])
        fetched = mgr.get_job(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.framework == "deepspeed"

    def test_get_missing_job_returns_none(self, mgr):
        assert mgr.get_job("nonexistent") is None

    def test_list_jobs_all(self, mgr):
        mgr.create_job("a", "torchrun", {}, [])
        mgr.create_job("b", "deepspeed", {}, [])
        jobs = mgr.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_by_status(self, mgr):
        j1 = mgr.create_job("a", "torchrun", {}, [])
        mgr.create_job("b", "deepspeed", {}, [])
        mgr.update_job_status(j1.id, JobStatus.RUNNING)
        running = mgr.list_jobs(status="running")
        assert len(running) == 1
        assert running[0].id == j1.id

    def test_update_job_status(self, mgr):
        job = mgr.create_job("j", "torchrun", {}, [])
        mgr.update_job_status(job.id, JobStatus.RUNNING)
        fetched = mgr.get_job(job.id)
        assert fetched.status == JobStatus.RUNNING

    def test_update_status_to_completed_sets_finished_at(self, mgr):
        job = mgr.create_job("j", "torchrun", {}, [])
        mgr.update_job_status(job.id, JobStatus.RUNNING)
        mgr.update_job_status(job.id, JobStatus.COMPLETED)
        fetched = mgr.get_job(job.id)
        assert fetched.status == JobStatus.COMPLETED
        assert fetched.finished_at is not None


# ── Step & Cost Tracking ──────────────────────────────────────────────────────


class TestStepCostTracking:
    def test_update_step(self, mgr):
        job = mgr.create_job("j", "torchrun", {}, [], total_steps=100)
        mgr.update_job_step(job.id, 50)
        fetched = mgr.get_job(job.id)
        assert fetched.current_step == 50

    def test_update_step_with_cost(self, mgr):
        job = mgr.create_job("j", "torchrun", {}, [], total_steps=100)
        mgr.update_job_step(job.id, 10, cost_usd=0.50)
        mgr.update_job_step(job.id, 20, cost_usd=0.50)
        fetched = mgr.get_job(job.id)
        assert fetched.current_step == 20
        assert fetched.cost_usd == pytest.approx(1.0, abs=0.01)

    def test_set_cost_rate(self, mgr):
        job = mgr.create_job("j", "torchrun", {}, [])
        mgr.set_cost_rate(job.id, 3.50)
        fetched = mgr.get_job(job.id)
        assert fetched.cost_per_gpu_hour == pytest.approx(3.50)


# ── Computed Properties ───────────────────────────────────────────────────────


class TestComputedProperties:
    def test_gpu_count_default(self):
        job = JobRecord(
            id="j1", name="test", framework="torchrun",
            status=JobStatus.RUNNING,
            nodes=["10.0.0.1", "10.0.0.2"],
            config={"gpus_per_node": 8},
        )
        assert job.gpu_count == 16

    def test_gpu_count_single_node(self):
        job = JobRecord(
            id="j1", name="test", framework="torchrun",
            status=JobStatus.RUNNING,
            nodes=[],
            config={},
        )
        # No nodes → max(0,1) * default 8 = 8
        assert job.gpu_count == 8

    def test_elapsed_hours(self):
        job = JobRecord(
            id="j1", name="test", framework="torchrun",
            status=JobStatus.RUNNING,
            started_at=datetime.now() - timedelta(hours=2),
        )
        assert 1.9 < job.elapsed_hours < 2.1

    def test_elapsed_hours_no_start(self):
        job = JobRecord(
            id="j1", name="test", framework="torchrun",
            status=JobStatus.CREATED,
        )
        assert job.elapsed_hours == 0.0

    def test_gpu_hours(self):
        job = JobRecord(
            id="j1", name="test", framework="torchrun",
            status=JobStatus.RUNNING,
            nodes=["10.0.0.1"],
            config={"gpus_per_node": 4},
            started_at=datetime.now() - timedelta(hours=1),
        )
        # ~1hr * 4 GPUs = ~4 GPU-hours
        assert 3.5 < job.gpu_hours < 4.5

    def test_eta_hours(self):
        job = JobRecord(
            id="j1", name="test", framework="torchrun",
            status=JobStatus.RUNNING,
            started_at=datetime.now() - timedelta(hours=1),
            current_step=500,
            total_steps=1000,
            config={},
        )
        # 500 steps in 1 hour → 500 remaining → ~1 hour ETA
        eta = job.eta_hours
        assert eta is not None
        assert 0.8 < eta < 1.2

    def test_eta_hours_no_progress(self):
        job = JobRecord(
            id="j1", name="test", framework="torchrun",
            status=JobStatus.RUNNING,
            started_at=datetime.now(),
            current_step=0,
            total_steps=1000,
        )
        assert job.eta_hours is None

    def test_efficiency(self):
        job = JobRecord(
            id="j1", name="test", framework="torchrun",
            status=JobStatus.RUNNING,
            nodes=["10.0.0.1"],
            config={"gpus_per_node": 8},
            started_at=datetime.now() - timedelta(hours=1),
            current_step=1000,
        )
        # 1000 steps / ~8 GPU-hours ≈ 125 steps/GPU-hour
        assert 100 < job.efficiency < 150

    def test_to_dict_keys(self):
        job = JobRecord(
            id="j1", name="test", framework="torchrun",
            status=JobStatus.RUNNING,
            started_at=datetime.now(),
            current_step=50,
            total_steps=100,
        )
        d = job.to_dict()
        assert "id" in d
        assert "gpu_hours" in d
        assert "eta_hours" in d
        assert "efficiency_steps_per_gpuh" in d
        assert "progress_pct" in d
        assert d["progress_pct"] == 50.0


# ── Checkpoint Recording ──────────────────────────────────────────────────────


class TestCheckpoints:
    def test_create_checkpoint(self, mgr):
        job = mgr.create_job("j", "torchrun", {}, [])
        ckpt = mgr.create_checkpoint(
            job_id=job.id,
            step=100,
            path="/tmp/ckpt-100",
            manifest={"shards": 8},
            size_bytes=1024 * 1024,
        )
        assert ckpt.step == 100
        assert ckpt.job_id == job.id
        assert ckpt.status == CheckpointStatus.WRITING

    def test_list_checkpoints(self, mgr):
        job = mgr.create_job("j", "torchrun", {}, [])
        mgr.create_checkpoint(job.id, 100, "/tmp/ckpt-100")
        mgr.create_checkpoint(job.id, 200, "/tmp/ckpt-200")
        mgr.create_checkpoint(job.id, 300, "/tmp/ckpt-300")
        ckpts = mgr.list_checkpoints(job.id)
        assert len(ckpts) == 3
        # Should be ordered by step desc
        assert ckpts[0].step == 300

    def test_commit_checkpoint(self, mgr):
        job = mgr.create_job("j", "torchrun", {}, [])
        ckpt = mgr.create_checkpoint(job.id, 100, "/tmp/ckpt-100")
        mgr.commit_checkpoint(ckpt.id)
        ckpts = mgr.list_checkpoints(job.id)
        assert ckpts[0].status == CheckpointStatus.COMMITTED


# ── Running Jobs Summary ──────────────────────────────────────────────────────


class TestRunningJobsSummary:
    def test_running_jobs_summary(self, mgr):
        j1 = mgr.create_job("a", "torchrun", {"gpus_per_node": 4}, ["n1"])
        mgr.update_job_status(j1.id, JobStatus.RUNNING)
        mgr.update_job_step(j1.id, 50)

        summary = mgr.running_jobs_summary()
        assert len(summary) == 1
        assert summary[0]["id"] == j1.id
        assert summary[0]["status"] == "running"

    def test_job_metrics(self, mgr):
        j = mgr.create_job("a", "torchrun", {}, [], total_steps=100)
        mgr.update_job_status(j.id, JobStatus.RUNNING)
        mgr.update_job_step(j.id, 30)
        mgr.create_checkpoint(j.id, 30, "/tmp/ckpt-30")

        metrics = mgr.job_metrics(j.id)
        assert metrics["current_step"] == 30
        assert metrics["checkpoint_count"] == 1
        assert metrics["last_checkpoint_step"] == 30

    def test_job_metrics_not_found(self, mgr):
        metrics = mgr.job_metrics("nonexistent")
        assert "error" in metrics


# ── Edge Cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_double_close(self, mgr):
        mgr.close()
        mgr.close()  # Should not raise

    def test_total_cost(self, mgr):
        j1 = mgr.create_job("a", "torchrun", {}, [])
        j2 = mgr.create_job("b", "torchrun", {}, [])
        mgr.update_job_step(j1.id, 10, cost_usd=5.0)
        mgr.update_job_step(j2.id, 10, cost_usd=3.0)
        total = mgr.total_cost()
        assert total == pytest.approx(8.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
