#!/usr/bin/env python3
"""Deep state-invariant tests for the JobStateManager.

These tests verify that failures (provisioning crash, checkpoint failure,
cancellation, preemption) never leave the state engine in an inconsistent
drift state such as orphaned RUNNING jobs, leaked active allocations, or
corrupted job records.
"""

from __future__ import annotations

import pytest

from terradev_cli.core.job_state_manager import JobStateManager, JobStatus, CheckpointStatus


@pytest.fixture
def state_manager(tmp_path):
    """A JobStateManager backed by a temporary SQLite database."""
    db_path = tmp_path / "jobs.db"
    return JobStateManager(db_path=str(db_path))


class TestProvisionFailureLeavesNoOrphans:
    """Invariant: a failed/terminated/cancelled job leaves no RUNNING/LAUNCHING state."""

    def test_failed_job_leaves_no_running_or_launching(self, state_manager):
        job = state_manager.create_job(
            name="test-fail-job",
            framework="pytorch",
            config={"gpus_per_node": 8},
            nodes=["node-1"],
            total_steps=100,
        )

        # Simulate the provisioning pipeline progressing then crashing
        state_manager.update_job_status(job.id, JobStatus.LAUNCHING)
        state_manager.update_job_status(job.id, JobStatus.RUNNING)
        state_manager.update_job_status(job.id, JobStatus.FAILED, error_message="network timeout")

        # Invariant: no active allocation state remains
        assert state_manager.list_jobs(status=JobStatus.RUNNING.value) == []
        assert state_manager.list_jobs(status=JobStatus.LAUNCHING.value) == []

        # Invariant: the job record is terminal and has an error and finish time
        updated = state_manager.get_job(job.id)
        assert updated.status == JobStatus.FAILED
        assert updated.error_message == "network timeout"
        assert updated.finished_at is not None

        # Invariant: no cost is leaked
        assert state_manager.total_cost() == 0.0
        assert state_manager.running_jobs_summary() == []

    def test_cancelled_job_leaves_no_running_state(self, state_manager):
        job = state_manager.create_job(
            name="test-cancel-job",
            framework="deepspeed",
            config={"gpus_per_node": 4},
            nodes=["node-a"],
        )

        state_manager.update_job_status(job.id, JobStatus.LAUNCHING)
        state_manager.update_job_status(job.id, JobStatus.RUNNING)
        state_manager.update_job_status(job.id, JobStatus.CANCELLED)

        assert state_manager.list_jobs(status=JobStatus.RUNNING.value) == []
        assert state_manager.list_jobs(status=JobStatus.LAUNCHING.value) == []

        updated = state_manager.get_job(job.id)
        assert updated.status == JobStatus.CANCELLED
        assert updated.finished_at is not None

    def test_preempted_job_cleans_active_state(self, state_manager):
        job = state_manager.create_job(
            name="test-preempt-job",
            framework="pytorch",
            config={},
            nodes=["spot-node-1"],
        )

        state_manager.update_job_status(job.id, JobStatus.RUNNING)
        state_manager.update_job_status(job.id, JobStatus.PREEMPTED)

        assert state_manager.running_jobs_summary() == []
        updated = state_manager.get_job(job.id)
        assert updated.status == JobStatus.PREEMPTED


class TestCheckpointFailureInvariants:
    """Invariant: a failed checkpoint does not corrupt or roll back job state."""

    def test_failed_checkpoint_does_not_leak_writing_state(self, state_manager):
        job = state_manager.create_job(
            name="test-ckpt-job",
            framework="pytorch",
            config={},
            nodes=["node-1"],
        )

        state_manager.update_job_status(job.id, JobStatus.RUNNING)
        ckpt = state_manager.create_checkpoint(job.id, step=10, path="/tmp/ckpt-10")
        assert ckpt.status == CheckpointStatus.WRITING

        state_manager.fail_checkpoint(ckpt.id)

        # Invariant: checkpoint is terminal, not left as WRITING
        ckpts = state_manager.list_checkpoints(job.id)
        assert all(c.status in (CheckpointStatus.FAILED, CheckpointStatus.COMMITTED, CheckpointStatus.PROMOTED, CheckpointStatus.DELETED) for c in ckpts)
        assert any(c.status == CheckpointStatus.FAILED for c in ckpts)

        # Invariant: the parent job is still in the expected state
        job = state_manager.get_job(job.id)
        assert job.status == JobStatus.RUNNING

    def test_checkpoint_retention_does_not_delete_active_job(self, state_manager):
        job = state_manager.create_job(
            name="test-ckpt-retention",
            framework="pytorch",
            config={},
            nodes=["node-1"],
        )

        state_manager.update_job_status(job.id, JobStatus.RUNNING)
        for step in range(5):
            ckpt = state_manager.create_checkpoint(job.id, step=step, path=f"/tmp/ckpt-{step}")
            state_manager.commit_checkpoint(ckpt.id)

        deleted = state_manager.delete_old_checkpoints(job.id, keep=2)
        assert deleted == 3

        # Invariant: job still exists and is still running after retention cleanup
        assert state_manager.get_job(job.id) is not None
        assert state_manager.get_job(job.id).status == JobStatus.RUNNING


class TestJobIsolationInvariants:
    """Invariant: concurrent job creation is isolated and IDs are unique."""

    def test_many_jobs_created_are_isolated(self, state_manager):
        jobs = []
        for i in range(20):
            job = state_manager.create_job(
                name=f"concurrent-job-{i % 5}",  # repeated names
                framework="pytorch",
                config={"id": i},
                nodes=[f"node-{i}"],
            )
            jobs.append(job)

        ids = {j.id for j in jobs}
        assert len(ids) == len(jobs)

        # Each record stores its own config independently
        for i, job in enumerate(jobs):
            loaded = state_manager.get_job(job.id)
            assert loaded.config["id"] == i
            assert loaded.nodes == [f"node-{i}"]
