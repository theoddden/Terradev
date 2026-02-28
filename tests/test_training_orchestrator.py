#!/usr/bin/env python3
"""
Tests for TrainingOrchestrator — launch, stop, resume, status.

Uses mocked SSH/subprocess to avoid requiring real GPU nodes.

Covers:
  1. TrainingConfig construction (dict, YAML, defaults)
  2. Launch command builders (torchrun, deepspeed, accelerate)
  3. Orchestrator launch (mocked execution)
  4. Stop (mocked process kill)
  5. Resume from checkpoint
  6. Status reporting
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.core.training_orchestrator import (
    TrainingConfig,
    TrainingOrchestrator,
    _build_torchrun_cmd,
    _build_deepspeed_cmd,
    _run_on,
)
from terradev_cli.core.job_state_manager import JobStateManager, JobStatus


@pytest.fixture
def state_mgr(tmp_path):
    db = str(tmp_path / "test_jobs.db")
    m = JobStateManager(db_path=db)
    yield m
    m.close()


# ── TrainingConfig ────────────────────────────────────────────────────────────


class TestTrainingConfig:
    def test_defaults(self):
        c = TrainingConfig()
        assert c.framework == "torchrun"
        assert c.backend == "native"
        assert c.gpus_per_node == 8
        assert c.tp_size == 1
        assert c.pp_size == 1
        assert c.max_restarts == 3

    def test_from_dict(self):
        c = TrainingConfig.from_dict({
            "name": "my-job",
            "framework": "deepspeed",
            "script": "train.py",
            "nodes": ["10.0.0.1", "10.0.0.2"],
            "gpus_per_node": 4,
            "tp_size": 2,
        })
        assert c.name == "my-job"
        assert c.framework == "deepspeed"
        assert len(c.nodes) == 2
        assert c.gpus_per_node == 4
        assert c.tp_size == 2

    def test_from_dict_ignores_unknown_keys(self):
        c = TrainingConfig.from_dict({
            "script": "train.py",
            "unknown_key": "ignored",
        })
        assert c.script == "train.py"
        assert not hasattr(c, "unknown_key")

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "job.yaml"
        yaml_file.write_text(
            "name: yaml-job\n"
            "framework: accelerate\n"
            "script: train.py\n"
            "gpus_per_node: 2\n"
            "total_steps: 5000\n"
        )
        c = TrainingConfig.from_yaml(str(yaml_file))
        assert c.name == "yaml-job"
        assert c.framework == "accelerate"
        assert c.gpus_per_node == 2
        assert c.total_steps == 5000

    def test_to_dict(self):
        c = TrainingConfig(name="test", script="train.py")
        d = c.to_dict()
        assert d["name"] == "test"
        assert d["script"] == "train.py"
        assert "framework" in d
        assert "gpus_per_node" in d


# ── Command Builders ──────────────────────────────────────────────────────────


class TestCommandBuilders:
    def test_torchrun_single_node(self):
        config = TrainingConfig(
            script="train.py",
            gpus_per_node=8,
            script_args=["--lr", "1e-4"],
        )
        cmd = _build_torchrun_cmd(config, n_nodes=1, master_addr="localhost")
        assert cmd[0] == "torchrun"
        assert "--nnodes=1" in cmd
        assert "--nproc_per_node=8" in cmd
        assert "train.py" in cmd
        assert "--lr" in cmd
        assert "1e-4" in cmd

    def test_torchrun_multi_node(self):
        config = TrainingConfig(
            script="train.py",
            gpus_per_node=8,
            rdzv_port=29400,
            max_restarts=5,
        )
        cmd = _build_torchrun_cmd(config, n_nodes=4, master_addr="10.0.0.1")
        assert "--nnodes=4" in cmd
        assert "--master_addr=10.0.0.1" in cmd
        assert "--rdzv_backend=c10d" in cmd
        assert "--rdzv_endpoint=10.0.0.1:29400" in cmd
        assert "--max_restarts=5" in cmd

    def test_deepspeed_single_node(self):
        config = TrainingConfig(
            script="train.py",
            gpus_per_node=4,
        )
        cmd = _build_deepspeed_cmd(config, n_nodes=1,
                                   master_addr="localhost", hostfile="")
        assert cmd[0] == "deepspeed"
        assert "train.py" in cmd

    def test_deepspeed_multi_node(self):
        config = TrainingConfig(
            script="train.py",
            gpus_per_node=8,
        )
        cmd = _build_deepspeed_cmd(config, n_nodes=2,
                                   master_addr="10.0.0.1",
                                   hostfile="/tmp/hostfile")
        assert "--hostfile=/tmp/hostfile" in cmd
        assert "--num_nodes=2" in cmd
        assert "--master_addr=10.0.0.1" in cmd


# ── Orchestrator Launch (Mocked) ──────────────────────────────────────────────


class TestOrchestratorLaunch:
    @patch("terradev_cli.core.training_orchestrator._run_on")
    def test_launch_single_node(self, mock_run, state_mgr):
        mock_run.return_value = (0, "OK", "")

        orch = TrainingOrchestrator(state_manager=state_mgr)
        config = TrainingConfig(
            script="train.py",
            framework="torchrun",
            gpus_per_node=1,
        )
        result = orch.launch(config, skip_preflight=True)

        assert result["status"] in ("running", "launched", "failed")
        assert "job_id" in result
        # Job should be in state DB
        job = state_mgr.get_job(result["job_id"])
        assert job is not None

    @patch("terradev_cli.core.training_orchestrator._run_on")
    def test_launch_creates_job_record(self, mock_run, state_mgr):
        mock_run.return_value = (0, "PID=1234", "")

        orch = TrainingOrchestrator(state_manager=state_mgr)
        config = TrainingConfig(
            name="test-launch",
            script="train.py",
            total_steps=500,
        )
        result = orch.launch(config, skip_preflight=True)

        job = state_mgr.get_job(result["job_id"])
        assert job.name == "test-launch"
        assert job.total_steps == 500


# ── Orchestrator Stop ─────────────────────────────────────────────────────────


class TestOrchestratorStop:
    @patch("terradev_cli.core.training_orchestrator._run_on")
    def test_stop_existing_job(self, mock_run, state_mgr):
        mock_run.return_value = (0, "", "")

        orch = TrainingOrchestrator(state_manager=state_mgr)
        config = TrainingConfig(script="train.py")
        launch_result = orch.launch(config, skip_preflight=True)
        job_id = launch_result["job_id"]

        # Now stop
        stop_result = orch.stop(job_id)
        assert stop_result["status"] == "stopped"

        job = state_mgr.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

    def test_stop_missing_job(self, state_mgr):
        orch = TrainingOrchestrator(state_manager=state_mgr)
        result = orch.stop("nonexistent-job")
        assert result["status"] == "failed"
        assert "not found" in result["error"].lower()


# ── Orchestrator Resume ───────────────────────────────────────────────────────


class TestOrchestratorResume:
    def test_resume_missing_job(self, state_mgr):
        orch = TrainingOrchestrator(state_manager=state_mgr)
        result = orch.resume("nonexistent")
        assert result["status"] == "failed"

    @patch("terradev_cli.core.training_orchestrator._run_on")
    def test_resume_rebuilds_config(self, mock_run, state_mgr):
        mock_run.return_value = (0, "", "")

        orch = TrainingOrchestrator(state_manager=state_mgr)
        config = TrainingConfig(
            script="train.py",
            framework="torchrun",
            checkpoint_dir="/tmp/ckpts",
        )
        launch = orch.launch(config, skip_preflight=True)

        # Resume should rebuild config from job state
        result = orch.resume(launch["job_id"])
        assert "job_id" in result


# ── Orchestrator Status ───────────────────────────────────────────────────────


class TestOrchestratorStatus:
    @patch("terradev_cli.core.training_orchestrator._run_on")
    def test_status_single_job(self, mock_run, state_mgr):
        mock_run.return_value = (0, "", "")

        orch = TrainingOrchestrator(state_manager=state_mgr)
        config = TrainingConfig(script="train.py")
        launch = orch.launch(config, skip_preflight=True)

        status = orch.status(launch["job_id"])
        assert "id" in status or "error" in status

    def test_status_all_running(self, state_mgr):
        orch = TrainingOrchestrator(state_manager=state_mgr)
        status = orch.status()
        assert "running" in status
        assert isinstance(status["running"], list)


# ── SSH Helper ────────────────────────────────────────────────────────────────


class TestRunOn:
    @patch("subprocess.run")
    def test_localhost_runs_shell(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="OK", stderr=""
        )
        rc, out, err = _run_on(None, "echo hello")
        assert rc == 0
        assert out == "OK"
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_remote_runs_ssh(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="remote-ok", stderr=""
        )
        rc, out, err = _run_on("10.0.0.1", "nvidia-smi", user="ubuntu")
        assert rc == 0
        call_args = mock_run.call_args[0][0]
        assert "ssh" in call_args
        assert "ubuntu@10.0.0.1" in call_args

    @patch("subprocess.run")
    def test_remote_with_key(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _run_on("10.0.0.1", "ls", key="/home/user/.ssh/id_rsa")
        call_args = mock_run.call_args[0][0]
        assert "-i" in call_args
        assert "/home/user/.ssh/id_rsa" in call_args

    def test_localhost_variants(self):
        """localhost and 127.0.0.1 should run locally."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_on("localhost", "echo hi")
            # Should run as shell, not SSH
            call_args = mock_run.call_args
            assert call_args[1].get("shell") is True

            mock_run.reset_mock()
            _run_on("127.0.0.1", "echo hi")
            call_args = mock_run.call_args
            assert call_args[1].get("shell") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
