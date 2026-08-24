"""Tests for training commands."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from terradev_cli.commands import cli


# training groups and top-level commands


class TestLora:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "lora" in result.output

    def test_activate_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "activate", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_activate_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "activate"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_add_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "add", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_add_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "add"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_cost_report_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "cost-report", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_drift_check_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "drift-check", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_drift_check_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "drift-check"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_list_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "list", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_list_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "list"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_lorax_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "lorax", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_peft_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "peft", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_register_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "register", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_register_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "register"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_remove_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "remove", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_remove_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "remove"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_rollback_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "rollback", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_rollback_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "rollback"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_sync_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "sync", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_sync_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "sync"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_versions_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "versions", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_versions_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lora", "versions"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestRetrain:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["retrain", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "retrain" in result.output

    def test_deploy_help(self, runner, mock_api):
        result = runner.invoke(cli, ["retrain", "deploy", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_deploy_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["retrain", "deploy"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_detect_help(self, runner, mock_api):
        result = runner.invoke(cli, ["retrain", "detect", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_detect_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["retrain", "detect"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_drift_help(self, runner, mock_api):
        result = runner.invoke(cli, ["retrain", "drift", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_drift_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["retrain", "drift"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_history_help(self, runner, mock_api):
        result = runner.invoke(cli, ["retrain", "history", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestTrain:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "train" in result.output

    def test_resume_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "resume", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_resume_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "resume"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_start_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "start", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_status_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "status", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_stop_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "stop", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_stop_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "stop"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestCheckpoint:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["checkpoint", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["checkpoint"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestMonitor:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["monitor", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0


class TestFunctionalTraining:
    """Functional DI tests for training commands."""

    @patch("terradev_cli.ml_services.lora_registry.get_lora_registry")
    def test_lora_list_registry_runs(self, mock_get_registry, runner, mock_api):
        reg = MagicMock()
        reg.get_registry_stats.return_value = {
            "total_adapter_names": 1,
            "total_versions": 2,
            "active_versions": 1,
            "total_replicas": 0,
            "total_tenants": 0,
        }
        reg.list_all_adapters.return_value = ["customer-a"]
        reg.get_active_version.return_value = None
        mock_get_registry.return_value = reg
        result = runner.invoke(cli, ["lora", "list", "--endpoint", "http://localhost:8000", "--registry"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.core.job_state_manager.JobStateManager")
    def test_train_status_job_runs(self, mock_jsm, runner, mock_api):
        sm = MagicMock()
        sm.job_metrics.return_value = {
            "id": "job-1",
            "name": "test-job",
            "status": "running",
            "framework": "pytorch",
            "current_step": 50,
            "total_steps": 100,
            "progress_pct": 50,
            "elapsed_hours": 1.5,
            "gpu_hours": 0.75,
            "eta_hours": 1.5,
            "cost_usd": 1.23,
            "efficiency_steps_per_gpuh": 66.7,
            "last_checkpoint_id": "ckpt-1",
            "error_message": None,
        }
        mock_jsm.return_value = sm

        result = runner.invoke(cli, ["train", "status", "-j", "job-1"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "job-1" in result.output
        assert "running" in result.output

    @patch("terradev_cli.core.job_state_manager.JobStateManager")
    def test_train_status_no_jobs_runs(self, mock_jsm, runner, mock_api):
        sm = MagicMock()
        sm.running_jobs_summary.return_value = []
        sm.total_cost.return_value = 0.0
        mock_jsm.return_value = sm

        result = runner.invoke(cli, ["train", "status"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.ml_services.peft_import_service.get_peft_import_service")
    def test_peft_list_runs(self, mock_get_svc, runner, mock_api):
        svc = MagicMock()
        svc.list_local_adapters.return_value = [
            SimpleNamespace(
                adapter_id="a1",
                local_path="/tmp/a1",
                base_model="m1",
                rank=8,
                alpha=16,
                peft_type="lora",
            )
        ]
        mock_get_svc.return_value = svc

        result = runner.invoke(cli, ["lora", "peft", "list"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "a1" in result.output


class TestTrainStages:
    """Tests for new train sft/dpo/grpo/pipeline commands."""

    def test_train_sft_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "sft", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--data" in result.output

    def test_train_dpo_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "dpo", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "base-checkpoint" in result.output

    def test_train_grpo_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "grpo", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "rollout-provider" in result.output

    def test_train_pipeline_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "pipeline", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_train_sft_requires_model_and_data(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "sft"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_train_pipeline_dry_run(self, runner, mock_api, tmp_path):
        from pathlib import Path

        example = Path(__file__).resolve().parents[2] / "examples" / "training_pipeline.yaml"
        if not example.exists():
            example = tmp_path / "pipeline.yaml"
            example.write_text(
                """
name: test-pipeline
checkpoint_bucket: s3://my-bucket/checkpoints
defaults:
  provider: auto
stages:
  - name: sft
    type: sft
    model: meta-llama/Llama-3-8B
    data: s3://my-bucket/sft-data.jsonl
    framework: unsloth
"""
            )
        result = runner.invoke(
            cli,
            ["train", "pipeline", "--config", str(example), "--dry-run"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "sft" in result.output