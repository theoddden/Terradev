"""Tests for training commands."""
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