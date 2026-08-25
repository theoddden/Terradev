"""Tests for mlops commands."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from terradev_cli.commands import cli


# mlops groups and top-level commands


class TestAgenticServing:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "agentic-serving", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "agentic-serving" in result.output

    def test_configure_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "agentic-serving", "configure", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_helm_values_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "agentic-serving", "helm-values", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_k8s_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "agentic-serving", "k8s", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_launch_args_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "agentic-serving", "launch-args", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_lmcache_env_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "agentic-serving", "lmcache-env", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_show_config_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "agentic-serving", "show-config", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestEnvironments:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["environments", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "environments" in result.output

    def test_approve_help(self, runner, mock_api):
        result = runner.invoke(cli, ["environments", "approve", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_approve_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["environments", "approve"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_history_help(self, runner, mock_api):
        result = runner.invoke(cli, ["environments", "history", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_list_help(self, runner, mock_api):
        result = runner.invoke(cli, ["environments", "list", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_promote_help(self, runner, mock_api):
        result = runner.invoke(cli, ["environments", "promote", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_promote_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["environments", "promote"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestEval:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["eval", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "eval" in result.output

    def test_compare_help(self, runner, mock_api):
        result = runner.invoke(cli, ["eval", "compare", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_compare_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["eval", "compare"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_evaluation_help(self, runner, mock_api):
        result = runner.invoke(cli, ["eval", "evaluation", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestLineage:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "lineage" in result.output

    def test_add_input_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "add-input", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_add_input_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "add-input"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_add_output_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "add-output", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_add_output_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "add-output"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_auto_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "auto", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_auto_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "auto"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_complete_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "complete", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_complete_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "complete"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_diff_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "diff", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_diff_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "diff"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_export_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "export", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_graph_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "graph", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_graph_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "graph"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_production_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "production", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_register_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "register", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_register_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "register"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_show_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "show", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_show_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "show"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_trace_help(self, runner, mock_api):
        result = runner.invoke(cli, ["lineage", "trace", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestMigrate:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["migrate", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "migrate" in result.output

    def test_list_workloads_help(self, runner, mock_api):
        result = runner.invoke(cli, ["migrate", "list-workloads", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_migration_help(self, runner, mock_api):
        result = runner.invoke(cli, ["migrate", "migration", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_migration_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["migrate", "migration"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestModelRouter:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["model-router", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "model-router" in result.output

    def test_classify_help(self, runner, mock_api):
        result = runner.invoke(cli, ["model-router", "classify", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_classify_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["model-router", "classify"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_configure_help(self, runner, mock_api):
        result = runner.invoke(cli, ["model-router", "configure", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_llmd_config_help(self, runner, mock_api):
        result = runner.invoke(cli, ["model-router", "llmd-config", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_stats_help(self, runner, mock_api):
        result = runner.invoke(cli, ["model-router", "stats", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["model-router", "test", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestRecord:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["record", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "record" in result.output

    def test_start_help(self, runner, mock_api):
        result = runner.invoke(cli, ["record", "start", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_start_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["record", "start"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_stop_help(self, runner, mock_api):
        result = runner.invoke(cli, ["record", "stop", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_stop_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["record", "stop"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestTriggers:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "triggers" in result.output

    def test_create_help(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "create", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_create_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "create"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_disable_help(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "disable", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_disable_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "disable"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_enable_help(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "enable", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_enable_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "enable"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_fire_help(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "fire", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_fire_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "fire"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_list_help(self, runner, mock_api):
        result = runner.invoke(cli, ["triggers", "list", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestExport:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["export", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["export"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestImport:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["import", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["import"], obj={"api": mock_api})
        assert result.exit_code != 0


class TestFunctionalMlops:
    """Functional DI tests for MLOps/governance commands."""

    @patch("terradev_cli.core.event_system.lineage_service")
    def test_environments_list_runs(self, mock_service, runner, mock_api):
        mock_service.artifacts = {}
        result = runner.invoke(cli, ["environments", "list"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.core.auto_lineage.auto_lineage")
    def test_lineage_show_runs(self, mock_auto, runner, mock_api):
        from datetime import datetime
        from terradev_cli.core.auto_lineage import LineageRecord
        from terradev_cli.core.event_system import Environment

        record = LineageRecord(
            id="exec-123",
            pipeline_id="train-pipeline",
            environment=Environment.PROD,
            status="completed",
            duration_seconds=120.0,
            gpu_hours=10.0,
            compute_cost=5.0,
            hyperparameters={"lr": 0.001},
            datasets=["dataset-1"],
            output_models=["mymodel"],
            timestamp=datetime(2026, 7, 1, 12, 0, 0),
        )
        mock_auto.get_lineage_for_model.return_value = [record]

        result = runner.invoke(cli, ["lineage", "show", "mymodel"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "mymodel" in result.output
        assert "train-pipeline" in result.output

    @patch("terradev_cli.core.event_system.trigger_manager")
    def test_triggers_list_runs(self, mock_tm, runner, mock_api):
        trigger = SimpleNamespace(
            name="t1",
            type=SimpleNamespace(value="event_based"),
            target_pipeline="p1",
            target_environment=SimpleNamespace(value="dev"),
            enabled=True,
            trigger_count=0,
        )
        mock_tm.triggers = {"t1": trigger}

        result = runner.invoke(cli, ["triggers", "list"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "t1" in result.output

    @patch("terradev_cli.core.evaluation_orchestrator.EvaluationOrchestrator")
    def test_eval_model_runs(self, mock_orchestrator, runner, mock_api):
        result = SimpleNamespace(
            evaluation_id="eval-1",
            model_path="/tmp/model",
            endpoint_url=None,
            workload_type="general",
            metrics={"accuracy": 0.95, "latency": 50.0},
            baseline_comparison=None,
            timestamp=datetime(2026, 7, 1, 12, 0, 0),
            duration_seconds=120.0,
            metadata={},
        )
        orch = MagicMock()
        orch.evaluate_model.return_value = result
        mock_orchestrator.return_value = orch

        result = runner.invoke(
            cli,
            ["eval", "evaluation", "--model", "/tmp/model", "--dataset", "/tmp/dataset"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "eval-1" in result.output
        assert "0.950" in result.output