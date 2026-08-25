"""Functional DI tests for terradev_cli/commands/mlops.py lineage/eval/router sections."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from terradev_cli.commands import cli


def _make_artifact():
    return SimpleNamespace(
        id="art_123",
        name="demo",
        type=SimpleNamespace(value="model"),
        environment=SimpleNamespace(value="dev"),
        version="v1",
        created_at=SimpleNamespace(strftime=lambda fmt: "2099-12-31 00:00"),
        created_by="tester",
    )


class TestLineageFunctional:
    @patch("terradev_cli.commands.mlops.TerradevAPI")
    @patch("terradev_cli.core.event_system.lineage_service")
    def test_lineage_register(self, mock_service, MockAPI, runner):
        from terradev_cli.core.event_system import ArtifactType, Environment

        mock_service.register_artifact.return_value = _make_artifact()
        mock_api = MagicMock()
        mock_api._provider_creds.return_value = {}
        MockAPI.return_value = mock_api

        result = runner.invoke(
            cli,
            [
                "lineage",
                "register",
                "model",
                "demo",
                "s3://demo",
                "--env",
                "dev",
                "--user",
                "tester",
            ],
        )
        assert result.exit_code == 0
        assert "Registered" in result.output

    @patch("terradev_cli.core.event_system.lineage_service")
    def test_lineage_graph_empty(self, mock_service, runner):
        mock_service.get_lineage.return_value = {"parents": [], "children": []}
        result = runner.invoke(cli, ["lineage", "graph", "art_123"])
        assert result.exit_code == 0
        assert "No lineage" in result.output

    @patch("terradev_cli.core.event_system.lineage_service")
    def test_lineage_graph_populated(self, mock_service, runner):
        art = _make_artifact()
        mock_service.get_lineage.return_value = {"parents": [art], "children": [art]}
        result = runner.invoke(cli, ["lineage", "graph", "art_123"])
        assert result.exit_code == 0
        assert "Parents" in result.output

    @patch("terradev_cli.core.event_system.lineage_service")
    def test_lineage_production_empty(self, mock_service, runner):
        mock_service.get_production_artifacts.return_value = []
        result = runner.invoke(cli, ["lineage", "production"])
        assert result.exit_code == 0
        assert "No production" in result.output

    @patch("terradev_cli.core.event_system.lineage_service")
    def test_lineage_production_populated(self, mock_service, runner):
        mock_service.get_production_artifacts.return_value = [_make_artifact()]
        result = runner.invoke(cli, ["lineage", "production"])
        assert result.exit_code == 0
        assert "demo" in result.output

    @patch("terradev_cli.core.auto_lineage.auto_lineage")
    def test_lineage_trace_checkpoint(self, mock_auto, runner):
        mock_auto.trace_from_checkpoint.return_value = {
            "created_by": {
                "execution_id": "ex_1",
                "pipeline_id": "pipe",
                "environment": "dev",
                "timestamp": "t",
            },
            "inputs": {
                "datasets": [{"name": "ds", "id": "id1234567890"}],
                "models": [{"name": "m", "id": "id1234567890"}],
            },
            "ancestors": [{
                "execution_id": "ex_0",
                "pipeline_id": "pipe0",
                "environment": "dev",
                "timestamp": "t0",
            }],
        }
        result = runner.invoke(cli, ["lineage", "trace", "--checkpoint", "ckpt_1"])
        assert result.exit_code == 0
        assert "Tracing lineage" in result.output

    @patch("terradev_cli.core.auto_lineage.auto_lineage")
    def test_lineage_trace_error(self, mock_auto, runner):
        mock_auto.trace_from_checkpoint.return_value = {"error": "not found"}
        result = runner.invoke(cli, ["lineage", "trace", "--checkpoint", "ckpt_1"])
        assert result.exit_code in (0, 1)
        assert "ERROR" in result.output

    @patch("terradev_cli.core.auto_lineage.auto_lineage")
    def test_lineage_start(self, mock_auto, runner):
        mock_auto.start_execution.return_value = SimpleNamespace(id="ex_1")
        result = runner.invoke(
            cli,
            ["lineage", "auto", "--pipeline", "pipe", "--env", "dev", "--triggered-by", "test"],
        )
        assert result.exit_code == 0
        assert "ex_1" in result.output

    @patch("terradev_cli.core.auto_lineage.auto_lineage")
    def test_lineage_export(self, mock_auto, runner, tmp_path):
        mock_auto.export_lineage.return_value = "{\"data\": 1}"
        out = tmp_path / "lineage.json"
        result = runner.invoke(
            cli,
            ["lineage", "export", "--format", "json", "--model", "m", "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()

    @patch("terradev_cli.core.auto_lineage.auto_lineage")
    def test_lineage_diff(self, mock_auto, runner):
        mock_auto.diff_executions.return_value = {
            "execution_1": {"id": "ex_1", "timestamp": "t1"},
            "execution_2": {"id": "ex_2", "timestamp": "t2"},
            "differences": {"resources": {"a": {"exec1": 1, "exec2": 2}}},
        }
        result = runner.invoke(cli, ["lineage", "diff", "ex_1", "ex_2"])
        assert result.exit_code == 0
        assert "COST: Resource Usage" in result.output


class TestEvalFunctional:
    @patch("terradev_cli.core.evaluation_orchestrator.EvaluationOrchestrator")
    @patch("terradev_cli.core.evaluation_orchestrator.EvaluationConfig")
    def test_evaluation(self, MockConfig, MockOrc, runner):
        result_obj = SimpleNamespace(
            evaluation_id="ev1",
            model_path="/model",
            endpoint_url=None,
            workload_type="general",
            metrics={"acc": 0.9},
            baseline_comparison=None,
            timestamp=SimpleNamespace(isoformat=lambda: "2026-01-01"),
            duration_seconds=1.0,
            metadata={},
        )
        orchestrator = MagicMock()
        orchestrator.evaluate_model.return_value = result_obj
        MockOrc.return_value = orchestrator
        MockConfig.return_value = SimpleNamespace()

        result = runner.invoke(
            cli,
            [
                "eval",
                "evaluation",
                "--model",
                "/model",
                "--dataset",
                "/data",
                "--metrics",
                "acc",
            ],
        )
        assert result.exit_code == 0


class TestModelRouterFunctional:
    @patch("terradev_cli.commands.mlops.TerradevAPI")
    @patch("terradev_cli.ml_services.model_router.create_router_from_credentials")
    def test_model_router_test(self, mock_create, MockAPI, runner):
        router = MagicMock()
        router.route.return_value = (
            SimpleNamespace(model_id="m", tier=SimpleNamespace(value="decode"), url="http://u"),
            SimpleNamespace(value="decode"),
            "ok",
        )
        mock_create.return_value = router
        mock_api = MagicMock()
        mock_api._provider_creds.return_value = {}
        MockAPI.return_value = mock_api

        result = runner.invoke(cli, ["model-router", "test", "--format", "json"])
        assert result.exit_code == 0
        assert "decode" in result.output

    @patch("terradev_cli.commands.mlops.TerradevAPI")
    @patch("terradev_cli.ml_services.model_router.create_router_from_credentials")
    def test_model_router_stats(self, mock_create, MockAPI, runner):
        router = MagicMock()
        router.get_routing_stats.return_value = {
            "total_decisions": 10,
            "strong_pct": 80,
            "weak_pct": 20,
            "by_step_type": {"reasoning": {"total": 5, "strong": 4, "weak": 1}},
        }
        mock_create.return_value = router
        mock_api = MagicMock()
        mock_api._provider_creds.return_value = {}
        MockAPI.return_value = mock_api

        result = runner.invoke(cli, ["model-router", "stats"])
        assert result.exit_code == 0
        assert "Total Decisions" in result.output
