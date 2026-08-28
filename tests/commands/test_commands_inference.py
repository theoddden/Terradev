"""Comprehensive tests for inference serving commands via ctx.obj DI."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terradev_cli.commands import cli


def _write_inferx_config(mock_api):
    """Write a fake InferX config into the temp config directory."""
    config_dir = mock_api.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "inferx_config.json"
    config_file.write_text(
        '{"api_key": "test", "api_endpoint": "https://api.inferx.net"}'
    )


# ===========================================================================
# Shared mock helpers
# ===========================================================================


def _make_orchestrator():
    """Return a mock ModelOrchestrator with async methods."""
    orch = MagicMock()
    orch.start = AsyncMock(return_value=True)
    orch.stop = AsyncMock(return_value=True)
    orch.register_model = AsyncMock(return_value=True)
    orch.load_model = AsyncMock(return_value=True)
    orch.evict_model = AsyncMock(return_value=True)
    orch.handle_request = AsyncMock(return_value=(True, 12.3))
    orch.get_status = MagicMock(
        return_value={
            "gpu_id": 0,
            "total_memory_gb": 80.0,
            "used_memory_gb": 10.0,
            "available_memory_gb": 70.0,
            "memory_utilization_percent": 12.5,
            "scaling_policy": "billing_optimized",
            "total_models": 1,
            "warm_models_count": 1,
            "warm_models_memory_gb": 10.0,
            "models_by_state": {"warm": ["m1"]},
        }
    )
    orch.get_model_details = MagicMock(
        return_value={
            "framework": "pytorch",
            "state": "warm",
            "priority": 1,
            "tags": ["test"],
            "metrics": {
                "memory_gb": 10.0,
                "load_time_s": 1.2,
                "warmup_time_s": 0.5,
                "requests_per_hour": 100.0,
                "avg_latency_ms": 5.0,
                "error_rate": 0.01,
            },
            "last_accessed": "now",
        }
    )
    orch.memory_threshold_gb = 70.0
    return orch


def _make_warm_pool():
    """Return a mock WarmPoolManager with async methods."""
    wp = MagicMock()
    wp.start = AsyncMock(return_value=True)
    wp.stop = AsyncMock(return_value=True)
    wp.register_model = MagicMock(return_value=True)
    wp.get_status = MagicMock(
        return_value={
            "warm_models_count": 1,
            "warming_models_count": 0,
            "total_models": 1,
            "strategy": "traffic_based",
            "cache_hit_rate": 0.85,
            "total_requests": 1000,
            "cold_starts": 10,
            "avg_warm_latency_ms": 5.0,
            "avg_cold_latency_ms": 50.0,
            "memory_saved_gb": 2.0,
            "cost_saved_usd": 1.5,
        }
    )
    return wp


    cs.get_cost_optimization_recommendations = MagicMock(return_value=[])
    cs.get_model_cost_details = MagicMock(
        return_value={
            "memory_gb": 10.0,
            "hourly_cost_usd": 1.0,
            "cold_start_penalty_usd": 0.1,
            "total_cost_today": 24.0,
            "cost_rank": 1,
        }
    )
    return cs


def _make_inferx():
    """Return a mock InferXProvider with async methods."""
    prov = MagicMock()
    prov.deploy_model = AsyncMock(
        return_value={
            "model_id": "m1",
            "endpoint": "http://example.com",
            "cold_start_time": 1.0,
            "gpu_utilization": 90,
            "models_per_node": 2,
            "openai_compatible": True,
        }
    )
    prov.get_model_status = AsyncMock(
        return_value={
            "status": "running",
            "gpu_type": "A100",
            "cold_start_time": 1.0,
            "requests_per_minute": 10,
            "gpu_utilization": 90,
            "models_on_gpu": 1,
            "error_rate": 1.0,
        }
    )
    prov.delete_model = AsyncMock(return_value=True)
    prov.list_models = AsyncMock(return_value=[])
    prov.get_usage_stats = AsyncMock(
        return_value={
            "total_requests": 1000,
            "total_cost": 10.0,
            "active_models": 2,
            "gpu_hours": 5.0,
            "average_latency": 10.0,
            "gpu_utilization": 90.0,
        }
    )
    prov.get_instance_quotes = AsyncMock(
        return_value=[
            {
                "gpu_type": "A100",
                "price_per_hour": 2.0,
                "price_per_request": 0.0001,
                "cold_start_time": 1.0,
                "gpu_utilization": 90,
                "models_per_node": 2,
                "region": "us-west-2",
                "features": ["snapshot"],
            }
        ]
    )
    prov.close = AsyncMock(return_value=None)
    return prov


# ===========================================================================
# infer group
# ===========================================================================


class TestInferGroup:
    """Tests for the infer group and its subcommands."""

    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["infer", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Deploy and manage inference endpoints" in result.output

    def test_deploy_requires_model(self, runner, mock_api):
        result = runner.invoke(cli, ["infer", "deploy"], obj={"api": mock_api})
        assert result.exit_code != 0

    @patch("terradev_cli.commands.inference.TerradevAPI")
    def test_endpoint_dry_run(self, MockAPI, runner, mock_api):
        api = MagicMock()
        api._get_provider_quotes = AsyncMock(
            return_value=[{"price": 1.0, "gpu_type": "A100-80GB"}]
        )
        api._provider_creds = MagicMock(return_value={"api_key": "test"})
        MockAPI.return_value = api

        result = runner.invoke(
            cli,
            [
                "infer",
                "endpoint",
                "/models/llama",
                "--name",
                "test-ep",
                "--dry-run",
            ],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()

    def test_endpoint_requires_model_path(self, runner, mock_api):
        result = runner.invoke(cli, ["infer", "endpoint"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_status_help(self, runner, mock_api):
        result = runner.invoke(cli, ["infer", "status", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

# ===========================================================================
# inferx group
# ===========================================================================


class TestInferxGroup:
    """Tests for the inferx group and its subcommands."""

    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["inferx", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "InferX" in result.output

    @patch("terradev_cli.commands.inference.Path.home")
    @patch("terradev_cli.providers.inferx_provider.InferXProvider")
    def test_list_runs(self, MockProvider, MockHome, runner, mock_api):
        MockHome.return_value = mock_api.config_dir.parent
        _write_inferx_config(mock_api)
        MockProvider.return_value = _make_inferx()
        result = runner.invoke(cli, ["inferx", "list"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.commands.inference.Path.home")
    @patch("terradev_cli.providers.inferx_provider.InferXProvider")
    def test_usage_runs(self, MockProvider, MockHome, runner, mock_api):
        MockHome.return_value = mock_api.config_dir.parent
        _write_inferx_config(mock_api)
        MockProvider.return_value = _make_inferx()
        result = runner.invoke(cli, ["inferx", "usage"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.commands.inference.Path.home")
    @patch("terradev_cli.providers.inferx_provider.InferXProvider")
    def test_quote_runs(self, MockProvider, MockHome, runner, mock_api):
        MockHome.return_value = mock_api.config_dir.parent
        _write_inferx_config(mock_api)
        MockProvider.return_value = _make_inferx()
        result = runner.invoke(
            cli, ["inferx", "inferx-quote", "--gpu-type", "A100"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_delete_requires_model_id(self, runner, mock_api):
        result = runner.invoke(cli, ["inferx", "inferx-delete"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_deploy_missing_model(self, runner, mock_api):
        result = runner.invoke(cli, ["inferx", "deploy"], obj={"api": mock_api})
        assert result.exit_code != 0


# ===========================================================================
# orchestrator group
# ===========================================================================


class TestOrchestratorGroup:
    """Tests for the orchestrator group."""

    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["orchestrator", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Model orchestrator" in result.output

    @patch("terradev_cli.commands.inference.asyncio.sleep")
    @patch("terradev_cli.core.model_orchestrator.ModelOrchestrator")
    def test_start_runs(self, MockOrchestrator, MockSleep, runner, mock_api):
        MockSleep.side_effect = KeyboardInterrupt
        MockOrchestrator.return_value = _make_orchestrator()
        result = runner.invoke(cli, ["orchestrator", "start"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.core.model_orchestrator.ModelOrchestrator")
    def test_status_runs(self, MockOrchestrator, runner, mock_api):
        MockOrchestrator.return_value = _make_orchestrator()
        result = runner.invoke(
            cli, ["orchestrator", "status", "--model-id", "m1"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_register_requires_model_id(self, runner, mock_api):
        result = runner.invoke(cli, ["orchestrator", "register"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_load_requires_model_id(self, runner, mock_api):
        result = runner.invoke(cli, ["orchestrator", "load"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_evict_requires_model_id(self, runner, mock_api):
        result = runner.invoke(cli, ["orchestrator", "evict"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_infer_requires_model_id(self, runner, mock_api):
        result = runner.invoke(cli, ["orchestrator", "infer"], obj={"api": mock_api})
        assert result.exit_code != 0


# ===========================================================================
# warm-pool group
# ===========================================================================


class TestWarmPoolGroup:
    """Tests for the warm-pool group."""

    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["warm-pool", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Warm pool" in result.output

    @patch("terradev_cli.commands.inference.asyncio.sleep")
    @patch("terradev_cli.core.warm_pool_manager.WarmPoolManager")
    def test_start_runs(self, MockManager, MockSleep, runner, mock_api):
        MockSleep.side_effect = KeyboardInterrupt
        MockManager.return_value = _make_warm_pool()
        result = runner.invoke(cli, ["warm-pool", "start"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.core.warm_pool_manager.WarmPoolManager")
    def test_status_runs(self, MockManager, runner, mock_api):
        MockManager.return_value = _make_warm_pool()
        result = runner.invoke(cli, ["warm-pool", "status"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_register_requires_model_id(self, runner, mock_api):
        result = runner.invoke(cli, ["warm-pool", "register"], obj={"api": mock_api})
        assert result.exit_code != 0


