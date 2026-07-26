"""Tests for infrastructure commands."""
from unittest.mock import AsyncMock, patch

import pytest

from terradev_cli.commands import cli


# infrastructure groups and top-level commands


class TestAvailability:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["availability", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestBudgetOptimize:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["budget-optimize", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["budget-optimize"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestHelmGenerate:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["helm-generate", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["helm-generate"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestPercentiles:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["percentiles", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["percentiles"], obj={"api": mock_api})
        assert result.exit_code != 0

class TestPriceDiscovery:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["price-discovery", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestReliability:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["reliability", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0


class TestFunctionalInfrastructure:
    """End-to-end DI tests for infrastructure intelligence commands."""

    def _price(self, provider="RunPod", price=2.0, instance_type="runpod-a100", capacity="high", confidence=0.95):
        from types import SimpleNamespace
        return SimpleNamespace(provider=provider, price=price, instance_type=instance_type, capacity=capacity, confidence=confidence)

    @patch("terradev_cli.core.price_discovery.PriceDiscoveryEngine")
    def test_price_discovery_runs(self, MockEngine, runner, mock_api):
        engine = MockEngine.return_value
        engine.__aenter__ = AsyncMock(return_value=engine)
        engine.__aexit__ = AsyncMock(return_value=False)
        engine.get_realtime_prices = AsyncMock(return_value=[self._price()])
        engine.get_price_trends = AsyncMock(return_value={"RunPod": {"metrics": {"avg_price": 2.0, "min_price": 1.5, "max_price": 2.5, "volatility": 0.05, "trend": "flat"}}})
        result = runner.invoke(cli, ["price-discovery", "--gpu-type", "A100", "--trends"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.core.price_discovery.BudgetOptimizationEngine")
    def test_budget_optimize_runs(self, MockEngine, runner, mock_api):
        engine = MockEngine.return_value
        engine.optimize_for_budget = AsyncMock(return_value=[{
            "provider": "RunPod", "instance_type": "a100", "price": 2.0,
            "risk_score": 0.1, "budget_utilization": 0.4, "confidence": 0.95,
            "predicted_cost": 2.0, "risk_adjusted_cost": 2.1, "capacity": "high", "spot": False,
        }])
        result = runner.invoke(cli, ["budget-optimize", "--gpu-type", "A100", "--budget", "5"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_helm_generate_dry_run(self, runner, mock_api):
        result = runner.invoke(cli, ["helm-generate", "--image", "nginx", "--dry-run"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.core.price_intelligence.compute_percentiles")
    def test_percentiles_runs(self, mock_func, runner, mock_api):
        mock_func.return_value = {
            "providers": {
                "RunPod": {"p10": 1.0, "p25": 1.5, "p50": 2.0, "p75": 2.5, "p90": 3.0, "p99": 4.0, "min": 0.5, "max": 5.0, "count": 100},
            }
        }
        result = runner.invoke(cli, ["percentiles", "--gpu-type", "A100"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.core.price_intelligence.get_availability")
    def test_availability_runs(self, mock_func, runner, mock_api):
        mock_func.return_value = {
            "providers": {
                "RunPod": {"available": True, "availability_rate": 0.95, "total_checks": 100, "available_checks": 95, "avg_response_ms": 50, "last_seen": "2026-01-01T00:00:00", "last_error": None},
            }
        }
        result = runner.invoke(cli, ["availability", "--gpu-type", "A100"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.core.price_intelligence.get_provider_ranking")
    def test_reliability_ranking_runs(self, mock_func, runner, mock_api):
        mock_func.return_value = [
            {"provider": "RunPod", "overall_score": 95.0, "quote_success_rate": 0.99, "provision_success_rate": 0.98, "avg_quote_latency_ms": 50, "avg_provision_latency_ms": 100, "total_events": 1000},
        ]
        result = runner.invoke(cli, ["reliability", "--ranking"], obj={"api": mock_api})
        assert result.exit_code == 0