"""Tests for terradev_cli.core.cost_scaler.

CostScaler makes budget-aware scaling decisions for inference workloads.
"""

from datetime import datetime

import pytest

from terradev_cli.core.cost_scaler import (
    CostConfig,
    CostMetrics,
    CostScaler,
    CostStrategy,
)


@pytest.fixture
def scaler(tmp_path):
    config = CostConfig(
        hourly_budget_usd=10.0,
        cost_per_gb_hour_usd=0.1,
        cold_start_cost_penalty_usd=0.05,
        peak_hour_multiplier=1.5,
        cost_threshold_for_warming=0.5,
        strategy=CostStrategy.BALANCE_COST_LATENCY,
    )
    return CostScaler(config, config_dir=tmp_path)


def test_cost_config_defaults():
    """CostConfig has sensible defaults."""
    config = CostConfig()
    assert config.hourly_budget_usd > 0
    assert config.strategy == CostStrategy.BALANCE_COST_LATENCY
    assert config.peak_hours


def test_cost_metrics_defaults():
    """CostMetrics starts at zero."""
    metrics = CostMetrics()
    assert metrics.total_cost_usd == 0.0
    assert metrics.budget_utilization_percent == 0.0


def test_memory_cost_and_peak_multiplier(scaler):
    """Memory cost scales with usage and peak hours."""
    cost = scaler.calculate_memory_cost(10.0, 1.0)
    if datetime.now().hour in scaler.config.peak_hours:
        assert cost == 10.0 * 0.1 * 1.5
    else:
        assert cost == 10.0 * 0.1


def test_cold_start_penalty(scaler):
    """Cold start penalty is a flat fee."""
    assert scaler.calculate_cold_start_penalty("m") == 0.05


def test_register_and_evict_model(scaler):
    """Registering and evicting models updates memory usage."""
    scaler.register_model_load("m1", 10.0, 5.0)
    assert scaler.model_memory_usage["m1"] == 10.0
    assert scaler.current_memory_usage_gb == 10.0
    assert "m1" in scaler.model_load_costs

    cost = scaler.get_current_hourly_cost()
    assert cost > 0

    scaler.register_model_eviction("m1")
    assert "m1" not in scaler.model_memory_usage
    assert scaler.current_memory_usage_gb == 0.0


def test_should_load_model_budget_constrained(tmp_path):
    """Budget-constrained strategy blocks loads that exceed budget."""
    config = CostConfig(
        hourly_budget_usd=1.0,
        cost_per_gb_hour_usd=1.0,
        strategy=CostStrategy.BUDGET_CONSTRAINED,
    )
    scaler = CostScaler(config, config_dir=tmp_path)

    should_load, reason = scaler.should_load_model("m1", 2.0)
    assert should_load is False
    assert "exceed budget" in reason.lower()


def test_should_load_latency_critical(tmp_path):
    """Latency-critical strategy allows loads even near budget."""
    config = CostConfig(
        hourly_budget_usd=1.0,
        cost_per_gb_hour_usd=0.5,
        strategy=CostStrategy.LATENCY_CRITICAL,
    )
    scaler = CostScaler(config, config_dir=tmp_path)

    should_load, reason = scaler.should_load_model("m1", 1.0)
    assert should_load is True
    assert "Latency critical" in reason


def test_predict_hourly_cost(scaler):
    """Predicted cost returns a non-negative value."""
    cost = scaler.predict_hourly_cost(1)
    assert cost >= 0.0

    scaler.config.enable_cost_prediction = False
    assert scaler.predict_hourly_cost(1) == scaler.get_current_hourly_cost()


def test_cost_savings(scaler):
    """Savings are the difference between baseline and current cost."""
    scaler.register_model_load("m1", 10.0, 1.0)
    savings = scaler.calculate_cost_savings(5.0)
    assert savings >= 0


def test_optimization_recommendations(scaler):
    """Recommendations are produced when budget or memory is high."""
    scaler.register_model_load("m1", 100.0, 1.0)
    recs = scaler.get_cost_optimization_recommendations()
    assert isinstance(recs, list)
    assert any(r["type"] in ("budget", "memory", "peak_hour") for r in recs)


@pytest.mark.asyncio
async def test_start_and_stop(scaler):
    """Start and stop create and cancel the background monitor."""
    await scaler.start()
    assert scaler._running is True
    assert scaler._cost_monitor_task is not None

    await scaler.stop()
    assert scaler._running is False
