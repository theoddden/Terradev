"""Unit tests for terradev_cli.cost_optimizer."""

import asyncio

import pytest

from terradev_cli.cost_optimizer import (
    CostTier,
    InferXCostOptimizer,
    OptimizationRecommendation,
)


@pytest.fixture
def sample_config():
    return {
        "nodes": [
            {"gpu_type": "A100", "gpu_count": 2, "spot": True},
            {"gpu_type": "A10G", "gpu_count": 1, "spot": False},
        ],
        "storage_gb": 200,
        "snapshot_gb": 500,
    }


@pytest.fixture
def sample_metrics():
    return {
        "gpu_utilization": 45.0,
        "memory_utilization": 70.0,
        "cpu_utilization": 25.0,
        "models_deployed": 25,
        "cold_start_time": 2.2,
        "requests_per_hour": 150,
    }


def test_optimizer_instantiates():
    opt = InferXCostOptimizer()
    assert opt.cost_models
    assert opt.optimization_rules


def test_analyze_current_costs(sample_config, sample_metrics):
    opt = InferXCostOptimizer()
    metrics = asyncio.run(opt.analyze_current_costs(sample_config, sample_metrics))
    assert metrics.hourly_cost > 0
    assert metrics.monthly_cost == metrics.hourly_cost * 730
    assert metrics.gpu_utilization == 45.0


def test_generate_optimization_recommendations(sample_config, sample_metrics):
    opt = InferXCostOptimizer()
    current = asyncio.run(opt.analyze_current_costs(sample_config, sample_metrics))
    recs = asyncio.run(
        opt.generate_optimization_recommendations(
            current, sample_config, CostTier.ECONOMY
        )
    )
    assert isinstance(recs, list)
    assert all(isinstance(r, OptimizationRecommendation) for r in recs)
    # Low GPU utilization, on-demand nodes, and low model density should trigger recs
    assert any(r.action == "optimize_gpu_utilization" for r in recs)
    assert any(r.action == "switch_to_spot_instances" for r in recs)


def test_simulate_optimization_scenario(sample_config, sample_metrics):
    opt = InferXCostOptimizer()
    current = asyncio.run(opt.analyze_current_costs(sample_config, sample_metrics))
    recs = asyncio.run(
        opt.generate_optimization_recommendations(
            current, sample_config, CostTier.ECONOMY
        )
    )
    scenario = asyncio.run(opt.simulate_optimization_scenario(current, recs))
    assert "optimized_monthly_cost" in scenario
    assert "total_savings" in scenario
    assert "savings_percentage" in scenario
    assert "risk_assessment" in scenario
    assert "roi_analysis" in scenario


def test_generate_cost_report(sample_config, sample_metrics):
    opt = InferXCostOptimizer()
    report = asyncio.run(opt.generate_cost_report(sample_config, sample_metrics))
    assert "current_metrics" in report
    assert "recommendations" in report
    assert "optimization_scenario" in report
    assert "summary" in report
    assert "key_insights" in report
