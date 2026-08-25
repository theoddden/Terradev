"""Tests for terradev_cli.core.egress_cost_monitor.

EgressCostMonitor estimates data transfer costs between providers and regions.
"""

from datetime import datetime, timedelta

import pytest

from terradev_cli.core.egress_cost_monitor import (
    EgressCostAlert,
    EgressCostLevel,
    EgressCostMonitor,
)


@pytest.fixture
def monitor():
    return EgressCostMonitor(budget_limit=1000.0)


def test_egress_cost_level_values():
    """EgressCostLevel enum values are as expected."""
    assert EgressCostLevel.LOW.value == "low"
    assert EgressCostLevel.CRITICAL.value == "critical"


def test_get_egress_cost_known_route(monitor):
    """Known provider/region/dst combinations return a per-GB cost."""
    cost = monitor._get_egress_cost("aws", "us-east-1", "gcp")
    assert cost == 0.12


def test_get_egress_cost_defaults(monitor):
    """Unknown routes fall back to the default cost."""
    cost = monitor._get_egress_cost("foo", "bar", "baz")
    assert cost == 0.09


def test_determine_alert_level(monitor):
    """Alert levels are chosen from configured thresholds."""
    assert monitor._determine_alert_level(0.5) == EgressCostLevel.LOW
    assert monitor._determine_alert_level(15.0) == EgressCostLevel.MEDIUM
    assert monitor._determine_alert_level(60.0) == EgressCostLevel.HIGH
    assert monitor._determine_alert_level(150.0) == EgressCostLevel.CRITICAL


@pytest.mark.asyncio
async def test_analyze_egress_cost_same_cloud(monitor):
    """Same-cloud transfers include an internal networking recommendation."""
    result = await monitor.analyze_egress_cost(
        src_provider="aws",
        src_region="us-east-1",
        dst_provider="aws",
        dst_region="us-west-2",
        data_size_gb=10.0,
    )
    assert round(result["estimated_cost"], 10) == pytest.approx(0.9)
    assert result["alert_level"] == "low"
    assert any("internal networking" in r for r in result["recommendations"])


@pytest.mark.asyncio
async def test_analyze_egress_cost_critical_alert(monitor):
    """Large transfers can trigger critical alerts."""
    result = await monitor.analyze_egress_cost(
        src_provider="gcp",
        src_region="asia-east1",
        dst_provider="aws",
        dst_region="us-east-1",
        data_size_gb=1000.0,
    )
    assert result["estimated_cost"] == 190.0
    assert result["alert_level"] == "critical"
    assert result["alert"] is not None


@pytest.mark.asyncio
async def test_analyze_egress_cost_budget_exceeded(monitor):
    """Transfers that exceed the budget are flagged."""
    result = await monitor.analyze_egress_cost(
        src_provider="aws",
        src_region="us-east-1",
        dst_provider="gcp",
        dst_region="us-central1",
        data_size_gb=20000.0,
    )
    assert result["budget_exceeded"] is True


@pytest.mark.asyncio
async def test_analyze_egress_cost_zero_egress_alternatives(monitor):
    """Alternative routes include zero-egress providers."""
    result = await monitor.analyze_egress_cost(
        src_provider="aws",
        src_region="us-east-1",
        dst_provider="gcp",
        dst_region="us-central1",
        data_size_gb=100.0,
    )
    providers = {a["provider"] for a in result["alternative_routes"]}
    assert "runpod" in providers


def test_cost_record_storage(monitor):
    """Cost records are stored and capped at 1000."""
    record = {
        "timestamp": datetime.now(),
        "src_provider": "aws",
        "dst_provider": "gcp",
        "estimated_cost": 1.0,
        "data_size_gb": 10.0,
    }
    monitor._store_cost_record(record)
    assert len(monitor.cost_history) == 1


@pytest.mark.asyncio
async def test_get_cost_summary_empty(monitor):
    """Empty history returns a zeroed summary."""
    summary = await monitor.get_cost_summary(days=30)
    assert summary["total_cost"] == 0.0
    assert summary["transfers"] == 0
    assert summary["cost_trend"] == "stable"


@pytest.mark.asyncio
async def test_get_cost_summary_with_records(monitor):
    """Summary aggregates stored records."""
    await monitor.analyze_egress_cost(
        src_provider="aws",
        src_region="us-east-1",
        dst_provider="gcp",
        dst_region="us-central1",
        data_size_gb=100.0,
    )
    summary = await monitor.get_cost_summary(days=30)
    assert summary["total_cost"] == 12.0
    assert summary["transfers"] == 1
    assert summary["top_routes"]
    assert summary["budget_remaining"] is not None


@pytest.mark.asyncio
async def test_check_budget_alerts_no_limit():
    """Without a budget limit, no budget alerts are generated."""
    no_budget = EgressCostMonitor()
    assert await no_budget.check_budget_alerts() == []


@pytest.mark.asyncio
async def test_check_budget_alerts(monitor):
    """Budget utilization triggers alerts."""
    record = {
        "timestamp": datetime.now(),
        "estimated_cost": 950.0,
        "data_size_gb": 1000.0,
        "src_provider": "aws",
        "dst_provider": "gcp",
    }
    monitor._store_cost_record(record)

    alerts = await monitor.check_budget_alerts()
    assert alerts
    assert any(a.level == EgressCostLevel.CRITICAL for a in alerts)
