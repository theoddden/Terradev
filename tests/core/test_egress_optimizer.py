"""Tests for terradev_cli.core.egress_optimizer.

Data transfer cost is a major client concern. These tests guard the egress
math, the multi-hop Dijkstra optimizer, and the staging-route helper.
"""

import pytest

from terradev_cli.core.egress_optimizer import (
    estimate_egress_cost,
    find_cheapest_multihop,
    find_cheapest_route,
    optimize_staging_route,
    optimize_transfer_plan,
)


def test_same_provider_same_region_is_free():
    """Transfers within the same provider and region cost nothing."""
    cost = estimate_egress_cost("aws", "us-east-1", "aws", "us-east-1", 1000)
    assert cost == 0.0


def test_same_provider_cross_region_cost():
    """Cross-region within the same provider uses the same-continent rate."""
    cost = estimate_egress_cost("aws", "us-east-1", "aws", "us-west-2", 1000)
    assert cost == pytest.approx(10.0, rel=0.05)


def test_cross_provider_internet_rate():
    """Cross-continent cross-provider transfers use the expensive internet rate."""
    cost = estimate_egress_cost("aws", "us-east-1", "gcp", "eu-west-1", 100)
    assert cost == pytest.approx(9.0, rel=0.05)


def test_cross_provider_same_continent_rate():
    """Same-continent cross-provider transfers get the cheaper same-continent rate."""
    cost = estimate_egress_cost("aws", "us-east-1", "gcp", "us-central1", 100)
    assert cost == pytest.approx(1.0, rel=0.05)


def test_zero_egress_provider_to_anywhere():
    """RunPod and similar providers advertise zero egress."""
    cost = estimate_egress_cost("runpod", "us-east-1", "aws", "us-east-1", 1000)
    assert cost == 0.0


def test_find_cheapest_multihop_same_provider_is_zero():
    """Multi-hop from a provider to itself is trivially free."""
    result = find_cheapest_multihop("aws", "aws", 100)
    assert result["hops"] == 0
    assert result["total_cost"] == 0.0


def test_find_cheapest_multihop_returns_lowest_cost_path():
    """Multi-hop search returns a path with cost and hop count within bounds."""
    result = find_cheapest_multihop("aws", "gcp", 100, max_hops=3)
    assert result["hops"] <= 3
    assert result["total_cost"] == pytest.approx(result["direct_cost"], rel=0.05)
    assert result["savings"] >= 0


def test_find_cheapest_route_ranks_candidates():
    """Route ranking returns destinations sorted by egress cost."""
    candidates = [
        {"provider": "aws", "region": "us-east-1"},
        {"provider": "runpod", "region": "us-east-1"},
        {"provider": "gcp", "region": "us-central1"},
    ]
    ranked = find_cheapest_route("aws", "us-east-1", candidates, 100)
    assert ranked[0]["egress_cost"] == 0.0
    assert ranked[-1]["egress_cost"] == pytest.approx(1.0, rel=0.05)


def test_optimize_transfer_plan_returns_routes():
    """The transfer plan returns routes, totals, and recommendations."""
    plan = optimize_transfer_plan(
        {"provider": "aws", "region": "us-east-1"},
        [{"provider": "aws", "region": "us-west-2"}],
        100,
    )
    assert "routes" in plan
    assert "total_egress_cost" in plan
    assert "recommendations" in plan
    assert plan["total_egress_cost"] == pytest.approx(0.0, rel=0.05)


def test_optimize_staging_route_recommends_direct_when_no_savings():
    """When multi-hop cannot beat direct, staging recommends a direct transfer."""
    result = optimize_staging_route(
        "aws",
        "us-east-1",
        [
            {"provider": "runpod", "region": "us-east-1"},
            {"provider": "coreweave", "region": "us-east-1"},
        ],
        1000,
    )
    assert result["strategy"] == "direct"
    assert "recommendation" in result
    assert result["total_cost"] >= 0
