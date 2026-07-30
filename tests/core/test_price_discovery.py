"""Tests for terradev_cli.core.price_discovery.

PriceDiscoveryEngine uses price tick data and availability signals to rank
providers. Optional price_intelligence imports are mocked away for isolation.
"""

import pytest

from terradev_cli.core.price_discovery import (
    BudgetOptimizationEngine,
    PriceDiscoveryEngine,
    PriceInfo,
)


def test_price_info_dataclass():
    """PriceInfo stores provider price details."""
    info = PriceInfo(
        provider="runpod",
        gpu_type="A100",
        price=0.5,
        instance_type="A100",
        region="us-east-1",
        capacity="Available",
        confidence=0.9,
        last_updated=None,
        spot=True,
    )
    assert info.spot is True
    assert info.provider == "runpod"


@pytest.fixture
def engine():
    return PriceDiscoveryEngine()


@pytest.mark.asyncio
async def test_get_realtime_prices_fallback(engine, monkeypatch):
    """Prices fall back to mock quotes when real price data is unavailable."""

    def raise_import(*args, **kwargs):
        raise ImportError("no price intelligence")

    monkeypatch.setattr(
        "terradev_cli.core.price_intelligence.get_price_series",
        raise_import,
    )

    prices = await engine.get_realtime_prices("A100", "us-east")
    assert prices
    providers = {p.provider for p in prices}
    assert "runpod" in providers


@pytest.mark.asyncio
async def test_get_realtime_prices_from_real_data(engine, monkeypatch):
    """Real price data is converted to PriceInfo objects."""
    fake_ticks = [
        {
            "provider": "vastai",
            "gpu_type": "A100",
            "price_hr": 1.2,
            "instance_type": "A100",
            "region": "us-east",
            "spot": True,
        },
        {
            "provider": "runpod",
            "gpu_type": "A100",
            "price_hr": 1.5,
            "instance_type": "A100",
            "region": "us-east",
            "spot": False,
        },
    ]
    async def fake_real_data(*a, **k):
        return fake_ticks

    async def fake_capacity(*a, **k):
        return "Available"

    async def fake_confidence(*a, **k):
        return 0.9

    monkeypatch.setattr(
        "terradev_cli.core.price_discovery.PriceDiscoveryEngine._get_real_price_data",
        fake_real_data,
    )
    monkeypatch.setattr(
        "terradev_cli.core.price_discovery.PriceDiscoveryEngine._check_capacity",
        fake_capacity,
    )
    monkeypatch.setattr(
        "terradev_cli.core.price_discovery.PriceDiscoveryEngine._calculate_confidence_from_real_data",
        fake_confidence,
    )

    prices = await engine.get_realtime_prices("A100", "us-east")
    assert len(prices) == 2
    assert prices[0].provider == "vastai"  # lower price/confidence comes first


@pytest.mark.asyncio
async def test_get_price_trends(engine, monkeypatch):
    """Trends are grouped and summarized per provider."""
    fake_series = [
        {"provider": "runpod", "price_hr": 1.0, "ts": "2024-01-01T00:00:00"},
        {"provider": "runpod", "price_hr": 1.1, "ts": "2024-01-01T01:00:00"},
    ]

    def fake_series_fn(*a, **k):
        return fake_series

    monkeypatch.setattr(
        "terradev_cli.core.price_intelligence.get_price_series",
        fake_series_fn,
    )

    trends = await engine.get_price_trends("A100", hours=24)
    assert "runpod" in trends
    assert trends["runpod"]["metrics"]["trend"] == "up"


@pytest.mark.asyncio
async def test_is_data_stale_no_data(engine, monkeypatch):
    """Missing data is reported as stale."""
    monkeypatch.setattr(
        "terradev_cli.core.price_intelligence.get_price_series",
        lambda *a, **k: [],
    )

    assert engine.is_data_stale("A100", "runpod") is True


def test_budget_optimization_engine():
    """BudgetOptimizationEngine exists and exposes optimize_for_budget."""
    boe = BudgetOptimizationEngine()
    assert hasattr(boe, "optimize_for_budget")
