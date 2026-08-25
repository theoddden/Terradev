"""Tests for terradev_cli.core.price_intelligence.

Price intelligence records ticks, computes delta/gamma/volatility, and
produces dashboards. Tests use an isolated SQLite database.
"""

from datetime import datetime, timedelta, timezone

import pytest

from terradev_cli.core import price_intelligence as pi


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Redirect price intelligence DB to a temp file."""
    db_path = tmp_path / "price_intelligence.db"
    monkeypatch.setattr("terradev_cli.core.price_intelligence.DB_PATH", db_path)
    # Force schema creation
    conn = pi._conn()
    conn.close()
    return db_path


def test_record_and_get_price_tick(tmp_db):
    """record_price_tick stores data and get_price_series retrieves it."""
    pi.record_price_tick(
        gpu_type="h100", provider="aws", price_hr=2.5, spot=False, workload_type="training"
    )
    series = pi.get_price_series("h100", provider="aws", spot=False, workload_type="training")
    assert len(series) == 1
    assert series[0]["price_hr"] == 2.5
    assert series[0]["provider"] == "aws"


def test_get_price_series_filters(tmp_db):
    """get_price_series filters by provider, spot, and workload."""
    pi.record_price_tick("h100", "aws", 2.0, spot=False, workload_type="training")
    pi.record_price_tick("h100", "gcp", 1.8, spot=False, workload_type="training")
    pi.record_price_tick("h100", "aws", 1.2, spot=True, workload_type="inference")

    assert len(pi.get_price_series("h100")) == 3
    assert len(pi.get_price_series("h100", provider="aws")) == 2
    assert len(pi.get_price_series("h100", spot=True)) == 1
    assert len(pi.get_price_series("h100", workload_type="inference")) == 1


def test_record_price_ticks_batch(tmp_db):
    """record_price_ticks_batch inserts many ticks."""
    ticks = [
        {"gpu_type": "a100", "provider": "aws", "price": 1.0},
        {"gpu_type": "a100", "provider": "gcp", "price": 0.9},
    ]
    pi.record_price_ticks_batch(ticks)
    series = pi.get_price_series("a100")
    assert len(series) == 2


def test_compute_delta(tmp_db):
    """compute_delta returns the percentage change over a window."""
    prices = [1.0, 1.1, 1.155]
    delta = pi.compute_delta(prices)
    assert round(delta, 6) == round((1.155 - 1.1) / 1.1, 6)

    assert pi.compute_delta([1.0]) is None
    assert pi.compute_delta([0.0, 1.0]) is None


def test_compute_delta_absolute():
    """compute_delta_absolute returns the raw price difference."""
    assert pi.compute_delta_absolute([1.0, 1.5]) == 0.5
    assert pi.compute_delta_absolute([1.0]) is None


def test_compute_gamma():
    """compute_gamma returns the acceleration of price change."""
    prices = [1.0, 1.1, 1.25]
    gamma = pi.compute_gamma(prices)
    assert gamma is not None
    assert pi.compute_gamma([1.0, 1.1]) is None


def test_compute_realized_volatility():
    """compute_realized_volatility annualizes log returns."""
    prices = [1.0, 1.01, 1.02]
    vol = pi.compute_realized_volatility(prices)
    assert vol is not None
    assert vol > 0

    assert pi.compute_realized_volatility([1.0, 1.1]) is None


def test_percentile():
    """_percentile computes percentile from a sorted list."""
    values = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pi._percentile(values, 0) == 1.0
    assert pi._percentile(values, 100) == 5.0
    assert pi._percentile(values, 50) == 3.0
    assert pi._percentile([], 50) == 0.0


def test_refresh_stats_and_insights(tmp_db):
    """refresh_stats populates the stats table and get_insights returns a dashboard."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    # Insert enough ticks for the 30-day window
    for i, price in enumerate([2.0, 2.1, 2.2]):
        ts = (now - timedelta(hours=i)).isoformat()
        conn = pi._conn()
        conn.execute(
            "INSERT INTO price_ticks (ts, gpu_type, provider, region, price_hr, spot, workload_type, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, "H100", "aws", "us-east-1", price, 0, "training", "quote"),
        )
        conn.commit()
        conn.close()

    count = pi.refresh_stats()
    assert count >= 1

    insights = pi.get_insights("h100", workload_type="training")
    assert insights["gpu_type"] == "H100"
    assert insights["providers"]
    assert insights["recommendations"]["cheapest"]["provider"] == "aws"


def test_get_all_tracked_gpus(tmp_db):
    """get_all_tracked_gpus returns distinct GPU types."""
    pi.record_price_tick("h100", "aws", 2.0)
    pi.record_price_tick("a100", "aws", 1.0)
    assert sorted(pi.get_all_tracked_gpus()) == ["A100", "H100"]


def test_training_vs_inference(tmp_db):
    """get_training_vs_inference compares workload type pricing."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for wtype, price in [("training", 2.0), ("inference", 2.5)]:
        conn = pi._conn()
        conn.execute(
            "INSERT INTO price_ticks (ts, gpu_type, provider, region, price_hr, spot, workload_type, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now.isoformat(), "H100", "aws", "us-east-1", price, 0, wtype, "quote"),
        )
        conn.commit()
        conn.close()

    pi.refresh_stats()
    result = pi.get_training_vs_inference("h100")
    assert "training" in result
    assert "inference" in result
    assert result["inference_premium"] is not None


def test_compute_percentiles(tmp_db):
    """compute_percentiles returns per-provider price percentiles."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for price in [1.0, 2.0, 3.0, 4.0, 5.0]:
        conn = pi._conn()
        conn.execute(
            "INSERT INTO price_ticks (ts, gpu_type, provider, region, price_hr, spot, workload_type, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now.isoformat(), "H100", "aws", "us-east-1", price, 0, "training", "quote"),
        )
        conn.commit()
        conn.close()

    percentiles = pi.compute_percentiles("h100", provider="aws")
    assert percentiles["providers"]["aws"]["p50"] == 3.0
    assert percentiles["providers"]["aws"]["count"] == 5


def test_record_and_get_availability(tmp_db):
    """record_availability stores checks and get_availability reports status."""
    pi.record_availability("h100", "aws", True, region="us-east-1", response_ms=120)
    pi.record_availability("h100", "aws", False, region="us-east-1", response_ms=300)
    pi.record_availability("h100", "gcp", True, region="us-west-1", response_ms=80)

    status = pi.get_availability("h100", hours=24)
    assert status["providers"]["aws"]["available"] is False
    assert status["providers"]["gcp"]["available"] is True


def test_get_availability_summary(tmp_db):
    """get_availability_summary returns the latest availability per provider."""
    pi.record_availability("h100", "aws", True)
    pi.record_availability("h100", "aws", False)
    summary = pi.get_availability_summary()
    assert summary["H100"]["aws"] is False


def test_record_provider_event_and_reliability(tmp_db):
    """record_provider_event feeds get_provider_reliability scoring."""
    pi.record_provider_event("aws", "quote", success=True, latency_ms=100)
    pi.record_provider_event("aws", "provision", success=True, latency_ms=200)
    pi.record_provider_event("aws", "quote", success=False, latency_ms=50, error="timeout")

    reliability = pi.get_provider_reliability("aws", hours=24)
    assert "aws" in reliability["providers"]
    assert reliability["providers"]["aws"]["total_events"] == 3
    assert reliability["providers"]["aws"]["overall_score"] > 0


def test_get_provider_ranking(tmp_db):
    """get_provider_ranking sorts providers by overall score."""
    pi.record_provider_event("aws", "quote", success=True)
    pi.record_provider_event("gcp", "quote", success=False)
    ranking = pi.get_provider_ranking()
    assert ranking[0]["provider"] == "aws"
