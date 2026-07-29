"""Tests for terradev_cli.core.cost_tracker.

Cost tracking is the other half of the moat: clients need accurate spend
data. These tests use a temporary database to avoid polluting ~/.terradev.
"""

from unittest.mock import patch

import pytest

from terradev_cli.core import cost_tracker


@pytest.fixture
def fresh_db(monkeypatch, tmp_path):
    """Force cost_tracker to use a temp SQLite database for each test."""
    db_path = tmp_path / "cost_tracking.db"
    with patch.object(cost_tracker, "DB_PATH", db_path):
        yield


def test_record_quotes_and_summary(fresh_db):
    """Quotes are persisted and returned by spend summary."""
    quotes = [
        {"gpu_type": "A100", "provider": "runpod", "region": "us-east-1", "price": 2.0},
        {"gpu_type": "A100", "provider": "aws", "region": "us-east-1", "price": 3.5},
    ]
    cost_tracker.record_quotes(quotes, selected_idx=0)

    summary = cost_tracker.get_spend_summary(days=30)
    assert summary["quotes_fetched"] == 2


def test_record_provision_and_spend(fresh_db):
    """Provisions are tracked and ended provisions compute cost."""
    cost_tracker.record_provision(
        instance_id="i-1",
        provider="runpod",
        gpu_type="A100",
        region="us-east-1",
        price_hr=2.0,
    )

    cost_tracker.end_provision("i-1")
    summary = cost_tracker.get_spend_summary(days=30)
    assert summary["total_provisions"] == 1
    assert summary["total_provision_cost"] >= 0

    by_provider = summary["by_provider"]
    assert "runpod" in by_provider


def test_record_egress(fresh_db):
    """Egress records show up in the spend summary."""
    cost_tracker.record_egress(
        src_provider="aws",
        src_region="us-east-1",
        dst_provider="runpod",
        dst_region="us-east-1",
        bytes_moved=1_000_000_000,
        cost=10.0,
        optimized=True,
    )
    summary = cost_tracker.get_spend_summary(days=30)
    assert summary["egress_cost"] == 10.0


def test_record_staging(fresh_db):
    """Staging events are persisted without error."""
    cost_tracker.record_staging(
        dataset="c4",
        original_size=1_000_000,
        compressed_size=500_000,
        compression="zstd",
        chunks=4,
        regions=["us-east-1", "us-west-2"],
    )


def test_parallel_group_summary_and_ip_ssh(fresh_db):
    """Parallel group helpers return grouped data and resolve IP/SSH paths."""
    cost_tracker.record_provision(
        instance_id="i-1",
        provider="runpod",
        gpu_type="A100",
        region="us-east-1",
        price_hr=2.0,
        parallel_group="group-a",
    )
    cost_tracker.set_instance_ip("i-1", "10.0.0.1")
    cost_tracker.set_ssh_key_path("group-a", "/keys/a.pem")

    rows = cost_tracker.get_parallel_group_summary("group-a")
    assert len(rows) == 1
    assert rows[0]["ip_address"] == "10.0.0.1"

    assert cost_tracker.get_provision_ssh_key_path("group-a") == "/keys/a.pem"
    assert cost_tracker.get_latest_parallel_group() == "group-a"

    active = cost_tracker.get_active_instances("group-a")
    assert len(active) == 1


def test_daily_spend_aggregation(fresh_db):
    """Daily spend is aggregated by provider."""
    cost_tracker.record_provision("i-1", "aws", "A100", "us-east-1", 3.0)
    cost_tracker.record_provision("i-2", "gcp", "A100", "us-central1", 2.0)

    daily = cost_tracker.get_daily_spend(days=30)
    assert len(daily) == 1
    assert daily[0]["cost"] == pytest.approx(5.0, rel=0.01)
    assert set(daily[0]["providers"].keys()) == {"aws", "gcp"}
