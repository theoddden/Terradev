"""Tests for terradev_cli.core.public_ip_billing_tracker.

Tracks public IP assignment costs and detects idle/unused IPs.
"""

from datetime import datetime, timedelta

import pytest

from terradev_cli.core.public_ip_billing_tracker import (
    IPStatus,
    PublicIPBillingTracker,
    PublicIPRecord,
)


@pytest.fixture
def tracker():
    return PublicIPBillingTracker()


@pytest.mark.asyncio
async def test_track_public_ip(tracker):
    """Tracking a public IP creates a record and returns billing info."""
    result = await tracker.track_public_ip(
        provider="aws",
        instance_id="i-1",
        ip_address="1.2.3.4",
        region="us-east-1",
    )
    assert result["ip_address"] == "1.2.3.4"
    assert result["provider"] == "aws"
    assert result["hourly_cost"] == 0.0045
    assert result["status"] == "active"


@pytest.mark.asyncio
async def test_track_public_ip_included_cost(tracker):
    """Some providers include the public IP in the instance cost."""
    result = await tracker.track_public_ip(
        provider="runpod",
        instance_id="i-2",
        ip_address="2.3.4.5",
        region="us-east-1",
    )
    assert result["included_in_instance_cost"] is True
    assert result["hourly_cost"] == 0.0


@pytest.mark.asyncio
async def test_update_ip_status(tracker):
    """IP status can be updated."""
    await tracker.track_public_ip(
        provider="aws",
        instance_id="i-1",
        ip_address="1.2.3.4",
        region="us-east-1",
    )

    update_time = datetime.now() - timedelta(hours=2)
    result = await tracker.update_ip_status(
        ip_address="1.2.3.4",
        status=IPStatus.IDLE,
        last_active=update_time,
    )
    assert result["ip_address"] == "1.2.3.4"
    assert result["new_status"] == "idle"
    assert result["idle_hours"] >= 2.0


@pytest.mark.asyncio
async def test_update_ip_status_missing(tracker):
    """Updating an unknown IP returns an error."""
    result = await tracker.update_ip_status(
        ip_address="1.2.3.4",
        status=IPStatus.IDLE,
    )
    assert result["error"]


@pytest.mark.asyncio
async def test_analyze_ip_costs(tracker):
    """Cost analysis aggregates active and idle records."""
    await tracker.track_public_ip("aws", "i-1", "1.2.3.4", "us-east-1")
    await tracker.update_ip_status(
        ip_address="1.2.3.4",
        status=IPStatus.IDLE,
        last_active=datetime.now() - timedelta(hours=48),
    )

    result = await tracker.analyze_ip_costs(days=30)
    assert result["total_ips"] == 1
    assert result["idle_ips"] == 1
    assert result["cost_by_provider"]["aws"] >= 0.0
    assert result["recommendations"]


@pytest.mark.asyncio
async def test_detect_idle_ips(tracker):
    """Idle IP detection returns records above the threshold."""
    await tracker.track_public_ip("aws", "i-1", "1.2.3.4", "us-east-1")
    await tracker.update_ip_status(
        ip_address="1.2.3.4",
        status=IPStatus.ACTIVE,
        last_active=datetime.now() - timedelta(hours=48),
    )

    idle = await tracker.detect_idle_ips(idle_threshold_hours=24)
    assert len(idle) == 1
    assert idle[0]["ip_address"] == "1.2.3.4"
    assert idle[0]["idle_hours"] >= 48.0


@pytest.mark.asyncio
async def test_get_billing_alerts(tracker):
    """Billing alerts are generated when idle cost is high."""
    await tracker.track_public_ip(
        provider="coreweave",
        instance_id="i-3",
        ip_address="3.4.5.6",
        region="us-east-1",
    )
    await tracker.update_ip_status(
        ip_address="3.4.5.6",
        status=IPStatus.IDLE,
        last_active=datetime.now() - timedelta(hours=2500),
    )

    # Make record old enough for both idle cost and budget alerts
    record = tracker.ip_records[0]
    record.created_at = datetime.now() - timedelta(hours=500)

    alerts = await tracker.get_billing_alerts(budget_limit=1.0)
    assert alerts
    assert any(a["type"] in ("high_idle_cost", "budget_exceeded") for a in alerts)


def test_ip_status_enum():
    """IPStatus enum values are as expected."""
    assert IPStatus.ACTIVE.value == "active"
    assert IPStatus.IDLE.value == "idle"
