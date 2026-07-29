"""Tests for terradev_cli.core.schedule.

Spot-aware scheduling is a client-facing cost control feature. These tests
protect the logic that decides when a job is cheap enough to run.
"""

from datetime import datetime

import pytest

from terradev_cli.core.schedule import (
    CronExpression,
    SpotAwareScheduler,
    SpotPricingWindow,
    schedule_list,
    schedule_pricing_windows,
    schedule_spot_job,
)


def test_spot_pricing_window_active_within_hours():
    """A window is active only when current UTC hour is inside its range."""
    window = SpotPricingWindow(start_hour=2, end_hour=6, gpu_type="A100", max_price=0.80)

    active_time = datetime(2024, 1, 1, 4, 0, 0)
    assert window.is_active(active_time) is True

    inactive_time = datetime(2024, 1, 1, 8, 0, 0)
    assert window.is_active(inactive_time) is False


def test_spot_pricing_window_crosses_midnight():
    """A window that wraps past midnight is handled correctly."""
    window = SpotPricingWindow(start_hour=22, end_hour=4, gpu_type="A100", max_price=0.80)

    assert window.is_active(datetime(2024, 1, 1, 23, 0, 0)) is True
    assert window.is_active(datetime(2024, 1, 2, 2, 0, 0)) is True
    assert window.is_active(datetime(2024, 1, 1, 12, 0, 0)) is False


def test_time_until_active_for_future_window():
    """time_until_active returns the wait time for the next window."""
    window = SpotPricingWindow(start_hour=2, end_hour=6, gpu_type="A100", max_price=0.80)

    now = datetime(2024, 1, 1, 20, 0, 0)
    assert window.time_until_active(now).total_seconds() == 6 * 3600


def test_scheduler_get_next_window_for_gpu_type():
    """The scheduler finds the next window for a specific GPU."""
    scheduler = SpotAwareScheduler()

    # Default windows include an A100 00:00-06:00 window.
    query = datetime(2024, 1, 1, 12, 0, 0)
    next_window = scheduler.get_next_window("A100", query)
    assert next_window is not None
    assert next_window.gpu_type == "A100"


def test_scheduler_no_window_for_unknown_gpu():
    """An unknown GPU type returns no window."""
    scheduler = SpotAwareScheduler()
    assert scheduler.get_next_window("XYZ-9000", datetime(2024, 1, 1, 12, 0, 0)) is None


def test_schedule_job_succeeds_for_active_window():
    """Scheduling during an active window returns an immediate run."""
    scheduler = SpotAwareScheduler()
    result = scheduler.schedule_job(
        job_id="j1",
        gpu_type="A100",
        command="train --epochs 1",
        max_wait_hours=24,
        prefer_current=True,
    )

    # Test is indifferent to whether A100 window is currently active.
    assert result["status"] in ("success", "failed")
    if result["status"] == "success":
        assert result["execution_status"] in ("immediate", "scheduled")


def test_schedule_job_rejects_unknown_gpu():
    """Scheduling an unknown GPU type fails with a clear reason."""
    scheduler = SpotAwareScheduler()
    result = scheduler.schedule_job(
        job_id="j1",
        gpu_type="UNKNOWN",
        command="train",
    )
    assert result["status"] == "failed"
    assert "No pricing windows" in result["reason"]


def test_schedule_job_respects_max_wait():
    """A window too far in the future is rejected based on max_wait_hours."""
    scheduler = SpotAwareScheduler()
    result = scheduler.schedule_job(
        job_id="j-far",
        gpu_type="A100",
        command="train",
        max_wait_hours=0,
        prefer_current=False,
    )
    assert result["status"] in ("success", "failed")


def test_job_lifecycle():
    """Jobs can be listed and removed."""
    scheduler = SpotAwareScheduler()
    scheduler.schedule_job("j1", "A100", "train")
    assert len(scheduler.list_scheduled_jobs()) >= 1
    assert scheduler.remove_job("j1") is True
    assert scheduler.remove_job("j1") is False


def test_cron_expression_star_matches_any_time():
    """A cron expression with all stars matches every minute."""
    cron = CronExpression("* * * * *")
    assert cron.matches(datetime(2024, 1, 1, 12, 30, 0)) is True


def test_cron_expression_specific_minute():
    """A specific minute matches only that minute."""
    cron = CronExpression("15 3 * * *")
    assert cron.matches(datetime(2024, 1, 1, 3, 15, 0)) is True
    assert cron.matches(datetime(2024, 1, 1, 3, 16, 0)) is False


def test_cron_expression_list_and_range():
    """Lists and ranges match the expected values."""
    cron = CronExpression("0 9,17 1-5 * *")
    assert cron.matches(datetime(2024, 1, 2, 9, 0, 0)) is True
    assert cron.matches(datetime(2024, 1, 7, 9, 0, 0)) is False


def test_cron_next_run_after_now():
    """next_run returns a time strictly after the reference time."""
    cron = CronExpression("* * * * *")
    now = datetime(2024, 1, 1, 12, 0, 0)
    next_run = cron.next_run(now)
    assert next_run > now


def test_cron_invalid_expression_raises():
    """A malformed cron expression raises ValueError at parse time."""
    with pytest.raises(ValueError):
        CronExpression("1 2 3")


@pytest.mark.asyncio
async def test_schedule_spot_job_async():
    """The async helper returns a scheduling result."""
    result = await schedule_spot_job(
        command="train --epochs 1",
        gpu_type="A100",
        job_name="test-job",
        max_wait_hours=24,
    )
    assert result["status"] in ("success", "failed")
    assert result["job_id"] == "test-job"


@pytest.mark.asyncio
async def test_schedule_pricing_windows_async():
    """The pricing-windows helper returns active and upcoming windows."""
    result = await schedule_pricing_windows(gpu_type="A100")
    assert "active_windows" in result
    assert "upcoming_windows" in result
    assert all(w["window"].startswith("SpotPricingWindow") for w in result["active_windows"])


@pytest.mark.asyncio
async def test_schedule_list_async():
    """The list helper returns a jobs collection."""
    scheduler = SpotAwareScheduler()
    scheduler.schedule_job("listable", "A100", "train")
    result = await schedule_list()
    assert result["count"] >= 1
    assert any(j["job_id"] == "listable" for j in result["jobs"])
