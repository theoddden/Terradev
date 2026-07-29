"""Tests for terradev_cli.core.rate_limiter.

Rate limiting protects provider APIs and client quotas. These tests cover the
Python fallback throttler, metrics, and adaptive delays.
"""

import asyncio

import pytest

from terradev_cli.core.rate_limiter import (
    ProviderRateLimit,
    RateLimitedSession,
    RateLimiter,
    RateLimitMetrics,
    RateLimitStrategy,
)


@pytest.fixture
def limiter():
    return RateLimiter()


def test_default_provider_limits_initialized(limiter):
    """Default limits are loaded for common providers."""
    for provider in ["aws", "gcp", "azure", "runpod"]:
        limit = limiter.get_provider_limit(provider)
        assert limit is not None
        assert limit.requests_per_second > 0
        assert limit.burst_limit > 0


def test_set_and_reset_provider_limits(limiter):
    """Limits can be overridden and metrics reset."""
    custom = ProviderRateLimit(
        requests_per_second=1.0,
        requests_per_minute=10,
        burst_limit=1,
        strategy=RateLimitStrategy.TOKEN_BUCKET,
    )
    limiter.set_provider_limit("test", custom)
    assert limiter.get_provider_limit("test") is custom
    assert limiter.get_provider_metrics("test") is not None

    limiter.reset_metrics("test")
    metrics = limiter.get_provider_metrics("test")
    assert metrics.total_requests == 0
    assert metrics.successful_requests == 0


def test_metrics_dataclass_defaults():
    """RateLimitMetrics has sensible defaults."""
    m = RateLimitMetrics()
    assert m.total_requests == 0
    assert m.current_rate == 0.0


@pytest.mark.asyncio
async def test_acquire_increments_metrics(limiter):
    """acquire increments total_requests for configured providers."""
    assert await limiter.acquire("aws") is True
    metrics = limiter.get_provider_metrics("aws")
    assert metrics.total_requests == 1


@pytest.mark.asyncio
async def test_acquire_unknown_provider_is_permissive(limiter):
    """Unknown providers are allowed through with a warning."""
    assert await limiter.acquire("unknown") is True


@pytest.mark.asyncio
async def test_execute_with_rate_limit_runs_func(limiter):
    """execute_with_rate_limit runs the wrapped async function."""
    async def dummy():
        return "ok"

    result = await limiter.execute_with_rate_limit("aws", dummy)
    assert result == "ok"

    metrics = limiter.get_provider_metrics("aws")
    assert metrics.successful_requests == 1


@pytest.mark.asyncio
async def test_execute_with_rate_limit_retries(limiter):
    """execute_with_rate_limit retries on aiohttp.ClientError."""
    import aiohttp

    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise aiohttp.ClientError("boom")
        return "recovered"

    # Use a provider with few retry attempts so the test is fast
    limiter.set_provider_limit(
        "flaky",
        ProviderRateLimit(
            requests_per_second=100,
            requests_per_minute=1000,
            retry_attempts=3,
            backoff_factor=1.1,
        ),
    )

    result = await limiter.execute_with_rate_limit("flaky", flaky)
    assert result == "recovered"
    assert attempts["n"] == 2


def test_calculate_current_rate_and_is_rate_limited(limiter):
    """Current rate and rate-limited status reflect recent activity."""
    assert limiter.calculate_current_rate("aws") == 0.0
    assert limiter.is_rate_limited("aws") is False


def test_get_adaptive_delay_scales(limiter):
    """Adaptive delay is zero for a provider with no recent requests."""
    assert limiter.get_adaptive_delay("aws") == 0.0


def test_status_report_and_all_metrics(limiter):
    """Status report and all metrics aggregate provider state."""
    report = limiter.get_status_report()
    assert "providers" in report
    assert "global_rate_limit" in report

    all_metrics = limiter.get_all_metrics()
    assert "aws" in all_metrics


def test_rate_limited_session_instantiation(limiter):
    """RateLimitedSession wraps a limiter and provider name."""
    session = RateLimitedSession(limiter, "aws")
    assert session.rate_limiter is limiter
    assert session.provider == "aws"
