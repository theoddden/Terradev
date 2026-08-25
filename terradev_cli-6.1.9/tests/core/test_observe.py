"""Tests for terradev_cli.core.observe.

The observability pipeline wires gateway traffic to W&B, Phoenix, and cost
analytics. Since those integrations are optional, tests exercise the no-op
fallback paths.
"""

from unittest.mock import patch

import pytest

from terradev_cli.core.observe import (
    ObservabilityPipeline,
    observe_gateway_traffic,
    observe_status,
)


@pytest.fixture
def pipeline():
    return ObservabilityPipeline()


def test_pipeline_initial_state(pipeline):
    """Pipeline starts with a trace ID and all destinations disabled."""
    assert pipeline.trace_id
    assert pipeline.destinations == {
        "wandb": False,
        "phoenix": False,
        "cost_analytics": False,
    }
    assert pipeline.active_destinations == []


@pytest.mark.asyncio
async def test_initialize_wandb_fallback(pipeline):
    """W&B init returns False when the integration is unavailable."""
    assert await pipeline.initialize_wandb("proj") is False
    assert pipeline.destinations["wandb"] is False


@pytest.mark.asyncio
async def test_initialize_phoenix_fallback(pipeline):
    """Phoenix init returns False when the integration is unavailable."""
    assert await pipeline.initialize_phoenix("http://localhost:6006") is False
    assert pipeline.destinations["phoenix"] is False


@pytest.mark.asyncio
async def test_initialize_cost_analytics_fallback(pipeline):
    """Cost analytics init returns False when the module is unavailable."""
    assert await pipeline.initialize_cost_analytics() is False
    assert pipeline.destinations["cost_analytics"] is False


@pytest.mark.asyncio
async def test_track_gateway_traffic_no_destinations(pipeline):
    """Tracking traffic succeeds even with no destinations enabled."""
    assert await pipeline.track_gateway_traffic({"request_count": 10}) is True


@pytest.mark.asyncio
async def test_get_trace_summary(pipeline):
    """get_trace_summary returns trace context and active destinations."""
    summary = await pipeline.get_trace_summary()
    assert summary["trace_id"] == pipeline.trace_id
    assert summary["active_destinations"] == []
    assert "shared_context" in summary


@pytest.mark.asyncio
async def test_cleanup_no_destinations(pipeline):
    """Cleanup is safe when no destinations are enabled."""
    assert await pipeline.cleanup() is True


@pytest.mark.asyncio
async def test_observe_status():
    """observe_status returns a synthetic trace status."""
    status = await observe_status("trace-1")
    assert status["trace_id"] == "trace-1"
    assert status["status"] == "completed"
    assert status["wandb"]["status"] == "active"


@pytest.mark.asyncio
async def test_observe_gateway_traffic():
    """observe_gateway_traffic runs the full pipeline and returns summary."""
    summary = await observe_gateway_traffic(
        "http://gateway:8080",
        wandb_project=None,
        phoenix_endpoint=None,
        enable_cost_analytics=False,
        duration_seconds=1,
        sample_rate=1.0,
    )
    assert summary["trace_id"]
    assert isinstance(summary["active_destinations"], list)


@pytest.mark.asyncio
async def test_observe_gateway_traffic_with_failed_init():
    """The pipeline completes even when integrations fail to initialize."""
    summary = await observe_gateway_traffic(
        "http://gateway:8080",
        wandb_project="test",
        phoenix_endpoint="http://localhost:6006",
        enable_cost_analytics=True,
        duration_seconds=1,
    )
    assert summary["trace_id"]
    assert summary["active_destinations"] == []
