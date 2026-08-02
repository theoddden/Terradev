"""Tests for multi-provider quote concurrency and resilience."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terradev_cli.core.terradev_engine import InstanceQuote, TerradevEngine


@pytest.fixture
def engine(tmp_path):
    """Engine with no real providers."""
    cfg = MagicMock()
    cfg.get_enabled_providers.return_value = []
    cfg.get_provider_reliability.return_value = 0.8
    auth = MagicMock()
    with patch("terradev_cli.core.terradev_engine.ProviderFactory"), patch(
        "terradev_cli.core.terradev_engine.ProviderRegistry"
    ):
        engine = TerradevEngine(config=cfg, auth=auth)
        engine.providers = {}
        return engine


def _make_provider(name, quotes, delay=0):
    provider = MagicMock()
    provider.name = name

    async def _get_quotes(gpu_type, region=None):
        if delay:
            await asyncio.sleep(delay)
        return quotes

    provider.get_instance_quotes = _get_quotes
    return provider


@pytest.mark.asyncio
class TestProviderConcurrency:
    async def test_get_quotes_calls_all_providers(self, engine):
        quotes_a = [
            {"price_per_hour": 1.0, "gpu_type": "A100", "region": "us-east-1", "available": True, "instance_type": "a"}
        ]
        quotes_b = [
            {"price_per_hour": 2.0, "gpu_type": "A100", "region": "us-west-1", "available": True, "instance_type": "b"}
        ]
        engine.providers = {
            "runpod": _make_provider("runpod", quotes_a),
            "vastai": _make_provider("vastai", quotes_b),
        }
        result = await engine.get_quotes(gpu_type="A100")
        providers = {q.provider for q in result}
        assert providers == {"runpod", "vastai"}

    async def test_get_quotes_ignores_unknown_providers(self, engine):
        engine.providers = {
            "runpod": _make_provider("runpod", []),
        }
        result = await engine.get_quotes(gpu_type="A100", providers=["runpod", "nonexistent"])
        assert result == []

    async def test_get_quotes_handles_exception_gracefully(self, engine):
        async def _boom(*args, **kwargs):
            raise Exception("provider down")

        healthy = _make_provider("runpod", [
            {"price_per_hour": 1.0, "gpu_type": "A100", "region": "us-east-1", "available": True, "instance_type": "a"}
        ])
        failing = MagicMock()
        failing.name = "vastai"
        failing.get_instance_quotes = _boom

        engine.providers = {"runpod": healthy, "vastai": failing}
        result = await engine.get_quotes(gpu_type="A100")
        assert len(result) == 1
        assert result[0].provider == "runpod"

    async def test_get_quotes_runs_in_parallel(self, engine):
        # Each provider sleeps 0.1s. Sequential would take ~0.2s; parallel < 0.15s.
        engine.providers = {
            "runpod": _make_provider("runpod", [
                {"price_per_hour": 1.0, "gpu_type": "A100", "region": "us-east-1", "available": True, "instance_type": "a"}
            ], delay=0.1),
            "vastai": _make_provider("vastai", [
                {"price_per_hour": 2.0, "gpu_type": "A100", "region": "us-west-1", "available": True, "instance_type": "b"}
            ], delay=0.1),
        }
        start = time.monotonic()
        result = await engine.get_quotes(gpu_type="A100")
        elapsed = time.monotonic() - start
        assert len(result) == 2
        assert elapsed < 0.18  # should be ~0.1s if truly parallel

    async def test_get_quotes_respects_provider_filter(self, engine):
        engine.providers = {
            "runpod": _make_provider("runpod", [
                {"price_per_hour": 1.0, "gpu_type": "A100", "region": "us-east-1", "available": True, "instance_type": "a"}
            ]),
            "vastai": _make_provider("vastai", [
                {"price_per_hour": 2.0, "gpu_type": "A100", "region": "us-west-1", "available": True, "instance_type": "b"}
            ]),
        }
        result = await engine.get_quotes(gpu_type="A100", providers=["runpod"])
        assert len(result) == 1
        assert result[0].provider == "runpod"

    async def test_get_quotes_returns_instance_quote_objects(self, engine):
        engine.providers = {
            "runpod": _make_provider("runpod", [
                {"price_per_hour": 1.5, "gpu_type": "A100", "region": "us-east-1", "available": True, "instance_type": "a"}
            ]),
        }
        result = await engine.get_quotes(gpu_type="A100")
        assert len(result) == 1
        assert isinstance(result[0], InstanceQuote)
        assert result[0].price_per_hour == 1.5
