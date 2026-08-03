#!/usr/bin/env python3
"""Fault-injection and network-degradation tests for the quote command.

These tests simulate real-world provider failures (HTTP 502s, timeouts,
truncated JSON, slow sockets) and verify that `terradev quote` does not hang,
does not crash, and gracefully degrades to the providers that remain healthy.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock

import aiohttp
import pytest

from terradev_cli.commands import cli


SAMPLE_QUOTE = {
    "provider": "RunPod",
    "price": 1.50,
    "gpu_type": "A100-80GB",
    "region": "us-east-1",
    "availability": "on-demand",
    "gpu_count": 1,
    "instance_type": "runpod-a100-80gb",
    "memory_gb": 80,
}


class TestQuoteFaultInjection:
    """Fault-injection tests for the quote aggregator."""

    def test_quote_returns_quotes_despite_provider_failures(self, runner, mock_api, monkeypatch):
        """Provider exceptions should not crash quote; healthy providers win."""
        # Fail a subset of providers with different real-world errors
        mock_api.get_vastai_quotes = AsyncMock(side_effect=aiohttp.ClientError("502 Bad Gateway"))
        mock_api.get_aws_quotes = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_api.get_gcp_quotes = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))
        mock_api.get_azure_quotes = AsyncMock(side_effect=ValueError("truncated JSON"))

        # Keep RunPod healthy
        mock_api.get_runpod_quotes = AsyncMock(return_value=[SAMPLE_QUOTE])

        result = runner.invoke(cli, ["quote", "-g", "A100"], obj={"api": mock_api})

        assert result.exit_code == 0, result.output
        assert "RunPod" in result.output
        assert "$1.50" in result.output or "$2" in result.output
        assert "ERROR" not in result.output or "No quotes returned" not in result.output

    def test_quote_gracefully_handles_malformed_responses(self, runner, mock_api, monkeypatch):
        """Non-list / malformed provider responses should be ignored."""
        mock_api.get_runpod_quotes = AsyncMock(return_value=[SAMPLE_QUOTE])
        mock_api.get_vastai_quotes = AsyncMock(return_value={"error": "not a list"})
        mock_api.get_aws_quotes = AsyncMock(return_value="truncated")
        mock_api.get_gcp_quotes = AsyncMock(return_value=12345)

        result = runner.invoke(cli, ["quote", "-g", "A100"], obj={"api": mock_api})

        assert result.exit_code == 0, result.output
        assert "RunPod" in result.output

    def test_quote_does_not_hang_with_slow_provider(self, runner, mock_api, monkeypatch):
        """A slow provider should not block the whole quote round-trip."""

        async def slow_but_healthy(*_args, **_kwargs):
            await asyncio.sleep(0.05)
            return [SAMPLE_QUOTE]

        async def slow_and_empty(*_args, **_kwargs):
            await asyncio.sleep(0.05)
            return []

        mock_api.get_runpod_quotes = AsyncMock(side_effect=slow_but_healthy)
        mock_api.get_vastai_quotes = AsyncMock(side_effect=slow_and_empty)
        mock_api.get_aws_quotes = AsyncMock(side_effect=slow_and_empty)

        start = time.monotonic()
        result = runner.invoke(cli, ["quote", "-g", "A100"], obj={"api": mock_api})
        elapsed = time.monotonic() - start

        assert result.exit_code == 0, result.output
        # Concurrency means the total should be well under the sum of each sleep
        assert elapsed < 0.5, f"quote hung for {elapsed:.2f}s"
        assert "RunPod" in result.output

    def test_quote_handles_all_providers_failing(self, runner, mock_api, monkeypatch):
        """When every provider fails, quote should report the empty state cleanly."""
        for attr in [
            "get_runpod_quotes",
            "get_vastai_quotes",
            "get_aws_quotes",
            "get_gcp_quotes",
            "get_azure_quotes",
            "get_tensordock_quotes",
            "get_lambda_quotes",
            "get_coreweave_quotes",
            "get_oracle_quotes",
            "get_crusoe_quotes",
        ]:
            setattr(mock_api, attr, AsyncMock(side_effect=aiohttp.ClientError("provider down")))

        result = runner.invoke(cli, ["quote", "-g", "A100"], obj={"api": mock_api})

        assert result.exit_code == 0, result.output
        assert "No quotes returned" in result.output

    def test_quote_region_filter_still_works_with_partial_failures(self, runner, mock_api, monkeypatch):
        """Region filtering should survive a mix of failures and good data."""
        eu_quote = dict(SAMPLE_QUOTE)
        eu_quote["region"] = "eu-west-1"

        mock_api.get_runpod_quotes = AsyncMock(return_value=[eu_quote])
        mock_api.get_vastai_quotes = AsyncMock(side_effect=aiohttp.ClientError("502"))
        mock_api.get_aws_quotes = AsyncMock(return_value=[])

        result = runner.invoke(cli, ["quote", "-g", "A100", "-r", "eu-west-1"], obj={"api": mock_api})

        assert result.exit_code == 0, result.output
        assert "eu-west-1" in result.output

    def test_quote_quick_path_after_degradation(self, runner, mock_api, monkeypatch):
        """The --quick hint path should work even when some providers are degraded."""
        cheap = dict(SAMPLE_QUOTE, price=1.50)

        mock_api.get_runpod_quotes = AsyncMock(return_value=[cheap])
        mock_api.get_vastai_quotes = AsyncMock(return_value=[])
        mock_api.get_aws_quotes = AsyncMock(side_effect=asyncio.TimeoutError())

        result = runner.invoke(cli, ["quote", "-g", "A100", "--quick"], obj={"api": mock_api})

        assert result.exit_code == 0, result.output
        assert "Quick provision" in result.output
        assert "No quotes returned" not in result.output
