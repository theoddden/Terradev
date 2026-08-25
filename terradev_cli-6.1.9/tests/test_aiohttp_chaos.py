"""Resilience/chaos tests for aiohttp client interactions."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from aiohttp.client_reqrep import ConnectionKey
import pytest

from terradev_cli.commands._api import TerradevAPI


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A real TerradevAPI backed by a temp config directory."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("TERRADEV_SKIP_ONBOARDING", "1")
    return TerradevAPI()


@pytest.mark.asyncio
class TestAiohttpResilience:
    async def _make_provider(self, exc=None, quotes=None):
        prov = MagicMock()
        prov.get_instance_quotes = AsyncMock(side_effect=exc, return_value=quotes)
        prov.session = MagicMock()
        prov.session.closed = False
        prov.session.close = AsyncMock()
        return prov

    async def test_client_error_returns_empty_and_closes_session(self, api):
        from terradev_cli.providers.provider_factory import ProviderFactory

        provider = await self._make_provider(exc=aiohttp.ClientError("connection refused"))
        with patch.object(ProviderFactory, "create_provider", return_value=provider):
            result = await api._get_provider_quotes("runpod", "A100")
        assert result == []
        provider.session.close.assert_awaited_once()

    async def test_timeout_error_returns_empty_and_closes_session(self, api):
        from terradev_cli.providers.provider_factory import ProviderFactory

        provider = await self._make_provider(exc=asyncio.TimeoutError())
        with patch.object(ProviderFactory, "create_provider", return_value=provider):
            result = await api._get_provider_quotes("runpod", "A100")
        assert result == []
        provider.session.close.assert_awaited_once()

    async def test_connection_reset_returns_empty(self, api):
        from terradev_cli.providers.provider_factory import ProviderFactory

        provider = await self._make_provider(exc=ConnectionResetError())
        with patch.object(ProviderFactory, "create_provider", return_value=provider):
            result = await api._get_provider_quotes("runpod", "A100")
        assert result == []

    async def test_ssl_error_returns_empty(self, api):
        from terradev_cli.providers.provider_factory import ProviderFactory

        key = ConnectionKey("localhost", 443, True, True, None, None, None)
        provider = await self._make_provider(
            exc=aiohttp.ClientConnectorSSLError(key, OSError("certificate verify failed"))
        )
        with patch.object(ProviderFactory, "create_provider", return_value=provider):
            result = await api._get_provider_quotes("runpod", "A100")
        assert result == []

    async def test_server_error_response_returns_empty(self, api):
        from terradev_cli.providers.provider_factory import ProviderFactory

        provider = await self._make_provider(exc=aiohttp.ServerDisconnectedError("peer closed"))
        with patch.object(ProviderFactory, "create_provider", return_value=provider):
            result = await api._get_provider_quotes("runpod", "A100")
        assert result == []

    async def test_success_returns_quotes_and_closes_session(self, api):
        from terradev_cli.providers.provider_factory import ProviderFactory

        raw = [
            {
                "price_per_hour": 1.0,
                "gpu_type": "A100",
                "region": "us-east-1",
                "gpu_count": 1,
                "instance_type": "t",
                "available": True,
                "latency_ms": 50,
            }
        ]
        provider = await self._make_provider(quotes=raw)
        with patch.object(ProviderFactory, "create_provider", return_value=provider):
            result = await api._get_provider_quotes("runpod", "A100")
        assert len(result) == 1
        assert result[0]["provider"] == "Runpod"
        provider.session.close.assert_awaited_once()
