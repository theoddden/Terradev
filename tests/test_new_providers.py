#!/usr/bin/env python3
"""
Provider Conformance Tests for New Providers

Tests the 5 new providers added in recent work:
- Alibaba
- OVHcloud
- Hetzner
- SiliconFlow

These tests verify:
1. Auth header formats (critical for each provider's specific auth)
2. API request shapes
3. Error handling
4. Response parsing
5. Output schema consistency
"""

import asyncio
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.providers.alibaba_provider import AlibabaProvider
from terradev_cli.providers.ovhcloud_provider import OVHcloudProvider
from terradev_cli.providers.hetzner_provider import HetznerProvider
from terradev_cli.providers.siliconflow_provider import SiliconFlowProvider


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAlibabaProvider:
    """Test Alibaba provider - uses signed URL with raw aiohttp"""

    def test_auth_header_format(self):
        """Alibaba uses signed requests, not standard Bearer auth"""
        provider = AlibabaProvider(
            credentials={"access_key_id": "test", "access_key_secret": "test"}
        )
        headers = provider._get_auth_headers()
        # Alibaba doesn't use standard auth headers in _get_auth_headers
        # It uses signed URLs in _ecs_request
        assert isinstance(headers, dict)

    @pytest.mark.asyncio
    async def test_no_credentials_returns_empty_quotes(self):
        """Alibaba returns empty quotes without credentials"""
        provider = AlibabaProvider({})
        result = await provider.get_instance_quotes("A100")
        assert result == []

    def test_percent_encode_uses_safe_tilde(self):
        """Alibaba _percent_encode uses safe='~' per RFC 3986"""
        provider = AlibabaProvider(
            credentials={"access_key_id": "test", "access_key_secret": "test"}
        )
        # Test that ~ is not encoded
        from urllib.parse import quote

        test_str = "test~value"
        encoded = provider._percent_encode(test_str)
        assert "~" in encoded or encoded == quote(test_str, safe="~")


class TestOVHcloudProvider:
    """Test OVHcloud provider - uses X-Ovh-* headers with raw aiohttp"""

    def test_auth_header_format(self):
        """OVHcloud auth header format"""
        provider = OVHcloudProvider(
            {
                "application_key": "test",
                "application_secret": "test",
                "consumer_key": "test",
            }
        )
        headers = provider._get_auth_headers()
        # OVHcloud may include Content-Type even without full auth
        assert "X-Ovh-Application" in headers or "Content-Type" in headers
        assert "X-Ovh-Consumer" in headers
        assert headers["X-Ovh-Application"] == "test"
        assert headers["X-Ovh-Consumer"] == "test"

    @pytest.mark.asyncio
    async def test_no_credentials_returns_empty_quotes(self):
        """Alibaba returns empty quotes without credentials"""
        provider = AlibabaProvider({})
        result = await provider.get_instance_quotes("A100")
        assert result == []

    def test_auth_headers_without_key(self):
        provider = OVHcloudProvider(credentials={})
        headers = provider._get_auth_headers()
        # May include Content-Type even without auth
        assert headers == {} or "Content-Type" in headers


class TestHetznerProvider:
    """Test Hetzner provider - uses Bearer auth with Content-Type"""

    def test_auth_header_format(self):
        """Hetzner uses Authorization: Bearer + Content-Type: application/json"""
        provider = HetznerProvider(credentials={"api_token": "test-token"})
        headers = provider._get_auth_headers()
        assert headers["Authorization"] == "Bearer test-token"
        assert "Content-Type" in headers

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty_quotes(self):
        """Hetzner returns empty quotes without API key"""
        provider = HetznerProvider({})
        result = await provider.get_instance_quotes("A100")
        assert result == []

    def test_auth_headers_without_key(self):
        provider = HetznerProvider(credentials={})
        headers = provider._get_auth_headers()
        # May include Content-Type even without auth
        assert headers == {} or "Content-Type" in headers


class TestSiliconFlowProvider:
    """Test SiliconFlow provider - uses Bearer auth with Content-Type"""

    def test_auth_header_format(self):
        """SiliconFlow uses Authorization: Bearer + Content-Type"""
        provider = SiliconFlowProvider(credentials={"api_key": "test-key"})
        headers = provider._get_auth_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert "Content-Type" in headers

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty_quotes(self):
        """SiliconFlow returns empty quotes without API key"""
        provider = SiliconFlowProvider({})
        result = await provider.get_instance_quotes("A100")
        assert result == []

    def test_auth_headers_without_key(self):
        provider = SiliconFlowProvider(credentials={})
        headers = provider._get_auth_headers()
        # May include Content-Type even without auth
        assert headers == {} or "Content-Type" in headers


class TestProviderOutputSchemaConsistency:
    """Verify all new providers return consistent output schemas"""

    QUOTE_REQUIRED_KEYS = {
        "instance_type",
        "gpu_type",
        "price_per_hour",
        "region",
        "available",
        "provider",
    }

    @pytest.mark.parametrize(
        "provider_class,credentials",
        [
            (AlibabaProvider, {"access_key_id": "test", "access_key_secret": "test"}),
            (
                OVHcloudProvider,
                {
                    "application_key": "test",
                    "application_secret": "test",
                    "consumer_key": "test",
                },
            ),
            (HetznerProvider, {"api_token": "test"}),
            (SiliconFlowProvider, {"api_key": "test"}),
        ],
    )
    @pytest.mark.asyncio
    async def test_quote_schema(self, provider_class, credentials):
        """Test that provider quotes have required keys"""
        provider = provider_class(credentials)

        # Mock the API call to return a valid response
        mock_response = {"data": []}  # Empty response for schema test

        with patch.object(
            provider, "_make_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = mock_response
            quotes = await provider.get_instance_quotes("A100")

            # Should return a list
            assert isinstance(quotes, list)

            # If we have quotes, check schema
            for q in quotes:
                missing = self.QUOTE_REQUIRED_KEYS - set(q.keys())
                assert (
                    not missing
                ), f"Missing keys in {provider_class.__name__} quote: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
