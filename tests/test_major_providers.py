#!/usr/bin/env python3
"""
Major Provider Conformance Tests

Tests the major cloud providers (AWS, GCP, Azure) that were previously untested.

These tests verify:
1. Auth header formats
2. Configuration initialization
3. Error handling without credentials
4. Output schema consistency
"""

import asyncio
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.providers.aws_provider import AWSProvider
from terradev_cli.providers.gcp_provider import GCPProvider
from terradev_cli.providers.azure_provider import AzureProvider


def run_async(coro):
    """Helper to run async functions in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestAWSProvider:
    """Test AWS provider"""

    def test_init_with_credentials(self):
        """AWS provider initialization with credentials"""
        provider = AWSProvider(
            {"aws_access_key_id": "test-key", "aws_secret_access_key": "test-secret"}
        )
        assert provider.credentials["aws_access_key_id"] == "test-key"
        assert provider.credentials["aws_secret_access_key"] == "test-secret"

    def test_no_credentials_returns_empty_quotes(self):
        """AWS provider returns empty quotes without credentials"""
        provider = AWSProvider({})
        result = run_async(provider.get_instance_quotes("A100"))
        assert result == []

    def test_auth_headers_with_key(self):
        """AWS provider auth headers with key"""
        provider = AWSProvider(
            {"aws_access_key_id": "test", "aws_secret_access_key": "test"}
        )
        headers = provider._get_auth_headers()
        # AWS uses boto3 which handles auth internally
        assert isinstance(headers, dict)

    def test_auth_headers_without_key(self):
        """AWS provider auth headers without key"""
        provider = AWSProvider({})
        headers = provider._get_auth_headers()
        assert isinstance(headers, dict)


class TestGCPProvider:
    """Test GCP provider"""

    def test_init_with_credentials(self):
        """GCP provider initialization with credentials"""
        provider = GCPProvider(
            {
                "gcp_project_id": "test-project",
                "gcp_credentials_path": "/path/to/credentials.json",
            }
        )
        assert provider.credentials["gcp_project_id"] == "test-project"

    def test_no_credentials_returns_empty_quotes(self):
        """GCP provider returns empty quotes without credentials"""
        provider = GCPProvider({})
        result = run_async(provider.get_instance_quotes("A100"))
        assert result == []

    def test_auth_headers_with_key(self):
        """GCP provider auth headers with key"""
        provider = GCPProvider(
            {
                "gcp_project_id": "test-project",
                "gcp_credentials_path": "/path/to/credentials.json",
            }
        )
        headers = provider._get_auth_headers()
        # GCP uses service account auth
        assert isinstance(headers, dict)

    def test_auth_headers_without_key(self):
        """GCP provider auth headers without key"""
        provider = GCPProvider({})
        headers = provider._get_auth_headers()
        assert isinstance(headers, dict)


class TestAzureProvider:
    """Test Azure provider"""

    def test_init_with_credentials(self):
        """Azure provider initialization with credentials"""
        provider = AzureProvider(
            {
                "azure_subscription_id": "test-sub",
                "azure_client_id": "test-client",
                "azure_client_secret": "test-secret",
                "azure_tenant_id": "test-tenant",
            }
        )
        assert provider.credentials["azure_subscription_id"] == "test-sub"
        assert provider.credentials["azure_client_id"] == "test-client"

    def test_no_credentials_returns_empty_quotes(self):
        """Azure provider returns empty quotes without credentials"""
        provider = AzureProvider({})
        result = run_async(provider.get_instance_quotes("A100"))
        assert result == []

    def test_auth_headers_with_key(self):
        """Azure provider auth headers with key"""
        provider = AzureProvider(
            {
                "azure_subscription_id": "test-sub",
                "azure_client_id": "test-client",
                "azure_client_secret": "test-secret",
                "azure_tenant_id": "test-tenant",
            }
        )
        headers = provider._get_auth_headers()
        # Azure uses OAuth2 token
        assert isinstance(headers, dict)

    def test_auth_headers_without_key(self):
        """Azure provider auth headers without key"""
        provider = AzureProvider({})
        headers = provider._get_auth_headers()
        assert isinstance(headers, dict)


class TestProviderOutputSchemaConsistency:
    """Verify all major providers return consistent output schemas"""

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
            (
                AWSProvider,
                {"aws_access_key_id": "test", "aws_secret_access_key": "test"},
            ),
            (
                GCPProvider,
                {
                    "gcp_project_id": "test",
                    "gcp_credentials_path": "/path/to/creds.json",
                },
            ),
            (
                AzureProvider,
                {
                    "azure_subscription_id": "test",
                    "azure_client_id": "test",
                    "azure_client_secret": "test",
                    "azure_tenant_id": "test",
                },
            ),
        ],
    )
    def test_quote_schema(self, provider_class, credentials):
        """Test that provider quotes have required keys"""
        provider = provider_class(credentials)

        # Mock the API call to return a valid response
        mock_response = {"data": []}  # Empty response for schema test

        with patch.object(
            provider, "_make_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = mock_response
            quotes = run_async(provider.get_instance_quotes("A100"))

            # Should return a list
            assert isinstance(quotes, list)

            # If we have quotes, check schema
            for q in quotes:
                missing = self.QUOTE_REQUIRED_KEYS - set(q.keys())
                assert (
                    not missing
                ), f"Missing keys in {provider_class.__name__} quote: {missing}"


class TestProviderErrorHandling:
    """Test provider error handling"""

    @pytest.mark.parametrize(
        "provider_class,credentials",
        [
            (AWSProvider, {}),
            (GCPProvider, {}),
            (AzureProvider, {}),
        ],
    )
    def test_no_api_key_raises_on_provision(self, provider_class, credentials):
        """Providers should raise error on provision without API key"""
        provider = provider_class(credentials)

        with pytest.raises(RuntimeError):
            run_async(provider.provision_instance("type", "region", "A100"))

    @pytest.mark.parametrize(
        "provider_class,credentials",
        [
            (AWSProvider, {}),
            (GCPProvider, {}),
            (AzureProvider, {}),
        ],
    )
    def test_no_api_key_returns_empty_list(self, provider_class, credentials):
        """Providers should return empty list for instances without API key"""
        provider = provider_class(credentials)
        result = run_async(provider.list_instances())
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
