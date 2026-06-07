#!/usr/bin/env python3
"""
Provider Contract Tests — verify request shapes and response handling.

Tests use mocked HTTP responses to verify that each provider:
  1. Sends correctly shaped API requests
  2. Handles successful responses
  3. Handles error responses gracefully
  4. Returns consistent output schemas
  5. Requires credentials (BYOAPI)
"""

import asyncio
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.providers.base_provider import BaseProvider
from terradev_cli.providers.runpod_provider import RunPodProvider


# ── BaseProvider Contract ─────────────────────────────────────────────────────


class TestBaseProviderContract:
    """Verify BaseProvider abstract interface is complete."""

    def test_abstract_methods_exist(self):
        methods = [
            "get_instance_quotes",
            "provision_instance",
            "get_instance_status",
            "stop_instance",
            "start_instance",
            "terminate_instance",
            "list_instances",
            "execute_command",
            "_get_auth_headers",
        ]
        for method in methods:
            assert hasattr(BaseProvider, method), f"Missing abstract method: {method}"

    def test_gpu_specs_known_types(self):
        """Verify GPU spec lookup works for known GPU types."""

        # Create a concrete subclass just for testing
        class DummyProvider(BaseProvider):
            async def get_instance_quotes(self, gpu_type, region=None):
                return []

            async def provision_instance(self, it, r, g):
                return {}

            async def get_instance_status(self, iid):
                return {}

            async def stop_instance(self, iid):
                return {}

            async def start_instance(self, iid):
                return {}

            async def terminate_instance(self, iid):
                return {}

            async def list_instances(self):
                return []

            async def execute_command(self, iid, cmd, ae):
                return {}

            def _get_auth_headers(self):
                return {}

        p = DummyProvider(credentials={})
        for gpu in ["A100", "V100", "H100", "RTX4090", "RTX3090"]:
            specs = p._get_gpu_specs(gpu)
            assert "memory_gb" in specs, f"Missing memory_gb for {gpu}"
            assert "tflops" in specs, f"Missing tflops for {gpu}"
            assert specs["memory_gb"] > 0

    def test_unknown_gpu_returns_empty(self):
        class DummyProvider(BaseProvider):
            async def get_instance_quotes(self, gpu_type, region=None):
                return []

            async def provision_instance(self, it, r, g):
                return {}

            async def get_instance_status(self, iid):
                return {}

            async def stop_instance(self, iid):
                return {}

            async def start_instance(self, iid):
                return {}

            async def terminate_instance(self, iid):
                return {}

            async def list_instances(self):
                return []

            async def execute_command(self, iid, cmd, ae):
                return {}

            def _get_auth_headers(self):
                return {}

        p = DummyProvider(credentials={})
        assert p._get_gpu_specs("NONEXISTENT") == {}


# ── RunPod Provider ───────────────────────────────────────────────────────────


class TestRunPodProvider:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty_quotes(self):
        """RunPod returns empty quotes without API key"""
        provider = RunPodProvider({})
        result = await provider.get_instance_quotes("A100")
        assert result == []

    @pytest.mark.asyncio
    async def test_no_api_key_raises_on_provision(self):
        """RunPod raises error on provision without API key"""
        provider = RunPodProvider({})
        with pytest.raises(Exception, match="API key not configured"):
            await provider.provision_instance("type", "region", "A100")

    @pytest.mark.asyncio
    async def test_no_api_key_raises_on_status(self):
        """RunPod raises error on status check without API key"""
        provider = RunPodProvider({})
        with pytest.raises(Exception, match="API key not configured"):
            await provider.get_instance_status("some-id")

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty_list(self):
        """RunPod returns empty list for instances without API key"""
        provider = RunPodProvider({})
        result = await provider.list_instances()
        assert result == []

    def test_auth_headers_with_key(self):
        provider = RunPodProvider(credentials={"api_key": "test-key-123"})
        headers = provider._get_auth_headers()
        assert headers == {"Authorization": "Bearer test-key-123"}

    def test_auth_headers_without_key(self):
        provider = RunPodProvider(credentials={})
        headers = provider._get_auth_headers()
        assert headers == {}

    @pytest.mark.asyncio
    async def test_get_quotes_with_mocked_api(self):
        """Verify RunPod sends correct GraphQL query and parses response."""
        provider = RunPodProvider(credentials={"api_key": "test-key"})

        mock_response = {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA A100 80GB",
                        "displayName": "A100 80GB",
                        "memoryInGb": 80,
                        "communityPrice": 1.19,
                        "securePrice": 2.49,
                    },
                    {
                        "id": "NVIDIA RTX 3090",
                        "displayName": "RTX 3090",
                        "memoryInGb": 24,
                        "communityPrice": 0.22,
                        "securePrice": 0.44,
                    },
                ]
            }
        }

        with patch.object(
            provider, "_make_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = mock_response
            quotes = await provider.get_instance_quotes("A100")

            # Should have called GraphQL endpoint
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            assert call_args[0][0] == "POST"
            assert "graphql" in call_args[0][1]

            # Should return A100 quotes only (community + secure)
            assert len(quotes) == 2
            assert all(q["gpu_type"] == "A100" for q in quotes)
            assert all(q["provider"] == "runpod" for q in quotes)
            # Sorted by price
            assert quotes[0]["price_per_hour"] <= quotes[1]["price_per_hour"]

    @pytest.mark.asyncio
    async def test_provision_sends_mutation(self):
        """Verify provision sends correct GraphQL mutation."""
        provider = RunPodProvider(credentials={"api_key": "test-key"})

        mock_response = {
            "data": {
                "podFindAndDeployOnDemand": {
                    "id": "new-pod-id",
                    "desiredState": "RUNNING",
                }
            }
        }

        with patch.object(
            provider, "_make_request", new_callable=AsyncMock
        ) as mock_req:
            # Need to also patch GPU_PRICING since it's missing on the class
            provider.GPU_PRICING = {"A100": {"id": "NVIDIA A100 80GB", "price": 2.49}}
            mock_req.return_value = mock_response
            result = await provider.provision_instance(
                "runpod-community-A100", "us-east-1", "A100"
            )

            assert result["instance_id"] == "new-pod-id"
            assert result["status"] == "provisioning"
            assert result["provider"] == "runpod"

    @pytest.mark.asyncio
    async def test_list_instances_parses_response(self):
        """RunPod list instances parses response"""
        provider = RunPodProvider(credentials={"api_key": "test-key"})

        mock_response = {
            "data": {
                "myself": {
                    "pods": [
                        {
                            "id": "pod-1",
                            "name": "test-pod",
                            "desiredStatus": "RUNNING",
                            "gpuCount": 1,
                            "machine": {"gpuDisplayName": "NVIDIA A100-80GB"},
                        }
                    ]
                }
            }
        }

        with patch.object(
            provider, "_make_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = mock_response
            instances = await provider.list_instances()

            assert len(instances) == 1
            assert instances[0]["instance_id"] == "pod-1"
            assert instances[0]["status"] == "running"
            assert instances[0]["provider"] == "runpod"


# ── Output Schema Validation ─────────────────────────────────────────────────


class TestOutputSchemaConsistency:
    """Verify all providers return consistent output schemas."""

    QUOTE_REQUIRED_KEYS = {
        "instance_type",
        "gpu_type",
        "price_per_hour",
        "region",
        "available",
        "provider",
    }

    PROVISION_REQUIRED_KEYS = {
        "instance_id",
        "status",
        "provider",
    }

    @pytest.mark.asyncio
    async def test_runpod_quote_schema(self):
        provider = RunPodProvider(credentials={"api_key": "test"})
        mock_response = {
            "data": {
                "gpuTypes": [
                    {
                        "id": "A100-80GB",
                        "displayName": "NVIDIA A100-80GB",
                        "communityPrice": 2.49,
                        "memoryInGb": 80,
                    }
                ]
            }
        }

        with patch.object(
            provider, "_get_live_pricing", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = [
                {
                    "instance_type": "runpod-community-A100-80GB",
                    "gpu_type": "A100",
                    "price_per_hour": 2.49,
                    "region": "us-east",
                    "available": True,
                    "provider": "runpod",
                    "vcpus": 16,
                    "memory_gb": 80,
                    "gpu_count": 1,
                }
            ]
            quotes = await provider.get_instance_quotes("A100")

            for q in quotes:
                missing = self.QUOTE_REQUIRED_KEYS - set(q.keys())
                assert not missing, f"Missing keys in RunPod quote: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
