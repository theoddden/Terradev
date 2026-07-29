"""Tests for terradev_cli.providers.provider_factory.

Provider instantiation is lazy, but the factory must reliably create and cache
instances when clients ask for them.
"""

from typing import Any, Dict, List

import pytest

from terradev_cli.providers.base_provider import BaseProvider
from terradev_cli.providers.provider_factory import ProviderFactory


class DummyProvider(BaseProvider):
    """Minimal provider for factory testing."""

    async def get_instance_quotes(self, gpu_type, region=None): return []
    async def provision_instance(self, instance_type, region, gpu_type, ssh_public_key=""): return {}
    async def get_instance_status(self, instance_id): return {}
    async def stop_instance(self, instance_id): return {}
    async def start_instance(self, instance_id): return {}
    async def terminate_instance(self, instance_id): return {}
    async def list_instances(self): return []
    async def execute_command(self, instance_id, command, async_exec): return {}
    def _get_auth_headers(self): return {}


def test_provider_factory_lists_supported_providers():
    """Factory exposes the providers shipped with Terradev."""
    factory = ProviderFactory()
    providers = factory.get_supported_providers()
    assert "aws" in providers
    assert "gcp" in providers
    assert "runpod" in providers
    assert "demo" in providers


def test_provider_factory_unknown_provider_raises():
    """Requesting an unknown provider raises a clear ValueError."""
    factory = ProviderFactory()
    with pytest.raises(ValueError, match="Unknown provider"):
        factory.get_provider("not-a-provider")


def test_provider_factory_caches_instances():
    """get_provider returns the same instance for repeated requests."""
    factory = ProviderFactory()
    p1 = factory.get_provider("demo", {})
    p2 = factory.get_provider("demo", {})
    assert p1 is p2


def test_provider_factory_register_custom_provider():
    """Clients can register a custom provider class and get a cached instance."""
    factory = ProviderFactory()
    factory.register_provider("dummy", DummyProvider)
    instance = factory.get_provider("dummy", {})
    assert isinstance(instance, DummyProvider)
    assert factory.get_provider("dummy", {}) is instance


def test_provider_factory_register_rejects_non_provider():
    """Only BaseProvider subclasses can be registered."""
    factory = ProviderFactory()

    class NotAProvider:
        pass

    with pytest.raises(ValueError, match="must inherit from BaseProvider"):
        factory.register_provider("bad", NotAProvider)
