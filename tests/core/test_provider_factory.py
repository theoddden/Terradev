"""Tests for terradev_cli.providers.provider_factory.

Provider instantiation is lazy, but the factory must reliably create and cache
instances when clients ask for them.
"""

import pytest

from terradev_cli.providers.provider_factory import ProviderFactory


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
    """Clients can register a custom provider class."""

    class DummyProvider:
        pass

    factory = ProviderFactory()
    factory.register_provider("dummy", DummyProvider)
    assert factory.get_provider("dummy", {}) is DummyProvider
