#!/usr/bin/env python3
"""Tests for providers/provider_factory.py"""

import pytest
from terradev_cli.providers.provider_factory import ProviderFactory, _lazy_import
from terradev_cli.providers.base_provider import BaseProvider


class TestLazyImport:
    """Test _lazy_import function"""

    def test_lazy_import_structure(self):
        """Lazy import returns a callable"""
        loader = _lazy_import(".base_provider.BaseProvider")
        assert callable(loader)

    def test_lazy_import_execution(self):
        """Lazy import loads the class when called"""
        loader = _lazy_import(".base_provider.BaseProvider")
        provider_class = loader()
        assert provider_class is BaseProvider

    def test_lazy_import_invalid_module(self):
        """Lazy import raises ImportError for invalid module"""
        loader = _lazy_import(".nonexistent.NonexistentProvider")
        with pytest.raises(ImportError):
            loader()


class TestProviderFactory:
    """Test ProviderFactory"""

    def test_initialization(self):
        """Factory initialization"""
        factory = ProviderFactory()
        assert factory._provider_classes == {}
        assert len(factory._loaders) > 0

    def test_get_supported_providers(self):
        """Get list of supported providers"""
        factory = ProviderFactory()
        providers = factory.get_supported_providers()
        
        assert isinstance(providers, list)
        assert "aws" in providers
        assert "gcp" in providers
        assert "azure" in providers
        assert "runpod" in providers
        assert "demo" in providers

    def test_create_provider_aws(self):
        """Create AWS provider"""
        factory = ProviderFactory()
        provider = factory.create_provider("aws", {"api_key": "test"})
        
        assert provider is not None
        assert isinstance(provider, BaseProvider)

    def test_create_provider_demo(self):
        """Create demo provider (no dependencies)"""
        factory = ProviderFactory()
        provider = factory.create_provider("demo", {})
        
        assert provider is not None
        # DemoModeProvider uses provider_name from initialization

    def test_create_provider_unknown(self):
        """Create unknown provider raises ValueError"""
        factory = ProviderFactory()
        with pytest.raises(ValueError) as exc_info:
            factory.create_provider("unknown_provider", {})
        
        assert "Unknown provider" in str(exc_info.value)

    def test_create_provider_missing_dependency(self):
        """Create provider with missing dependency raises ImportError"""
        factory = ProviderFactory()
        
        # This test is skipped - requires mocking missing dependencies
        pass

    def test_register_provider(self):
        """Register a new provider class"""
        factory = ProviderFactory()
        
        class MockProvider(BaseProvider):
            def __init__(self, credentials):
                super().__init__(credentials)
                self.name = "mock"
            
            async def get_instance_quotes(self, gpu_type, region=None):
                return []
            
            async def provision_instance(self, instance_type, region, gpu_type, ssh_public_key=""):
                return {}
            
            async def get_instance_status(self, instance_id):
                return {}
            
            async def stop_instance(self, instance_id):
                return {}
            
            async def start_instance(self, instance_id):
                return {}
            
            async def terminate_instance(self, instance_id):
                return {}
            
            async def list_instances(self):
                return []
            
            async def execute_command(self, instance_id, command, async_exec=False):
                return {}
            
            def _get_auth_headers(self):
                return {}
        
        factory.register_provider("mock", MockProvider)
        provider = factory.create_provider("mock", {})
        
        assert provider.name == "mock"

    def test_register_provider_invalid(self):
        """Register non-BaseProvider class raises ValueError"""
        factory = ProviderFactory()
        
        class NotAProvider:
            pass
        
        with pytest.raises(ValueError) as exc_info:
            factory.register_provider("invalid", NotAProvider)
        
        assert "must inherit from BaseProvider" in str(exc_info.value)

    def test_create_all_providers(self):
        """Create all configured providers"""
        factory = ProviderFactory()
        
        credentials = {
            "demo": {},
            "aws": {"api_key": "test"},
        }
        
        providers = factory.create_all_providers(credentials)
        
        assert "demo" in providers
        assert "aws" in providers  # May fail if boto3 not installed

    def test_create_all_providers_empty(self):
        """Create all providers with empty credentials"""
        factory = ProviderFactory()
        providers = factory.create_all_providers({})
        
        assert providers == {}

    def test_create_all_providers_handles_errors(self):
        """Create all providers handles errors gracefully"""
        factory = ProviderFactory()
        
        credentials = {
            "demo": {},
            "unknown_provider": {},  # This should fail silently
        }
        
        providers = factory.create_all_providers(credentials)
        
        # Should still create demo provider
        assert "demo" in providers
        # Unknown provider should be skipped
        assert "unknown_provider" not in providers

    def test_resolve_caches_provider_class(self):
        """Resolve caches provider class after first load"""
        factory = ProviderFactory()
        
        # First call
        provider1 = factory.create_provider("demo", {})
        # Second call should use cached class
        provider2 = factory.create_provider("demo", {})
        
        assert provider1 is not provider2  # Different instances
        assert "demo" in factory._provider_classes  # Cached

    def test_supported_providers_count(self):
        """Check number of supported providers"""
        factory = ProviderFactory()
        providers = factory.get_supported_providers()
        
        # Should have at least the major providers
        assert len(providers) >= 19

    def test_provider_loaders_registry(self):
        """Provider loaders registry contains expected entries"""
        factory = ProviderFactory()
        
        expected_providers = [
            "aws", "gcp", "azure", "runpod", "vastai",
            "tensordock", "demo", "latitude",
        ]
        
        for provider in expected_providers:
            assert provider in factory._loaders

    def test_create_provider_with_credentials(self):
        """Create provider with credentials"""
        factory = ProviderFactory()
        credentials = {"api_key": "test_key", "secret": "test_secret"}
        
        provider = factory.create_provider("demo", credentials)
        
        # DemoModeProvider doesn't use credentials like other providers
        assert provider is not None
