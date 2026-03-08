#!/usr/bin/env python3
"""
Provider Factory - Creates and manages cloud provider instances

All provider imports are lazy — each is loaded on first use so that a
missing optional dependency (e.g. boto3) does not crash the whole CLI.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

from .base_provider import BaseProvider


def _lazy_import(module_attr: str):
    """Return a callable that imports a provider class on first call."""
    module_path, class_name = module_attr.rsplit(".", 1)

    def _load():
        import importlib
        mod = importlib.import_module(module_path, package=__package__)
        return getattr(mod, class_name)
    return _load


# Registry of provider name → lazy loader.  The actual import only fires
# when ProviderFactory.create_provider() is called for that specific name.
_PROVIDER_LOADERS = {
    "aws":          _lazy_import(".aws_provider.AWSProvider"),
    "gcp":          _lazy_import(".gcp_provider.GCPProvider"),
    "azure":        _lazy_import(".azure_provider.AzureProvider"),
    "runpod":       _lazy_import(".runpod_provider.RunPodProvider"),
    "vastai":       _lazy_import(".vastai_provider.VastAIProvider"),
    "lambda":       _lazy_import(".lambda_labs_provider.LambdaLabsProvider"),
    "coreweave":    _lazy_import(".coreweave_provider.CoreWeaveProvider"),
    "tensordock":   _lazy_import(".tensordock_provider.TensorDockProvider"),
    "huggingface":  _lazy_import(".huggingface_provider.HuggingFaceProvider"),
    "baseten":      _lazy_import(".baseten_provider.BasetenProvider"),
    "oracle":       _lazy_import(".oracle_provider.OracleProvider"),
    "crusoe":       _lazy_import(".crusoe_provider.CrusoeProvider"),
    "hyperstack":   _lazy_import(".hyperstack_provider.HyperstackProvider"),
    "digitalocean": _lazy_import(".digitalocean_provider.DigitalOceanProvider"),
    "alibaba":      _lazy_import(".alibaba_provider.AlibabaProvider"),
    "ovhcloud":     _lazy_import(".ovhcloud_provider.OVHcloudProvider"),
    "fluidstack":   _lazy_import(".fluidstack_provider.FluidStackProvider"),
    "hetzner":      _lazy_import(".hetzner_provider.HetznerProvider"),
    "siliconflow":  _lazy_import(".siliconflow_provider.SiliconFlowProvider"),
    "demo":         _lazy_import(".demo_mode.DemoModeProvider"),
}


class ProviderFactory:
    """Factory for creating cloud provider instances"""

    def __init__(self):
        self._provider_classes: Dict[str, Any] = {}
        self._loaders = dict(_PROVIDER_LOADERS)

    def _resolve(self, provider_name: str):
        """Lazy-load a provider class on first access."""
        if provider_name not in self._provider_classes:
            loader = self._loaders.get(provider_name)
            if loader is None:
                raise ValueError(f"Unknown provider: {provider_name}")
            try:
                self._provider_classes[provider_name] = loader()
            except ImportError as e:
                raise ImportError(
                    f"Provider '{provider_name}' requires a missing dependency: {e}. "
                    f"Install it with: pip install <package>"
                ) from e
        return self._provider_classes[provider_name]

    def create_provider(
        self, provider_name: str, credentials: Dict[str, str]
    ) -> BaseProvider:
        """Create a provider instance"""
        provider_class = self._resolve(provider_name)
        return provider_class(credentials)

    def get_supported_providers(self) -> list:
        """Get list of supported providers"""
        return list(self._loaders.keys())

    def register_provider(self, provider_name: str, provider_class: type) -> None:
        """Register a new provider class"""
        if not issubclass(provider_class, BaseProvider):
            raise ValueError("Provider class must inherit from BaseProvider")

        self._provider_classes[provider_name] = provider_class

    def create_all_providers(
        self, credentials: Dict[str, Dict[str, str]]
    ) -> Dict[str, BaseProvider]:
        """Create all configured providers"""
        providers = {}

        for provider_name, provider_credentials in credentials.items():
            try:
                provider = self.create_provider(provider_name, provider_credentials)
                providers[provider_name] = provider
            except Exception as e:
                logger.debug(f"Failed to create provider {provider_name}: {e}")

        return providers
