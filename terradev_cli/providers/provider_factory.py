#!/usr/bin/env python3
"""
Provider Factory - Creates and manages cloud provider instances

All provider imports are lazy — each is loaded on first use so that a
missing optional dependency (e.g. boto3) does not crash the whole CLI.
"""

import logging
from typing import Dict, Any, Optional

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
    "aws": _lazy_import(".aws_provider.AWSProvider"),
    "gcp": _lazy_import(".gcp_provider.GCPProvider"),
    "azure": _lazy_import(".azure_provider.AzureProvider"),
    "runpod": _lazy_import(".runpod_provider.RunPodProvider"),
    "vastai": _lazy_import(".vastai_provider.VastAIProvider"),
    "tensordock": _lazy_import(".tensordock_provider.TensorDockProvider"),
    "huggingface": _lazy_import(".huggingface_provider.HuggingFaceProvider"),
    "baseten": _lazy_import(".baseten_provider.BasetenProvider"),
    "crusoe": _lazy_import(".crusoe_provider.CrusoeProvider"),
    "hyperstack": _lazy_import(".hyperstack_provider.HyperstackProvider"),
    "digitalocean": _lazy_import(".digitalocean_provider.DigitalOceanProvider"),
    "ovhcloud": _lazy_import(".ovhcloud_provider.OVHcloudProvider"),
    "siliconflow": _lazy_import(".siliconflow_provider.SiliconFlowProvider"),
    "inferx": _lazy_import(".inferx_provider.InferXProvider"),
    "latitude": _lazy_import(".latitude_provider.LatitudeProvider"),
    "demo": _lazy_import(".demo_mode.DemoModeProvider"),
    "yottalabs": _lazy_import(".yottalabs_provider.YottaLabsProvider"),
    "e2enetworks": _lazy_import(".e2e_networks_provider.E2ENetworksProvider"),
}


class ProviderFactory:
    """Factory for creating cloud provider instances"""

    def __init__(self):
        self._provider_classes: Dict[str, Any] = {}
        self._provider_instances: Dict[str, Any] = {}
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

    def get_provider(self, provider_name: str, credentials: Optional[Dict[str, str]] = None):
        """Get or create a provider instance, cached for reuse."""
        if provider_name in self._provider_instances:
            return self._provider_instances[provider_name]

        credentials = credentials or {}
        provider_class = self._provider_classes.get(provider_name)
        if not provider_class:
            provider_class = self._resolve(provider_name)

        instance = provider_class(credentials)
        self._provider_instances[provider_name] = instance
        return instance

    def get_supported_providers(self) -> list:
        """Get list of supported providers"""
        return list(self._loaders.keys())

    def requires_auth(self, provider_name: str) -> bool:
        """Return False if the provider can quote without credentials."""
        try:
            provider_class = self._resolve(provider_name)
        except (ValueError, ImportError):
            return True
        return getattr(provider_class, "REQUIRES_AUTH", True)

    def register_provider(self, provider_name: str, provider_class: type) -> None:
        """Register a new provider class"""
        if not issubclass(provider_class, BaseProvider):
            raise ValueError("Provider class must inherit from BaseProvider")

        self._provider_classes[provider_name] = provider_class
        self._provider_instances.pop(provider_name, None)

    def create_all_providers(
        self, credentials: Dict[str, Dict[str, str]]
    ) -> Dict[str, BaseProvider]:
        """Create all configured providers"""
        providers = {}

        for provider_name, provider_credentials in credentials.items():
            try:
                provider = self.create_provider(provider_name, provider_credentials)
                providers[provider_name] = provider
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to create provider {provider_name}: {e}")

        return providers
