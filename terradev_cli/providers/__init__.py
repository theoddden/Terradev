# terradev_cli.providers
#
# This module exports typed domain contracts and the provider factory.
# Individual provider classes are loaded lazily via ProviderFactory to avoid
# import errors when optional dependencies (boto3, google-cloud-*, etc.) are missing.

from .types import (
    GPUDescriptor,
    GPUVendor,
    InstanceStatus,
    Quote,
    QuoteRequest,
    ProvisionRequest,
    ProvisionResult,
    InstanceInfo,
    ProviderEvent,
    CredentialField,
    ProviderCapabilities,
    HealthStatus,
    ProviderHealth,
)

from .base_provider import BaseProvider
from .provider_factory import ProviderFactory
from .registry import ProviderRegistry
from .gpu_catalog import (
    normalize,
    get_canonical_name,
    list_all_canonical_gpus,
    list_providers_for_gpu,
)

__all__ = [
    # Types
    "GPUDescriptor",
    "GPUVendor",
    "InstanceStatus",
    "Quote",
    "QuoteRequest",
    "ProvisionRequest",
    "ProvisionResult",
    "InstanceInfo",
    "ProviderEvent",
    "CredentialField",
    "ProviderCapabilities",
    "HealthStatus",
    "ProviderHealth",
    # Core classes
    "BaseProvider",
    "ProviderFactory",
    "ProviderRegistry",
    # GPU catalog functions
    "normalize",
    "get_canonical_name",
    "list_all_canonical_gpus",
    "list_providers_for_gpu",
]
