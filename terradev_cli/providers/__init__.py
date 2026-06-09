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
    ProviderProfile,
)

from .base_provider import BaseProvider
from .provider_factory import ProviderFactory
from .registry import ProviderRegistry
from .provider_profiles import (
    get_profile,
    list_all_profiles,
    register_profile,
    register_profiles_from_dict,
    load_profiles_from_file,
    unregister_profile,
)
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
    "ProviderProfile",
    # Core classes
    "BaseProvider",
    "ProviderFactory",
    "ProviderRegistry",
    # GPU catalog functions
    "normalize",
    "get_canonical_name",
    "list_all_canonical_gpus",
    "list_providers_for_gpu",
    # Provider profiles
    "get_profile",
    "list_all_profiles",
    "register_profile",
    "register_profiles_from_dict",
    "load_profiles_from_file",
    "unregister_profile",
]
