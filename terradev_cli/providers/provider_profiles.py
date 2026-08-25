#!/usr/bin/env python3
"""
Provider Profiles — Encodes provider-specific quirks and behaviors.

Each provider has unique API patterns, auth methods, capacity constraints,
and lifecycle quirks. These profiles enable intelligent routing decisions
in ProviderRegistry and CLI.
"""

from .types import ProviderProfile

# Provider profiles based on audit of provider implementations
PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "runpod": ProviderProfile(
        name="runpod",
        api_style="graphql",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="graphql",
        volume_required_for_persistence=True,
        volume_cost_separate=True,
        data_loss_on_restart=True,
        egress_cost=0.0,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=100,
        compute_model="pod",
        isolation_level="container",
        has_multi_tier_cloud=True,  # Community vs Secure
        supports_stop_start=True,
    ),
    "vastai": ProviderProfile(
        name="vastai",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="post_filter",  # POST /bundles/ with JSON filter
        egress_cost=0.01,
        ssh_port_fixed=False,  # Dynamic SSH port per instance
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "aws": ProviderProfile(
        name="aws",
        api_style="rest",
        auth_type="service_account",  # access_key + secret_key
        requires_instance_type_mapping=True,  # gpu_type → ec2_instance_type
        quote_method="get",
        egress_cost=0.09,  # $0.09/GB outbound
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=2,
        spot_preemption_webhook=True,  # CloudWatch Events
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        requires_boto3=True,
        has_capacity_reservations=True,
        supports_stop_start=True,
    ),
    "gcp": ProviderProfile(
        name="gcp",
        api_style="rest",
        auth_type="service_account",  # JSON credentials file
        requires_instance_type_mapping=True,
        quote_method="get",
        egress_cost=0.12,  # $0.12/GB outbound
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        requires_gcp_sdk=True,
        has_capacity_reservations=True,
        requires_zone_probing=True,  # H100 availability varies by zone
        supports_stop_start=True,
    ),
    "azure": ProviderProfile(
        name="azure",
        api_style="rest",
        auth_type="service_account",
        requires_instance_type_mapping=True,
        quote_method="get",
        egress_cost=0.087,  # $0.087/GB outbound
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "crusoe": ProviderProfile(
        name="crusoe",
        api_style="rest",
        auth_type="hmac_sha256",  # Per-request HMAC-SHA256 signing
        requires_instance_type_mapping=False,
        quote_method="get",
        provision_requires_location_id=True,
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "digitalocean": ProviderProfile(
        name="digitalocean",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=False,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        region_specific_availability=True,
        supports_stop_start=True,
    ),
    "yottalabs": ProviderProfile(
        name="yottalabs",
        api_style="rest",
        auth_type="x_api_key",  # X-Api-Key header (NOT Bearer)
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="pod",
        isolation_level="container",
        supports_stop_start=True,
    ),
    "tensordock": ProviderProfile(
        name="tensordock",
        api_style="jsonapi",  # JSON:API-style envelope
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        provision_requires_location_id=True,
        egress_cost=0.005,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=False,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "hyperstack": ProviderProfile(
        name="hyperstack",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "oracle": ProviderProfile(
        name="oracle",
        api_style="rest",
        auth_type="service_account",  # RSA signing limitation
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "ovhcloud": ProviderProfile(
        name="ovhcloud",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "hetzner": ProviderProfile(
        name="hetzner",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=False,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "siliconflow": ProviderProfile(
        name="siliconflow",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "alibaba": ProviderProfile(
        name="alibaba",
        api_style="rest",
        auth_type="service_account",
        requires_instance_type_mapping=True,
        quote_method="get",
        egress_cost=0.08,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "e2enetworks": ProviderProfile(
        name="e2enetworks",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=False,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "baseten": ProviderProfile(
        name="baseten",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=False,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "inferx": ProviderProfile(
        name="inferx",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "latitude": ProviderProfile(
        name="latitude",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
    "huggingface": ProviderProfile(
        name="huggingface",
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=False,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    ),
}


def get_profile(provider_name: str) -> ProviderProfile:
    """Get provider profile by name. Returns default profile if not found."""
    return PROVIDER_PROFILES.get(provider_name.lower(), _default_profile(provider_name))


def _default_profile(provider_name: str) -> ProviderProfile:
    """Default profile for unknown providers (conservative defaults)."""
    return ProviderProfile(
        name=provider_name,
        api_style="rest",
        auth_type="bearer",
        requires_instance_type_mapping=False,
        quote_method="get",
        egress_cost=0.0,
        ssh_port_fixed=True,
        ssh_default_port=22,
        supports_spot=True,
        spot_interruption_notice_minutes=0,
        rate_limit_per_minute=0,
        compute_model="vm",
        isolation_level="vm",
        supports_stop_start=True,
    )


def list_all_profiles() -> dict[str, ProviderProfile]:
    """Get all provider profiles."""
    return PROVIDER_PROFILES.copy()


def register_profile(profile: ProviderProfile, override: bool = False) -> None:
    """
    Register a custom provider profile.

    Args:
        profile: ProviderProfile instance to register
        override: If True, replace existing profile with same name. If False, raise error.

    Raises:
        ValueError: If profile with same name exists and override=False
    """
    name = profile.name.lower()
    if name in PROVIDER_PROFILES and not override:
        raise ValueError(f"Profile '{name}' already exists. Use override=True to replace.")
    PROVIDER_PROFILES[name] = profile


def register_profiles_from_dict(profiles_dict: dict, override: bool = False) -> None:
    """
    Register multiple provider profiles from a dictionary.

    Args:
        profiles_dict: Dict of {name: profile_dict} where profile_dict contains ProviderProfile fields
        override: If True, replace existing profiles. If False, skip existing.

    Example:
        profiles_dict = {
            "my_provider": {
                "name": "my_provider",
                "api_style": "rest",
                "auth_type": "bearer",
                "egress_cost": 0.05,
                "supports_spot": True,
            }
        }
        register_profiles_from_dict(profiles_dict)
    """
    for name, profile_data in profiles_dict.items():
        if isinstance(profile_data, dict):
            profile = ProviderProfile(name=name, **profile_data)
            register_profile(profile, override=override)
        elif isinstance(profile_data, ProviderProfile):
            register_profile(profile_data, override=override)


def load_profiles_from_file(path: str, override: bool = False) -> None:
    """
    Load provider profiles from a YAML or JSON file.

    Args:
        path: Path to config file (.yaml, .yml, or .json)
        override: If True, replace existing profiles. If False, skip existing.

    Example YAML:
        profiles:
          my_provider:
            api_style: rest
            auth_type: bearer
            egress_cost: 0.05
            supports_spot: true
    """
    from pathlib import Path

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    ext = path_obj.suffix.lower()

    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required to load YAML profiles. Install with: pip install pyyaml")

        with open(path_obj) as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        import json
        with open(path_obj) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .yaml, .yml, or .json")

    # Extract profiles dict (support both direct dict and nested under 'profiles' key)
    if isinstance(data, dict):
        profiles_data = data.get("profiles", data)
        register_profiles_from_dict(profiles_data, override=override)
    else:
        raise ValueError(f"Invalid profile file format: {path}")


def unregister_profile(provider_name: str) -> bool:
    """
    Remove a provider profile from the registry.

    Args:
        provider_name: Name of provider to unregister

    Returns:
        True if profile was removed, False if it didn't exist
    """
    name = provider_name.lower()
    if name in PROVIDER_PROFILES:
        del PROVIDER_PROFILES[name]
        return True
    return False
