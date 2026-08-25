#!/usr/bin/env python3
"""Tests for providers/provider_profiles.py"""

import pytest
import tempfile
import json
from terradev_cli.providers.provider_profiles import (
    get_profile,
    _default_profile,
    list_all_profiles,
    register_profile,
    register_profiles_from_dict,
    unregister_profile,
    PROVIDER_PROFILES,
)
from terradev_cli.providers.types import ProviderProfile


class TestGetProfile:
    """Test get_profile function"""

    def test_get_profile_existing_provider(self):
        """Get profile for existing provider"""
        profile = get_profile("aws")
        assert profile.name == "aws"
        assert profile.api_style == "rest"
        assert profile.auth_type == "service_account"

    def test_get_profile_case_insensitive(self):
        """Get profile is case-insensitive"""
        profile_lower = get_profile("aws")
        profile_upper = get_profile("AWS")
        profile_mixed = get_profile("Aws")
        
        assert profile_lower.name == profile_upper.name
        assert profile_lower.name == profile_mixed.name

    def test_get_profile_unknown_provider(self):
        """Get profile for unknown provider returns default"""
        profile = get_profile("unknown_provider")
        assert profile.name == "unknown_provider"
        assert profile.api_style == "rest"
        assert profile.auth_type == "bearer"

    def test_get_profile_runpod(self):
        """Get RunPod profile with specific attributes"""
        profile = get_profile("runpod")
        assert profile.name == "runpod"
        assert profile.api_style == "graphql"
        assert profile.supports_spot is True
        assert profile.rate_limit_per_minute == 100

class TestDefaultProfile:
    """Test _default_profile function"""

    def test_default_profile_structure(self):
        """Default profile has conservative defaults"""
        profile = _default_profile("test_provider")
        assert profile.name == "test_provider"
        assert profile.api_style == "rest"
        assert profile.auth_type == "bearer"
        assert profile.supports_spot is True
        assert profile.egress_cost == 0.0

    def test_default_profile_spot_notice(self):
        """Default profile has no spot notice"""
        profile = _default_profile("test_provider")
        assert profile.spot_interruption_notice_minutes == 0

    def test_default_profile_ssh_defaults(self):
        """Default profile has SSH defaults"""
        profile = _default_profile("test_provider")
        assert profile.ssh_port_fixed is True
        assert profile.ssh_default_port == 22


class TestListAllProfiles:
    """Test list_all_profiles function"""

    def test_list_all_profiles_returns_dict(self):
        """List all profiles returns dictionary"""
        profiles = list_all_profiles()
        assert isinstance(profiles, dict)

    def test_list_all_profiles_contains_major_providers(self):
        """List contains major cloud providers"""
        profiles = list_all_profiles()
        assert "aws" in profiles
        assert "gcp" in profiles
        assert "azure" in profiles
        assert "runpod" in profiles

    def test_list_all_profiles_count(self):
        """List contains expected number of providers"""
        profiles = list_all_profiles()
        assert len(profiles) == 17

    def test_list_all_profiles_returns_copy(self):
        """List returns a copy, not the original"""
        profiles1 = list_all_profiles()
        profiles2 = list_all_profiles()
        assert profiles1 is not profiles2


class TestRegisterProfile:
    """Test register_profile function"""

    def test_register_new_profile(self):
        """Register a new provider profile"""
        new_profile = ProviderProfile(
            name="test_provider",
            api_style="rest",
            auth_type="bearer",
            egress_cost=0.05,
            supports_spot=True,
        )
        
        register_profile(new_profile)
        retrieved = get_profile("test_provider")
        assert retrieved.name == "test_provider"
        assert retrieved.egress_cost == 0.05
        
        # Cleanup
        unregister_profile("test_provider")

    def test_register_profile_override(self):
        """Register profile with override=True replaces existing"""
        profile1 = ProviderProfile(
            name="test_provider",
            api_style="rest",
            auth_type="bearer",
            egress_cost=0.05,
        )
        
        register_profile(profile1)
        
        profile2 = ProviderProfile(
            name="test_provider",
            api_style="graphql",
            auth_type="bearer",
            egress_cost=0.10,
        )
        
        register_profile(profile2, override=True)
        retrieved = get_profile("test_provider")
        assert retrieved.api_style == "graphql"
        assert retrieved.egress_cost == 0.10
        
        # Cleanup
        unregister_profile("test_provider")

    def test_register_profile_duplicate_error(self):
        """Register duplicate profile without override raises error"""
        profile = ProviderProfile(
            name="test_provider",
            api_style="rest",
            auth_type="bearer",
        )
        
        register_profile(profile)
        
        with pytest.raises(ValueError) as exc_info:
            register_profile(profile, override=False)
        
        assert "already exists" in str(exc_info.value)
        
        # Cleanup
        unregister_profile("test_provider")


class TestRegisterProfilesFromDict:
    """Test register_profiles_from_dict function"""

    def test_register_from_dict(self):
        """Register profiles from dictionary"""
        profiles_dict = {
            "provider1": {
                "api_style": "rest",
                "auth_type": "bearer",
                "egress_cost": 0.05,
            },
            "provider2": {
                "api_style": "graphql",
                "auth_type": "bearer",
                "egress_cost": 0.10,
            },
        }
        
        register_profiles_from_dict(profiles_dict)
        
        profile1 = get_profile("provider1")
        profile2 = get_profile("provider2")
        
        assert profile1.egress_cost == 0.05
        assert profile2.api_style == "graphql"
        
        # Cleanup
        unregister_profile("provider1")
        unregister_profile("provider2")

    def test_register_from_dict_with_profile_objects(self):
        """Register profiles from dict with Profile objects"""
        profile1 = ProviderProfile(
            name="provider1",
            api_style="rest",
            auth_type="bearer",
        )
        
        profiles_dict = {
            "provider1": profile1,
        }
        
        register_profiles_from_dict(profiles_dict)
        
        retrieved = get_profile("provider1")
        assert retrieved.api_style == "rest"
        
        # Cleanup
        unregister_profile("provider1")

    def test_register_from_dict_override(self):
        """Register from dict with override=True"""
        profiles_dict = {
            "provider1": {
                "api_style": "rest",
                "auth_type": "bearer",
                "egress_cost": 0.05,
            },
        }
        
        register_profiles_from_dict(profiles_dict)
        
        profiles_dict["provider1"]["egress_cost"] = 0.15
        register_profiles_from_dict(profiles_dict, override=True)
        
        retrieved = get_profile("provider1")
        assert retrieved.egress_cost == 0.15
        
        # Cleanup
        unregister_profile("provider1")


class TestUnregisterProfile:
    """Test unregister_profile function"""

    def test_unregister_existing_profile(self):
        """Unregister existing profile returns True"""
        profile = ProviderProfile(
            name="test_provider",
            api_style="rest",
            auth_type="bearer",
        )
        
        register_profile(profile)
        result = unregister_profile("test_provider")
        
        assert result is True
        assert "test_provider" not in PROVIDER_PROFILES

    def test_unregister_nonexistent_profile(self):
        """Unregister nonexistent profile returns False"""
        result = unregister_profile("nonexistent_provider")
        assert result is False

    def test_unregister_case_insensitive(self):
        """Unregister is case-insensitive"""
        profile = ProviderProfile(
            name="test_provider",
            api_style="rest",
            auth_type="bearer",
        )
        
        register_profile(profile)
        result = unregister_profile("TEST_PROVIDER")
        
        assert result is True


class TestLoadProfilesFromFile:
    """Test load_profiles_from_file function"""

    def test_load_from_json_file(self):
        """Load profiles from JSON file"""
        import json
        from terradev_cli.providers.provider_profiles import load_profiles_from_file
        
        profiles_data = {
            "profiles": {
                "test_provider": {
                    "api_style": "rest",
                    "auth_type": "bearer",
                    "egress_cost": 0.05,
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(profiles_data, f)
            temp_path = f.name
        
        try:
            load_profiles_from_file(temp_path)
            profile = get_profile("test_provider")
            assert profile.egress_cost == 0.05
            unregister_profile("test_provider")
        finally:
            import os
            os.unlink(temp_path)

    def test_load_from_file_not_found(self):
        """Load from non-existent file raises FileNotFoundError"""
        from terradev_cli.providers.provider_profiles import load_profiles_from_file
        
        with pytest.raises(FileNotFoundError):
            load_profiles_from_file("/nonexistent/file.json")

    def test_load_from_unsupported_format(self):
        """Load from unsupported file format raises ValueError"""
        from terradev_cli.providers.provider_profiles import load_profiles_from_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_profiles_from_file(temp_path)
            assert "Unsupported file format" in str(exc_info.value)
        finally:
            import os
            os.unlink(temp_path)


class TestProviderProfilesData:
    """Test provider profiles data integrity"""

    def test_aws_profile_attributes(self):
        """AWS profile has expected attributes"""
        profile = PROVIDER_PROFILES["aws"]
        assert profile.requires_instance_type_mapping is True
        assert profile.spot_interruption_notice_minutes == 2
        assert profile.spot_preemption_webhook is True
        assert profile.requires_boto3 is True

    def test_gcp_profile_attributes(self):
        """GCP profile has expected attributes"""
        profile = PROVIDER_PROFILES["gcp"]
        assert profile.requires_zone_probing is True
        assert profile.requires_gcp_sdk is True
        assert profile.egress_cost == 0.12

    def test_vastai_profile_attributes(self):
        """Vast.ai profile has expected attributes"""
        profile = PROVIDER_PROFILES["vastai"]
        assert profile.ssh_port_fixed is False
        assert profile.egress_cost == 0.01

    def test_all_profiles_have_required_fields(self):
        """All profiles have required fields"""
        required_fields = ["name", "api_style", "auth_type", "egress_cost", "supports_spot"]
        
        for provider_name, profile in PROVIDER_PROFILES.items():
            for field in required_fields:
                assert hasattr(profile, field), f"{provider_name} missing {field}"

    def test_profile_names_match_keys(self):
        """Profile names match dictionary keys"""
        for provider_name, profile in PROVIDER_PROFILES.items():
            assert profile.name == provider_name.lower()
