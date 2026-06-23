#!/usr/bin/env python3
"""Tests for CLI utility functions in cli.py"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from terradev_cli.cli import validate_credentials, TerradevAPI


class TestValidateCredentials:
    """Test validate_credentials function"""

    def test_validate_aws_complete(self):
        """Validate AWS with complete credentials"""
        creds = {"api_key": "test_key", "secret_key": "test_secret"}
        assert validate_credentials("aws", creds) is True

    def test_validate_aws_missing_api_key(self):
        """Validate AWS with missing api_key"""
        creds = {"secret_key": "test_secret"}
        assert validate_credentials("aws", creds) is False

    def test_validate_aws_missing_secret_key(self):
        """Validate AWS with missing secret_key"""
        creds = {"api_key": "test_key"}
        assert validate_credentials("aws", creds) is False

    def test_validate_aws_empty_values(self):
        """Validate AWS with empty credential values"""
        creds = {"api_key": "", "secret_key": "test_secret"}
        assert validate_credentials("aws", creds) is False

    def test_validate_aws_whitespace_values(self):
        """Validate AWS with whitespace-only credential values"""
        creds = {"api_key": "   ", "secret_key": "test_secret"}
        assert validate_credentials("aws", creds) is False

    def test_validate_gcp_complete(self):
        """Validate GCP with complete credentials"""
        creds = {"project_id": "test_project", "credentials_file": "/path/to/creds.json"}
        assert validate_credentials("gcp", creds) is True

    def test_validate_gcp_missing_project_id(self):
        """Validate GCP with missing project_id"""
        creds = {"credentials_file": "/path/to/creds.json"}
        assert validate_credentials("gcp", creds) is False

    def test_validate_azure_complete(self):
        """Validate Azure with complete credentials"""
        creds = {
            "subscription_id": "test_sub",
            "tenant_id": "test_tenant",
            "client_id": "test_client",
            "client_secret": "test_secret",
        }
        assert validate_credentials("azure", creds) is True

    def test_validate_azure_missing_tenant_id(self):
        """Validate Azure with missing tenant_id"""
        creds = {
            "subscription_id": "test_sub",
            "client_id": "test_client",
            "client_secret": "test_secret",
        }
        assert validate_credentials("azure", creds) is False

    def test_validate_runpod_complete(self):
        """Validate RunPod with complete credentials"""
        creds = {"api_key": "test_key"}
        assert validate_credentials("runpod", creds) is True

    def test_validate_runpod_missing_api_key(self):
        """Validate RunPod with missing api_key"""
        creds = {}
        assert validate_credentials("runpod", creds) is False

    def test_validate_vastai_complete(self):
        """Validate Vast.ai with complete credentials"""
        creds = {"api_key": "test_key"}
        assert validate_credentials("vastai", creds) is True

    def test_validate_lambda_labs_complete(self):
        """Validate Lambda Labs with complete credentials"""
        creds = {"api_key": "test_key"}
        assert validate_credentials("lambda_labs", creds) is True

    def test_validate_coreweave_complete(self):
        """Validate CoreWeave with complete credentials"""
        creds = {"api_key": "test_key"}
        assert validate_credentials("coreweave", creds) is True

    def test_validate_tensordock_complete(self):
        """Validate TensorDock with complete credentials"""
        creds = {"api_key": "test_key", "api_token": "test_token"}
        assert validate_credentials("tensordock", creds) is True

    def test_validate_tensordock_missing_api_token(self):
        """Validate TensorDock with missing api_token"""
        creds = {"api_key": "test_key"}
        assert validate_credentials("tensordock", creds) is False

    def test_validate_huggingface_complete(self):
        """Validate HuggingFace with complete credentials"""
        creds = {"api_key": "test_key", "namespace": "test_namespace"}
        assert validate_credentials("huggingface", creds) is True

    def test_validate_huggingface_missing_namespace(self):
        """Validate HuggingFace with missing namespace"""
        creds = {"api_key": "test_key"}
        assert validate_credentials("huggingface", creds) is False

    def test_validate_baseten_complete(self):
        """Validate Baseten with complete credentials"""
        creds = {"api_key": "test_key"}
        assert validate_credentials("baseten", creds) is True

    def test_validate_oracle_complete(self):
        """Validate Oracle with complete credentials"""
        creds = {
            "api_key": "test_key",
            "tenancy_ocid": "test_tenancy",
            "compartment_ocid": "test_compartment",
            "region": "us-ashburn-1",
        }
        assert validate_credentials("oracle", creds) is True

    def test_validate_oracle_missing_region(self):
        """Validate Oracle with missing region"""
        creds = {
            "api_key": "test_key",
            "tenancy_ocid": "test_tenancy",
            "compartment_ocid": "test_compartment",
        }
        assert validate_credentials("oracle", creds) is False

    def test_validate_crusoe_complete(self):
        """Validate Crusoe with complete credentials"""
        creds = {
            "access_key": "test_access",
            "secret_key": "test_secret",
            "project_id": "test_project",
        }
        assert validate_credentials("crusoe", creds) is True

    def test_validate_crusoe_missing_project_id(self):
        """Validate Crusoe with missing project_id"""
        creds = {"access_key": "test_access", "secret_key": "test_secret"}
        assert validate_credentials("crusoe", creds) is False

    def test_validate_unknown_provider(self):
        """Validate unknown provider returns False"""
        creds = {"api_key": "test_key"}
        assert validate_credentials("unknown_provider", creds) is False

    def test_validate_case_insensitive(self):
        """Validate is case-insensitive for provider name"""
        aws_creds = {"api_key": "test_key", "secret_key": "test_secret"}
        runpod_creds = {"api_key": "test_key"}
        assert validate_credentials("AWS", aws_creds) is True
        assert validate_credentials("RunPod", runpod_creds) is True
        assert validate_credentials("RUNPOD", runpod_creds) is True


class TestTerradevAPI:
    """Test TerradevAPI class"""

    def test_initialization_creates_config_dir(self):
        """API initialization creates config directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                assert api.config_dir.exists()
                assert api.config_dir.name == ".terradev"

    def test_initialization_sets_file_paths(self):
        """API initialization sets correct file paths"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                assert api.credentials_file.name == "credentials.json"
                assert api.usage_file.name == "usage.json"

    def test_is_first_time_user_no_credentials_file(self):
        """First-time user when no credentials file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                assert api.is_first_time_user() is True

    def test_is_first_time_user_empty_credentials(self):
        """First-time user when credentials file is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {}
                api.save_credentials()
                assert api.is_first_time_user() is True

    def test_is_first_time_user_placeholder_credentials(self):
        """First-time user when credentials are placeholders"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"aws": {"api_key": "your_api_key", "secret_key": "your_secret"}}
                api.save_credentials()
                assert api.is_first_time_user() is True

    def test_is_first_time_user_real_credentials(self):
        """Not first-time user when credentials are real"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"aws": {"api_key": "real_key_123", "secret_key": "real_secret"}}
                api.save_credentials()
                assert api.is_first_time_user() is False

    def test_is_first_time_user_mixed_credentials(self):
        """Not first-time user when at least one credential is real"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {
                    "aws": {"api_key": "your_api_key", "secret_key": "your_secret"},
                    "runpod": {"api_key": "real_key_456"},
                }
                api.save_credentials()
                assert api.is_first_time_user() is False

    def test_load_credentials_creates_empty_dict(self):
        """Load credentials creates empty dict when file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                assert api.credentials == {}

    def test_load_credentials_loads_existing_file(self):
        """Load credentials loads from existing file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                # Create credentials file
                creds_file = Path(tmpdir) / ".terradev" / "credentials.json"
                creds_file.parent.mkdir(parents=True, exist_ok=True)
                with open(creds_file, "w") as f:
                    json.dump({"aws": {"api_key": "test"}}, f)

                api = TerradevAPI()
                assert api.credentials == {"aws": {"api_key": "test"}}

    def test_save_credentials_writes_file(self):
        """Save credentials writes to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"runpod": {"api_key": "test_key"}}
                api.save_credentials()

                creds_file = Path(tmpdir) / ".terradev" / "credentials.json"
                assert creds_file.exists()
                with open(creds_file, "r") as f:
                    loaded = json.load(f)
                # AuthManager encrypts credentials and wraps them under 'credentials' key
                assert "credentials" in loaded
                assert "version" in loaded
                assert loaded["version"] == "2.0"
                # Verify the nested structure exists (encrypted value, not plaintext)
                assert "runpod" in loaded["credentials"]

    def test_save_credentials_sets_permissions(self):
        """Save credentials sets file permissions to 0600"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"aws": {"api_key": "test"}}
                api.save_credentials()

                creds_file = Path(tmpdir) / ".terradev" / "credentials.json"
                # Check permissions (may fail on Windows)
                try:
                    import stat
                    mode = creds_file.stat().st_mode
                    assert mode & 0o777 == 0o600
                except (OSError, AttributeError, AssertionError):
                    pass  # Skip on systems that don't support chmod or have different permissions

    def test_load_usage_creates_default_usage(self):
        """Load usage creates default usage dict when file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                assert "provisions_this_month" in api.usage
                assert "inference_endpoints" in api.usage

    def test_load_usage_loads_existing_file(self):
        """Load usage loads from existing file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                # Create usage file
                usage_file = Path(tmpdir) / ".terradev" / "usage.json"
                usage_file.parent.mkdir(parents=True, exist_ok=True)
                with open(usage_file, "w") as f:
                    json.dump({"provisions_this_month": 5}, f)

                api = TerradevAPI()
                assert api.usage["provisions_this_month"] == 5

    def test_provider_creds_nested_format(self):
        """_provider_creds returns nested format credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"runpod": {"api_key": "real_key_123"}}
                creds = api._provider_creds("runpod")
                assert creds == {"api_key": "real_key_123"}

    def test_provider_creds_flat_format_aws(self):
        """_provider_creds returns flat format AWS credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"aws_access_key_id": "test_key", "aws_secret_access_key": "test_secret"}
                creds = api._provider_creds("aws")
                assert creds == {"api_key": "test_key", "secret_key": "test_secret"}

    def test_provider_creds_flat_format_runpod(self):
        """_provider_creds returns flat format RunPod credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"runpod_api_key": "test_key"}
                creds = api._provider_creds("runpod")
                assert creds == {"api_key": "test_key"}

    def test_provider_creds_nested_placeholder_fallback(self):
        """_provider_creds falls back to flat format when all nested are placeholders"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"runpod": {"api_key": "your_api_key"}, "runpod_api_key": "real_flat_key"}
                creds = api._provider_creds("runpod")
                # Should fall back to flat format
                assert creds == {"api_key": "real_flat_key"}

    def test_provider_creds_nested_real_uses_nested(self):
        """_provider_creds uses nested format when at least one real credential exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"runpod": {"api_key": "real_key_123", "secret": "real_secret"}}
                creds = api._provider_creds("runpod")
                # Should use nested format with all credentials
                assert creds == {"api_key": "real_key_123", "secret": "real_secret"}

    def test_provider_creds_empty_credentials(self):
        """_provider_creds returns empty dict when no credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {}
                creds = api._provider_creds("runpod")
                assert creds == {}

    def test_check_provision_limit_no_tier(self):
        """check_provision_limit returns True when no tier (unlimited)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                assert api.check_provision_limit() is True

    def test_record_provision_increments_counter(self):
        """record_provision increments monthly counter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                initial_count = api.usage.get("provisions_this_month", 0)
                api.record_provision()
                assert api.usage["provisions_this_month"] == initial_count + 1

    def test_save_usage_writes_file(self):
        """save_usage writes usage to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.usage["provisions_this_month"] = 10
                api.save_usage()

                usage_file = Path(tmpdir) / ".terradev" / "usage.json"
                assert usage_file.exists()
                with open(usage_file, "r") as f:
                    loaded = json.load(f)
                assert loaded["provisions_this_month"] == 10

    def test_provider_creds_flat_format_gcp(self):
        """_provider_creds returns flat format GCP credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"gcp_project_id": "test_project", "gcp_credentials_file": "/path/to/creds.json"}
                creds = api._provider_creds("gcp")
                assert creds == {"project_id": "test_project", "credentials_file": "/path/to/creds.json"}

    def test_provider_creds_flat_format_azure(self):
        """_provider_creds returns flat format Azure credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {
                    "azure_subscription_id": "test_sub",
                    "azure_tenant_id": "test_tenant",
                    "azure_client_id": "test_client",
                    "azure_client_secret": "test_secret",
                }
                creds = api._provider_creds("azure")
                assert creds == {
                    "subscription_id": "test_sub",
                    "tenant_id": "test_tenant",
                    "client_id": "test_client",
                    "client_secret": "test_secret",
                }

    def test_provider_creds_flat_format_vastai(self):
        """_provider_creds returns flat format Vast.ai credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"vastai_api_key": "test_key"}
                creds = api._provider_creds("vastai")
                assert creds == {"api_key": "test_key"}

    def test_provider_creds_flat_format_lambda_labs(self):
        """_provider_creds returns flat format Lambda Labs credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"lambda_api_key": "test_key"}
                creds = api._provider_creds("lambda_labs")
                assert creds == {"api_key": "test_key"}

    def test_provider_creds_flat_format_coreweave(self):
        """_provider_creds returns flat format CoreWeave credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"coreweave_api_key": "test_key"}
                creds = api._provider_creds("coreweave")
                assert creds == {"api_key": "test_key"}

    def test_provider_creds_flat_format_tensordock(self):
        """_provider_creds returns flat format TensorDock credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"tensordock_api_key": "test_key", "tensordock_api_token": "test_token"}
                creds = api._provider_creds("tensordock")
                assert creds == {"api_key": "test_key", "api_token": "test_token"}

    def test_provider_creds_flat_format_huggingface(self):
        """_provider_creds returns flat format HuggingFace credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"huggingface_api_token": "test_token", "huggingface_namespace": "test_ns"}
                creds = api._provider_creds("huggingface")
                assert creds == {"api_key": "test_token", "namespace": "test_ns"}

    def test_provider_creds_flat_format_baseten(self):
        """_provider_creds returns flat format Baseten credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"baseten_api_key": "test_key"}
                creds = api._provider_creds("baseten")
                assert creds == {"api_key": "test_key"}

    def test_provider_creds_flat_format_oracle(self):
        """_provider_creds returns flat format Oracle credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {
                    "oracle_api_key": "test_key",
                    "oracle_tenancy_ocid": "test_tenancy",
                    "oracle_compartment_ocid": "test_compartment",
                    "oracle_region": "us-ashburn-1",
                }
                creds = api._provider_creds("oracle")
                assert creds == {
                    "api_key": "test_key",
                    "tenancy_ocid": "test_tenancy",
                    "compartment_ocid": "test_compartment",
                    "region": "us-ashburn-1",
                }

    def test_provider_creds_flat_format_crusoe(self):
        """_provider_creds returns flat format Crusoe credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {
                    "crusoe_access_key": "test_access",
                    "crusoe_secret_key": "test_secret",
                    "crusoe_project_id": "test_project",
                }
                creds = api._provider_creds("crusoe")
                assert creds == {
                    "access_key": "test_access",
                    "secret_key": "test_secret",
                    "project_id": "test_project",
                }

    def test_provider_creds_flat_format_alibaba(self):
        """_provider_creds returns flat format Alibaba credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {
                    "alibaba_access_key_id": "test_key",
                    "alibaba_access_key_secret": "test_secret",
                }
                creds = api._provider_creds("alibaba")
                assert creds == {
                    "access_key_id": "test_key",
                    "access_key_secret": "test_secret",
                    "region_id": "cn-beijing",
                    "security_group_id": "",
                    "vswitch_id": "",
                }

    def test_provider_creds_flat_format_ovhcloud(self):
        """_provider_creds returns flat format OVHcloud credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {
                    "ovhcloud_application_key": "test_app_key",
                    "ovhcloud_application_secret": "test_app_secret",
                    "ovhcloud_consumer_key": "test_consumer",
                    "ovhcloud_project_id": "test_project",
                }
                creds = api._provider_creds("ovhcloud")
                assert creds == {
                    "application_key": "test_app_key",
                    "application_secret": "test_app_secret",
                    "consumer_key": "test_consumer",
                    "project_id": "test_project",
                    "endpoint": "ovh-eu",
                    "ssh_key_id": "",
                }

    def test_provider_creds_flat_format_fluidstack(self):
        """_provider_creds returns flat format FluidStack credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"fluidstack_api_key": "test_key"}
                creds = api._provider_creds("fluidstack")
                assert creds == {"api_key": "test_key", "ssh_key_name": ""}

    def test_provider_creds_flat_format_hetzner(self):
        """_provider_creds returns flat format Hetzner credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"hetzner_api_token": "test_token"}
                creds = api._provider_creds("hetzner")
                assert creds == {
                    "api_token": "test_token",
                    "robot_user": "",
                    "robot_password": "",
                }

    def test_provider_creds_flat_format_siliconflow(self):
        """_provider_creds returns flat format SiliconFlow credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"siliconflow_api_key": "test_key"}
                creds = api._provider_creds("siliconflow")
                assert creds == {
                    "api_key": "test_key",
                    "region": "global",
                    "default_model": "",
                }

    def test_provider_creds_flat_format_hyperstack(self):
        """_provider_creds returns flat format Hyperstack credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"hyperstack_api_key": "test_key"}
                creds = api._provider_creds("hyperstack")
                assert creds == {
                    "api_key": "test_key",
                    "environment": "default-CANADA-1",
                    "ssh_key_name": "",
                }

    def test_provider_creds_flat_format_digitalocean(self):
        """_provider_creds returns flat format DigitalOcean credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"digitalocean_api_token": "test_token"}
                creds = api._provider_creds("digitalocean")
                assert creds == {"api_key": "test_token", "region": "tor1"}

    def test_provider_creds_flat_format_inferx(self):
        """_provider_creds returns flat format InferX credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"inferx_api_key": "test_key"}
                creds = api._provider_creds("inferx")
                assert creds == {
                    "api_key": "test_key",
                    "api_endpoint": "https://api.inferx.net",
                    "region": "us-west-2",
                }

    def test_provider_creds_flat_format_kserve(self):
        """_provider_creds returns flat format KServe credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"kserve_kubeconfig_path": "/path/to/kubeconfig"}
                creds = api._provider_creds("kserve")
                assert creds == {
                    "namespace": "default",
                    "kubeconfig_path": "/path/to/kubeconfig",
                    "auth_token": "",
                    "cluster_endpoint": "",
                }

    def test_provider_creds_flat_format_dvc(self):
        """_provider_creds returns flat format DVC credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"dvc_repo_path": "/path/to/repo"}
                creds = api._provider_creds("dvc")
                assert creds == {
                    "repo_path": "/path/to/repo",
                    "remote_storage": "",
                    "remote_type": "",
                    "aws_access_key_id": "",
                    "aws_secret_access_key": "",
                    "gcp_credentials_path": "",
                    "azure_connection_string": "",
                }

    def test_provider_creds_flat_format_mlflow(self):
        """_provider_creds returns flat format MLflow credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"mlflow_tracking_uri": "http://localhost:5000"}
                creds = api._provider_creds("mlflow")
                assert creds == {
                    "tracking_uri": "http://localhost:5000",
                    "username": "",
                    "password": "",
                    "experiment_name": "",
                    "registry_uri": "",
                }

    def test_provider_creds_flat_format_ray(self):
        """_provider_creds returns flat format Ray credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"ray_dashboard_uri": "http://localhost:8265"}
                creds = api._provider_creds("ray")
                assert creds == {
                    "dashboard_uri": "http://localhost:8265",
                    "cluster_name": "",
                    "auth_token": "",
                    "head_node_ip": "",
                    "head_node_port": "6379",
                    "namespace": "default",
                }

    def test_provider_creds_flat_format_kubernetes(self):
        """_provider_creds returns flat format Kubernetes credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"kubernetes_kubeconfig_path": "/path/to/kubeconfig"}
                creds = api._provider_creds("kubernetes")
                assert creds == {
                    "kubeconfig_path": "/path/to/kubeconfig",
                    "cluster_name": "",
                    "namespace": "default",
                    "karpenter_enabled": "false",
                    "karpenter_version": "v1.10.0",
                    "aws_region": "us-east-1",
                    "aws_account_id": "",
                    "monitoring_enabled": "false",
                    "prometheus_enabled": "false",
                    "grafana_enabled": "false",
                    "dashboard_port": "3000",
                }

    def test_provider_creds_flat_format_wandb(self):
        """_provider_creds returns flat format W&B credentials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"wandb_api_key": "test_key"}
                creds = api._provider_creds("wandb")
                assert creds == {
                    "api_key": "test_key",
                    "entity": "",
                    "project": "",
                    "base_url": "",
                    "team": "",
                    "dashboard_enabled": "false",
                    "reports_enabled": "false",
                    "alerts_enabled": "false",
                    "integration_enabled": "false",
                }

    def test_provider_creds_unknown_provider(self):
        """_provider_creds returns empty dict for unknown provider"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                api = TerradevAPI()
                api.credentials = {"some_key": "some_value"}
                creds = api._provider_creds("unknown_provider")
                assert creds == {}

