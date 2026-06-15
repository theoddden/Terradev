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
                assert loaded == {"runpod": {"api_key": "test_key"}}

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
                except (OSError, AttributeError):
                    pass  # Skip on systems that don't support chmod

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

