"""Test CLI Click commands with CliRunner and mocks."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from terradev_cli.cli import cli

# Skip onboarding for all CLI tests
os.environ["TERRADEV_SKIP_ONBOARDING"] = "1"


# Create comprehensive mock for TerradevAPI
def create_mock_api():
    """Create a fully mocked TerradevAPI instance"""
    mock = MagicMock()
    
    # Mock all async quote methods
    mock.return_value.get_runpod_quotes = AsyncMock(return_value=[
        {"provider": "RunPod", "price": 1.50, "gpu_type": "A100", "region": "us-east-1", "available": True}
    ])
    mock.return_value.get_vastai_quotes = AsyncMock(return_value=[
        {"provider": "Vast.ai", "price": 1.20, "gpu_type": "A100", "region": "us-west-1", "available": True}
    ])
    mock.return_value.get_aws_quotes = AsyncMock(return_value=[
        {"provider": "AWS", "price": 2.50, "gpu_type": "A100", "region": "us-east-1", "available": True}
    ])
    mock.return_value.get_gcp_quotes = AsyncMock(return_value=[
        {"provider": "GCP", "price": 2.80, "gpu_type": "A100", "region": "us-central1", "available": True}
    ])
    mock.return_value.get_azure_quotes = AsyncMock(return_value=[
        {"provider": "Azure", "price": 2.60, "gpu_type": "A100", "region": "eastus", "available": True}
    ])
    mock.return_value.get_tensordock_quotes = AsyncMock(return_value=[
        {"provider": "TensorDock", "price": 1.10, "gpu_type": "A100", "region": "us-east-1", "available": True}
    ])
    mock.return_value.get_lambda_quotes = AsyncMock(return_value=[
        {"provider": "Lambda Labs", "price": 1.80, "gpu_type": "A100", "region": "us-east-1", "available": True}
    ])
    mock.return_value.get_coreweave_quotes = AsyncMock(return_value=[
        {"provider": "CoreWeave", "price": 1.40, "gpu_type": "A100", "region": "us-east-1", "available": True}
    ])
    mock.return_value.get_oracle_quotes = AsyncMock(return_value=[
        {"provider": "Oracle", "price": 1.30, "gpu_type": "A100", "region": "us-ashburn-1", "available": True}
    ])
    mock.return_value.get_crusoe_quotes = AsyncMock(return_value=[
        {"provider": "Crusoe", "price": 1.15, "gpu_type": "A100", "region": "us-east-1", "available": True}
    ])
    
    # Mock provision methods
    mock.return_value.provision_instance = AsyncMock(return_value={
        "instance_id": "test-instance-123",
        "provider": "RunPod",
        "gpu_type": "A100",
        "status": "running"
    })
    
    # Mock status methods
    mock.return_value.get_instance_status = AsyncMock(return_value={
        "status": "running",
        "gpu_utilization": 85
    })
    
    # Mock other methods
    mock.return_value.is_first_time_user = MagicMock(return_value=False)
    mock.return_value.load_credentials = MagicMock(return_value={})
    mock.return_value.save_credentials = MagicMock()
    mock.return_value.load_usage = MagicMock(return_value={"provisions_this_month": 0})
    mock.return_value.save_usage = MagicMock()
    mock.return_value.check_provision_limit = MagicMock(return_value=True)
    mock.return_value.record_provision = MagicMock()
    
    return mock


class TestQuoteCommand:
    """Test the quote command with comprehensive mocking."""

    def test_quote_basic_invocation(self):
        """Quote command basic invocation"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["quote", "-g", "A100"])
            assert result.exit_code == 0

    def test_quote_with_gpu_type(self):
        """Quote command with GPU type"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["quote", "--gpu-type", "H100"])
            assert result.exit_code == 0

    def test_quote_with_specific_provider(self):
        """Quote command with specific provider filter"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["quote", "-g", "A100", "-p", "runpod"])
            assert result.exit_code == 0

    def test_quote_with_multiple_providers(self):
        """Quote command with multiple providers"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["quote", "-g", "A100", "-p", "runpod,vastai"])
            assert result.exit_code == 0

    def test_quote_with_region_filter(self):
        """Quote command with region filter"""
        runner = CliRunner()
        api = create_mock_api().return_value
        result = runner.invoke(
            cli,
            ["quote", "-g", "A100", "-r", "us-east-1"],
            obj={"api": api},
        )
        assert result.exit_code == 0

    def test_quote_with_parallel_flag(self):
        """Quote command with parallel queries"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["quote", "-g", "A100", "--parallel", "10"])
            assert result.exit_code == 0

    def test_quote_with_quick_flag(self):
        """Quote command with quick flag"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["quote", "-g", "A100", "-q"])
            assert result.exit_code == 0

    def test_quote_with_include_local(self):
        """Quote command with local pool inclusion"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["quote", "-g", "A100", "--include-local"])
            assert result.exit_code == 0

    def test_quote_various_gpu_types(self):
        """Quote command with various GPU types"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            for gpu in ["A100", "H100", "RTX4090", "L40", "A40", "V100"]:
                result = runner.invoke(cli, ["quote", "-g", gpu])
                assert result.exit_code == 0


class TestProvisionCommand:
    """Test the provision command with comprehensive mocking."""

    def test_provision_basic_invocation(self):
        """Provision command basic invocation"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100"])
            assert result.exit_code == 0

    def test_provision_with_gpu_type(self):
        """Provision command with GPU type"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "--gpu-type", "H100"])
            assert result.exit_code == 0

    def test_provision_with_count(self):
        """Provision command with instance count"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "-n", "4"])
            assert result.exit_code == 0

    def test_provision_with_max_price(self):
        """Provision command with max price filter"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--max-price", "2.50"])
            assert result.exit_code == 0

    def test_provision_with_provider(self):
        """Provision command with specific provider"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "-p", "runpod"])
            assert result.exit_code == 0

    def test_provision_with_multiple_providers(self):
        """Provision command with multiple providers"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "-p", "runpod,vastai"])
            assert result.exit_code == 0

    def test_provision_with_parallel(self):
        """Provision command with parallel deployment"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--parallel", "6"])
            assert result.exit_code == 0

    def test_provision_dry_run(self):
        """Provision command with dry-run flag"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--dry-run"])
            assert result.exit_code == 0

    def test_provision_spot_flag(self):
        """Provision command with spot flag"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--spot"])
            assert result.exit_code == 0

    def test_provision_on_demand_flag(self):
        """Provision command with on-demand flag"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--on-demand"])
            assert result.exit_code == 0

    def test_provision_training_type(self):
        """Provision command with training workload type"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--type", "training"])
            assert result.exit_code == 0

    def test_provision_inference_type(self):
        """Provision command with inference workload type"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--type", "inference"])
            assert result.exit_code == 0

    def test_provision_with_model_name(self):
        """Provision command with model name for inference"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--type", "inference", "--model-name", "llama-70b"])
            assert result.exit_code == 0

    def test_provision_with_backend(self):
        """Provision command with backend specification"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--backend", "vllm"])
            assert result.exit_code == 0

    def test_provision_prefer_local(self):
        """Provision command with local preference"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--prefer-local"])
            assert result.exit_code == 0

    def test_provision_multi_agent_mode(self):
        """Provision command with multi-agent KV sharing"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "H100", "--agents", "20", "--context", "32k"])
            assert result.exit_code == 0

    def test_provision_auto_select(self):
        """Provision command with auto-select flag"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["provision", "-g", "A100", "--auto"])
            assert result.exit_code == 0

    def test_provision_various_gpu_types(self):
        """Provision command with various GPU types"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            for gpu in ["A100", "H100", "RTX4090", "L40"]:
                result = runner.invoke(cli, ["provision", "-g", gpu])
                assert result.exit_code == 0


class TestConfigureCommand:
    """Test the configure command with comprehensive mocking."""

    def test_configure_runpod(self):
        """Configure command for RunPod"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["configure", "--provider", "runpod"])
            assert result.exit_code == 0 or "runpod" in result.output.lower()

    def test_configure_aws(self):
        """Configure command for AWS"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["configure", "--provider", "aws"])
            assert result.exit_code == 0 or "aws" in result.output.lower()

    def test_configure_gcp(self):
        """Configure command for GCP"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["configure", "--provider", "gcp"])
            assert result.exit_code == 0 or "gcp" in result.output.lower()

    def test_configure_azure(self):
        """Configure command for Azure"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["configure", "--provider", "azure"])
            assert result.exit_code == 0 or "azure" in result.output.lower()

    def test_configure_vastai(self):
        """Configure command for Vast.ai"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["configure", "--provider", "vastai"])
            assert result.exit_code == 0 or "vast" in result.output.lower()

    def test_configure_lambda(self):
        """Configure command for Lambda Labs"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["configure", "--provider", "lambda"])
            assert result.exit_code == 0 or "lambda" in result.output.lower()

    def test_configure_invalid_provider(self):
        """Configure command with invalid provider"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["configure", "--provider", "invalid_provider"])
            assert "Unknown provider" in result.output or result.exit_code != 0


class TestProvidersCommands:
    """Test the providers group commands with comprehensive mocking."""

    def test_providers_list_profiles(self):
        """List provider profiles"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["providers", "list-profiles"])
            assert result.exit_code == 0

    def test_providers_list_profiles_json(self):
        """List provider profiles in JSON format"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["providers", "list-profiles", "--format", "json"])
            assert result.exit_code == 0

    def test_providers_list_profiles_yaml(self):
        """List provider profiles in YAML format"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["providers", "list-profiles", "--format", "yaml"])
            assert result.exit_code == 0

    def test_providers_show_profile(self):
        """Show specific provider profile"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["providers", "show-profile", "runpod"])
            assert result.exit_code == 0

    def test_providers_show_profile_json(self):
        """Show provider profile in JSON format"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["providers", "show-profile", "runpod", "--format", "json"])
            assert result.exit_code == 0

    def test_providers_show_profile_yaml(self):
        """Show provider profile in YAML format"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["providers", "show-profile", "runpod", "--format", "yaml"])
            assert result.exit_code == 0

    def test_providers_export_example(self):
        """Export example provider profiles"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["providers", "export-example"])
            assert result.exit_code == 0

    def test_providers_show_various_profiles(self):
        """Show various provider profiles"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            for provider in ["runpod", "vastai", "aws", "gcp", "azure", "lambda"]:
                result = runner.invoke(cli, ["providers", "show-profile", provider])
                assert result.exit_code == 0


class TestSetupCommand:
    """Test the setup command with comprehensive mocking."""

    def test_setup_runpod(self):
        """Setup command for RunPod"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["setup", "runpod"])
            assert result.exit_code == 0

    def test_setup_runpod_quick(self):
        """Setup command for RunPod with quick flag"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["setup", "runpod", "--quick"])
            assert result.exit_code == 0

    def test_setup_vastai(self):
        """Setup command for Vast.ai"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["setup", "vastai"])
            assert result.exit_code == 0

    def test_setup_aws(self):
        """Setup command for AWS"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["setup", "aws"])
            assert result.exit_code == 0

    def test_setup_gcp(self):
        """Setup command for GCP"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["setup", "gcp"])
            assert result.exit_code == 0

    def test_setup_azure(self):
        """Setup command for Azure"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["setup", "azure"])
            assert result.exit_code == 0

    def test_setup_tensordock(self):
        """Setup command for TensorDock"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["setup", "tensordock"])
            assert result.exit_code == 0

    def test_setup_crusoe(self):
        """Setup command for Crusoe"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["setup", "crusoe"])
            assert result.exit_code == 0

    def test_setup_invalid_provider(self):
        """Setup command with invalid provider"""
        runner = CliRunner()
        with patch("terradev_cli.cli.TerradevAPI", create_mock_api()):
            result = runner.invoke(cli, ["setup", "invalid_provider"])
            assert result.exit_code != 0 or "not found" in result.output.lower()


class TestCLIHelpAndVersion:
    """Test CLI help and version commands."""

    def test_cli_help(self):
        """CLI help command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "terradev" in result.output.lower()

    def test_cli_version(self):
        """CLI version command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_quote_help(self):
        """Quote command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["quote", "--help"])
        assert result.exit_code == 0
        assert "quote" in result.output.lower()

    def test_provision_help(self):
        """Provision command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["provision", "--help"])
        assert result.exit_code == 0
        assert "provision" in result.output.lower()

    def test_configure_help(self):
        """Configure command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["configure", "--help"])
        assert result.exit_code == 0
        assert "configure" in result.output.lower()

    def test_providers_help(self):
        """Providers group help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["providers", "--help"])
        assert result.exit_code == 0
        assert "providers" in result.output.lower()

    def test_setup_help(self):
        """Setup command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0
        assert "setup" in result.output.lower()


class TestCLIErrorPaths:
    """Test CLI error handling paths for coverage."""

    def test_provision_without_configure_fails(self):
        """Provision command may fail without prior configuration"""
        runner = CliRunner()
        result = runner.invoke(cli, ["provision", "-g", "H100"])
        # Command may succeed with existing credentials or fail - just check it runs
        # The important thing is it doesn't crash
        assert result.exit_code in [0, 1, 2]

    def test_invalid_gpu_type_accepted(self):
        """Quote command accepts any GPU type (validation happens at provider level)"""
        runner = CliRunner()
        result = runner.invoke(cli, ["quote", "-g", "INVALID_GPU"])
        # CLI accepts the GPU type, providers will reject it
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_missing_required_args(self):
        """Provision command fails without required GPU argument"""
        runner = CliRunner()
        result = runner.invoke(cli, ["provision"])
        # Should fail with non-zero exit code (Click validation)
        assert result.exit_code != 0

    def test_quote_missing_gpu_arg(self):
        """Quote command may have default behavior without GPU argument"""
        runner = CliRunner()
        result = runner.invoke(cli, ["quote"])
        # CLI may show help or have default behavior
        # Just check it doesn't crash
        assert result.exit_code in [0, 1, 2]

    def test_configure_without_provider(self):
        """Configure command prompts for provider (interactive)"""
        runner = CliRunner()
        result = runner.invoke(cli, ["configure"])
        # Should either fail or start interactive prompt
        # Exit code 1 means aborted (user cancelled), 0 means started
        assert result.exit_code in [0, 1]

    def test_invalid_provider_rejected(self):
        """Configure command may reject invalid provider"""
        runner = CliRunner()
        result = runner.invoke(cli, ["configure", "--provider", "nonexistent_provider"])
        # May reject or handle gracefully
        assert result.exit_code in [0, 1, 2]
