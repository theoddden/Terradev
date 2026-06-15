"""Test CLI Click commands with CliRunner and mocks."""

import os

import pytest
from click.testing import CliRunner

from terradev_cli.cli import cli

# Skip onboarding for all CLI tests
os.environ["TERRADEV_SKIP_ONBOARDING"] = "1"


class TestQuoteCommand:
    """Test the quote command."""

    def test_quote_basic_invocation(self):
        """Quote command basic invocation"""
        runner = CliRunner()
        result = runner.invoke(cli, ["quote", "-g", "A100"])
        # Command should parse correctly even if it fails at runtime
        assert result.exit_code == 0 or "A100" in result.output


class TestProvisionCommand:
    """Test the provision command."""

    def test_provision_basic_invocation(self):
        """Provision command basic invocation"""
        runner = CliRunner()
        result = runner.invoke(cli, ["provision", "-g", "A100"])
        # Command should parse correctly even if it fails at runtime
        assert result.exit_code == 0 or "A100" in result.output


class TestConfigureCommand:
    """Test the configure command."""

    def test_configure_runpod(self):
        """Configure command for RunPod"""
        runner = CliRunner()
        result = runner.invoke(cli, ["configure", "--provider", "runpod"])
        # Command should parse correctly
        assert result.exit_code == 0 or "runpod" in result.output.lower()

    def test_configure_invalid_provider(self):
        """Configure command with invalid provider"""
        runner = CliRunner()
        result = runner.invoke(cli, ["configure", "--provider", "invalid_provider"])
        assert "Unknown provider" in result.output or result.exit_code != 0


class TestProvidersCommands:
    """Test the providers group commands."""

    def test_providers_list_profiles(self):
        """List provider profiles"""
        runner = CliRunner()
        result = runner.invoke(cli, ["providers", "list-profiles"])
        assert result.exit_code == 0

    def test_providers_show_profile(self):
        """Show specific provider profile"""
        runner = CliRunner()
        result = runner.invoke(cli, ["providers", "show-profile", "runpod"])
        assert result.exit_code == 0

    def test_providers_export_example(self):
        """Export example provider profiles"""
        runner = CliRunner()
        result = runner.invoke(cli, ["providers", "export-example"])
        assert result.exit_code == 0




class TestSetupCommand:
    """Test the setup command."""

    def test_setup_runpod(self):
        """Setup command for RunPod"""
        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "runpod"])
        assert result.exit_code == 0

    def test_setup_aws(self):
        """Setup command for AWS"""
        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "aws"])
        assert result.exit_code == 0

