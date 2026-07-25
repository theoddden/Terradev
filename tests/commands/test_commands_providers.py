"""Comprehensive tests for providers commands via ctx.obj DI."""

from unittest.mock import AsyncMock, patch

from terradev_cli.commands import cli


# ===========================================================================
# quote
# ===========================================================================


class TestQuoteCommand:
    """Quote command fetches prices from the injected API."""

    def test_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["quote", "-g", "A100-80GB"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_calls_runpod_quotes(self, runner, mock_api):
        runner.invoke(cli, ["quote", "-g", "A100-80GB"], obj={"api": mock_api})
        mock_api.get_runpod_quotes.assert_called_once_with("A100-80GB")

    def test_calls_all_providers_by_default(self, runner, mock_api):
        runner.invoke(cli, ["quote", "-g", "A100-80GB"], obj={"api": mock_api})
        assert mock_api.get_vastai_quotes.called
        assert mock_api.get_tensordock_quotes.called

    def test_provider_filter_runpod_only(self, runner, mock_api):
        runner.invoke(
            cli, ["quote", "-g", "A100-80GB", "--providers", "runpod"], obj={"api": mock_api}
        )
        assert mock_api.get_runpod_quotes.called
        mock_api.get_vastai_quotes.assert_not_called()

    def test_provider_filter_vastai_only(self, runner, mock_api):
        runner.invoke(
            cli, ["quote", "-g", "A100-80GB", "--providers", "vastai"], obj={"api": mock_api}
        )
        assert mock_api.get_vastai_quotes.called
        mock_api.get_runpod_quotes.assert_not_called()

    def test_output_shows_best_price(self, runner, mock_api):
        result = runner.invoke(cli, ["quote", "-g", "A100-80GB"], obj={"api": mock_api})
        assert "Best:" in result.output or "best" in result.output.lower()

    def test_output_shows_table_header(self, runner, mock_api):
        result = runner.invoke(cli, ["quote", "-g", "A100-80GB"], obj={"api": mock_api})
        assert "Provider" in result.output

    def test_region_filter_matching(self, runner, mock_api):
        result = runner.invoke(
            cli, ["quote", "-g", "A100-80GB", "--region", "us-east-1"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_region_filter_no_match_reports_error(self, runner, mock_api):
        """A region with no quotes should report an error, not crash."""
        result = runner.invoke(
            cli, ["quote", "-g", "A100-80GB", "--region", "ap-southeast-99"], obj={"api": mock_api}
        )
        assert result.exit_code == 0
        assert "ERROR" in result.output or "No quotes" in result.output

    def test_no_quotes_from_all_providers(self, runner, mock_api):
        """When all providers return empty, a helpful error should appear."""
        mock_api.get_runpod_quotes = AsyncMock(return_value=[])
        mock_api.get_vastai_quotes = AsyncMock(return_value=[])
        mock_api.get_tensordock_quotes = AsyncMock(return_value=[])
        result = runner.invoke(cli, ["quote", "-g", "A100-80GB"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "ERROR" in result.output or "No quotes" in result.output

    def test_various_gpu_types_accepted(self, runner, mock_api):
        for gpu in ["H100", "RTX4090", "L40S", "V100", "A40"]:
            result = runner.invoke(cli, ["quote", "-g", gpu], obj={"api": mock_api})
            assert result.exit_code == 0, f"Failed for GPU: {gpu}"

    def test_quick_flag(self, runner, mock_api):
        result = runner.invoke(
            cli, ["quote", "-g", "A100-80GB", "--quick"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_default_gpu_type_used_when_omitted(self, runner, mock_api):
        """quote without -g should use the default GPU type (A100) and succeed."""
        result = runner.invoke(cli, ["quote"], obj={"api": mock_api})
        assert result.exit_code == 0


# ===========================================================================
# configure
# ===========================================================================


class TestConfigureCommand:
    """Configure command writes provider credentials."""

    def test_runpod_echoes_provider_name(self, runner, mock_api):
        result = runner.invoke(
            cli, ["configure", "--provider", "runpod"], obj={"api": mock_api}, input="test-key\n"
        )
        assert result.exit_code == 0
        assert "RUNPOD" in result.output

    def test_aws_echoes_provider_name(self, runner, mock_api):
        result = runner.invoke(
            cli, ["configure", "--provider", "aws"], obj={"api": mock_api}, input="test-key\n"
        )
        assert result.exit_code == 0
        assert "AWS" in result.output

    def test_gcp_echoes_provider_name(self, runner, mock_api):
        result = runner.invoke(
            cli, ["configure", "--provider", "gcp"], obj={"api": mock_api}, input="test-key\n/project\n"
        )
        assert result.exit_code == 0
        assert "GCP" in result.output

    def test_azure_echoes_provider_name(self, runner, mock_api):
        # Azure prompts for Client ID, Subscription ID, Tenant ID, Client ID, Client Secret
        result = runner.invoke(
            cli,
            ["configure", "--provider", "azure"],
            obj={"api": mock_api},
            input="client-id\nsub-id\ntenant-id\nclient-id\nsecret\n",
        )
        assert result.exit_code == 0
        assert "AZURE" in result.output

    def test_vastai_echoes_provider_name(self, runner, mock_api):
        result = runner.invoke(
            cli, ["configure", "--provider", "vastai"], obj={"api": mock_api}, input="test-key\n"
        )
        assert result.exit_code == 0
        assert "VASTAI" in result.output

    def test_lambda_echoes_provider_name(self, runner, mock_api):
        result = runner.invoke(
            cli, ["configure", "--provider", "lambda_labs"], obj={"api": mock_api}, input="test-key\n"
        )
        assert result.exit_code == 0

    def test_unknown_provider_accepted_not_crash(self, runner, mock_api):
        """configure rejects unknown provider names gracefully."""
        result = runner.invoke(
            cli, ["configure", "--provider", "my-custom-cloud"], obj={"api": mock_api}
        )
        assert "Unknown provider" in result.output or result.exit_code != 0


# ===========================================================================
# setup
# ===========================================================================


class TestSetupCommand:
    """Setup command shows instructions for a specific provider."""

    def test_setup_runpod_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "runpod"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_setup_runpod_quick_flag(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "runpod", "--quick"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_setup_vastai_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "vastai"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_setup_aws_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "aws"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_setup_gcp_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "gcp"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_setup_azure_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "azure"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_setup_tensordock_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "tensordock"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_setup_lambda_labs_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "lambda_labs"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_setup_crusoe_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "crusoe"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_setup_invalid_provider_reports_error(self, runner, mock_api):
        result = runner.invoke(cli, ["setup", "not_a_real_provider"], obj={"api": mock_api})
        assert result.exit_code != 0 or "not found" in result.output.lower() or "Unknown" in result.output

    def test_setup_missing_provider_rejected(self, runner, mock_api):
        result = runner.invoke(cli, ["setup"], obj={"api": mock_api})
        assert result.exit_code != 0


# ===========================================================================
# providers group subcommands
# ===========================================================================


class TestProvidersGroup:
    """providers group subcommands."""

    def test_providers_help(self, runner):
        result = runner.invoke(cli, ["providers", "--help"])
        assert result.exit_code == 0
        assert "providers" in result.output.lower()

    def test_list_profiles_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["providers", "list-profiles"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_list_profiles_json(self, runner, mock_api):
        result = runner.invoke(
            cli, ["providers", "list-profiles", "--format", "json"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_list_profiles_yaml(self, runner, mock_api):
        result = runner.invoke(
            cli, ["providers", "list-profiles", "--format", "yaml"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_show_profile_runpod(self, runner, mock_api):
        result = runner.invoke(
            cli, ["providers", "show-profile", "runpod"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_show_profile_vastai(self, runner, mock_api):
        result = runner.invoke(
            cli, ["providers", "show-profile", "vastai"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_show_profile_aws(self, runner, mock_api):
        result = runner.invoke(
            cli, ["providers", "show-profile", "aws"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_show_profile_json(self, runner, mock_api):
        result = runner.invoke(
            cli,
            ["providers", "show-profile", "runpod", "--format", "json"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0

    def test_show_profile_unknown_returns_zero(self, runner, mock_api):
        result = runner.invoke(
            cli, ["providers", "show-profile", "nonexistent"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_export_example_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["providers", "export-example"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_export_example_contains_yaml_structure(self, runner, mock_api):
        result = runner.invoke(cli, ["providers", "export-example"], obj={"api": mock_api})
        assert "profiles:" in result.output
        assert "my_internal_cluster" in result.output

    def test_remove_profile_unknown_exits_zero(self, runner, mock_api):
        result = runner.invoke(
            cli, ["providers", "remove-profile", "nonexistent", "--force"], obj={"api": mock_api}
        )
        assert result.exit_code == 0


# ===========================================================================
# onboarding
# ===========================================================================


class TestOnboardingCommand:
    """Onboarding command conditional on first-time-user status."""

    def test_already_configured_skips_onboarding(self, runner, mock_api):
        mock_api.is_first_time_user.return_value = False
        result = runner.invoke(cli, ["onboarding"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "already" in result.output.lower()

    def test_force_flag_triggers_onboarding(self, runner, mock_api):
        mock_api.is_first_time_user.return_value = False
        with patch("terradev_cli.commands.providers.run_interactive_onboarding") as mock_onboard:
            result = runner.invoke(cli, ["onboarding", "--force"], obj={"api": mock_api})
        assert result.exit_code == 0
        mock_onboard.assert_called_once_with(mock_api)

    def test_first_time_user_triggers_onboarding(self, runner, mock_api):
        mock_api.is_first_time_user.return_value = True
        with patch("terradev_cli.commands.providers.run_interactive_onboarding") as mock_onboard:
            result = runner.invoke(cli, ["onboarding"], obj={"api": mock_api})
        assert result.exit_code == 0
        mock_onboard.assert_called_once_with(mock_api)


# ===========================================================================
# CLI structure / help / version
# ===========================================================================


class TestCLIStructure:
    """Structural CLI tests (no API needed)."""

    def test_root_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "terradev" in result.output.lower()

    def test_version_flag(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "5." in result.output

    def test_quote_help(self, runner):
        result = runner.invoke(cli, ["quote", "--help"])
        assert result.exit_code == 0
        assert "GPU" in result.output or "gpu" in result.output

    def test_provision_help(self, runner):
        result = runner.invoke(cli, ["provision", "--help"])
        assert result.exit_code == 0

    def test_configure_help(self, runner):
        result = runner.invoke(cli, ["configure", "--help"])
        assert result.exit_code == 0

    def test_setup_help(self, runner):
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0

    def test_manage_help(self, runner):
        result = runner.invoke(cli, ["manage", "--help"])
        assert result.exit_code == 0

    def test_status_help(self, runner):
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0

    def test_execute_help(self, runner):
        result = runner.invoke(cli, ["execute", "--help"])
        assert result.exit_code == 0
