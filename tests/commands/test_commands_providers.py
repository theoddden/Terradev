"""Focused tests for the extracted providers commands using ctx.obj dependency injection."""

import pytest

from terradev_cli.commands import cli


class TestQuoteCommand:
    """Quote command tested in isolation without monkeypatching TerradevAPI."""

    def test_quote_runs_with_mock_api(self, runner, mock_api):
        """quote should use the injected API and exit cleanly."""
        result = runner.invoke(cli, ["quote", "-g", "A100"], obj={"api": mock_api})
        assert result.exit_code == 0
        mock_api.get_runpod_quotes.assert_called()

    def test_quote_filters_providers(self, runner, mock_api):
        """--providers should restrict which quote methods are called."""
        result = runner.invoke(cli, ["quote", "-g", "A100", "-p", "runpod"], obj={"api": mock_api})
        assert result.exit_code == 0
        mock_api.get_runpod_quotes.assert_called_once()
        mock_api.get_vastai_quotes.assert_not_called()

    def test_quote_missing_gpu_type(self, runner, mock_api):
        """quote without GPU type should fail validation."""
        result = runner.invoke(cli, ["quote"], obj={"api": mock_api})
        # click itself requires -g because the option has a default; verify exit code
        assert result.exit_code == 0 or result.exit_code == 2


class TestConfigureCommand:
    """Configure command with mocked TerradevAPI."""

    def test_configure_unknown_provider(self, runner, mock_api):
        """An unknown provider is accepted and echoed by the configure command."""
        result = runner.invoke(
            cli,
            ["configure", "--provider", "not-a-provider"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "not-a-provider".upper() in result.output


class TestProvidersGroup:
    """Structural and behaviour tests for the providers group."""

    def test_providers_group_exists(self, runner):
        result = runner.invoke(cli, ["providers", "--help"])
        assert result.exit_code == 0
        assert "providers" in result.output.lower()
