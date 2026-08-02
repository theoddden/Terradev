"""Tests for terradev_cli.commands.platform."""

import pytest
from unittest.mock import MagicMock, patch

from terradev_cli.commands import cli


# ── SSO tests ────────────────────────────────────────────────────────────────

@pytest.fixture
def sso_api(mock_api):
    """Mock API with an enterprise_auth stub for SSO commands."""
    mock_api.enterprise_auth = MagicMock()
    mock_api.enterprise_auth.list_enabled_providers.return_value = []
    mock_api.enterprise_auth.get_sso_provider_config.return_value = {"enabled": False}
    mock_api.enterprise_auth.enable_sso_provider = MagicMock()
    return mock_api


class TestSso:
    def test_sso_status_no_auth(self, runner, mock_api):
        mock_api.enterprise_auth = None
        with patch("terradev_cli.commands.platform.TerradevAPI", return_value=mock_api):
            result = runner.invoke(cli, ["sso", "status"])
        assert result.exit_code == 0
        assert "Enterprise auth not initialized" in result.output

    def test_sso_status_no_providers(self, runner, sso_api):
        with patch("terradev_cli.commands.platform.TerradevAPI", return_value=sso_api):
            result = runner.invoke(cli, ["sso", "status"])
        assert result.exit_code == 0
        assert "No SSO providers configured" in result.output

    def test_sso_status_configured(self, runner, sso_api):
        sso_api.enterprise_auth.list_enabled_providers.return_value = ["okta"]
        with patch("terradev_cli.commands.platform.TerradevAPI", return_value=sso_api):
            result = runner.invoke(cli, ["sso", "status"])
        assert result.exit_code == 0
        assert "SSO is configured" in result.output
        assert "okta" in result.output

    def test_sso_configure_oidc_requires_id_and_secret(self, runner, sso_api):
        with patch("terradev_cli.commands.platform.TerradevAPI", return_value=sso_api):
            result = runner.invoke(cli, ["sso", "configure", "--provider", "google_workspace"])
        assert result.exit_code == 0
        assert "Client ID and secret required" in result.output

    def test_sso_configure_oidc_success(self, runner, sso_api):
        with patch("terradev_cli.commands.platform.TerradevAPI", return_value=sso_api):
            result = runner.invoke(
                cli,
                ["sso", "configure", "--provider", "google_workspace", "--client-id", "id", "--client-secret", "sec"],
            )
        assert result.exit_code == 0
        assert "configured successfully" in result.output
        sso_api.enterprise_auth.enable_sso_provider.assert_called_once()

    def test_sso_configure_auth0_requires_domain(self, runner, sso_api):
        with patch("terradev_cli.commands.platform.TerradevAPI", return_value=sso_api):
            result = runner.invoke(
                cli,
                ["sso", "configure", "--provider", "auth0", "--client-id", "id", "--client-secret", "sec"],
            )
        assert result.exit_code == 0
        assert "Domain required for Auth0" in result.output

    def test_sso_test_all(self, runner, sso_api):
        sso_api.enterprise_auth.list_enabled_providers.return_value = ["okta"]
        with patch("terradev_cli.commands.platform.TerradevAPI", return_value=sso_api):
            result = runner.invoke(cli, ["sso", "test"])
        assert result.exit_code == 0
        assert "okta" in result.output

    def test_sso_test_specific_missing(self, runner, sso_api):
        sso_api.enterprise_auth.get_sso_provider_config.return_value = {"enabled": False}
        with patch("terradev_cli.commands.platform.TerradevAPI", return_value=sso_api):
            result = runner.invoke(cli, ["sso", "test", "--provider", "okta"])
        assert result.exit_code == 0
        assert "not configured" in result.output


# ── Smoke / help tests for other platform groups ─────────────────────────────

class TestPlatformHelp:
    @pytest.mark.parametrize("path", [
        "sso",
        "sso status",
        "sso configure",
        "sso test",
        "mcp",
        "local",
        "local scan",
        "local register",
        "local pool",
        "agent",
        "agent plan",
        "agent deploy",
        "agent status",
        "agent scale",
        "agent cost",
        "agent list",
        "agent teardown",
        "gateway",
        "observe",
        "observe gateway",
        "observe status",
        "schedule",
        "schedule job",
        "schedule list",
        "schedule windows",
    ])
    def test_help(self, runner, mock_api, path):
        with patch("terradev_cli.commands.platform.TerradevAPI", return_value=mock_api):
            result = runner.invoke(cli, path.split() + ["--help"])
        assert result.exit_code == 0, f"help failed for {path}: {result.output}"
