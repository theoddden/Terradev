"""Tests for the MCP command integration."""
from unittest.mock import patch, MagicMock

import pytest

pytest.importorskip("mcp")

from terradev_cli.commands import cli


class TestMcp:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["mcp", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "mcp" in result.output

    def test_requires_action(self, runner, mock_api):
        result = runner.invoke(cli, ["mcp"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_list_tools(self, runner, mock_api):
        result = runner.invoke(cli, ["mcp", "list-tools"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Terradev MCP tools" in result.output

    def test_install_requires_client(self, runner, mock_api):
        result = runner.invoke(cli, ["mcp", "install"], obj={"api": mock_api})
        assert result.exit_code != 0

    @patch("terradev_cli.mcp.install_config")
    def test_install_runs(self, mock_install, runner, mock_api):
        result = runner.invoke(
            cli, ["mcp", "install", "--client", "cursor"], obj={"api": mock_api}
        )
        assert result.exit_code == 0
        mock_install.assert_called_once_with("cursor")


class TestMcpCheck:
    def _vault_mock(self, configured=None):
        m = MagicMock()
        m.verify.return_value = {"configured": configured or [], "missing": {}, "valid": True}
        return m

    def test_check_no_credentials_exits_1(self, runner, mock_api):
        with patch("terradev_cli.core.vault_adapter.VaultAdapter", return_value=self._vault_mock()):
            with patch.dict("os.environ", {}, clear=False):
                result = runner.invoke(cli, ["mcp", "check"], obj={"api": mock_api})
        assert result.exit_code == 1
        assert "No providers configured" in result.output

    def test_check_env_var_shows_ok(self, runner, mock_api):
        with patch("terradev_cli.core.vault_adapter.VaultAdapter", return_value=self._vault_mock()):
            with patch.dict("os.environ", {"RUNPOD_API_KEY": "rpa_test123"}):
                result = runner.invoke(cli, ["mcp", "check"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "OK" in result.output
        assert "runpod" in result.output
        assert "env" in result.output

    def test_check_vault_provider_shows_ok(self, runner, mock_api):
        with patch("terradev_cli.core.vault_adapter.VaultAdapter", return_value=self._vault_mock(configured=["vastai"])):
            result = runner.invoke(cli, ["mcp", "check"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "OK" in result.output
        assert "vastai" in result.output
        assert "vault" in result.output

    def test_check_shows_unconfigured_providers(self, runner, mock_api):
        with patch("terradev_cli.core.vault_adapter.VaultAdapter", return_value=self._vault_mock(configured=["runpod"])):
            result = runner.invoke(cli, ["mcp", "check"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Not configured" in result.output

    def test_check_output_includes_help_text(self, runner, mock_api):
        with patch("terradev_cli.core.vault_adapter.VaultAdapter", return_value=self._vault_mock(configured=["runpod"])):
            result = runner.invoke(cli, ["mcp", "check"], obj={"api": mock_api})
        assert "terradev mcp serve" in result.output
