"""Tests for the MCP command integration."""
from unittest.mock import patch

import pytest

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
