"""Tests for the `terradev agent mem0` command group."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from terradev_cli.commands import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_api():
    api = MagicMock()
    api.is_first_time_user.return_value = False
    return api


class TestMem0Help:
    def test_mem0_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "mem0", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "add" in result.output
        assert "search" in result.output

    def test_mem0_configure_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "mem0", "configure", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--api-key" in result.output

    def test_mem0_add_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "mem0", "add", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--text" in result.output


class TestMem0MissingClient:
    def test_configure_without_mem0ai(self, runner, mock_api):
        with patch.dict(sys.modules, {"mem0": None}):
            result = runner.invoke(
                cli,
                ["agent", "mem0", "configure", "--api-key", "test"],
                obj={"api": mock_api},
            )
        assert result.exit_code == 0

    def test_add_without_mem0ai(self, runner, mock_api):
        mock_api._provider_creds.return_value = {"api_key": "test"}
        with patch.dict(sys.modules, {"mem0": None}):
            result = runner.invoke(
                cli,
                ["agent", "mem0", "add", "--text", "hello", "--user", "u"],
                obj={"api": mock_api},
            )
        assert result.exit_code == 1
        assert "mem0ai" in result.output
