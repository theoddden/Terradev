"""Tests for the `terradev agent letta` command group."""

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


class TestLettaHelp:
    def test_letta_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "letta", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "create" in result.output

    def test_letta_create_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "letta", "create", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--name" in result.output

    def test_letta_chat_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "letta", "chat", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--message" in result.output


class TestLettaMissingClient:
    def test_create_without_letta_client(self, runner, mock_api):
        with patch.dict(sys.modules, {"letta_client": None}):
            result = runner.invoke(
                cli, ["agent", "letta", "create", "--name", "test"], obj={"api": mock_api}
            )
        assert result.exit_code == 1
        assert "letta-client is not installed" in result.output
