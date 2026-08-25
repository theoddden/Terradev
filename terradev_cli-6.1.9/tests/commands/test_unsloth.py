"""Tests for the `terradev train unsloth` command group."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from terradev_cli.commands import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_api():
    from unittest.mock import MagicMock

    api = MagicMock()
    api.is_first_time_user.return_value = False
    return api


class TestUnslothHelp:
    def test_unsloth_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "unsloth", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_unsloth_start_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "unsloth", "start", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "claude" in result.output

    def test_unsloth_run_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "unsloth", "run", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_unsloth_stop_help(self, runner, mock_api):
        result = runner.invoke(cli, ["train", "unsloth", "stop", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--pid-file" in result.output


class TestUnslothMissingCli:
    def test_start_without_unsloth_installed(self, runner, mock_api):
        with patch("terradev_cli.commands.unsloth.shutil.which", return_value=None):
            result = runner.invoke(
                cli, ["train", "unsloth", "start", "claude"], obj={"api": mock_api}
            )
        assert result.exit_code == 1
        assert "unsloth CLI not found" in result.output

    def test_run_without_unsloth_installed(self, runner, mock_api):
        with patch("terradev_cli.commands.unsloth.shutil.which", return_value=None):
            result = runner.invoke(
                cli, ["train", "unsloth", "run", "--model", "unsloth/Llama-3.1-8B"], obj={"api": mock_api}
            )
        assert result.exit_code == 1
        assert "unsloth CLI not found" in result.output
