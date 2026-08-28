"""Tests for platform commands."""
from unittest.mock import MagicMock, patch

import pytest

from terradev_cli.commands import cli


# platform groups and top-level commands


class TestAgent:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "agent" in result.output

    def test_cost_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "cost", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_cost_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "cost"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_deploy_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "deploy", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_list_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "list", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_plan_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "plan", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_plan_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "plan"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_scale_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "scale", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_scale_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "scale"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_status_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "status", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_status_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "status"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_teardown_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "teardown", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_teardown_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "teardown"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_vector_db_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "vector-db", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_vector_db_up_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "vector-db", "up", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_skill_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "skill", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_skill_init_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "skill", "init", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_skill_attach_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "skill", "attach", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_letta_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "letta", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_letta_create_help(self, runner, mock_api):
        result = runner.invoke(cli, ["agent", "letta", "create", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestLocal:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["local", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "local" in result.output

    def test_pool_help(self, runner, mock_api):
        result = runner.invoke(cli, ["local", "pool", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_register_help(self, runner, mock_api):
        result = runner.invoke(cli, ["local", "register", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_register_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["local", "register"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_scan_help(self, runner, mock_api):
        result = runner.invoke(cli, ["local", "scan", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0


class TestSso:
    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["sso", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "sso" in result.output

    def test_configure_help(self, runner, mock_api):
        result = runner.invoke(cli, ["sso", "configure", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_configure_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["sso", "configure"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_status_help(self, runner, mock_api):
        result = runner.invoke(cli, ["sso", "status", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["sso", "test", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestGateway:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["gateway", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

class TestMcp:
    def test_help(self, runner, mock_api):
        result = runner.invoke(cli, ["mcp", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_requires_args(self, runner, mock_api):
        result = runner.invoke(cli, ["mcp"], obj={"api": mock_api})
        assert result.exit_code != 0


class TestFunctionalPlatform:
    """Functional DI tests for platform service commands."""

    @patch("terradev_cli.commands.platform.TerradevAPI")
    def test_sso_status_runs(self, MockAPI, runner, mock_api):
        api = MagicMock()
        api.enterprise_auth.list_enabled_providers.return_value = []
        MockAPI.return_value = api
        result = runner.invoke(cli, ["sso", "status"], obj={"api": api})
        assert result.exit_code == 0