"""Focused tests for the extracted compute/provisioning commands using ctx.obj DI."""

from terradev_cli.commands import cli


class TestOptimizeCommand:
    """Optimize command directly exercises the injected TerradevAPI quote methods."""

    def test_optimize_fetches_quotes(self, runner, mock_api):
        """optimize should call quote methods on the injected API."""
        result = runner.invoke(cli, ["optimize"], obj={"api": mock_api})
        assert result.exit_code == 0
        # One or more quote methods should have been invoked for the tracked GPU type.
        assert mock_api.get_runpod_quotes.called or mock_api.get_vastai_quotes.called


class TestStatusCommand:
    """Status command uses the injected API usage data."""

    def test_status_shows_tracked_instances(self, runner, mock_api):
        result = runner.invoke(cli, ["status"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Active Instances" in result.output


class TestManageCommand:
    """Manage command lifecycle operations on tracked instances."""

    def test_manage_unknown_instance_reports_error(self, runner, mock_api):
        result = runner.invoke(
            cli,
            ["manage", "-i", "missing-id", "-a", "status"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "missing-id" in result.output


class TestCleanupCommand:
    """Cleanup command uses the injected API usage data."""

    def test_cleanup_no_old_instances(self, runner, mock_api):
        """cleanup should report no old instances when the tracked instance is new."""
        result = runner.invoke(cli, ["cleanup"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "No old instances found" in result.output
        mock_api.save_usage.assert_not_called()
