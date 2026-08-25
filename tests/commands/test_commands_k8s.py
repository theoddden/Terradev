"""Comprehensive tests for Kubernetes / GitOps commands via ctx.obj DI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terradev_cli.commands import cli


# ===========================================================================
# Shared mock helpers
# ===========================================================================


def _make_terraform_wrapper():
    """Return a mock TerraformWrapper for k8s commands."""
    tw = MagicMock()
    tw.create_cluster.return_value = True
    tw.destroy_cluster.return_value = True
    tw.list_clusters.return_value = [
        {
            "name": "test-cluster",
            "gpu_type": "A100",
            "node_count": 1,
            "status": "running",
        }
    ]
    tw.get_cluster_info.return_value = {
        "name": "test-cluster",
        "status": "running",
        "outputs": {
            "gpu_summary": {
                "gpu_type": "A100",
                "total_gpus": 1,
                "max_price": 4.0,
                "actual_average": 2.5,
                "prefer_spot": True,
            },
            "cost_breakdown": {
                "aws": {"nodes": 1, "cost_hr": 2.5, "cost_mo": 1800.0},
            },
            "savings_analysis": {
                "aws_only_cost_per_hour": 3.0,
                "multi_cloud_cost_per_hour": 2.5,
                "savings_per_hour": 0.5,
                "savings_percentage": 16.7,
            },
            "next_steps": ["Run workloads"],
        },
    }
    return tw


# ===========================================================================
# k8s group
# ===========================================================================


class TestK8sGroup:
    """Tests for the k8s group."""

    def test_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["k8s", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Kubernetes" in result.output

    @patch("terradev_cli.commands.k8s.TerraformWrapper")
    def test_create_runs(self, MockWrapper, runner, mock_api):
        MockWrapper.return_value = _make_terraform_wrapper()
        result = runner.invoke(
            cli, ["k8s", "create", "test-cluster", "--gpu", "A100", "--count", "1"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    @patch("terradev_cli.commands.k8s.TerraformWrapper")
    def test_destroy_runs(self, MockWrapper, runner, mock_api):
        MockWrapper.return_value = _make_terraform_wrapper()
        result = runner.invoke(
            cli, ["k8s", "destroy", "test-cluster"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    @patch("terradev_cli.commands.k8s.TerraformWrapper")
    def test_list_runs(self, MockWrapper, runner, mock_api):
        MockWrapper.return_value = _make_terraform_wrapper()
        result = runner.invoke(cli, ["k8s", "list"], obj={"api": mock_api})
        assert result.exit_code == 0

    @patch("terradev_cli.commands.k8s.TerraformWrapper")
    def test_info_runs(self, MockWrapper, runner, mock_api):
        MockWrapper.return_value = _make_terraform_wrapper()
        result = runner.invoke(
            cli, ["k8s", "info", "test-cluster"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_create_requires_cluster_name(self, runner, mock_api):
        result = runner.invoke(cli, ["k8s", "create"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_destroy_requires_cluster_name(self, runner, mock_api):
        result = runner.invoke(cli, ["k8s", "destroy"], obj={"api": mock_api})
        assert result.exit_code != 0

