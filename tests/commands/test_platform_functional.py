"""Functional DI tests for terradev_cli/commands/platform.py"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terradev_cli.commands import cli


def _make_status():
    tier = SimpleNamespace(
        instances=2,
        healthy=2,
        failed=0,
        kv_hit_rate=0.95,
        decode_queue_depth=3,
        ttft_p95_ms=1200.0,
        cost_hr=1.5,
        gpu_type="A100",
    )
    return SimpleNamespace(
        fleet_id="ag_123",
        model="meta-llama/Llama-3.1-70B-Instruct",
        n_agents=8,
        kv_cache_pressure="healthy",
        total_cost_hr=4.5,
        uptime_s=3600,
        warnings=["demo warning"],
        tiers={"reasoning": tier, "decode": tier, "cpu_tools": tier},
    )


def _make_spec():
    tiers = {
        "reasoning": SimpleNamespace(count=1, gpu_type="H100"),
        "decode": SimpleNamespace(count=2, gpu_type="A100"),
        "cpu_tools": SimpleNamespace(count=1, gpu_type="CPU"),
    }
    return SimpleNamespace(
        model="meta-llama/Llama-3.1-70B-Instruct",
        n_agents=8,
        fleet_id="ag_123",
        tiers=tiers,
        to_dict=lambda: {
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "n_agents": 8,
            "tiers": {k: {"count": v.count, "gpu_type": v.gpu_type} for k, v in tiers.items()},
        },
    )


def _make_cost():
    return SimpleNamespace(
        reasoning_hr=2.0,
        decode_hr=2.0,
        cpu_hr=0.5,
        total_hr=4.5,
        daily=108.0,
        monthly=3240.0,
        cost_per_agent_hr=0.56,
    )


def _make_provision_result():
    return SimpleNamespace(
        fleet_id="ag_123",
        success=True,
        total_wall_ms=1234.5,
        state_path="/tmp/ag_123.json",
        errors=[],
        cost_estimate=SimpleNamespace(total_hr=4.5, daily=108.0, monthly=3240.0),
    )


class TestSsoFunctional:
    @patch("terradev_cli.commands.platform.TerradevAPI")
    def test_sso_status_uninitialized(self, MockAPI, runner):
        api = MagicMock()
        api.enterprise_auth = None
        MockAPI.return_value = api
        result = runner.invoke(cli, ["sso", "status"])
        assert result.exit_code == 0
        assert "not initialized" in result.output.lower()

    @patch("terradev_cli.commands.platform.TerradevAPI")
    def test_sso_status_empty(self, MockAPI, runner):
        api = MagicMock()
        api.enterprise_auth.list_enabled_providers.return_value = []
        MockAPI.return_value = api
        result = runner.invoke(cli, ["sso", "status"])
        assert result.exit_code == 0
        assert "No SSO providers" in result.output

    @patch("terradev_cli.commands.platform.TerradevAPI")
    def test_sso_status_configured(self, MockAPI, runner):
        api = MagicMock()
        api.enterprise_auth.list_enabled_providers.return_value = ["okta"]
        MockAPI.return_value = api
        result = runner.invoke(cli, ["sso", "status"])
        assert result.exit_code == 0
        assert "OK" in result.output

    @patch("terradev_cli.commands.platform.TerradevAPI")
    def test_sso_configure_oidc_missing_secret(self, MockAPI, runner):
        api = MagicMock()
        api.enterprise_auth = MagicMock()
        MockAPI.return_value = api
        result = runner.invoke(cli, ["sso", "configure", "-p", "google_workspace", "--client-id", "id"])
        assert result.exit_code == 0
        assert "secret required" in result.output.lower()

    @patch("terradev_cli.commands.platform.TerradevAPI")
    def test_sso_configure_google(self, MockAPI, runner):
        api = MagicMock()
        api.enterprise_auth.get_sso_provider_config.return_value = {}
        MockAPI.return_value = api
        result = runner.invoke(
            cli,
            [
                "sso",
                "configure",
                "-p",
                "google_workspace",
                "--client-id",
                "id",
                "--client-secret",
                "secret",
            ],
        )
        assert result.exit_code == 0
        assert "configured" in result.output.lower()

    @patch("terradev_cli.commands.platform.TerradevAPI")
    def test_sso_test_specific_missing(self, MockAPI, runner):
        api = MagicMock()
        api.enterprise_auth.get_sso_provider_config.return_value = {}
        MockAPI.return_value = api
        result = runner.invoke(cli, ["sso", "test", "-p", "okta"])
        assert result.exit_code == 0
        assert "not configured" in result.output.lower()


class TestLocalFunctional:
    @patch("terradev_cli.commands.platform._register_local_pool", create=True)
    @patch("subprocess.run")
    def test_scan_local_found(self, mock_sub, mock_register, runner, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_sub.return_value = MagicMock(returncode=0, stdout="0,RTX4090,24564,535.104,0,45,280,450,4,16,8.9\n")
        result = runner.invoke(cli, ["local", "scan", "--register"], input="n\n")
        assert result.exit_code == 0
        assert "RTX4090" in result.output

    @patch("terradev_cli.commands.platform._register_local_pool", create=True)
    @patch("subprocess.run")
    def test_scan_local_not_found(self, mock_sub, mock_register, runner, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_sub.return_value = MagicMock(returncode=0, stdout="")
        result = runner.invoke(cli, ["local", "scan"])
        assert result.exit_code == 0
        assert "No GPUs" in result.output

    @patch("terradev_cli.commands.platform._register_local_pool", create=True)
    @patch("subprocess.run")
    def test_register_local(self, mock_sub, mock_register, runner, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_sub.return_value = MagicMock(returncode=0, stdout="0,RTX4090,24564,535.104,0,45\n")
        result = runner.invoke(cli, ["local", "register", "--name", "ws"])
        assert result.exit_code == 0
        assert "Registered" in result.output

    def test_pool_empty(self, runner, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        result = runner.invoke(cli, ["local", "pool"])
        assert result.exit_code == 0
        assert "No local pool" in result.output

    def test_pool_json_and_remove(self, runner, monkeypatch, tmp_path):
        import json as _json
        import os

        # On Windows os.path.expanduser("~") ignores HOME and uses USERPROFILE,
        # so we set both to the temp directory for a portable cross-platform home.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        os.makedirs(tmp_path / ".terradev", exist_ok=True)
        pool = {
            "ws": {
                "gpus": [{"name": "RTX4090", "memory_total_mb": 24564}],
                "provider": "local",
                "price_per_hour": 0.0,
                "host": "localhost",
            }
        }
        (tmp_path / ".terradev" / "local_pool.json").write_text(_json.dumps(pool))

        result = runner.invoke(cli, ["local", "pool", "--format", "json"])
        assert result.exit_code == 0
        assert "ws" in result.output

        result = runner.invoke(cli, ["local", "pool", "--remove", "ws"])
        assert result.exit_code == 0
        assert "Removed" in result.output


class TestAgentFunctional:
    @patch("terradev_cli.core.agentic_topology.AgentTopologyPlanner")
    def test_agent_plan_table(self, MockPlanner, runner):
        planner = MagicMock()
        planner.infer_from_agent_count.return_value = _make_spec()
        planner.estimate_cost.return_value = _make_cost()
        MockPlanner.return_value = planner

        result = runner.invoke(cli, ["agent", "plan", "--agents", "8"])
        assert result.exit_code == 0
        assert "To provision" in result.output

    @patch("terradev_cli.core.agentic_topology.AgentTopologyPlanner")
    def test_agent_plan_json(self, MockPlanner, runner):
        planner = MagicMock()
        planner.infer_from_agent_count.return_value = _make_spec()
        planner.estimate_cost.return_value = _make_cost()
        MockPlanner.return_value = planner

        result = runner.invoke(cli, ["agent", "plan", "--agents", "8", "--format", "json"])
        assert result.exit_code == 0
        assert '"reasoning"' in result.output

    @patch("terradev_cli.core.agentic_topology.AgentTopologyPlanner")
    @patch("terradev_cli.core.agentic_provisioner.AgenticProvisioner")
    def test_agent_deploy_dry_run(self, MockProv, MockPlanner, runner):
        planner = MagicMock()
        planner.infer_from_agent_count.return_value = _make_spec()
        planner.estimate_cost.return_value = _make_cost()
        MockPlanner.return_value = planner

        provisioner = MagicMock()
        provisioner.provision_fleet = AsyncMock(return_value=_make_provision_result())
        MockProv.return_value = provisioner

        result = runner.invoke(cli, ["agent", "deploy", "--agents", "8", "--dry-run"])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    @patch("terradev_cli.core.agentic_topology.AgentTopologyPlanner")
    @patch("terradev_cli.core.agentic_provisioner.AgenticProvisioner")
    def test_agent_deploy_and_status(self, MockProv, MockPlanner, runner):
        planner = MagicMock()
        planner.infer_from_agent_count.return_value = _make_spec()
        planner.estimate_cost.return_value = _make_cost()
        MockPlanner.return_value = planner

        provisioner = MagicMock()
        provisioner.provision_fleet = AsyncMock(return_value=_make_provision_result())
        provisioner.fleet_status = AsyncMock(return_value=_make_status())
        MockProv.return_value = provisioner

        result = runner.invoke(cli, ["agent", "deploy", "--agents", "8"])
        assert result.exit_code == 0
        assert "PROVISIONED" in result.output

        result = runner.invoke(cli, ["agent", "status", "--fleet-id", "ag_123", "--format", "json"])
        assert result.exit_code == 0
        assert "ag_123" in result.output

    @patch("terradev_cli.core.agentic_provisioner.AgenticProvisioner")
    def test_agent_cost_and_list(self, MockProv, runner):
        provisioner = MagicMock()
        provisioner.fleet_cost.return_value = {
            "uptime_hr": 1.0,
            "cost_per_hr": 4.5,
            "accrued_cost": 4.5,
            "projected_daily": 108.0,
            "projected_monthly": 3240.0,
            "cost_per_agent_hr": 0.56,
            "breakdown": {"reasoning": 2.0, "decode": 2.0, "cpu_tools": 0.5},
        }
        provisioner.list_fleets.return_value = [
            {
                "fleet_id": "ag_123",
                "model": "meta-llama/Llama-3.1-70B-Instruct",
                "n_agents": 8,
                "cost_hr": 4.5,
                "success": True,
                "created_at": 1700000000,
            }
        ]
        MockProv.return_value = provisioner

        result = runner.invoke(cli, ["agent", "cost", "--fleet-id", "ag_123"])
        assert result.exit_code == 0
        assert "$4.50" in result.output or "4.50" in result.output

        result = runner.invoke(cli, ["agent", "list"])
        assert result.exit_code == 0
        assert "ag_123" in result.output


class TestGatewayFunctional:
    @patch("terradev_cli.commands.gateway.create_gateway_config")
    @patch("terradev_cli.commands.gateway.GatewayService")
    def test_gateway_runs(self, MockSvc, MockCfg, runner):
        MockCfg.return_value = SimpleNamespace()
        MockSvc.return_value = MagicMock()
        result = runner.invoke(cli, ["gateway", "--port", "18080"])
        assert result.exit_code == 0
        MockSvc.return_value.run_sync.assert_called_once()


