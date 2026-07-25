"""Comprehensive tests for compute/provisioning commands via ctx.obj DI."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terradev_cli.commands import cli


def make_mock_provider() -> MagicMock:
    """Return an async mock satisfying the provider interface."""
    prov = AsyncMock()
    prov.get_instance_status.return_value = {"status": "running", "ip": "10.0.0.1"}
    prov.stop_instance.return_value = {"status": "stopped"}
    prov.start_instance.return_value = {"status": "running"}
    prov.terminate_instance.return_value = {"status": "terminated"}
    prov.execute_command.return_value = {"stdout": "ok", "stderr": "", "exit_code": 0}
    prov.provision_instance.return_value = {
        "instance_id": "new-prov-inst",
        "price_per_hour": 1.50,
    }
    return prov


# ===========================================================================
# provision – dry-run paths (no ProviderFactory needed)
# ===========================================================================


class TestProvisionDryRun:
    """Provision --dry-run exercises the allocation planner without launching."""

    def test_dry_run_exit_zero(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli, ["provision", "-g", "A100-80GB", "--dry-run"], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_dry_run_shows_plan(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli, ["provision", "-g", "A100-80GB", "--dry-run"], obj={"api": mock_api}
        )
        assert "DRY RUN" in result.output or "plan" in result.output.lower()

    def test_dry_run_shows_provider_in_plan(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli, ["provision", "-g", "A100-80GB", "--dry-run"], obj={"api": mock_api}
        )
        assert "RunPod" in result.output or "Vast.ai" in result.output or "TensorDock" in result.output

    def test_dry_run_provision_instance_not_called(self, runner, mock_api, patch_registry):
        runner.invoke(
            cli, ["provision", "-g", "A100-80GB", "--dry-run"], obj={"api": mock_api}
        )
        mock_api.provision_instance.assert_not_called()

    def test_dry_run_max_price_filters_out(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--dry-run", "--max-price", "0.10"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "No instances" in result.output or "ERROR" in result.output

    def test_dry_run_provider_filter_runpod_only(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--dry-run", "--providers", "runpod"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0

    def test_dry_run_count_two(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--dry-run", "-n", "2"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0

    def test_dry_run_spot_flag(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--dry-run", "--spot"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0

    def test_dry_run_on_demand_flag(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--dry-run", "--on-demand"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0

    def test_dry_run_training_type(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--dry-run", "--type", "training"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0

    def test_dry_run_inference_type(self, runner, mock_api, patch_registry):
        result = runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--dry-run", "--type", "inference"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0

    def test_dry_run_no_quotes_reports_error(self, runner, mock_api_empty, patch_registry):
        """When all providers return empty lists the command should report an error."""
        mock_api_empty.get_runpod_quotes = AsyncMock(return_value=[])
        mock_api_empty.get_vastai_quotes = AsyncMock(return_value=[])
        mock_api_empty.get_tensordock_quotes = AsyncMock(return_value=[])
        result = runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--dry-run"],
            obj={"api": mock_api_empty},
        )
        assert result.exit_code == 0
        assert "ERROR" in result.output or "No quotes" in result.output or "No instances" in result.output

    def test_dry_run_missing_gpu_arg_rejected(self, runner, mock_api, patch_registry):
        result = runner.invoke(cli, ["provision", "--dry-run"], obj={"api": mock_api})
        assert result.exit_code != 0


class TestProvisionLive:
    """Provision without --dry-run – ProviderFactory mocked at source."""

    @patch("terradev_cli.core.ssh_key_manager.generate_provision_keypair", return_value=("/tmp/k", "ssh-ed25519 AAAA"))
    @patch("terradev_cli.core.rate_limiter.get_rate_limiter")
    @patch("terradev_cli.providers.provider_factory.ProviderFactory")
    def test_live_calls_provider_factory(self, MockFactory, mock_get_rl, _mock_kp, runner, mock_api, patch_registry):
        mock_prov = make_mock_provider()
        MockFactory.return_value.create_provider.return_value = mock_prov

        async def _passthrough(pname, fn):
            return await fn()

        mock_rl = MagicMock()
        mock_rl.execute_with_rate_limit = _passthrough
        mock_get_rl.return_value = mock_rl

        result = runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--providers", "runpod", "--auto", "--on-demand"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert MockFactory.called

    @patch("terradev_cli.core.ssh_key_manager.generate_provision_keypair", return_value=("/tmp/k", "ssh-ed25519 AAAA"))
    @patch("terradev_cli.core.rate_limiter.get_rate_limiter")
    @patch("terradev_cli.providers.provider_factory.ProviderFactory")
    def test_live_records_provision_on_success(self, MockFactory, mock_get_rl, _mock_kp, runner, mock_api, patch_registry):
        mock_prov = make_mock_provider()
        MockFactory.return_value.create_provider.return_value = mock_prov

        async def _passthrough(pname, fn):
            return await fn()

        mock_rl = MagicMock()
        mock_rl.execute_with_rate_limit = _passthrough
        mock_get_rl.return_value = mock_rl

        runner.invoke(
            cli,
            ["provision", "-g", "A100-80GB", "--providers", "runpod", "--auto", "--on-demand"],
            obj={"api": mock_api},
        )
        mock_api.record_provision.assert_called()


# ===========================================================================
# manage
# ===========================================================================


class TestManageCommand:
    """Manage command lifecycle operations."""

    def test_unknown_instance_reports_error(self, runner, mock_api):
        result = runner.invoke(
            cli, ["manage", "-i", "does-not-exist", "-a", "status"], obj={"api": mock_api}
        )
        assert result.exit_code == 0
        assert "does-not-exist" in result.output
        assert "ERROR" in result.output

    def test_missing_instance_id_rejected(self, runner, mock_api):
        result = runner.invoke(cli, ["manage", "-a", "status"], obj={"api": mock_api})
        assert result.exit_code != 0

    @patch("terradev_cli.providers.provider_factory.ProviderFactory")
    def test_status_known_instance_calls_provider(self, MockFactory, runner, mock_api):
        mock_prov = make_mock_provider()
        MockFactory.return_value.create_provider.return_value = mock_prov

        result = runner.invoke(
            cli,
            ["manage", "-i", "test-inst-abc123", "-a", "status"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        mock_prov.get_instance_status.assert_called_once_with("test-inst-abc123")

    @patch("terradev_cli.providers.provider_factory.ProviderFactory")
    def test_stop_known_instance(self, MockFactory, runner, mock_api):
        mock_prov = make_mock_provider()
        MockFactory.return_value.create_provider.return_value = mock_prov

        result = runner.invoke(
            cli,
            ["manage", "-i", "test-inst-abc123", "-a", "stop"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        mock_prov.stop_instance.assert_called_once()

    @patch("terradev_cli.providers.provider_factory.ProviderFactory")
    def test_start_known_instance(self, MockFactory, runner, mock_api):
        mock_prov = make_mock_provider()
        MockFactory.return_value.create_provider.return_value = mock_prov

        result = runner.invoke(
            cli,
            ["manage", "-i", "test-inst-abc123", "-a", "start"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        mock_prov.start_instance.assert_called_once()

    @patch("terradev_cli.providers.provider_factory.ProviderFactory")
    def test_terminate_removes_instance_from_usage(self, MockFactory, runner, mock_api):
        mock_prov = make_mock_provider()
        MockFactory.return_value.create_provider.return_value = mock_prov

        runner.invoke(
            cli,
            ["manage", "-i", "test-inst-abc123", "-a", "terminate"],
            obj={"api": mock_api},
        )
        remaining = [
            i for i in mock_api.usage["instances_created"] if i["id"] == "test-inst-abc123"
        ]
        assert remaining == []

    @patch("terradev_cli.providers.provider_factory.ProviderFactory")
    def test_terminate_calls_save_usage(self, MockFactory, runner, mock_api):
        mock_prov = make_mock_provider()
        MockFactory.return_value.create_provider.return_value = mock_prov

        runner.invoke(
            cli,
            ["manage", "-i", "test-inst-abc123", "-a", "terminate"],
            obj={"api": mock_api},
        )
        mock_api.save_usage.assert_called()


# ===========================================================================
# execute
# ===========================================================================


class TestExecuteCommand:
    """Execute command runs shell commands on remote instances."""

    def test_unknown_instance_reports_error(self, runner, mock_api):
        result = runner.invoke(
            cli,
            ["execute", "-i", "nonexistent", "--cmd", "nvidia-smi"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "nonexistent" in result.output
        assert "ERROR" in result.output

    @patch("terradev_cli.providers.provider_factory.ProviderFactory")
    def test_known_instance_executes_command(self, MockFactory, runner, mock_api):
        mock_prov = make_mock_provider()
        MockFactory.return_value.create_provider.return_value = mock_prov

        result = runner.invoke(
            cli,
            ["execute", "-i", "test-inst-abc123", "--cmd", "nvidia-smi"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        mock_prov.execute_command.assert_called_once_with("test-inst-abc123", "nvidia-smi", False)

    @patch("terradev_cli.providers.provider_factory.ProviderFactory")
    def test_async_exec_flag(self, MockFactory, runner, mock_api):
        mock_prov = make_mock_provider()
        mock_prov.execute_command.return_value = {"job_id": "job-xyz", "status": "submitted"}
        MockFactory.return_value.create_provider.return_value = mock_prov

        result = runner.invoke(
            cli,
            ["execute", "-i", "test-inst-abc123", "--cmd", "python train.py", "--async-exec"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        mock_prov.execute_command.assert_called_once_with("test-inst-abc123", "python train.py", True)

    def test_missing_instance_id_rejected(self, runner, mock_api):
        result = runner.invoke(cli, ["execute", "--cmd", "ls"], obj={"api": mock_api})
        assert result.exit_code != 0

    def test_missing_command_rejected(self, runner, mock_api):
        result = runner.invoke(
            cli, ["execute", "-i", "test-inst-abc123"], obj={"api": mock_api}
        )
        assert result.exit_code != 0


# ===========================================================================
# status
# ===========================================================================


class TestStatusCommand:
    """Status command reads instance data from the injected API."""

    def test_shows_active_instances_header(self, runner, mock_api):
        result = runner.invoke(cli, ["status"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Active Instances" in result.output

    def test_lists_tracked_instance_id(self, runner, mock_api):
        result = runner.invoke(cli, ["status"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "test-inst-abc123" in result.output

    def test_empty_usage_shows_tip(self, runner, mock_api_empty):
        result = runner.invoke(cli, ["status"], obj={"api": mock_api_empty})
        assert result.exit_code == 0
        assert "No active instances" in result.output or "Tip" in result.output

    def test_json_format_output(self, runner, mock_api):
        result = runner.invoke(cli, ["status", "--format", "json"], obj={"api": mock_api})
        assert result.exit_code == 0
        import json
        import re
        # Status prints a header before the JSON block; extract the JSON array.
        match = re.search(r"\[[\s\S]*\]", result.output)
        assert match, "No JSON array found in status output"
        parsed = json.loads(match.group(0))
        assert isinstance(parsed, list)
        assert parsed[0]["id"] == "test-inst-abc123"

    def test_shows_mode_info(self, runner, mock_api):
        result = runner.invoke(cli, ["status"], obj={"api": mock_api})
        assert "Open Source" in result.output or "Provisions" in result.output


# ===========================================================================
# cleanup
# ===========================================================================


class TestCleanupCommand:
    """Cleanup removes instances older than 30 days."""

    def test_no_old_instances_reports_ok(self, runner, mock_api):
        result = runner.invoke(cli, ["cleanup"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "No old instances found" in result.output

    def test_no_old_instances_does_not_save_usage(self, runner, mock_api):
        runner.invoke(cli, ["cleanup"], obj={"api": mock_api})
        mock_api.save_usage.assert_not_called()

    def test_old_instance_is_removed(self, runner, mock_api_old_instance):
        runner.invoke(cli, ["cleanup"], obj={"api": mock_api_old_instance})
        remaining = mock_api_old_instance.usage["instances_created"]
        assert all(i["id"] != "old-inst-xyz789" for i in remaining)

    def test_old_instance_triggers_save(self, runner, mock_api_old_instance):
        runner.invoke(cli, ["cleanup"], obj={"api": mock_api_old_instance})
        mock_api_old_instance.save_usage.assert_called_once()

    def test_mixed_instances_keeps_fresh(self, runner, mock_api_mixed_instances):
        runner.invoke(cli, ["cleanup"], obj={"api": mock_api_mixed_instances})
        remaining_ids = [
            i["id"] for i in mock_api_mixed_instances.usage["instances_created"]
        ]
        assert "test-inst-abc123" in remaining_ids
        assert "old-inst-xyz789" not in remaining_ids


# ===========================================================================
# optimize
# ===========================================================================


class TestOptimizeCommand:
    """Optimize command fetches quotes and analyses instances."""

    def test_no_instances_exits_zero(self, runner, mock_api_empty):
        result = runner.invoke(cli, ["optimize"], obj={"api": mock_api_empty})
        assert result.exit_code == 0

    def test_with_instance_calls_quote_methods(self, runner, mock_api):
        runner.invoke(cli, ["optimize"], obj={"api": mock_api})
        assert mock_api.get_runpod_quotes.called or mock_api.get_vastai_quotes.called

    def test_output_contains_analysis_header(self, runner, mock_api):
        result = runner.invoke(cli, ["optimize"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "OPTIMIZATION" in result.output or "optimize" in result.output.lower()

    def test_instance_id_filter(self, runner, mock_api):
        result = runner.invoke(
            cli,
            ["optimize", "--instance-id", "test-inst-abc123"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0

    def test_unknown_instance_id_filter_empty_result(self, runner, mock_api):
        result = runner.invoke(
            cli,
            ["optimize", "--instance-id", "nonexistent-id"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0


# ===========================================================================
# analytics
# ===========================================================================


class TestAnalyticsCommand:
    """Analytics command reads from cost_tracker (or falls back gracefully)."""

    def test_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["analytics"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_custom_days_flag(self, runner, mock_api):
        result = runner.invoke(cli, ["analytics", "--days", "30"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_json_format(self, runner, mock_api):
        result = runner.invoke(
            cli, ["analytics", "--format", "json"], obj={"api": mock_api}
        )
        assert result.exit_code == 0


# ===========================================================================
# integrations
# ===========================================================================


class TestIntegrationsCommand:
    """Integrations command shows observability status."""

    def test_exits_zero(self, runner, mock_api):
        result = runner.invoke(cli, ["integrations"], obj={"api": mock_api})
        assert result.exit_code == 0

    def test_shows_integrations_header(self, runner, mock_api):
        result = runner.invoke(cli, ["integrations"], obj={"api": mock_api})
        assert "Integrations" in result.output or "integration" in result.output.lower()


# ===========================================================================
# stage
# ===========================================================================


class TestStageCommand:
    """Stage command compresses and pre-positions datasets."""

    @patch("terradev_cli.core.dataset_stager.DatasetStager")
    def test_plan_only_flag(self, MockStager, runner, mock_api):
        plan = MagicMock()
        plan.to_dict.return_value = {
            "original_size": "100 MB",
            "compressed_size": "60 MB",
            "compression_ratio": "40%",
            "compression_algo": "zstd",
            "chunks": 10,
            "chunk_size": "10 MB",
            "regions": ["us-east-1"],
        }
        MockStager.return_value.plan.return_value = plan

        result = runner.invoke(
            cli,
            ["stage", "-d", "my-dataset", "--plan-only"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "Staging Plan" in result.output or "Plan" in result.output

    def test_missing_dataset_rejected(self, runner, mock_api):
        result = runner.invoke(cli, ["stage"], obj={"api": mock_api})
        assert result.exit_code != 0


# ===========================================================================
# job
# ===========================================================================


class TestJobCommand:
    """Job command loads YAML and runs it."""

    def test_missing_file_rejected_by_click(self, runner, mock_api):
        result = runner.invoke(
            cli, ["job", "nonexistent.yaml"], obj={"api": mock_api}
        )
        assert result.exit_code != 0

    def test_valid_yaml_runs_without_error(self, runner, mock_api, tmp_path):
        import yaml

        job_file = tmp_path / "test_job.yaml"
        job_file.write_text(
            yaml.dump({"name": "test-job", "gpu_type": "A100", "count": 1, "max_price": 2.0})
        )
        result = runner.invoke(
            cli, ["job", str(job_file)], obj={"api": mock_api}
        )
        assert result.exit_code == 0

    def test_optimize_flag_passed_through(self, runner, mock_api, tmp_path):
        import yaml

        job_file = tmp_path / "test_job.yaml"
        job_file.write_text(yaml.dump({"name": "j", "gpu_type": "H100"}))
        result = runner.invoke(
            cli, ["job", str(job_file), "--optimize", "cost"], obj={"api": mock_api}
        )
        assert result.exit_code == 0
        assert "cost" in result.output.lower() or "optimiz" in result.output.lower()


# ===========================================================================
# run (one-command GPU deploy)
# ===========================================================================


class TestRunCommand:
    """Run command combines provision + execute in one step."""

    def test_dry_run_shows_deployment_info(self, runner, mock_api):
        result = runner.invoke(
            cli,
            [
                "run",
                "-g", "A100-80GB",
                "--image", "pytorch/pytorch:latest",
                "--cmd", "python train.py",
                "--dry-run",
            ],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "A100-80GB" in result.output
        assert "pytorch/pytorch:latest" in result.output

    def test_dry_run_missing_image_rejected(self, runner, mock_api):
        result = runner.invoke(
            cli,
            ["run", "-g", "A100-80GB", "--dry-run"],
            obj={"api": mock_api},
        )
        assert result.exit_code != 0

    def test_dry_run_keep_alive_flag(self, runner, mock_api):
        result = runner.invoke(
            cli,
            [
                "run",
                "-g", "A100-80GB",
                "--image", "vllm/vllm-openai:latest",
                "--keep-alive",
                "--dry-run",
            ],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0
        assert "keep-alive" in result.output.lower()
