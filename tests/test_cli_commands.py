"""Test CLI commands with CliRunner and ctx.obj dependency injection.

All tests use runner.invoke(..., obj={"api": mock_api}) – no patching of the
terradev_cli.cli module, which is now a thin shim.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from terradev_cli import __version__
from terradev_cli.commands import cli
from terradev_cli.commands._api import TerradevAPI

os.environ["TERRADEV_SKIP_ONBOARDING"] = "1"


# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------


def _make_api() -> MagicMock:
    """Build a fully-mocked TerradevAPI for DI injection."""
    api = MagicMock(spec=TerradevAPI)
    api.credentials = {"runpod": {"api_key": "rpa_test"}}
    api._auth_manager = None
    api._provider_creds = MagicMock(return_value={"api_key": "rpa_test"})

    config_dir = Path(tempfile.mkdtemp())
    api.config_dir = config_dir
    api.credentials_file = config_dir / "credentials.json"
    api.usage_file = config_dir / "usage.json"

    _quotes = [
        {
            "provider": "RunPod",
            "price": 1.50,
            "gpu_type": "A100",
            "region": "us-east-1",
            "availability": "on-demand",
            "gpu_count": 1,
            "instance_type": "runpod-a100",
            "memory_gb": 80,
        },
        {
            "provider": "Vast.ai",
            "price": 1.20,
            "gpu_type": "A100",
            "region": "us-west-1",
            "availability": "on-demand",
            "gpu_count": 1,
            "instance_type": "vast-a100",
            "memory_gb": 80,
        },
        {
            "provider": "TensorDock",
            "price": 0.90,
            "gpu_type": "A100",
            "region": "us-east-1",
            "availability": "spot",
            "gpu_count": 1,
            "instance_type": "tensordock-a100",
            "memory_gb": 80,
        },
    ]
    api.get_runpod_quotes = AsyncMock(return_value=_quotes[:1])
    api.get_vastai_quotes = AsyncMock(return_value=_quotes[1:2])
    api.get_aws_quotes = AsyncMock(return_value=[])
    api.get_gcp_quotes = AsyncMock(return_value=[])
    api.get_azure_quotes = AsyncMock(return_value=[])
    api.get_tensordock_quotes = AsyncMock(return_value=_quotes[2:])
    api.get_oracle_quotes = AsyncMock(return_value=[])
    api.get_crusoe_quotes = AsyncMock(return_value=[])
    api.get_alibaba_quotes = AsyncMock(return_value=[])
    api.get_baseten_quotes = AsyncMock(return_value=[])
    api.get_digitalocean_quotes = AsyncMock(return_value=[])
    api.get_e2enetworks_quotes = AsyncMock(return_value=[])
    api.get_huggingface_quotes = AsyncMock(return_value=[])
    api.get_hyperstack_quotes = AsyncMock(return_value=[])
    api.get_inferx_quotes = AsyncMock(return_value=[])
    api.get_latitude_quotes = AsyncMock(return_value=[])
    api.get_siliconflow_quotes = AsyncMock(return_value=[])
    api.get_yottalabs_quotes = AsyncMock(return_value=[])

    api.provision_instance = AsyncMock(
        return_value={
            "instance_id": "test-instance-123",
            "provider": "RunPod",
            "gpu_type": "A100",
            "status": "running",
        }
    )
    api.get_instance_status = AsyncMock(
        return_value={"status": "running", "gpu_utilization": 85}
    )

    api.is_first_time_user = MagicMock(return_value=False)
    api.load_credentials = MagicMock(return_value={})
    api.save_credentials = MagicMock()
    api.load_usage = MagicMock(return_value={"provisions_this_month": 0})
    api.save_usage = MagicMock()
    api.check_provision_limit = MagicMock(return_value=True)
    api.record_provision = MagicMock()
    api.usage = {
        "instances_created": [
            {
                "id": "test-instance-123",
                "provider": "RunPod",
                "gpu_type": "A100",
                "price": 1.50,
                "region": "us-east-1",
                "created_at": "2099-12-31T00:00:00",
            }
        ],
        "inference_endpoints": [],
        "provisions_this_month": 0,
    }
    return api


# ---------------------------------------------------------------------------
# quote
# ---------------------------------------------------------------------------


class TestQuoteCommand:
    """Quote command via DI."""

    def test_basic_invocation(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["quote", "-g", "A100"], obj={"api": api})
        assert result.exit_code == 0

    def test_calls_runpod_quote_method(self):
        api = _make_api()
        CliRunner().invoke(cli, ["quote", "-g", "A100"], obj={"api": api})
        api.get_runpod_quotes.assert_called_once_with("A100")

    def test_calls_all_providers_by_default(self):
        api = _make_api()
        CliRunner().invoke(cli, ["quote", "-g", "A100"], obj={"api": api})
        assert api.get_vastai_quotes.called
        assert api.get_tensordock_quotes.called

    def test_provider_filter_runpod_only(self):
        api = _make_api()
        CliRunner().invoke(cli, ["quote", "-g", "A100", "--providers", "runpod"], obj={"api": api})
        assert api.get_runpod_quotes.called
        api.get_vastai_quotes.assert_not_called()

    def test_region_filter_match(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["quote", "-g", "A100", "-r", "us-east-1"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_region_filter_no_match_reports_error(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["quote", "-g", "A100", "-r", "ap-southeast-99"], obj={"api": api}
        )
        assert result.exit_code == 1
        assert "ERROR" in result.output or "No quotes" in result.output

    def test_no_quotes_shows_error(self):
        api = _make_api()
        api.get_runpod_quotes = AsyncMock(return_value=[])
        api.get_vastai_quotes = AsyncMock(return_value=[])
        api.get_tensordock_quotes = AsyncMock(return_value=[])
        result = CliRunner().invoke(cli, ["quote", "-g", "A100"], obj={"api": api})
        assert result.exit_code == 1
        assert "ERROR" in result.output or "No quotes" in result.output

    def test_output_contains_provider_table(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["quote", "-g", "A100"], obj={"api": api})
        assert "Provider" in result.output

    def test_output_contains_best_price(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["quote", "-g", "A100"], obj={"api": api})
        assert "Best:" in result.output

    def test_quick_flag(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["quote", "-g", "A100", "-q"], obj={"api": api})
        assert result.exit_code == 0

    def test_include_local_flag(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["quote", "-g", "A100", "--include-local"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_various_gpu_types(self):
        for gpu in ["H100", "RTX4090", "L40S", "A40", "V100"]:
            api = _make_api()
            result = CliRunner().invoke(cli, ["quote", "-g", gpu], obj={"api": api})
            assert result.exit_code == 0, f"GPU={gpu}: {result.output}"

    def test_parallel_flag(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["quote", "-g", "A100", "--parallel", "10"], obj={"api": api}
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# provision (dry-run; live paths covered in tests/commands/)
# ---------------------------------------------------------------------------


class TestProvisionCommand:
    """Provision command via DI – focusing on dry-run paths."""

    pytestmark = [pytest.mark.usefixtures("patch_registry")]

    def test_dry_run_exits_zero(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["provision", "-g", "A100", "--dry-run"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_dry_run_shows_plan(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["provision", "-g", "A100", "--dry-run"], obj={"api": api}
        )
        assert "DRY RUN" in result.output

    def test_dry_run_does_not_provision(self):
        api = _make_api()
        CliRunner().invoke(
            cli, ["provision", "-g", "A100", "--dry-run"], obj={"api": api}
        )
        api.provision_instance.assert_not_called()

    def test_dry_run_with_count(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["provision", "-g", "A100", "--dry-run", "-n", "2"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_dry_run_max_price_too_low(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["provision", "-g", "A100", "--dry-run", "--max-price", "0.01"],
            obj={"api": api},
        )
        assert result.exit_code == 0
        assert "ERROR" in result.output or "No instances" in result.output

    def test_dry_run_spot_flag(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["provision", "-g", "A100", "--dry-run", "--spot"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_dry_run_on_demand_flag(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["provision", "-g", "A100", "--dry-run", "--on-demand"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_dry_run_training_type(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli,
            ["provision", "-g", "A100", "--dry-run", "--type", "training"],
            obj={"api": api},
        )
        assert result.exit_code == 0

    def test_dry_run_inference_type(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli,
            ["provision", "-g", "A100", "--dry-run", "--type", "inference"],
            obj={"api": api},
        )
        assert result.exit_code == 0

    def test_dry_run_backend_vllm(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli,
            ["provision", "-g", "A100", "--dry-run", "--backend", "vllm"],
            obj={"api": api},
        )
        assert result.exit_code == 0

    def test_dry_run_provider_filter(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli,
            ["provision", "-g", "A100", "--dry-run", "--providers", "runpod", "--on-demand"],
            obj={"api": api},
        )
        assert result.exit_code == 0
        assert "RunPod" in result.output

    def test_dry_run_multi_agent(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli,
            ["provision", "-g", "H100", "--dry-run", "--agents", "20", "--context", "32k"],
            obj={"api": api},
        )
        assert result.exit_code == 0

    def test_dry_run_prefer_local(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["provision", "-g", "A100", "--dry-run", "--prefer-local"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_missing_gpu_arg_rejected(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["provision"], obj={"api": api})
        assert result.exit_code != 0

    def test_various_gpu_types_dry_run(self):
        for gpu in ["H100", "RTX4090", "L40S"]:
            api = _make_api()
            result = CliRunner().invoke(
                cli, ["provision", "-g", gpu, "--dry-run"], obj={"api": api}
            )
            assert result.exit_code == 0, f"GPU={gpu}: {result.output}"


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------


class TestConfigureCommand:
    """Configure command via DI."""

    def test_runpod_echoes_name(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["configure", "--provider", "runpod"], obj={"api": api}, input="test-key\n"
        )
        assert result.exit_code == 0
        assert "RUNPOD" in result.output

    def test_aws_echoes_name(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli,
            ["configure", "--provider", "aws"],
            obj={"api": api},
            input="test-key\ntest-secret\n",
        )
        assert result.exit_code == 0
        assert "AWS" in result.output

    def test_gcp_echoes_name(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["configure", "--provider", "gcp"], obj={"api": api}, input="service-account\nproject\n"
        )
        assert result.exit_code == 0
        assert "GCP" in result.output

    def test_azure_echoes_name(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli,
            ["configure", "--provider", "azure"],
            obj={"api": api},
            input="client-id\nsub-id\ntenant-id\nclient-id\nsecret\n",
        )
        assert result.exit_code == 0
        assert "AZURE" in result.output

    def test_vastai_echoes_name(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["configure", "--provider", "vastai"], obj={"api": api}, input="test-key\n"
        )
        assert result.exit_code == 0
        assert "VASTAI" in result.output


    def test_unknown_provider_rejected(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["configure", "--provider", "my-cloud"], obj={"api": api}
        )
        assert result.exit_code == 1 or "Unknown provider" in result.output


# ---------------------------------------------------------------------------
# providers group
# ---------------------------------------------------------------------------


class TestProvidersCommands:
    """providers group commands via DI."""

    def test_list_profiles(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["providers", "list-profiles"], obj={"api": api})
        assert result.exit_code == 0

    def test_list_profiles_json(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["providers", "list-profiles", "--format", "json"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_list_profiles_yaml(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["providers", "list-profiles", "--format", "yaml"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_show_profile_runpod(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["providers", "show-profile", "runpod"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_show_profile_json(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["providers", "show-profile", "runpod", "--format", "json"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_show_profile_yaml(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["providers", "show-profile", "runpod", "--format", "yaml"], obj={"api": api}
        )
        assert result.exit_code == 0

    def test_show_various_profiles(self):
        for provider in ["runpod", "vastai", "aws", "gcp", "azure", "lambda"]:
            api = _make_api()
            result = CliRunner().invoke(
                cli, ["providers", "show-profile", provider], obj={"api": api}
            )
            assert result.exit_code == 0, f"provider={provider}: {result.output}"

    def test_export_example(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["providers", "export-example"], obj={"api": api})
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


class TestSetupCommand:
    """Setup command via DI."""

    def test_setup_runpod(self):
        api = _make_api()
        assert CliRunner().invoke(cli, ["setup", "runpod"], obj={"api": api}).exit_code == 0

    def test_setup_runpod_quick(self):
        api = _make_api()
        assert CliRunner().invoke(cli, ["setup", "runpod", "--quick"], obj={"api": api}).exit_code == 0

    def test_setup_vastai(self):
        api = _make_api()
        assert CliRunner().invoke(cli, ["setup", "vastai"], obj={"api": api}).exit_code == 0

    def test_setup_aws(self):
        api = _make_api()
        assert CliRunner().invoke(cli, ["setup", "aws"], obj={"api": api}).exit_code == 0

    def test_setup_gcp(self):
        api = _make_api()
        assert CliRunner().invoke(cli, ["setup", "gcp"], obj={"api": api}).exit_code == 0

    def test_setup_azure(self):
        api = _make_api()
        assert CliRunner().invoke(cli, ["setup", "azure"], obj={"api": api}).exit_code == 0

    def test_setup_tensordock(self):
        api = _make_api()
        assert CliRunner().invoke(cli, ["setup", "tensordock"], obj={"api": api}).exit_code == 0

    def test_setup_crusoe(self):
        api = _make_api()
        assert CliRunner().invoke(cli, ["setup", "crusoe"], obj={"api": api}).exit_code == 0

    def test_setup_invalid_provider(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["setup", "invalid_provider"], obj={"api": api})
        assert result.exit_code != 0 or "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# help / version (no API needed)
# ---------------------------------------------------------------------------


class TestCLIHelpAndVersion:
    """CLI structural tests."""

    def test_cli_help(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "terradev" in result.output.lower()

    def test_cli_version(self):
        result = CliRunner().invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_quote_help(self):
        result = CliRunner().invoke(cli, ["quote", "--help"])
        assert result.exit_code == 0
        assert "quote" in result.output.lower()

    def test_provision_help(self):
        result = CliRunner().invoke(cli, ["provision", "--help"])
        assert result.exit_code == 0
        assert "provision" in result.output.lower()

    def test_configure_help(self):
        result = CliRunner().invoke(cli, ["configure", "--help"])
        assert result.exit_code == 0

    def test_providers_help(self):
        result = CliRunner().invoke(cli, ["providers", "--help"])
        assert result.exit_code == 0
        assert "providers" in result.output.lower()

    def test_setup_help(self):
        result = CliRunner().invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0

    def test_manage_help(self):
        result = CliRunner().invoke(cli, ["manage", "--help"])
        assert result.exit_code == 0

    def test_status_help(self):
        result = CliRunner().invoke(cli, ["status", "--help"])
        assert result.exit_code == 0

    def test_execute_help(self):
        result = CliRunner().invoke(cli, ["execute", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


class TestCLIErrorPaths:
    """Edge-case error handling."""

    def test_provision_missing_gpu_rejected(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["provision"], obj={"api": api})
        assert result.exit_code != 0

    def test_manage_missing_instance_id_rejected(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["manage", "-a", "status"], obj={"api": api})
        assert result.exit_code != 0

    def test_manage_unknown_instance_error_message(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["manage", "-i", "ghost-id", "-a", "status"], obj={"api": api}
        )
        assert result.exit_code == 0
        assert "ghost-id" in result.output
        assert "ERROR" in result.output

    def test_execute_missing_args_rejected(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["execute"], obj={"api": api})
        assert result.exit_code != 0

    def test_quote_invalid_gpu_accepted_gracefully(self):
        api = _make_api()
        api.get_runpod_quotes = AsyncMock(return_value=[])
        api.get_vastai_quotes = AsyncMock(return_value=[])
        api.get_tensordock_quotes = AsyncMock(return_value=[])
        result = CliRunner().invoke(cli, ["quote", "-g", "NOT_A_GPU"], obj={"api": api})
        assert result.exit_code in [0, 1]

    def test_status_shows_open_source_mode(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["status"], obj={"api": api})
        assert result.exit_code == 0

    def test_status_json_format(self):
        api = _make_api()
        result = CliRunner().invoke(
            cli, ["status", "--format", "json"], obj={"api": api}
        )
        assert result.exit_code == 0
        match = re.search(r"\[[\s\S]*\]", result.output)
        assert match, "No JSON array found in status output"
        parsed = json.loads(match.group(0))
        assert isinstance(parsed, list)

    def test_cleanup_no_old_instances(self):
        api = _make_api()
        result = CliRunner().invoke(cli, ["cleanup"], obj={"api": api})
        assert result.exit_code == 0
        assert "No old instances found" in result.output
