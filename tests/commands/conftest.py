"""Shared fixtures for testing commands in isolation via dependency injection."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

os.environ["TERRADEV_SKIP_ONBOARDING"] = "1"

# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

_RUNPOD_QUOTE = {
    "provider": "RunPod",
    "price": 1.50,
    "gpu_type": "A100-80GB",
    "region": "us-east-1",
    "availability": "on-demand",
    "gpu_count": 1,
    "instance_type": "runpod-a100-80gb",
    "memory_gb": 80,
}

_VASTAI_QUOTE = {
    "provider": "Vast.ai",
    "price": 1.20,
    "gpu_type": "A100-80GB",
    "region": "us-west-1",
    "availability": "on-demand",
    "gpu_count": 1,
    "instance_type": "vast-a100-80gb",
    "memory_gb": 80,
}

_SPOT_QUOTE = {
    "provider": "TensorDock",
    "price": 0.80,
    "gpu_type": "A100-80GB",
    "region": "us-east-1",
    "availability": "spot",
    "gpu_count": 1,
    "instance_type": "tensordock-a100-spot",
    "memory_gb": 80,
}

FRESH_INSTANCE = {
    "id": "test-inst-abc123",
    "provider": "RunPod",
    "gpu_type": "A100-80GB",
    "price": 1.50,
    "region": "us-east-1",
    "created_at": "2099-12-31T00:00:00",
}

OLD_INSTANCE = {
    "id": "old-inst-xyz789",
    "provider": "Vast.ai",
    "gpu_type": "A100-80GB",
    "price": 1.20,
    "region": "us-west-1",
    "created_at": "2020-01-01T00:00:00",
}


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Isolated Click test runner."""
    return CliRunner()


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """A fresh temporary config directory for each test."""
    config = tmp_path / ".terradev"
    config.mkdir()
    return config


def _build_mock_api(tmp_config_dir: Path, instances: list) -> MagicMock:
    """Build a fully-mocked TerradevAPI with the given instance list."""
    from terradev_cli.commands._api import TerradevAPI

    api = MagicMock(spec=TerradevAPI)
    api.credentials = {"runpod": {"api_key": "test-key"}}
    api._auth_manager = None
    api.config_dir = tmp_config_dir
    api.credentials_file = tmp_config_dir / "credentials.json"
    api.usage_file = tmp_config_dir / "usage.json"

    api.get_runpod_quotes = AsyncMock(return_value=[_RUNPOD_QUOTE])
    api.get_vastai_quotes = AsyncMock(return_value=[_VASTAI_QUOTE])
    api.get_aws_quotes = AsyncMock(return_value=[])
    api.get_gcp_quotes = AsyncMock(return_value=[])
    api.get_azure_quotes = AsyncMock(return_value=[])
    api.get_tensordock_quotes = AsyncMock(return_value=[_SPOT_QUOTE])
    api.get_oracle_quotes = AsyncMock(return_value=[])
    api.get_crusoe_quotes = AsyncMock(return_value=[])
    api.get_alibaba_quotes = AsyncMock(return_value=[])
    api.get_baseten_quotes = AsyncMock(return_value=[])
    api.get_digitalocean_quotes = AsyncMock(return_value=[])
    api.get_e2enetworks_quotes = AsyncMock(return_value=[])
    api.get_hetzner_quotes = AsyncMock(return_value=[])
    api.get_huggingface_quotes = AsyncMock(return_value=[])
    api.get_hyperstack_quotes = AsyncMock(return_value=[])
    api.get_inferx_quotes = AsyncMock(return_value=[])
    api.get_latitude_quotes = AsyncMock(return_value=[])
    api.get_ovhcloud_quotes = AsyncMock(return_value=[])
    api.get_siliconflow_quotes = AsyncMock(return_value=[])
    api.get_yottalabs_quotes = AsyncMock(return_value=[])

    api.provision_instance = AsyncMock(
        return_value={
            "instance_id": "test-inst-abc123",
            "provider": "RunPod",
            "gpu_type": "A100-80GB",
            "status": "running",
            "ip": "10.0.0.1",
        }
    )
    api.get_instance_status = AsyncMock(
        return_value={
            "status": "running",
            "gpu_utilization": 72,
            "instance_id": "test-inst-abc123",
        }
    )

    api.is_first_time_user = MagicMock(return_value=False)
    api.load_credentials = MagicMock()
    api.save_credentials = MagicMock()
    api.load_usage = MagicMock(return_value={"provisions_this_month": 0})
    api.save_usage = MagicMock()
    api.check_provision_limit = MagicMock(return_value=True)
    api.record_provision = MagicMock()
    api._provider_creds = MagicMock(return_value={"api_key": "test-key"})
    api.usage = {
        "instances_created": list(instances),
        "inference_endpoints": [],
        "provisions_this_month": 0,
    }
    return api


@pytest.fixture
def mock_api(tmp_config_dir: Path):
    """TerradevAPI mock with one fresh (future) instance."""
    return _build_mock_api(tmp_config_dir, [FRESH_INSTANCE])


@pytest.fixture
def mock_api_empty(tmp_config_dir: Path):
    """TerradevAPI mock with no tracked instances."""
    return _build_mock_api(tmp_config_dir, [])


@pytest.fixture
def mock_api_old_instance(tmp_config_dir: Path):
    """TerradevAPI mock with one old instance (>30 days) for cleanup tests."""
    return _build_mock_api(tmp_config_dir, [OLD_INSTANCE])


@pytest.fixture
def mock_api_mixed_instances(tmp_config_dir: Path):
    """TerradevAPI mock with one fresh and one old instance."""
    return _build_mock_api(tmp_config_dir, [FRESH_INSTANCE, OLD_INSTANCE])


# ---------------------------------------------------------------------------
# Provider-factory mock helpers
# ---------------------------------------------------------------------------


def make_mock_provider() -> MagicMock:
    """Return an async mock that satisfies the provider interface."""
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


@pytest.fixture
def patch_registry():
    """Patch ProviderRegistry so it returns a fixed, predictable provider ranking."""
    with patch("terradev_cli.providers.registry.ProviderRegistry") as MockRegistry:
        instance = MagicMock()
        instance.ranked_providers.return_value = [
            "runpod",
            "vastai",
            "aws",
            "gcp",
            "azure",
            "tensordock",
            "oracle",
            "crusoe",
        ]
        MockRegistry.return_value = instance
        yield MockRegistry


@pytest.fixture
def patch_rate_limiter():
    """Replace the global rate limiter with a synchronous passthrough."""
    from terradev_cli.core.rate_limiter import get_rate_limiter

    original = get_rate_limiter("global")

    class _PassThrough:
        async def execute_with_rate_limit(self, provider, fn, *args, **kwargs):
            return await fn(*args, **kwargs)

    get_rate_limiter.cache_clear()
    with patch("terradev_cli.core.rate_limiter.get_rate_limiter", return_value=_PassThrough()):
        yield
    get_rate_limiter.cache_clear()
    get_rate_limiter("global")  # warm cache back to original
