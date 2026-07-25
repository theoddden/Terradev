"""Shared fixtures for testing commands in isolation via dependency injection."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

os.environ["TERRADEV_SKIP_ONBOARDING"] = "1"


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


@pytest.fixture
def mock_api(tmp_config_dir: Path):
    """A fully-mocked TerradevAPI instance suitable for command-level DI."""
    from terradev_cli.commands._api import TerradevAPI

    api = MagicMock(spec=TerradevAPI)
    api.credentials = {"runpod": {"api_key": "test-key"}}
    api._auth_manager = None
    api.config_dir = tmp_config_dir
    api.credentials_file = tmp_config_dir / "credentials.json"
    api.usage_file = tmp_config_dir / "usage.json"

    _sample_quotes = [
        {
            "provider": "RunPod",
            "price": 1.50,
            "gpu_type": "A100-80GB",
            "region": "us-east-1",
            "availability": "on-demand",
            "gpu_count": 1,
            "instance_type": "runpod-a100-80gb",
            "memory_gb": 80,
        },
        {
            "provider": "Vast.ai",
            "price": 1.20,
            "gpu_type": "A100-80GB",
            "region": "us-west-1",
            "availability": "on-demand",
            "gpu_count": 1,
            "instance_type": "vast-a100-80gb",
            "memory_gb": 80,
        },
    ]

    api.get_runpod_quotes = AsyncMock(return_value=_sample_quotes[:1])
    api.get_vastai_quotes = AsyncMock(return_value=_sample_quotes[1:])
    api.get_aws_quotes = AsyncMock(return_value=[])
    api.get_gcp_quotes = AsyncMock(return_value=[])
    api.get_azure_quotes = AsyncMock(return_value=[])
    api.get_lambda_quotes = AsyncMock(return_value=[])
    api.get_tensordock_quotes = AsyncMock(return_value=[])
    api.get_coreweave_quotes = AsyncMock(return_value=[])
    api.get_oracle_quotes = AsyncMock(return_value=[])
    api.get_crusoe_quotes = AsyncMock(return_value=[])

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
    api.usage = {
        "instances_created": [
            {
                "id": "test-inst-abc123",
                "provider": "RunPod",
                "gpu_type": "A100-80GB",
                "price": 1.50,
                "created_at": "2099-12-31T00:00:00",
            }
        ],
        "inference_endpoints": [],
    }
    return api
