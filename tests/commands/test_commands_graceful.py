"""Graceful-failure and idempotency tests for gateway and ml command groups."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terradev_cli.commands import cli


def _make_mock_provider():
    """Return a mock provider with all async methods needed by the gateway adapters."""
    prov = MagicMock()
    prov.credentials = {}
    prov.default_model = ""
    prov.GPU_PRICING = {
        "A100": {"instance_type": "gpu-xlarge-a100", "price": 2.0},
        "H100": {"instance_type": "gpu-2xlarge-h100", "price": 4.0},
        "A10G": {"instance_type": "gpu-large-a10g", "price": 1.0},
        "T4": {"instance_type": "gpu-medium-t4", "price": 0.5},
    }
    prov.list_instances = AsyncMock(return_value=[{"id": "ep-1", "status": "running"}])
    prov.get_instance_status = AsyncMock(return_value={"id": "ep-1", "status": "running"})
    prov.terminate_instance = AsyncMock(return_value={"success": True})
    prov.provision_instance = AsyncMock(
        return_value={
            "instance_id": "ep-1",
            "status": "provisioning",
            "endpoint_url": "http://ep.example.com",
        }
    )
    prov.deploy_model = AsyncMock(
        return_value={
            "model_id": "m1",
            "endpoint": "http://ep.example.com",
            "status": "deploying",
        }
    )
    prov.run_inference = AsyncMock(return_value={"result": "hello"})
    prov.execute_command = AsyncMock(return_value={"exit_code": 0, "output": "hello"})
    prov.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "hello"}}]}
    )
    prov.list_models = AsyncMock(return_value=[{"id": "m1"}, {"id": "m2"}])
    prov.api_endpoint = "https://api.inferx.net"
    prov._get_session = AsyncMock(
        return_value=MagicMock(
            post=AsyncMock(
                __aenter__=AsyncMock(
                    return_value=MagicMock(
                        status=200,
                        json=AsyncMock(
                            return_value={
                                "choices": [{"message": {"content": "hello"}}]
                            }
                        ),
                    )
                ),
                __aexit__=AsyncMock(return_value=False),
            )
        )
    )
    return prov


@pytest.fixture
def gateway_api(mock_api):
    """Return a mock API pre-configured for gateway provider tests."""
    mock_api._provider_creds = MagicMock(return_value={"api_key": "test-key"})
    mock_api._save_provider_creds = MagicMock()
    mock_api.usage = {}
    mock_api.save_usage = MagicMock()
    return mock_api


class TestGatewayIdempotency:
    """Tests for idempotent endpoint registration and graceful failure handling."""

    def test_register_endpoint_is_idempotent(self):
        """Re-registering the same endpoint should replace, not duplicate."""
        from terradev_cli.commands.gateway import _register_endpoint

        api = MagicMock()
        api.usage = {
            "inference_endpoints": [
                {
                    "id": "ep-1",
                    "provider": "huggingface",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "gpu_type": "A100",
                    "region": "us-east-1",
                    "url": "http://old",
                    "price": 1.0,
                    "created_at": "2024-01-01T00:00:00",
                }
            ]
        }

        with patch("terradev_cli.commands.gateway.InferenceRouter") as MockRouter:
            _register_endpoint(
                api,
                "huggingface",
                {"instance_id": "ep-1", "endpoint_url": "http://new", "price_per_hour": "2.5"},
                "meta-llama/Llama-3.1-8B-Instruct",
                "A100",
                "us-east-1",
            )

        endpoints = api.usage["inference_endpoints"]
        assert len(endpoints) == 1
        assert endpoints[0]["url"] == "http://new"
        assert endpoints[0]["price"] == "2.5"
        api.save_usage.assert_called_once()


class TestGatewayGracefulFailure:
    """Tests that provider subcommands fail gracefully on runtime errors."""

    @pytest.mark.parametrize("provider", ["huggingface", "baseten", "siliconflow", "inferx"])
    def test_provider_status_runtime_error(self, runner, gateway_api, provider):
        """A provider runtime exception should produce a clean non-zero exit."""
        prov = _make_mock_provider()
        prov.get_instance_status = AsyncMock(side_effect=RuntimeError("provider down"))

        with patch("terradev_cli.providers.provider_factory.ProviderFactory") as MockFactory:
            MockFactory.return_value.create_provider = MagicMock(return_value=prov)
            result = runner.invoke(
                cli,
                ["gateway", provider, "status", "ep-1"],
                obj={"api": gateway_api},
            )

        assert result.exit_code == 1, result.output
        assert "ERROR:" in result.output


class TestMlGracefulFailure:
    """Tests for graceful failure and non-zero exit codes across ML commands."""

    def test_ml_wandb_unconfigured_exits_nonzero(self, runner, mock_api):
        """A handled error path that prints 'ERROR:' should exit 1."""
        mock_api._provider_creds.return_value = {}
        result = runner.invoke(
            cli,
            ["ml", "wandb", "list-projects"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 1, result.output
        assert "ERROR:" in result.output

    def test_ml_wandb_missing_dependency_exits_nonzero(self, runner, mock_api):
        """Missing an optional dependency should exit 1, not traceback."""
        mock_api._provider_creds.return_value = {"api_key": "test-key"}
        with patch.dict(
            sys.modules,
            {"terradev_cli.ml_services.wandb_enhanced": None},
        ):
            result = runner.invoke(
                cli,
                ["ml", "wandb", "test"],
                obj={"api": mock_api},
            )
        assert result.exit_code == 1, result.output
        assert "ERROR:" in result.output

    def test_ml_wandb_service_runtime_error_exits_nonzero(self, runner, mock_api):
        """An unexpected runtime error in a service should exit 1."""
        mock_api._provider_creds.return_value = {"api_key": "test-key"}
        with patch(
            "terradev_cli.ml_services.wandb_enhanced.create_enhanced_wandb_service_from_credentials"
        ) as mock_create:
            mock_create.side_effect = RuntimeError("service down")
            result = runner.invoke(
                cli,
                ["ml", "wandb", "list-projects"],
                obj={"api": mock_api},
            )
        assert result.exit_code == 1, result.output
        assert "ERROR:" in result.output
