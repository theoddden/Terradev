"""Tests for the gateway command group and inference provider subcommands."""

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


class TestGatewayGroup:
    """Tests for the top-level gateway group."""

    def test_group_help(self, runner, gateway_api):
        result = runner.invoke(cli, ["gateway", "--help"], obj={"api": gateway_api})
        assert result.exit_code == 0
        assert "Launch an API gateway for inference serving" in result.output
        assert "huggingface" in result.output
        assert "baseten" in result.output
        assert "siliconflow" in result.output
        assert "inferx" in result.output

    def test_serve_help(self, runner, gateway_api):
        result = runner.invoke(cli, ["gateway", "serve", "--help"], obj={"api": gateway_api})
        assert result.exit_code == 0
        assert "Start the API gateway server" in result.output

    def test_status_help(self, runner, gateway_api):
        result = runner.invoke(cli, ["gateway", "status", "--help"], obj={"api": gateway_api})
        assert result.exit_code == 0
        assert "Show the running gateway server status" in result.output


class TestProviderGroups:
    """Tests for provider subcommand groups under gateway."""

    @pytest.mark.parametrize("provider", ["huggingface", "baseten", "siliconflow", "inferx"])
    def test_provider_group_help(self, runner, gateway_api, provider):
        result = runner.invoke(cli, ["gateway", provider, "--help"], obj={"api": gateway_api})
        assert result.exit_code == 0
        assert f"{provider}" in result.output

    @pytest.mark.parametrize("provider", ["huggingface", "baseten", "siliconflow", "inferx"])
    def test_provider_configure(self, runner, gateway_api, provider):
        with patch("terradev_cli.providers.provider_factory.ProviderFactory") as MockFactory:
            MockFactory.return_value.create_provider = MagicMock(return_value=_make_mock_provider())
            with patch("terradev_cli.commands.gateway.InferenceRouter") as MockRouter:
                MockRouter.return_value.register_endpoint = MagicMock()
                extra = []
                if provider == "huggingface":
                    extra = ["--namespace", "hf-user"]
                result = runner.invoke(
                    cli,
                    ["gateway", provider, "configure", "--api-key", "test", *extra],
                    obj={"api": gateway_api},
                )
                assert result.exit_code == 0
                assert "credentials saved" in result.output.lower()
                gateway_api._save_provider_creds.assert_called_once()

    @pytest.mark.parametrize("provider", ["huggingface", "baseten", "siliconflow", "inferx"])
    def test_provider_deploy(self, runner, gateway_api, provider):
        with patch("terradev_cli.providers.provider_factory.ProviderFactory") as MockFactory:
            MockFactory.return_value.create_provider = MagicMock(return_value=_make_mock_provider())
            with patch("terradev_cli.commands.gateway.InferenceRouter") as MockRouter:
                MockRouter.return_value.register_endpoint = MagicMock()
                result = runner.invoke(
                    cli,
                    [
                        "gateway",
                        provider,
                        "deploy",
                        "--model",
                        "meta-llama/Llama-3.1-8B-Instruct",
                        "--gpu-type",
                        "A100",
                    ],
                    obj={"api": gateway_api},
                )
                assert result.exit_code == 0
                assert "deployed" in result.output.lower()
                MockRouter.return_value.register_endpoint.assert_called_once()

    @pytest.mark.parametrize("provider", ["huggingface", "baseten", "siliconflow", "inferx"])
    def test_provider_list(self, runner, gateway_api, provider):
        with patch("terradev_cli.providers.provider_factory.ProviderFactory") as MockFactory:
            MockFactory.return_value.create_provider = MagicMock(return_value=_make_mock_provider())
            result = runner.invoke(
                cli,
                ["gateway", provider, "list"],
                obj={"api": gateway_api},
            )
            assert result.exit_code == 0

    @pytest.mark.parametrize("provider", ["huggingface", "baseten", "siliconflow", "inferx"])
    def test_provider_status(self, runner, gateway_api, provider):
        with patch("terradev_cli.providers.provider_factory.ProviderFactory") as MockFactory:
            MockFactory.return_value.create_provider = MagicMock(return_value=_make_mock_provider())
            result = runner.invoke(
                cli,
                ["gateway", provider, "status", "ep-1"],
                obj={"api": gateway_api},
            )
            assert result.exit_code == 0

    @pytest.mark.parametrize("provider", ["huggingface", "baseten", "siliconflow", "inferx"])
    def test_provider_delete(self, runner, gateway_api, provider):
        with patch("terradev_cli.providers.provider_factory.ProviderFactory") as MockFactory:
            MockFactory.return_value.create_provider = MagicMock(return_value=_make_mock_provider())
            result = runner.invoke(
                cli,
                ["gateway", provider, "delete", "ep-1"],
                obj={"api": gateway_api},
            )
            assert result.exit_code == 0
            assert "deletion initiated" in result.output.lower()

    @pytest.mark.parametrize("provider", ["huggingface", "baseten", "siliconflow", "inferx"])
    def test_provider_chat(self, runner, gateway_api, provider):
        with patch("terradev_cli.providers.provider_factory.ProviderFactory") as MockFactory:
            MockFactory.return_value.create_provider = MagicMock(return_value=_make_mock_provider())
            result = runner.invoke(
                cli,
                ["gateway", provider, "chat", "--model", "m1", "--prompt", "hi"],
                obj={"api": gateway_api},
            )
            assert result.exit_code == 0
            assert "hello" in result.output.lower()

    @pytest.mark.parametrize("provider", ["huggingface", "baseten", "siliconflow", "inferx"])
    def test_provider_models(self, runner, gateway_api, provider):
        with patch("terradev_cli.providers.provider_factory.ProviderFactory") as MockFactory:
            MockFactory.return_value.create_provider = MagicMock(return_value=_make_mock_provider())
            result = runner.invoke(
                cli,
                ["gateway", provider, "models"],
                obj={"api": gateway_api},
            )
            assert result.exit_code == 0
            # HF/Baseten use list_instances; SiliconFlow/InferX have list_models
            expected_id = "m1" if provider in ("siliconflow", "inferx") else "ep-1"
            assert expected_id in result.output
