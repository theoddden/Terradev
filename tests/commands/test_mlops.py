"""Tests for terradev_cli.commands.mlops."""

import pytest
from unittest.mock import MagicMock, patch

from terradev_cli.commands import cli


@pytest.fixture
def agentic_api(mock_api):
    """Mock API with agentic_serving credentials for mlops commands."""
    mock_api._provider_creds.return_value = {
        "engine": "vllm",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
    }
    return mock_api


@pytest.fixture
def fake_agentic_serving():
    """Return a fake (config, instructions) tuple and patched helpers."""
    config = MagicMock()
    config.engine = "vllm"
    config.model = "meta-llama/Llama-3.1-8B-Instruct"
    config.tensor_parallel_size = 1
    config.max_model_len = 32768
    config.gpu_memory_utilization = 0.85
    config.enable_prefix_caching = True
    config.lmcache_enabled = True
    config.lmcache_backend = "cpu"
    config.disaggregation_enabled = False
    config.ttl_min = 30
    config.ttl_max = 3600
    config.ttl_multiplier = 2.0
    return config, "Agentic serving is ready."


class TestAgenticServing:
    def _patch_agentic(self, fake):
        config, instructions = fake
        return patch("terradev_cli.commands.mlops._get_api", return_value=agentic_api), patch(
            "terradev_cli.ml_services.agentic_serving.create_agentic_serving_from_credentials",
            return_value=(config, instructions),
        ), patch(
            "terradev_cli.ml_services.agentic_serving.generate_vllm_args", return_value=["--model", config.model]
        ), patch(
            "terradev_cli.ml_services.agentic_serving.generate_sglang_args", return_value=["--model-path", config.model]
        ), patch(
            "terradev_cli.ml_services.agentic_serving.generate_lmcache_config", return_value={"local_cpu": True}
        ), patch(
            "terradev_cli.ml_services.agentic_serving.generate_lmcache_env", return_value={"LMCACHE_LOCAL_CPU": "1"}
        ), patch(
            "terradev_cli.ml_services.agentic_serving.generate_k8s_deployment", return_value="apiVersion: v1\nkind: Deployment"
        ), patch(
            "terradev_cli.ml_services.agentic_serving.generate_helm_values",
            return_value={"image": "vllm/vllm-openai", "model": config.model},
        )

    def test_show_config_text(self, runner, agentic_api, fake_agentic_serving):
        config, instructions = fake_agentic_serving
        with patch("terradev_cli.commands.mlops._get_api", return_value=agentic_api), patch(
            "terradev_cli.ml_services.agentic_serving.create_agentic_serving_from_credentials",
            return_value=(config, instructions),
        ), patch("terradev_cli.ml_services.agentic_serving.generate_vllm_args", return_value=["--model", config.model]), patch(
            "terradev_cli.ml_services.agentic_serving.generate_lmcache_config", return_value={"local_cpu": True}
        ):
            result = runner.invoke(cli, ["agent", "agentic-serving", "show-config"])
        assert result.exit_code == 0, result.output
        assert config.model in result.output

    def test_show_config_json(self, runner, agentic_api, fake_agentic_serving):
        config, instructions = fake_agentic_serving
        with patch("terradev_cli.commands.mlops._get_api", return_value=agentic_api), patch(
            "terradev_cli.ml_services.agentic_serving.create_agentic_serving_from_credentials",
            return_value=(config, instructions),
        ), patch("terradev_cli.ml_services.agentic_serving.generate_vllm_args", return_value=["--model", config.model]), patch(
            "terradev_cli.ml_services.agentic_serving.generate_lmcache_config", return_value={"local_cpu": True}
        ):
            result = runner.invoke(cli, ["agent", "agentic-serving", "show-config", "--format", "json"])
        assert result.exit_code == 0, result.output
        assert '"engine"' in result.output

    def test_launch_args(self, runner, agentic_api, fake_agentic_serving):
        config, instructions = fake_agentic_serving
        with patch("terradev_cli.commands.mlops._get_api", return_value=agentic_api), patch(
            "terradev_cli.ml_services.agentic_serving.create_agentic_serving_from_credentials",
            return_value=(config, instructions),
        ), patch("terradev_cli.ml_services.agentic_serving.generate_vllm_args", return_value=["--model", config.model]):
            result = runner.invoke(cli, ["agent", "agentic-serving", "launch-args"])
        assert result.exit_code == 0, result.output
        assert config.model in result.output

    def test_lmcache_env(self, runner, agentic_api, fake_agentic_serving):
        config, instructions = fake_agentic_serving
        with patch("terradev_cli.commands.mlops._get_api", return_value=agentic_api), patch(
            "terradev_cli.ml_services.agentic_serving.create_agentic_serving_from_credentials",
            return_value=(config, instructions),
        ), patch("terradev_cli.ml_services.agentic_serving.generate_lmcache_env", return_value={"LMCACHE_LOCAL_CPU": "1"}):
            result = runner.invoke(cli, ["agent", "agentic-serving", "lmcache-env"])
        assert result.exit_code == 0, result.output
        assert "LMCACHE_LOCAL_CPU" in result.output

    def test_k8s_manifest(self, runner, agentic_api, fake_agentic_serving):
        config, instructions = fake_agentic_serving
        with patch("terradev_cli.commands.mlops._get_api", return_value=agentic_api), patch(
            "terradev_cli.ml_services.agentic_serving.create_agentic_serving_from_credentials",
            return_value=(config, instructions),
        ), patch("terradev_cli.ml_services.agentic_serving.generate_k8s_deployment", return_value="apiVersion: v1"):
            result = runner.invoke(cli, ["agent", "agentic-serving", "k8s"])
        assert result.exit_code == 0, result.output
        assert "apiVersion" in result.output

    def test_helm_values(self, runner, agentic_api, fake_agentic_serving):
        config, instructions = fake_agentic_serving
        with patch("terradev_cli.commands.mlops._get_api", return_value=agentic_api), patch(
            "terradev_cli.ml_services.agentic_serving.create_agentic_serving_from_credentials",
            return_value=(config, instructions),
        ), patch(
            "terradev_cli.ml_services.agentic_serving.generate_helm_values",
            return_value={"image": "vllm/vllm-openai"},
        ):
            result = runner.invoke(cli, ["agent", "agentic-serving", "helm-values"])
        assert result.exit_code == 0, result.output
        assert "image" in result.output


class TestModelRouter:
    def test_model_router_configure(self, runner, mock_api):
        mock_api._provider_creds.return_value = {}
        with patch("terradev_cli.commands.mlops._get_api", return_value=mock_api):
            result = runner.invoke(
                cli,
                [
                    "model-router",
                    "configure",
                    "--strong-url",
                    "https://api.openai.com",
                    "--strong-model",
                    "gpt-4",
                    "--strong-api-key",
                    "sk-test",
                    "--weak-url",
                    "http://localhost:8000",
                    "--weak-model",
                    "llama-3.1-8b",
                    "--strategy",
                    "threshold",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Model router configured" in result.output

    def test_model_router_configure_missing_prompts(self, runner, mock_api):
        # interactive prompts are not provided; click prompts and aborts
        with patch("terradev_cli.commands.mlops._get_api", return_value=mock_api):
            result = runner.invoke(cli, ["model-router", "configure"], input="\n\n\n\n\n")
        assert result.exit_code != 0


# ── Smoke / help tests for mlops command groups ──────────────────────────────

class TestMLOpsHelp:
    @pytest.mark.parametrize("path", [
        "agent agentic-serving",
        "agent agentic-serving configure",
        "agent agentic-serving show-config",
        "agent agentic-serving launch-args",
        "agent agentic-serving lmcache-env",
        "agent agentic-serving k8s",
        "agent agentic-serving helm-values",
        "model-router",
        "model-router configure",
        "model-router test",
        "model-router classify",
        "model-router stats",
        "model-router llmd-config",
        "migrate",
        "migrate migration",
        "migrate list-workloads",
        "eval",
        "eval evaluation",
        "eval compare",
        "export",
        "import",
        "record",
        "record start",
        "record stop",
        "triggers",
        "triggers create",
        "triggers list",
        "triggers enable",
        "triggers disable",
        "triggers fire",
        "environments",
        "environments list",
        "environments promote",
        "environments approve",
        "environments history",
        "lineage",
        "lineage register",
        "lineage graph",
        "lineage production",
        "lineage show",
        "lineage diff",
        "lineage export",
        "lineage trace",
        "lineage auto",
        "lineage add-input",
        "lineage add-output",
        "lineage complete",
    ])
    def test_help(self, runner, mock_api, path):
        with patch("terradev_cli.commands.mlops._get_api", return_value=mock_api):
            result = runner.invoke(cli, path.split() + ["--help"])
        assert result.exit_code == 0, f"help failed for {path}: {result.output}"
