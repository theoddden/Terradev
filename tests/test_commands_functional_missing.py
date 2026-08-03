"""Targeted functional tests for previously missing top-level CLI commands.

These tests exercise `up`, `rollback`, `manifests`, `hf-space`, and a subset of
`hf-spaces`/`karpenter` subcommands with heavy dependencies mocked so they run
fast and do not perform real network or cloud I/O.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure top-level commands (up, rollback, manifests, hf-space, hf-spaces,
# karpenter) are registered on the shared `cli` group before use.
import terradev_cli.cli  # noqa: F401
from terradev_cli.commands import cli


class TestUpCommand:
    @patch("terradev_cli.core.manifest_cache.ManifestCache")
    @patch("terradev_cli.core.deployment_router.SmartDeploymentRouter")
    def test_up_provisions_job(
        self, MockRouter, MockCache, runner, mock_api, tmp_path
    ):
        """`up` should provision a job and persist a manifest."""
        router = MockRouter.return_value
        best_option = SimpleNamespace(
            provider="RunPod",
            instance_type="a100",
            price_per_hour=2.0,
            confidence=0.95,
            score=1.0,
        )
        router.recommend_deployments = AsyncMock(return_value=[best_option])
        router.execute_deployment = AsyncMock(
            return_value={"instance_id": "inst-1", "deployment_id": "dep-1"}
        )

        cache = MockCache.return_value
        cache.list_versions = MagicMock(return_value=[])
        cache.compute_dataset_hash = MagicMock(return_value="sha256:abc")
        cache.store_manifest = MagicMock(return_value=str(tmp_path / "test.v1.json"))

        result = runner.invoke(
            cli,
            [
                "up",
                "--job",
                "test",
                "--cache-dir",
                str(tmp_path),
                "--gpu-type",
                "A100",
            ],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0, result.output
        assert "provisioned successfully" in result.output
        assert cache.store_manifest.called


class TestRollbackCommand:
    @patch("terradev_cli.core.drift_detector.DriftDetector")
    def test_rollback_job_version(self, MockDetector, runner, mock_api, tmp_path):
        """`rollback <job>@<version>` should roll back a cached manifest."""
        detector = MockDetector.return_value
        detector.rollback = AsyncMock(
            return_value={
                "status": "rolled_back",
                "target_version": "v1",
                "terminated": 0,
                "recreated": 1,
            }
        )

        result = runner.invoke(
            cli,
            ["rollback", "test@v1", "--cache-dir", str(tmp_path)],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0, result.output
        assert "Rollback completed" in result.output
        assert "v1" in result.output


class TestManifestsCommand:
    def test_manifests_empty_cache(self, runner, mock_api):
        """`manifests` with an empty cache should report no cached manifests."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ["manifests", "--cache-dir", "manifests"], obj={"api": mock_api}
            )
        assert result.exit_code == 0, result.output
        assert "No cached manifests" in result.output


class TestHfSpaceCommand:
    @patch("terradev_cli.core.hf_spaces.HFSpacesDeployer")
    @patch("terradev_cli.core.hf_spaces.HFSpaceTemplates")
    def test_hf_space_deploys_with_template(
        self, MockTemplates, MockDeployer, runner, mock_api, monkeypatch
    ):
        """Top-level `hf-space` should deploy using a template."""
        monkeypatch.setenv("HF_TOKEN", "test-token")

        config = MagicMock()
        config.hardware = "cpu-basic"
        config.sdk = "gradio"
        config.private = False
        config.env_vars = {}
        config.secrets = None
        MockTemplates.get_llm_template.return_value = config

        deployer = MockDeployer.return_value
        deployer.create_space = AsyncMock(
            return_value={
                "status": "created",
                "space_url": "https://huggingface.co/spaces/test/space",
                "hardware": "cpu-basic",
                "model_id": "meta-llama/Llama-2-7b",
            }
        )

        result = runner.invoke(
            cli,
            [
                "hf-space",
                "test-space",
                "--model-id",
                "meta-llama/Llama-2-7b",
                "--template",
                "llm",
            ],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0, result.output
        assert "OK" in result.output
        assert deployer.create_space.called


class TestHfSpacesGroup:
    @pytest.fixture
    def _hf_token_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

    @patch("terradev_cli.cli_hf_spaces._hf_api")
    def test_hf_spaces_list(self, mock_hf_api, runner, mock_api, _hf_token_env):
        mock_hf_api.return_value = [
            {"id": "user/space1", "sdk": "gradio"},
            {"id": "user/space2", "sdk": "streamlit"},
        ]
        result = runner.invoke(
            cli, ["hf-spaces", "list"], obj={"api": mock_api}
        )
        assert result.exit_code == 0, result.output
        assert "user/space1" in result.output

    @patch("terradev_cli.cli_hf_spaces._hf_api")
    def test_hf_spaces_info(self, mock_hf_api, runner, mock_api, _hf_token_env):
        mock_hf_api.return_value = {
            "id": "user/space1",
            "sdk": "gradio",
            "private": False,
            "runtime": {"hardware": {"current": "cpu-basic"}},
        }
        result = runner.invoke(
            cli, ["hf-spaces", "info", "user/space1"], obj={"api": mock_api}
        )
        assert result.exit_code == 0, result.output
        assert "user/space1" in result.output

    @patch("terradev_cli.cli_hf_spaces._hf_api")
    def test_hf_spaces_restart(self, mock_hf_api, runner, mock_api, _hf_token_env):
        mock_hf_api.return_value = {"status": "restarted"}
        result = runner.invoke(
            cli, ["hf-spaces", "restart", "user/space1"], obj={"api": mock_api}
        )
        assert result.exit_code == 0, result.output
        assert "Restarting" in result.output


