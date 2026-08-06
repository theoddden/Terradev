"""Tests for the `terradev database weaviate` command group."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from terradev_cli.commands import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_api():
    api = MagicMock()
    api.is_first_time_user.return_value = False
    return api


class TestWeaviateHelp:
    def test_weaviate_group_help(self, runner, mock_api):
        result = runner.invoke(cli, ["database", "weaviate", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "create-collection" in result.output

    def test_weaviate_create_collection_help(self, runner, mock_api):
        result = runner.invoke(cli, ["database", "weaviate", "create-collection", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--name" in result.output

    def test_weaviate_hybrid_search_help(self, runner, mock_api):
        result = runner.invoke(cli, ["database", "weaviate", "hybrid-search", "--help"], obj={"api": mock_api})
        assert result.exit_code == 0
        assert "--alpha" in result.output


class TestWeaviateAdapter:
    def test_adapter_registered(self):
        from terradev_cli.core.adapters.registry import REGISTRY

        adapter = REGISTRY.resolve("vector_store", "weaviate", {"environment": "local"})
        assert adapter.name == "weaviate"

    @pytest.mark.anyio
    async def test_adapter_health_missing_client(self):
        from terradev_cli.core.adapters.registry import REGISTRY

        adapter = REGISTRY.resolve("vector_store", "weaviate", {"environment": "local"})
        health = await adapter.health()
        assert not health.healthy
        assert "weaviate-client" in health.message
