"""Tests for ML service drift credential loading."""

import pytest

from terradev_cli.commands.canary import _load_drift_credentials


@pytest.mark.unit
class TestMlDriftCredentials:
    """Verify TERRADEV_* tokens for new ML services are loaded correctly."""

    def test_langfuse_public_and_secret_keys_loaded(self, monkeypatch):
        monkeypatch.setenv("TERRADEV_LANGFUSE_PUBLIC_KEY", "pk-lf-abc")
        monkeypatch.setenv("TERRADEV_LANGFUSE_SECRET_KEY", "sk-lf-xyz")
        creds = _load_drift_credentials(["langfuse"])
        assert "langfuse" in creds
        assert creds["langfuse"]["public_key"] == "pk-lf-abc"
        assert creds["langfuse"]["secret_key"] == "sk-lf-xyz"
        assert creds["langfuse"]["bearer_token"] == "pk-lf-abc:sk-lf-xyz"

    def test_letta_api_key_loaded(self, monkeypatch):
        monkeypatch.setenv("TERRADEV_LETTA_API_KEY", "letta-key")
        creds = _load_drift_credentials(["letta"])
        assert "letta" in creds
        assert creds["letta"]["api_key"] == "letta-key"
        assert creds["letta"]["bearer_token"] == "letta-key"

    def test_weaviate_api_key_loaded(self, monkeypatch):
        monkeypatch.setenv("TERRADEV_WEAVIATE_API_KEY", "weaviate-key")
        creds = _load_drift_credentials(["weaviate"])
        assert "weaviate" in creds
        assert creds["weaviate"]["api_key"] == "weaviate-key"
        assert creds["weaviate"]["bearer_token"] == "weaviate-key"

    def test_missing_tokens_return_empty(self, monkeypatch):
        monkeypatch.delenv("TERRADEV_WEAVIATE_API_KEY", raising=False)
        creds = _load_drift_credentials(["weaviate"])
        assert "weaviate" not in creds
