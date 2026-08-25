"""Unit tests for terradev_cli.commands._api.

Covers credential validation, provider credential extraction, usage tracking,
quote helpers, and basic resilience/secret-handling behavior.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terradev_cli.commands._api import TerradevAPI, validate_credentials


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A real TerradevAPI backed by a temp config directory."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    # Ensure clean state and skip onboarding wizard
    monkeypatch.setenv("TERRADEV_SKIP_ONBOARDING", "1")
    return TerradevAPI()


class TestValidateCredentials:
    def test_valid_runpod(self):
        assert validate_credentials("runpod", {"api_key": "rpa_abc"}) is True

    def test_missing_runpod_api_key(self):
        assert validate_credentials("runpod", {}) is False

    def test_blank_runpod_api_key(self):
        assert validate_credentials("runpod", {"api_key": "   "}) is False

    def test_valid_aws(self):
        assert validate_credentials("aws", {"api_key": "AKIA...", "secret_key": "secret"}) is True

    def test_missing_aws_secret(self):
        assert validate_credentials("aws", {"api_key": "AKIA..."}) is False

    def test_valid_azure(self):
        assert validate_credentials(
            "azure",
            {"subscription_id": "s", "tenant_id": "t", "client_id": "c", "client_secret": "cs"},
        ) is True

    def test_unknown_provider(self):
        assert validate_credentials("unknown_provider", {"api_key": "x"}) is False


class TestIsFirstTimeUser:
    def test_no_credentials_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setenv("TERRADEV_SKIP_ONBOARDING", "1")
        api = TerradevAPI()
        assert api.is_first_time_user() is True

    def test_placeholder_credentials(self, api):
        api.credentials = {"runpod": {"api_key": "your_runpod_api_key"}}
        assert api.is_first_time_user() is True

    def test_real_flat_credentials(self, api):
        api.credentials = {"runpod_api_key": "rpa_real"}
        assert api.is_first_time_user() is False

    def test_real_nested_credentials(self, api):
        api.credentials = {"runpod": {"api_key": "rpa_real"}}
        assert api.is_first_time_user() is False


class TestProviderCreds:
    def test_nested_real_creds(self, api):
        api.credentials = {"runpod": {"api_key": "real-key"}}
        creds = api._provider_creds("runpod")
        assert creds["api_key"] == "real-key"

    def test_nested_placeholder_fallback_to_flat(self, api):
        api.credentials = {
            "runpod": {"api_key": "your_runpod_api_key"},
            "runpod_api_key": "flat-key",
        }
        creds = api._provider_creds("runpod")
        assert creds["api_key"] == "flat-key"

    def test_empty_nested_uses_flat(self, api):
        api.credentials = {
            "runpod": {},
            "runpod_api_key": "flat-key",
        }
        creds = api._provider_creds("runpod")
        assert creds["api_key"] == "flat-key"

    def test_unknown_provider_returns_empty(self, api):
        assert api._provider_creds("unknown") == {}

    def test_aws_nested(self, api):
        api.credentials = {"aws": {"api_key": "AKIA", "secret_key": "wJalr..."}}
        creds = api._provider_creds("aws")
        assert creds["api_key"] == "AKIA"
        assert creds["secret_key"] == "wJalr..."


class TestUsageTracking:
    def test_load_usage_defaults(self, api):
        assert api.usage["provisions_this_month"] == 0

    def test_record_provision_increments(self, api):
        api.record_provision()
        assert api.usage["provisions_this_month"] == 1
        # File was persisted
        assert api.usage_file.exists()

    def test_check_provision_limit_open_source(self, api):
        # tier is None in open-source mode
        assert api.check_provision_limit() is True


class TestSaveAndLoadCredentials:
    def test_credentials_persist_and_decrypt(self, api):
        api.credentials = {"runpod": {"api_key": "secret-key"}}
        api.save_credentials()

        # Reload into a new API instance
        from pathlib import Path as _Path
        original = _Path.home
        _Path.home = lambda: api.config_dir.parent
        try:
            api2 = TerradevAPI()
            assert api2._provider_creds("runpod")["api_key"] == "secret-key"
        finally:
            _Path.home = original


class TestQuoteHelpers:
    @pytest.mark.asyncio
    async def test_get_provider_quotes_handles_exception(self, api):
        from terradev_cli.providers.provider_factory import ProviderFactory

        fake_provider = MagicMock()
        fake_provider.get_instance_quotes = AsyncMock(side_effect=Exception("boom"))

        with patch.object(ProviderFactory, "create_provider", return_value=fake_provider):
            quotes = await api._get_provider_quotes("runpod", "A100")
        assert quotes == []

    @pytest.mark.asyncio
    async def test_get_provider_quotes_closes_session(self, api):
        from terradev_cli.providers.provider_factory import ProviderFactory

        fake_provider = MagicMock()
        fake_provider.get_instance_quotes = AsyncMock(return_value=[
            {"price_per_hour": 1.0, "gpu_type": "A100", "region": "us-east-1", "gpu_count": 1, "instance_type": "t"}
        ])
        fake_session = MagicMock()
        fake_session.closed = False
        fake_session.close = AsyncMock()
        fake_provider.session = fake_session

        with patch.object(ProviderFactory, "create_provider", return_value=fake_provider):
            quotes = await api._get_provider_quotes("runpod", "A100")
        assert len(quotes) == 1
        fake_session.close.assert_awaited_once()
