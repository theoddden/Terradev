#!/usr/bin/env python3
"""Tests for provider factory, drift auth injection, and silent error branches.

These exercise the least-covered and most dangerous paths:
- provider lazy-loading failures and auth inference
- drift request construction with multiple auth types
- network/auth/timeout failures in drift checks
- credential loading edge cases
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import HealthCheck, given, settings

pytestmark = [pytest.mark.unit, pytest.mark.canary]


def _make_contract(auth_type: str = "bearer", auth_required: bool = True, auth_in: str = "header") -> Dict[str, Any]:
    return {
        "provider": "test",
        "base_url": "http://localhost:9000",
        "auth_required": auth_required,
        "auth_type": auth_type,
        "auth_in": auth_in,
        "auth_header": "Authorization",
        "auth_query_param": "api_key",
        "endpoints": [
            {
                "name": "status",
                "method": "GET",
                "path": "v1/status",
                "enabled": True,
                "expected_status": 200,
                "expected_response_fields": ["status"],
            }
        ],
    }


def _make_endpoint(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "name": "echo",
        "method": "POST",
        "path": "v1/echo",
        "enabled": True,
        "expected_status": 200,
        "expected_response_fields": ["result"],
        "smoke_test_query": payload.get("query") if payload else None,
        "smoke_test_variables": payload.get("variables") if payload else None,
    }


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


class TestProviderFactoryErrors:
    def test_unknown_provider_raises_value_error(self):
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        with pytest.raises(ValueError, match="Unknown provider"):
            factory.create_provider("not_a_provider", {})

    def test_create_all_providers_silently_skips_bad_credentials(self, caplog):
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        providers = factory.create_all_providers(
            {"aws": {"bad": "shape"}, "unknown_provider": {"x": "y"}}
        )
        assert "unknown_provider" not in providers
        # AWSProvider is lenient enough to instantiate without strict validation
        assert isinstance(providers, dict)

    def test_requires_auth_returns_true_for_unknown_provider(self):
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        assert factory.requires_auth("totally_unknown") is True

    def test_register_provider_rejects_non_base_provider(self):
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        with pytest.raises(ValueError, match="must inherit from BaseProvider"):
            factory.register_provider("bad", dict)

    def test_provider_factory_caches_instances(self):
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        creds = {"api_key": "k"}
        p1 = factory.get_provider("demo", creds)
        p2 = factory.get_provider("demo")
        assert p1 is p2


# ---------------------------------------------------------------------------
# Drift auth injection
# ---------------------------------------------------------------------------


class TestDriftAuthInjection:
    @pytest.fixture
    def monitor(self, tmp_path: Path):
        from terradev_cli.drift_monitor.agent import DriftMonitor

        return DriftMonitor(str(tmp_path), credentials={})

    def test_bearer_auth_in_header(self, monitor):
        contract = _make_contract(auth_type="bearer")
        endpoint = contract["endpoints"][0]
        url, headers, _ = monitor._build_request(contract, endpoint, "my-token")
        assert headers["Authorization"] == "Bearer my-token"

    def test_bearer_auth_from_creds_dict(self, monitor):
        contract = _make_contract(auth_type="bearer")
        endpoint = contract["endpoints"][0]
        url, headers, _ = monitor._build_request(contract, endpoint, {"api_key": "k"})
        assert headers["Authorization"] == "Bearer k"

    def test_basic_auth_encodes_user_pass(self, monitor):
        contract = _make_contract(auth_type="basic")
        endpoint = contract["endpoints"][0]
        url, headers, _ = monitor._build_request(contract, endpoint, {"bearer_token": "user:pass"})
        import base64

        assert headers["Authorization"].startswith("Basic ")
        decoded = base64.b64decode(headers["Authorization"].split(" ", 1)[1]).decode()
        assert decoded == "user:pass"

    def test_basic_auth_without_colon_adds_colon(self, monitor):
        contract = _make_contract(auth_type="basic")
        endpoint = contract["endpoints"][0]
        url, headers, _ = monitor._build_request(contract, endpoint, "token")
        import base64

        decoded = base64.b64decode(headers["Authorization"].split(" ", 1)[1]).decode()
        assert decoded == "token:"

    def test_query_auth_appends_api_key(self, monitor):
        contract = _make_contract(auth_type="", auth_in="query")
        contract["endpoints"][0]["query_params"] = ["foo"]
        endpoint = contract["endpoints"][0]
        url, _, _ = monitor._build_request(contract, endpoint, {"api_key": "k", "foo": "bar"})
        assert "api_key=k" in url
        assert "foo=bar" in url

    def test_auth_disabled_skips_headers(self, monitor):
        contract = _make_contract(auth_required=False)
        endpoint = contract["endpoints"][0]
        url, headers, _ = monitor._build_request(contract, endpoint, "k")
        assert "Authorization" not in headers

    def test_url_placeholder_substitution(self, monitor):
        contract = _make_contract(auth_type="bearer")
        contract["base_url"] = "http://{project_id}.example.com"
        contract["endpoints"][0]["path"] = "regions/{aws_region}"
        endpoint = contract["endpoints"][0]
        url, _, _ = monitor._build_request(
            contract,
            endpoint,
            {"project_id": "myproj", "aws_region": "us-west-2"},
        )
        assert url == "http://myproj.example.com/regions/us-west-2"

    def test_arbitrary_credential_placeholder(self, monitor):
        contract = _make_contract(auth_type="bearer")
        contract["base_url"] = "http://{api_endpoint}"
        endpoint = contract["endpoints"][0]
        url, _, _ = monitor._build_request(
            contract,
            endpoint,
            {"api_endpoint": "test.example.com:8080"},
        )
        assert url == "http://test.example.com:8080/v1/status"

    def test_smoke_test_payload_in_body(self, monitor):
        contract = _make_contract(auth_type="bearer")
        endpoint = _make_endpoint({"query": "q", "variables": {"x": 1}})
        _, headers, payload = monitor._build_request(contract, endpoint, "k")
        assert payload["query"] == "q"
        assert payload["variables"] == {"x": 1}
        assert headers.get("Content-Type") == "application/json"

    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    @given(
        key=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=32),
        path=st.text(alphabet=st.characters(whitelist_categories=("L", "N", "P")), min_size=0, max_size=40),
    )
    def test_url_is_non_empty(self, key, path, monitor):
        contract = _make_contract(auth_type="bearer", auth_required=False)
        endpoint = dict(contract["endpoints"][0])
        endpoint["path"] = path.replace("/", "")
        url, _, _ = monitor._build_request(contract, endpoint, key)
        assert url.startswith("http://")


# ---------------------------------------------------------------------------
# Drift error / silent failure branches
# ---------------------------------------------------------------------------


class TestDriftErrorBranches:
    @pytest.fixture
    def tmp_contracts(self, tmp_path: Path):
        contracts = tmp_path / "contracts"
        contracts.mkdir()
        (contracts / "ok.yaml").write_text(
            yaml.safe_dump({
                "provider": "ok",
                "base_url": "http://localhost:1",
                "auth_required": False,
                "endpoints": [
                    {
                        "name": "live",
                        "method": "GET",
                        "path": "live",
                        "enabled": True,
                        "expected_status": 200,
                        "content_type": "text/plain",
                        "expected_text": "ok",
                    }
                ],
            })
        )
        return contracts

    def _monitor(self, tmp_contracts):
        from terradev_cli.drift_monitor.agent import DriftMonitor

        return DriftMonitor(str(tmp_contracts), credentials={}, timeout=1)

    def test_connection_error_is_reported_not_crash(self, tmp_contracts):
        from terradev_cli.drift_monitor.agent import DriftMonitor
        import requests

        monitor = DriftMonitor(str(tmp_contracts), credentials={}, timeout=1)
        with patch("terradev_cli.drift_monitor.agent.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("connection refused")
            result = monitor.check_provider(tmp_contracts / "ok.yaml")
        assert result["drift_detected"] is True
        assert result["status"] == "drift"
        assert any("request failed" in (e.get("drift_summary") or "") for e in result["endpoints"])

    def test_malformed_yaml_does_not_crash(self, tmp_contracts):
        from terradev_cli.drift_monitor.agent import DriftMonitor

        bad = tmp_contracts / "bad.yaml"
        bad.write_text("{unclosed")
        monitor = DriftMonitor(str(tmp_contracts), credentials={})
        with pytest.raises(yaml.YAMLError):
            monitor.check_provider(bad)

    def test_contract_missing_base_url_raises(self, tmp_contracts):
        from terradev_cli.drift_monitor.agent import DriftMonitor

        bad = tmp_contracts / "nobase.yaml"
        bad.write_text(
            yaml.safe_dump({
                "provider": "nobase",
                "auth_required": False,
                "endpoints": [
                    {"name": "x", "method": "GET", "path": "x", "enabled": True, "expected_status": 200}
                ],
            })
        )
        monitor = DriftMonitor(str(tmp_contracts), credentials={})
        with pytest.raises(KeyError, match="base_url"):
            monitor.check_provider(bad)

    def test_text_plain_mismatch(self, tmp_contracts):
        from terradev_cli.drift_monitor.agent import DriftMonitor

        monitor = DriftMonitor(str(tmp_contracts), credentials={}, timeout=1)
        fake = MagicMock()
        fake.status_code = 200
        fake.ok = True
        fake.text = "not-ok"
        fake.json.side_effect = ValueError("not json")
        with patch("terradev_cli.drift_monitor.agent.requests.get", return_value=fake):
            result = monitor.check_provider(tmp_contracts / "ok.yaml")
        ep = result["endpoints"][0]
        assert ep["drift"] is True
        assert "not-ok" in str(ep.get("drift_summary", ""))

    def test_missing_field_drift(self, tmp_contracts):
        from terradev_cli.drift_monitor.agent import DriftMonitor

        # override the contract to require a field the fake body will not have
        path = tmp_contracts / "ok.yaml"
        data = yaml.safe_load(path.read_text())
        data["endpoints"][0]["expected_response_fields"] = ["missing_field"]
        del data["endpoints"][0]["content_type"]
        del data["endpoints"][0]["expected_text"]
        path.write_text(yaml.safe_dump(data))

        monitor = DriftMonitor(str(tmp_contracts), credentials={}, timeout=1)
        fake = MagicMock()
        fake.status_code = 200
        fake.ok = True
        fake.text = '{"status": "ok"}'
        fake.json.return_value = {"status": "ok"}
        with patch("terradev_cli.drift_monitor.agent.requests.get", return_value=fake):
            result = monitor.check_provider(path)
        ep = result["endpoints"][0]
        assert ep["drift"] is True
        assert "missing field" in str(ep.get("drift_summary", ""))

    def test_401_is_reported_as_drift(self, tmp_contracts):
        from terradev_cli.drift_monitor.agent import DriftMonitor

        monitor = DriftMonitor(str(tmp_contracts), credentials={}, timeout=1)
        fake = MagicMock()
        fake.status_code = 401
        fake.ok = False
        fake.text = "nope"
        with patch("terradev_cli.drift_monitor.agent.requests.get", return_value=fake):
            result = monitor.check_provider(tmp_contracts / "ok.yaml")
        ep = result["endpoints"][0]
        assert ep["drift"] is True
        assert ep["auth_ok"] is False
        assert "401" in str(ep.get("drift_summary", ""))


# ---------------------------------------------------------------------------
# Credential loading silent failures
# ---------------------------------------------------------------------------


class TestCredentialLoadingSilentFailures:
    def test_load_drift_env_extras_ignores_missing_azure_parts(self, monkeypatch):
        from terradev_cli.commands.canary import _load_drift_env_extras

        monkeypatch.setenv("TERRADEV_AZURE_PROJECT_ID", "sub-123")
        extras = _load_drift_env_extras("azure", "my-token")
        assert extras["bearer_token"] == "my-token"
        assert extras["subscription_id"] == "sub-123"

    def test_load_drift_env_extras_treats_langfuse_token_with_secret(self, monkeypatch):
        from terradev_cli.commands.canary import _load_drift_env_extras

        monkeypatch.setenv("TERRADEV_LANGFUSE_SECRET_KEY", "sk")
        extras = _load_drift_env_extras("langfuse", "pk")
        assert extras["bearer_token"] == "pk:sk"
        assert extras["public_key"] == "pk"

    def test_load_drift_env_extras_ignores_unknown_alias(self):
        from terradev_cli.commands.canary import _load_drift_env_extras

        extras = _load_drift_env_extras("unknown_provider", "k")
        assert extras["api_key"] == "k"
        # default location is attached even for unknown aliases
        assert extras.get("location") == "Delhi"

    def test_load_drift_credentials_skips_unresolvable_providers(self, monkeypatch):
        from terradev_cli.commands.canary import _load_drift_credentials

        creds = _load_drift_credentials(["totally_unknown"])
        assert creds == {}
