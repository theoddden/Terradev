"""Unit + contract tests for the API drift monitor.

These mock the ``requests`` library so they run offline while still proving that:
- the correct auth header/query param is built for each auth type,
- response shape (not just HTTP 200) is validated,
- missing/extra fields and unexpected status codes are reported as drift.
"""

from pathlib import Path
from unittest import mock

import pytest

from terradev_cli.drift_monitor.agent import DriftMonitor

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "terradev_cli" / "providers" / "contracts"


class _FakeResponse:
    def __init__(self, status_code, json_data, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text or str(json_data)
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._json


@pytest.fixture
def monitor(tmp_path):
    return DriftMonitor(str(CONTRACTS_DIR), credentials={}, timeout=5)


@pytest.mark.unit
def test_drift_status_validation_detects_wrong_status(monitor):
    contract = {
        "provider": "demo",
        "base_url": "https://demo.example.com",
        "auth_required": False,
        "endpoints": [
            {
                "name": "list",
                "method": "GET",
                "path": "items",
                "expected_status": 201,
                "expected_response_fields": ["items"],
            }
        ],
    }
    with mock.patch("requests.get", return_value=_FakeResponse(200, {"items": []})):
        result = monitor._check_endpoint(contract, contract["endpoints"][0], "")
    assert result["drift"] is True
    assert "expected HTTP 201, got 200" in result["drift_reasons"]


@pytest.mark.unit
def test_drift_missing_fields_detected_with_200(monitor):
    contract = {
        "provider": "demo",
        "base_url": "https://demo.example.com",
        "auth_required": False,
        "endpoints": [
            {
                "name": "list",
                "method": "GET",
                "path": "items",
                "expected_status": 200,
                "expected_response_fields": ["gpus"],
            }
        ],
    }
    with mock.patch("requests.get", return_value=_FakeResponse(200, {"items": []})):
        result = monitor._check_endpoint(contract, contract["endpoints"][0], "")
    assert result["drift"] is True
    assert "missing field(s): gpus" in result["drift_reasons"]


@pytest.mark.unit
def test_drift_empty_array_is_not_enough(monitor):
    """A 200 with an empty list should still drift if the expected wrapper key is missing."""
    contract = {
        "provider": "demo",
        "base_url": "https://demo.example.com",
        "auth_required": False,
        "endpoints": [
            {
                "name": "list",
                "method": "GET",
                "path": "items",
                "expected_status": 200,
                "expected_response_fields": ["offers"],
            }
        ],
    }
    with mock.patch("requests.get", return_value=_FakeResponse(200, [])):
        result = monitor._check_endpoint(contract, contract["endpoints"][0], "")
    assert result["drift"] is True
    assert "missing field(s): offers" in result["drift_reasons"]


@pytest.mark.unit
def test_bearer_auth_injected_in_header(monitor):
    contract = {
        "provider": "demo",
        "base_url": "https://demo.example.com",
        "auth_required": True,
        "auth_type": "Bearer",
        "auth_header": "Authorization",
        "endpoints": [
            {
                "name": "list",
                "method": "GET",
                "path": "items",
                "expected_status": 200,
                "expected_response_fields": ["items"],
            }
        ],
    }
    with mock.patch("requests.get", return_value=_FakeResponse(200, {"items": []})) as m:
        monitor._check_endpoint(contract, contract["endpoints"][0], "secret123")
    call = m.call_args
    assert call.kwargs["headers"]["Authorization"] == "Bearer secret123"


@pytest.mark.unit
def test_query_param_auth_injected(monitor):
    contract = {
        "provider": "demo",
        "base_url": "https://demo.example.com/api",
        "auth_required": True,
        "auth_in": "query",
        "auth_query_param": "api_key",
        "endpoints": [
            {
                "name": "list",
                "method": "GET",
                "path": "items",
                "expected_status": 200,
                "expected_response_fields": ["items"],
            }
        ],
    }
    with mock.patch("requests.get", return_value=_FakeResponse(200, {"items": []})) as m:
        monitor._check_endpoint(contract, contract["endpoints"][0], "secret123")
    url = call.kwargs["url"] if not hasattr(m.call_args, "args") or not m.call_args.args else m.call_args[0][0]
    assert "api_key=secret123" in m.call_args[0][0]


@pytest.mark.unit
def test_custom_header_auth_injected(monitor):
    contract = {
        "provider": "demo",
        "base_url": "https://demo.example.com",
        "auth_required": True,
        "auth_header": "X-Api-Key",
        "endpoints": [
            {
                "name": "list",
                "method": "GET",
                "path": "items",
                "expected_status": 200,
                "expected_response_fields": ["items"],
            }
        ],
    }
    with mock.patch("requests.get", return_value=_FakeResponse(200, {"items": []})) as m:
        monitor._check_endpoint(contract, contract["endpoints"][0], "secret123")
    assert m.call_args.kwargs["headers"]["X-Api-Key"] == "secret123"


@pytest.mark.unit
def test_strict_mode_flags_extra_top_level_fields(monitor):
    contract = {
        "provider": "demo",
        "base_url": "https://demo.example.com",
        "auth_required": False,
        "endpoints": [
            {
                "name": "list",
                "method": "GET",
                "path": "items",
                "expected_status": 200,
                "expected_response_fields": ["items"],
                "strict": True,
            }
        ],
    }
    with mock.patch("requests.get", return_value=_FakeResponse(200, {"items": [], "unexpected": 1})):
        result = monitor._check_endpoint(contract, contract["endpoints"][0], "")
    assert result["drift"] is True
    assert "unexpected field(s): unexpected" in result["drift_reasons"]


@pytest.mark.unit
def test_unauthenticated_public_endpoint_does_not_inject_auth(monitor):
    contract = {
        "provider": "public",
        "base_url": "https://public.example.com",
        "auth_required": False,
        "endpoints": [
            {
                "name": "list",
                "method": "GET",
                "path": "items",
                "expected_status": 200,
                "expected_response_fields": ["items"],
            }
        ],
    }
    with mock.patch("requests.get", return_value=_FakeResponse(200, {"items": []})) as m:
        monitor._check_endpoint(contract, contract["endpoints"][0], "")
    assert m.call_args.kwargs["headers"] == {}
