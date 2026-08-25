"""Tests for terradev_cli.integrations.datadog_integration helpers."""

from unittest.mock import MagicMock, patch

from terradev_cli.integrations import datadog_integration as dd


def test_get_credential_prompts():
    prompts = dd.get_credential_prompts()
    assert len(prompts) == 3
    assert all("key" in p and "prompt" in p for p in prompts)


def test_base_url_and_site():
    creds = {"datadog_site": "datadoghq.eu"}
    assert dd._get_site(creds) == "datadoghq.eu"
    assert dd._base_url(creds) == "https://api.datadoghq.eu"

    creds_default = {}
    assert dd._get_site(creds_default) == "datadoghq.com"


def test_auth_headers():
    creds = {"datadog_api_key": "api", "datadog_app_key": "app"}
    headers = dd._auth_headers(creds)
    assert headers["DD-API-KEY"] == "api"
    assert headers["DD-APPLICATION-KEY"] == "app"


def test_is_configured():
    assert dd.is_configured({"datadog_api_key": "x", "datadog_app_key": "y"}) is True
    assert dd.is_configured({"datadog_api_key": "x"}) is False
    assert dd.is_configured({}) is False


def test_get_status_summary():
    creds = {"datadog_api_key": "x", "datadog_site": "us3.datadoghq.com"}
    summary = dd.get_status_summary(creds)
    assert summary["integration"] == "datadog"
    assert summary["configured"] is False
    assert summary["site"] == "us3.datadoghq.com"
    assert summary["api_key_set"] is True


def test_build_series():
    metrics = [
        {"metric": "terradev.gpu.cost_per_hour", "value": 1.5, "tags": {"provider": "runpod"}}
    ]
    series = dd._build_series(metrics)
    assert "series" in series
    assert len(series["series"]) == 1
    assert series["series"][0]["metric"] == "terradev.gpu.cost_per_hour"
    assert series["series"][0]["points"][0]["value"] == 1.5


def test_build_event():
    event = dd._build_event("provision", "created instance", "success", {"gpu": "A100"})
    assert event["title"] == "provision"
    assert event["alert_type"] == "success"
    assert "source:terradev" in event["tags"]


def test_submit_metrics_sync():
    mock_resp = MagicMock()
    mock_resp.status = 202
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = dd.submit_metrics_sync(
            {"datadog_api_key": "k", "datadog_app_key": "a"},
            [{"metric": "terradev.gpu.cost_per_hour", "value": 1.5}],
        )
    assert result["success"] is True
    assert result["status_code"] == 202
    assert result["metrics_sent"] == 1


def test_send_event_sync():
    mock_resp = MagicMock()
    mock_resp.status = 202
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = dd.send_event_sync(
            {"datadog_api_key": "k", "datadog_app_key": "a"},
            "test",
            "event",
        )
    assert result["success"] is True
    assert result["status_code"] == 202
