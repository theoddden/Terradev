"""Tests for terradev_cli.integrations.prometheus_integration helpers."""

from unittest.mock import MagicMock, patch

from terradev_cli.integrations import prometheus_integration as prom


def test_get_credential_prompts():
    prompts = prom.get_credential_prompts()
    assert len(prompts) == 3
    assert prompts[0]["key"] == "prometheus_pushgateway_url"


def test_build_metric_payload():
    payload = prom.build_metric_payload(
        "terradev_gpu_cost_per_hour", 1.5, {"provider": "runpod", "gpu_type": "A100"}
    )
    assert "# HELP terradev_gpu_cost_per_hour" in payload
    assert "# TYPE terradev_gpu_cost_per_hour gauge" in payload
    assert 'provider="runpod"' in payload
    assert "1.5" in payload


def test_build_provision_metrics():
    payload = prom.build_provision_metrics("runpod", "A100", "us-east", "i-1", 1.5)
    assert "terradev_provisions_total" in payload
    assert "terradev_gpu_cost_per_hour" in payload
    assert "i-1" in payload


def test_build_terminate_metrics():
    payload = prom.build_terminate_metrics("runpod", "i-1", 10.0, 3600.0)
    assert "terradev_total_cost_usd" in payload
    assert "terradev_provision_duration_seconds" in payload


def test_get_push_url_and_auth_headers():
    creds = {"prometheus_pushgateway_url": "http://pushgateway:9091"}
    assert prom.get_push_url(creds) == "http://pushgateway:9091/metrics/job/terradev"

    auth_creds = {"prometheus_username": "u", "prometheus_password": "p"}
    headers = prom.get_auth_headers(auth_creds)
    assert headers["Authorization"].startswith("Basic ")


def test_push_metrics():
    mock_resp = MagicMock()
    mock_resp.status = 202
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = prom.push_metrics(
            {"prometheus_pushgateway_url": "http://pg:9091"}, "payload"
        )
    assert result["success"] is True
    assert result["status_code"] == 202


def test_generate_scrape_config():
    config = prom.generate_scrape_config()
    assert "job_name: 'terradev'" in config
    assert "pushgateway:9091" in config


def test_generate_grafana_dashboard_json():
    dashboard = prom.generate_grafana_dashboard_json()
    assert dashboard["dashboard"]["title"] == "Terradev GPU Cost Dashboard"
    assert len(dashboard["dashboard"]["panels"]) == 5


def test_is_configured_and_status_summary():
    creds = {"prometheus_pushgateway_url": "http://pg:9091", "prometheus_username": "u"}
    assert prom.is_configured(creds) is True
    summary = prom.get_status_summary(creds)
    assert summary["integration"] == "prometheus"
    assert summary["configured"] is True
    assert summary["auth_enabled"] is True
