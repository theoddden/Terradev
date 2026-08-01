"""Tests for ML integrations — config, validation, helpers (no live API calls)."""

import pytest


# ══════════════════════════════════════════════════════════════════════════
# Databricks Integration
# ══════════════════════════════════════════════════════════════════════════

class TestDatabricksIntegration:
    @pytest.fixture
    def mod(self):
        from terradev_cli.integrations import databricks_integration
        return databricks_integration

    @pytest.fixture
    def full_creds(self):
        return {
            "databricks_host": "https://dbc-abc123.cloud.databricks.com",
            "databricks_token": "dapi-test-token",
        }

    def test_get_credential_prompts_returns_list(self, mod):
        prompts = mod.get_credential_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) >= 2

    def test_credential_prompts_have_required_keys(self, mod):
        for p in mod.get_credential_prompts():
            assert "key" in p
            assert "prompt" in p
            assert "required" in p

    def test_base_url_uses_host(self, mod, full_creds):
        url = mod._base_url(full_creds)
        assert "dbc-abc123" in url

    def test_base_url_adds_https_if_missing(self, mod):
        creds = {"databricks_host": "dbc-abc123.cloud.databricks.com"}
        url = mod._base_url(creds)
        assert url.startswith("https://")

    def test_base_url_strips_trailing_slash(self, mod):
        creds = {"databricks_host": "https://dbc-abc123.cloud.databricks.com/"}
        url = mod._base_url(creds)
        assert not url.endswith("/")

    def test_auth_headers_contains_bearer(self, mod, full_creds):
        headers = mod._auth_headers(full_creds)
        assert "Authorization" in headers
        assert "dapi-test-token" in headers["Authorization"]

    def test_required_credentials_defined(self, mod):
        assert "host" in mod.REQUIRED_CREDENTIALS
        assert "token" in mod.REQUIRED_CREDENTIALS


# ══════════════════════════════════════════════════════════════════════════
# Datadog Integration
# ══════════════════════════════════════════════════════════════════════════

class TestDatadogIntegration:
    @pytest.fixture
    def mod(self):
        from terradev_cli.integrations import datadog_integration
        return datadog_integration

    @pytest.fixture
    def full_creds(self):
        return {
            "datadog_api_key": "api-key-123",
            "datadog_app_key": "app-key-456",
            "datadog_site": "datadoghq.com",
        }

    @pytest.fixture
    def empty_creds(self):
        return {}

    def test_get_credential_prompts_returns_list(self, mod):
        prompts = mod.get_credential_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) >= 2

    def test_is_configured_with_both_keys(self, mod, full_creds):
        assert mod.is_configured(full_creds) is True

    def test_is_configured_missing_api_key(self, mod):
        assert mod.is_configured({"datadog_app_key": "x"}) is False

    def test_is_configured_missing_app_key(self, mod):
        assert mod.is_configured({"datadog_api_key": "x"}) is False

    def test_is_configured_empty(self, mod, empty_creds):
        assert mod.is_configured(empty_creds) is False

    def test_get_site_default(self, mod, empty_creds):
        assert mod._get_site(empty_creds) == "datadoghq.com"

    def test_get_site_custom(self, mod):
        assert mod._get_site({"datadog_site": "datadoghq.eu"}) == "datadoghq.eu"

    def test_get_site_strips_whitespace(self, mod):
        assert mod._get_site({"datadog_site": "  datadoghq.com  "}) == "datadoghq.com"

    def test_base_url_format(self, mod, full_creds):
        url = mod._base_url(full_creds)
        assert url == "https://api.datadoghq.com"

    def test_base_url_eu_site(self, mod):
        url = mod._base_url({"datadog_site": "datadoghq.eu"})
        assert url == "https://api.datadoghq.eu"

    def test_auth_headers_has_api_key(self, mod, full_creds):
        headers = mod._auth_headers(full_creds)
        assert headers["DD-API-KEY"] == "api-key-123"
        assert headers["DD-APPLICATION-KEY"] == "app-key-456"

    def test_auth_headers_empty_creds(self, mod, empty_creds):
        headers = mod._auth_headers(empty_creds)
        assert headers["DD-API-KEY"] == ""

    def test_get_status_summary_configured(self, mod, full_creds):
        summary = mod.get_status_summary(full_creds)
        assert summary["configured"] is True
        assert summary["integration"] == "datadog"
        assert summary["api_key_set"] is True
        assert summary["app_key_set"] is True

    def test_get_status_summary_unconfigured(self, mod, empty_creds):
        summary = mod.get_status_summary(empty_creds)
        assert summary["configured"] is False

    def test_metric_catalog_has_entries(self, mod):
        assert len(mod.METRIC_CATALOG) > 0

    def test_metric_catalog_prefix(self, mod):
        for key in mod.METRIC_CATALOG:
            assert key.startswith("terradev.")

    def test_metric_catalog_has_type_field(self, mod):
        for key, meta in mod.METRIC_CATALOG.items():
            assert "type" in meta, f"Missing 'type' in {key}"

    def test_required_credentials_defined(self, mod):
        assert "api_key" in mod.REQUIRED_CREDENTIALS
        assert "app_key" in mod.REQUIRED_CREDENTIALS

    def test_optional_credentials_defined(self, mod):
        assert "site" in mod.OPTIONAL_CREDENTIALS


# ══════════════════════════════════════════════════════════════════════════
# Helicone Integration
# ══════════════════════════════════════════════════════════════════════════

class TestHeliconeIntegration:
    @pytest.fixture
    def mod(self):
        from terradev_cli.integrations import helicone_integration
        return helicone_integration

    @pytest.fixture
    def full_creds(self):
        return {"helicone_api_key": "sk-helicone-test-key"}

    @pytest.fixture
    def eu_creds(self):
        return {"helicone_api_key": "eu-helicone-test", "helicone_eu": "true"}

    def test_get_credential_prompts(self, mod):
        prompts = mod.get_credential_prompts()
        assert len(prompts) >= 1
        keys = [p["key"] for p in prompts]
        assert "helicone_api_key" in keys

    def test_is_eu_false_by_default(self, mod, full_creds):
        assert mod._is_eu(full_creds) is False

    def test_is_eu_true_with_true_string(self, mod, eu_creds):
        assert mod._is_eu(eu_creds) is True

    def test_is_eu_true_with_1(self, mod):
        assert mod._is_eu({"helicone_eu": "1"}) is True

    def test_is_eu_true_with_yes(self, mod):
        assert mod._is_eu({"helicone_eu": "yes"}) is True

    def test_gateway_url_default(self, mod, full_creds):
        url = mod._gateway_url(full_creds)
        assert url == "https://gateway.helicone.ai"

    def test_gateway_url_eu(self, mod, eu_creds):
        url = mod._gateway_url(eu_creds)
        assert "eu" in url

    def test_api_url_default(self, mod, full_creds):
        url = mod._api_url(full_creds)
        assert url == "https://api.helicone.ai"

    def test_api_url_eu(self, mod, eu_creds):
        url = mod._api_url(eu_creds)
        assert "eu" in url

    def test_api_auth_headers_bearer(self, mod, full_creds):
        headers = mod._api_auth_headers(full_creds)
        assert "Authorization" in headers
        assert "sk-helicone-test-key" in headers["Authorization"]

    def test_required_credentials_defined(self, mod):
        assert "api_key" in mod.REQUIRED_CREDENTIALS

    def test_optional_credentials_defined(self, mod):
        assert "eu" in mod.OPTIONAL_CREDENTIALS


# ══════════════════════════════════════════════════════════════════════════
# WandB Integration
# ══════════════════════════════════════════════════════════════════════════

class TestWandbIntegration:
    @pytest.fixture
    def mod(self):
        from terradev_cli.integrations import wandb_integration
        return wandb_integration

    @pytest.fixture
    def full_creds(self):
        return {
            "wandb_api_key": "abc123key",
            "wandb_entity": "my-team",
            "wandb_project": "my-project",
        }

    @pytest.fixture
    def empty_creds(self):
        return {}

    def test_get_credential_prompts(self, mod):
        prompts = mod.get_credential_prompts()
        keys = [p["key"] for p in prompts]
        assert "wandb_api_key" in keys

    def test_is_configured_with_key(self, mod, full_creds):
        assert mod.is_configured(full_creds) is True

    def test_is_configured_without_key(self, mod, empty_creds):
        assert mod.is_configured(empty_creds) is False

    def test_build_env_vars_sets_api_key(self, mod, full_creds):
        env = mod.build_env_vars(full_creds)
        assert env["WANDB_API_KEY"] == "abc123key"

    def test_build_env_vars_sets_entity(self, mod, full_creds):
        env = mod.build_env_vars(full_creds)
        assert env["WANDB_ENTITY"] == "my-team"

    def test_build_env_vars_sets_project(self, mod, full_creds):
        env = mod.build_env_vars(full_creds)
        assert env["WANDB_PROJECT"] == "my-project"

    def test_build_env_vars_default_project(self, mod):
        env = mod.build_env_vars({"wandb_api_key": "key"})
        assert env["WANDB_PROJECT"] == "terradev"

    def test_build_env_vars_no_api_key_omits_key(self, mod, empty_creds):
        env = mod.build_env_vars(empty_creds)
        assert "WANDB_API_KEY" not in env

    def test_build_env_vars_base_url(self, mod):
        env = mod.build_env_vars({"wandb_api_key": "k", "wandb_base_url": "https://myserver.com"})
        assert env["WANDB_BASE_URL"] == "https://myserver.com"

    def test_build_env_vars_no_base_url_omitted(self, mod, full_creds):
        env = mod.build_env_vars(full_creds)
        assert "WANDB_BASE_URL" not in env

    def test_build_run_config_has_required_fields(self, mod):
        cfg = mod.build_run_config(
            gpu_type="H100", provider="runpod",
            price_per_hour=2.5, region="us-east-1", instance_id="inst-123"
        )
        assert cfg["terradev_gpu_type"] == "H100"
        assert cfg["terradev_provider"] == "runpod"
        assert cfg["terradev_price_per_hour"] == 2.5
        assert cfg["terradev_region"] == "us-east-1"
        assert cfg["terradev_instance_id"] == "inst-123"

    def test_build_run_config_merges_extra(self, mod):
        cfg = mod.build_run_config(
            "A100", "vastai", 1.5, "eu-west-1", "i-abc",
            extra={"batch_size": 32}
        )
        assert cfg["batch_size"] == 32

    def test_generate_setup_script_is_bash(self, mod, full_creds):
        script = mod.generate_setup_script(full_creds)
        assert script.startswith("#!/bin/bash")

    def test_generate_setup_script_exports_api_key(self, mod, full_creds):
        script = mod.generate_setup_script(full_creds)
        assert "WANDB_API_KEY" in script
        assert "abc123key" in script

    def test_generate_setup_script_no_key_no_export(self, mod, empty_creds):
        script = mod.generate_setup_script(empty_creds)
        # No `export WANDB_API_KEY=...` line (variable reference in body is ok)
        assert 'export WANDB_API_KEY=' not in script

    def test_get_status_summary_configured(self, mod, full_creds):
        s = mod.get_status_summary(full_creds)
        assert s["configured"] is True
        assert s["integration"] == "wandb"
        assert s["entity"] == "my-team"
        assert s["project"] == "my-project"

    def test_get_status_summary_unconfigured(self, mod, empty_creds):
        s = mod.get_status_summary(empty_creds)
        assert s["configured"] is False
        assert s["project"] == "terradev"

    def test_get_status_summary_self_hosted_flag(self, mod):
        s = mod.get_status_summary({"wandb_api_key": "k", "wandb_base_url": "https://myserver.com"})
        assert s["self_hosted"] is True

    def test_get_status_summary_not_self_hosted_by_default(self, mod, full_creds):
        s = mod.get_status_summary(full_creds)
        assert s["self_hosted"] is False

    def test_required_credentials_defined(self, mod):
        assert "api_key" in mod.REQUIRED_CREDENTIALS


# ══════════════════════════════════════════════════════════════════════════
# Prometheus Integration
# ══════════════════════════════════════════════════════════════════════════

class TestPrometheusIntegration:
    @pytest.fixture
    def mod(self):
        from terradev_cli.integrations import prometheus_integration
        return prometheus_integration

    @pytest.fixture
    def full_creds(self):
        return {
            "prometheus_pushgateway_url": "http://pushgateway:9091",
            "prometheus_username": "admin",
            "prometheus_password": "secret",
        }

    @pytest.fixture
    def empty_creds(self):
        return {}

    def test_get_credential_prompts_returns_list(self, mod):
        prompts = mod.get_credential_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) >= 1

    def test_credential_prompts_have_required_keys(self, mod):
        for p in mod.get_credential_prompts():
            assert "key" in p
            assert "prompt" in p
            assert "required" in p

    def test_build_metric_payload_format(self, mod):
        payload = mod.build_metric_payload(
            "terradev_provisions_total",
            1,
            {"provider": "runpod", "gpu_type": "RTX4090", "region": "us-east-1"},
        )
        assert "# HELP terradev_provisions_total" in payload
        assert "# TYPE terradev_provisions_total counter" in payload
        assert 'provider="runpod"' in payload
        assert 'gpu_type="RTX4090"' in payload
        assert "terradev_provisions_total" in payload

    def test_build_provision_metrics_has_counter_and_gauge(self, mod):
        payload = mod.build_provision_metrics(
            "runpod", "RTX4090", "us-east-1", "pod-123", 1.25
        )
        assert "terradev_provisions_total" in payload
        assert "terradev_gpu_cost_per_hour" in payload
        assert 'instance_id="pod-123"' in payload
        assert "1.25" in payload

    def test_build_terminate_metrics(self, mod):
        payload = mod.build_terminate_metrics("runpod", "pod-123", 4.5, 3600)
        assert "terradev_total_cost_usd" in payload
        assert "terradev_provision_duration_seconds" in payload
        assert "4.5" in payload
        assert "3600" in payload

    def test_get_push_url(self, mod, full_creds):
        url = mod.get_push_url(full_creds)
        assert url == "http://pushgateway:9091/metrics/job/terradev"

    def test_get_push_url_custom_job(self, mod, full_creds):
        url = mod.get_push_url(full_creds, job="myjob")
        assert url == "http://pushgateway:9091/metrics/job/myjob"

    def test_get_auth_headers_with_basic_auth(self, mod, full_creds):
        headers = mod.get_auth_headers(full_creds)
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")

    def test_get_auth_headers_no_auth(self, mod, empty_creds):
        headers = mod.get_auth_headers(empty_creds)
        assert "Authorization" not in headers

    def test_is_configured_with_url(self, mod, full_creds):
        assert mod.is_configured(full_creds) is True

    def test_is_configured_without_url(self, mod, empty_creds):
        assert mod.is_configured(empty_creds) is False

    def test_get_status_summary_configured(self, mod, full_creds):
        s = mod.get_status_summary(full_creds)
        assert s["configured"] is True
        assert s["integration"] == "prometheus"
        assert s["pushgateway_url"] == "http://pushgateway:9091"
        assert s["auth_enabled"] is True

    def test_get_status_summary_unconfigured(self, mod, empty_creds):
        s = mod.get_status_summary(empty_creds)
        assert s["configured"] is False
        assert s["pushgateway_url"] == "(not set)"

    def test_generate_scrape_config(self, mod):
        cfg = mod.generate_scrape_config()
        assert "- job_name: 'terradev'" in cfg
        assert "honor_labels: true" in cfg

    def test_generate_grafana_dashboard_json(self, mod):
        dashboard = mod.generate_grafana_dashboard_json()
        assert "dashboard" in dashboard
        assert dashboard["dashboard"]["title"].startswith("Terradev")
        assert len(dashboard["dashboard"]["panels"]) > 0

    def test_push_metrics_success(self, mod, full_creds, monkeypatch):
        class _FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a, **k):
                return None

        def _fake_urlopen(req, *a, **k):
            assert req.get_method() == "POST"
            assert "text/plain" in req.headers["Content-type"]
            return _FakeResponse()

        monkeypatch.setattr(
            "urllib.request.urlopen",
            _fake_urlopen,
        )

        result = mod.push_metrics(full_creds, "test payload\n")
        assert result["success"] is True
        assert result["status_code"] == 200

    def test_push_metrics_connection_failure(self, mod, full_creds, monkeypatch):
        import urllib.error

        def _fake_urlopen(*a, **k):
            raise urllib.error.URLError(" refused")

        monkeypatch.setattr(
            "urllib.request.urlopen",
            _fake_urlopen,
        )

        result = mod.push_metrics(full_creds, "test payload\n")
        assert result["success"] is False
        assert "Connection failed" in result["error"]

    def test_required_credentials_defined(self, mod):
        assert "pushgateway_url" in mod.REQUIRED_CREDENTIALS

    def test_optional_credentials_defined(self, mod):
        assert "username" in mod.OPTIONAL_CREDENTIALS
        assert "password" in mod.OPTIONAL_CREDENTIALS
