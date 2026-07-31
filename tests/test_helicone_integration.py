"""Tests for terradev_cli.integrations.helicone_integration helpers."""

import asyncio
from unittest.mock import patch

from terradev_cli.integrations import helicone_integration as hi


def test_get_credential_prompts():
    prompts = hi.get_credential_prompts()
    assert len(prompts) == 2
    assert prompts[0]["key"] == "helicone_api_key"


def test_is_eu():
    assert hi._is_eu({"helicone_eu": "true"}) is True
    assert hi._is_eu({"helicone_eu": "1"}) is True
    assert hi._is_eu({"helicone_eu": "false"}) is False
    assert hi._is_eu({}) is False


def test_gateway_and_api_url():
    eu_creds = {"helicone_eu": "true"}
    assert "eu.gateway.helicone.ai" in hi._gateway_url(eu_creds)
    assert "eu.api.helicone.ai" in hi._api_url(eu_creds)

    us_creds = {}
    assert hi._gateway_url(us_creds) == "https://gateway.helicone.ai"
    assert hi._api_url(us_creds) == "https://api.helicone.ai"


def test_auth_headers():
    creds = {"helicone_api_key": "sk-test"}
    api_headers = hi._api_auth_headers(creds)
    assert api_headers["Authorization"] == "Bearer sk-test"

    gw_headers = hi._gateway_auth_headers(creds)
    assert gw_headers["Helicone-Auth"] == "Bearer sk-test"


def test_is_configured_and_status_summary():
    creds = {"helicone_api_key": "sk-test", "helicone_eu": "true"}
    assert hi.is_configured(creds) is True
    summary = hi.get_status_summary(creds)
    assert summary["integration"] == "helicone"
    assert summary["region"] == "EU"
    assert summary["configured"] is True


def test_build_request_query():
    body = hi._build_request_query(
        limit=10,
        model="gpt-4o",
        status_gte=200,
        status_lte=299,
        created_after="2024-01-01T00:00:00Z",
        user_id="u1",
        properties={"source": "terradev"},
    )
    assert body["limit"] == 10
    assert body["filter"]["request_response_rmt"]["model"]["equals"] == "gpt-4o"
    assert body["filter"]["request_response_rmt"]["status"]["gte"] == 200
    assert body["filter"]["request_response_rmt"]["status"]["lte"] == 299


def test_generate_gateway_config():
    creds = {"helicone_api_key": "sk-test"}
    config = hi.generate_gateway_config(
        creds,
        provider_base_url="https://api.openai.com",
        cache_enabled=True,
        retry_enabled=True,
        rate_limit="1000:day",
        custom_properties={"source": "terradev"},
    )
    assert config["base_url"] == "https://gateway.helicone.ai/v1"
    assert config["headers"]["Helicone-Target-Url"] == "https://api.openai.com"
    assert config["headers"]["Helicone-Cache-Enabled"] == "true"
    assert config["headers"]["Helicone-RateLimit-Policy"] == "1000:day"
    assert config["headers"]["Helicone-Property-source"] == "terradev"


def test_generate_vllm_gateway_env():
    env = hi.generate_vllm_gateway_env({"helicone_api_key": "sk-test"})
    assert env["OPENAI_API_BASE"] == "https://gateway.helicone.ai/v1"
    assert env["HELICONE_TARGET_URL"] == "http://localhost:8000"


def test_generate_gateway_snippet():
    snippet = hi.generate_gateway_snippet({"helicone_api_key": "sk-test"}, provider="openai")
    assert "OpenAI" in snippet
    assert "gateway.helicone.ai" in snippet

    bad = hi.generate_gateway_snippet({}, provider="foo")
    assert "Unsupported" in bad


def test_get_helicone_setup_instructions():
    instructions = hi.get_helicone_setup_instructions()
    assert "Helicone Setup Instructions" in instructions


def test_query_requests_sync():
    response_data = b'{"data": [{"cost_usd": 0.01, "total_tokens": 100, "model": "gpt-4"}]}'
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = type("R", (), {"read": lambda self: response_data, "__enter__": lambda self, *a: self, "__exit__": lambda *a: None})()
        mock_urlopen.return_value = mock_resp
        result = hi._query_requests_sync({"helicone_api_key": "pk-test"}, limit=1)
    assert result["success"] is True
    assert result["data"]["data"][0]["model"] == "gpt-4"


def test_get_cost_summary():
    sample_data = [
        {"cost_usd": 0.01, "total_tokens": 10, "model": "gpt-4"},
        {"cost_usd": 0.02, "total_tokens": 20, "model": "gpt-4"},
    ]
    with patch.object(hi, "query_requests", return_value={"success": True, "data": sample_data}):
        result = asyncio.run(hi.get_cost_summary({"helicone_api_key": "pk-test"}, hours=1))
    assert result["success"] is True
    assert result["total_cost_usd"] == 0.03
    assert result["total_requests"] == 2
    assert result["by_model"]["gpt-4"]["requests"] == 2


def test_test_connection():
    with patch.object(hi, "query_requests", return_value={"success": True, "data": []}):
        result = asyncio.run(hi.test_connection({"helicone_api_key": "pk-test"}))
    assert result["status"] == "connected"
