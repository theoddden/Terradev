"""Tests for terradev_cli.core.gateway_service.

The gateway service exposes OpenAI/Anthropic-compatible HTTP endpoints.
These tests validate configuration and route setup without bringing up a
real server.
"""

import pytest

fastapi = pytest.importorskip("fastapi")

from terradev_cli.core.gateway_service import (
    APIProvider,
    GatewayConfig,
    GatewayService,
    create_gateway_config,
)


def test_gateway_config_defaults():
    """GatewayConfig has safe default values."""
    cfg = GatewayConfig()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.enable_openai is True
    assert cfg.enable_anthropic is True
    assert cfg.enable_custom is True
    assert cfg.max_concurrent_requests == 100


def test_create_gateway_config():
    """create_gateway_config builds a populated GatewayConfig."""
    cfg = create_gateway_config(
        host="127.0.0.1",
        port=9000,
        enable_openai=False,
        enable_anthropic=False,
        max_concurrent_requests=10,
    )
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9000
    assert cfg.enable_openai is False
    assert cfg.enable_anthropic is False
    assert cfg.max_concurrent_requests == 10


def test_gateway_service_sets_up_routes():
    """GatewayService registers the expected OpenAI and management routes."""
    cfg = GatewayConfig(enable_openai=True, enable_anthropic=True, enable_custom=True)
    service = GatewayService(cfg)

    paths = {r.path for r in service.app.routes}
    assert "/health" in paths
    assert "/v1/gateway/status" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/completions" in paths
    assert "/v1/messages" in paths
    assert "/v1/custom/entry/{workflow_id}" in paths
    assert "/v1/custom/exit/{workflow_id}" in paths


def test_gateway_service_disabled_providers():
    """Disabling a provider removes its routes."""
    cfg = GatewayConfig(enable_openai=False, enable_anthropic=False, enable_custom=False)
    service = GatewayService(cfg)

    paths = {r.path for r in service.app.routes}
    assert "/v1/chat/completions" not in paths
    assert "/v1/messages" not in paths
    assert "/v1/custom/entry/{workflow_id}" not in paths


@pytest.mark.asyncio
async def test_process_openai_request():
    """The fallback OpenAI processor returns a mock chat completion."""
    cfg = GatewayConfig()
    service = GatewayService(cfg)

    request = fastapi.Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "query_string": b"",
        }
    )
    # Need real OpenAIChatRequest dataclass/pydantic model
    chat = service.app.openapi_schema
    chat_request = fastapi.Request({"type": "http"})

    # Build a request using the Pydantic model if available
    from terradev_cli.core.gateway_service import OpenAIChatRequest, OpenAIMessage

    req = OpenAIChatRequest(
        model="test-model",
        messages=[OpenAIMessage(role="user", content="hello")],
    )
    response = await service._process_openai_request(req, "req-1")
    assert response["object"] == "chat.completion"
    assert response["model"] == "test-model"
    assert response["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_process_anthropic_request():
    """The fallback Anthropic processor returns a mock message."""
    cfg = GatewayConfig()
    service = GatewayService(cfg)

    from terradev_cli.core.gateway_service import AnthropicRequest, AnthropicMessage

    req = AnthropicRequest(
        model="test-model",
        messages=[AnthropicMessage(role="user", content="hello")],
    )
    response = await service._process_anthropic_request(req, "req-1")
    assert response["type"] == "message"
    assert response["model"] == "test-model"
    assert response["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_route_request_returns_mock():
    """_route_request returns a structured mock response."""
    cfg = GatewayConfig()
    service = GatewayService(cfg)

    response = await service._route_request(
        "req-1",
        APIProvider.OPENAI,
        "test-model",
        [{"role": "user", "content": "hi"}],
        {"temperature": 0.5},
    )
    assert response["id"] == "req-1"
    assert "choices" in response
    assert "usage" in response


@pytest.mark.asyncio
async def test_stream_openai_response():
    """_stream_openai_response yields Server-Sent Events."""
    cfg = GatewayConfig()
    service = GatewayService(cfg)

    response_data = {"model": "test"}
    chunks = [c async for c in service._stream_openai_response(response_data, "req-1")]
    assert chunks
    assert "data: [DONE]" in chunks[-1]


@pytest.mark.asyncio
async def test_stream_anthropic_response():
    """_stream_anthropic_response yields SSE-formatted Anthropic events."""
    cfg = GatewayConfig()
    service = GatewayService(cfg)

    response_data = {"model": "test"}
    chunks = [c async for c in service._stream_anthropic_response(response_data, "req-1")]
    assert chunks
    assert any("event: message_start" in c for c in chunks)
