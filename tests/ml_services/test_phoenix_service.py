"""Unit tests for the Arize Phoenix ML service."""
import asyncio
import pytest

from terradev_cli.ml_services.phoenix_service import (
    PhoenixConfig,
    PhoenixService,
    create_phoenix_service_from_credentials,
    get_phoenix_setup_instructions,
)


@pytest.fixture
def svc():
    return PhoenixService(
        PhoenixConfig(collector_endpoint="http://phoenix:6006", api_key="test")
    )


def test_get_setup_instructions():
    assert "Phoenix" in get_phoenix_setup_instructions()


def test_create_service_from_credentials():
    svc = create_phoenix_service_from_credentials(
        {"collector_endpoint": "http://phoenix:6006", "api_key": "k", "project_name": "proj"}
    )
    assert isinstance(svc, PhoenixService)
    assert svc.config.collector_endpoint == "http://phoenix:6006"
    assert svc.config.project_name == "proj"


def test_auth_headers(svc):
    assert svc._get_auth_headers()["Authorization"] == "Bearer test"
    svc.config.api_key = None
    assert svc._get_auth_headers() == {}


def test_ensure_session(svc, fake_aiohttp):
    session = svc._ensure_session()
    assert session.headers["Authorization"] == "Bearer test"


@pytest.mark.asyncio
async def test_test_connection_success(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"data": [{"name": "default"}]}, ""),
    ]
    result = await svc.test_connection()
    assert result["status"] == "connected"
    assert result["collector_endpoint"] == "http://phoenix:6006"
    assert result["projects_found"] == 1


@pytest.mark.asyncio
async def test_test_connection_failure(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (500, {}, "boom"),
        (500, {}, "boom"),
        (500, {}, "boom"),
    ]
    result = await svc.test_connection()
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_retry_then_success(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (503, {}, ""),
        (200, {"data": []}, ""),
    ]
    result = await svc.test_connection()
    assert result["status"] == "connected"
    assert result["projects_found"] == 0


@pytest.mark.asyncio
async def test_list_projects(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"data": [{"id": "1", "name": "proj"}], "next_cursor": "c"}, ""),
    ]
    result = await svc.list_projects(cursor="c")
    assert result["data"][0]["name"] == "proj"


@pytest.mark.asyncio
async def test_list_spans(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"data": [{"span_id": "s1"}]}, ""),
    ]
    result = await svc.list_spans(project_identifier="proj", filter_condition="span_kind == 'RETRIEVER'")
    assert result["data"][0]["span_id"] == "s1"


@pytest.mark.asyncio
async def test_get_trace(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"data": [{"trace_id": "t1"}]}, ""),
    ]
    result = await svc.get_trace("t1", project_identifier="proj")
    assert result["data"][0]["trace_id"] == "t1"


def test_generate_otel_env(svc):
    env = svc.generate_otel_env(project_name="custom")
    assert env["PHOENIX_PROJECT_NAME"] == "custom"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://phoenix:6006"


def test_generate_otel_env_with_key(svc):
    env = svc.generate_otel_env()
    assert env["PHOENIX_API_KEY"] == "test"
    assert "PHOENIX_CLIENT_HEADERS" in env


def test_generate_instrumentation_snippet(svc):
    snippet = svc.generate_instrumentation_snippet("myproj")
    assert "arize-phoenix-otel" in snippet
    assert 'project_name="myproj"' in snippet


def test_generate_k8s_deployment(svc):
    yaml = svc.generate_k8s_deployment(namespace="obs")
    assert "phoenix-server" in yaml
    assert "obs" in yaml


def test_generate_helm_values(svc):
    values = svc.generate_helm_values()
    assert values["phoenix"]["image"] == "arizephoenix/phoenix:latest"
    assert values["phoenix"]["database"]["backend"] == "sqlite"


def test_generate_helm_values_postgres(svc):
    svc.config.db_backend = "postgresql"
    svc.config.postgres_dsn = "postgresql://db"
    values = svc.generate_helm_values()
    assert values["phoenix"]["database"]["dsn"] == "postgresql://db"


def test_context_manager(fake_aiohttp):
    svc = PhoenixService(PhoenixConfig())
    session = None

    async def run():
        nonlocal session
        async with svc as s:
            session = s.session
            assert session is not None
        assert session.closed

    asyncio.run(run())
