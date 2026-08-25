"""Unit tests for the Qdrant ML service."""
import asyncio
import pytest

from terradev_cli.ml_services.qdrant_service import (
    QdrantConfig,
    QdrantService,
    create_qdrant_service_from_credentials,
    get_qdrant_setup_instructions,
)


@pytest.fixture
def svc():
    return QdrantService(QdrantConfig(url="http://localhost:6333", api_key="test"))


def test_get_setup_instructions():
    assert "Qdrant" in get_qdrant_setup_instructions()


def test_create_service_from_credentials():
    svc = create_qdrant_service_from_credentials(
        {"url": "http://qdrant:6333", "api_key": "k", "embedding_model": "BAAI/bge-large-en-v1.5"}
    )
    assert isinstance(svc, QdrantService)
    assert svc.config.url == "http://qdrant:6333"
    assert svc.config.api_key == "k"


def test_auth_headers(svc):
    assert svc._get_auth_headers()["api-key"] == "test"
    assert svc._get_auth_headers()["Content-Type"] == "application/json"


def test_ensure_session(svc, fake_aiohttp):
    session = svc._ensure_session()
    assert session is not None
    assert session.headers["api-key"] == "test"


@pytest.mark.asyncio
async def test_test_connection_success(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"result": {"collections": [{"name": "c1"}]}}, ""),
    ]
    result = await svc.test_connection()
    assert result["status"] == "connected"
    assert result["url"] == "http://localhost:6333"
    assert result["collections"] == ["c1"]


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
        (200, {"result": {"collections": []}}, ""),
    ]
    result = await svc.test_connection()
    assert result["status"] == "connected"


@pytest.mark.asyncio
async def test_list_collections(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"result": {"collections": [{"name": "c1"}, {"name": "c2"}]}}, ""),
    ]
    assert await svc.list_collections() == ["c1", "c2"]


@pytest.mark.asyncio
async def test_get_collection_info(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"result": {"name": "c1"}}, ""),
    ]
    result = await svc.get_collection_info("c1")
    assert result["result"]["name"] == "c1"


@pytest.mark.asyncio
async def test_create_collection(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"result": {"operation_id": 1}}, ""),
    ]
    result = await svc.create_collection(name="my", vector_size=128, distance="Dot")
    assert result["result"]["operation_id"] == 1


@pytest.mark.asyncio
async def test_create_collection_on_disk_quantization(svc, fake_aiohttp):
    svc.config.on_disk = True
    svc.config.quantization = "scalar"
    fake_aiohttp.responses = [
        (200, {"result": {"operation_id": 2}}, ""),
    ]
    result = await svc.create_collection(name="my2")
    assert result["result"]["operation_id"] == 2


@pytest.mark.asyncio
async def test_delete_collection(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"result": True}, ""),
    ]
    result = await svc.delete_collection("c1")
    assert result["result"] is True


@pytest.mark.asyncio
async def test_upsert_points(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"result": {"operation_id": 10}}, ""),
    ]
    points = [{"id": "p1", "vector": [0.1, 0.2], "payload": {}}]
    result = await svc.upsert_points(points, name="c1")
    assert result["result"]["operation_id"] == 10


@pytest.mark.asyncio
async def test_search(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"result": [{"id": "p1", "score": 0.9}]}, ""),
    ]
    result = await svc.search([0.1, 0.2], name="c1", limit=5, score_threshold=0.5)
    assert result["result"][0]["id"] == "p1"


@pytest.mark.asyncio
async def test_count_points(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"result": {"count": 42}}, ""),
    ]
    assert await svc.count_points("c1") == 42


@pytest.mark.asyncio
async def test_configure_rag_collection(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"result": {"operation_id": 99}}, ""),
    ]
    result = await svc.configure_rag_collection(name="rag", embedding_model="text-embedding-3-small")
    assert result["collection"] == "rag"
    assert result["vector_size"] == 1536
    assert result["embedding_model"] == "text-embedding-3-small"


def test_generate_k8s_deployment(svc):
    yaml = svc.generate_k8s_deployment(namespace="qdrant-ns")
    assert "qdrant-ns" in yaml
    assert "qdrant" in yaml


def test_generate_helm_values(svc):
    values = svc.generate_helm_values()
    assert values["qdrant"]["image"] == "qdrant/qdrant:latest"
    assert values["qdrant"]["defaultCollection"]["vectorSize"] == 1024


def test_context_manager(fake_aiohttp):
    svc = QdrantService(QdrantConfig())
    session = None

    async def run():
        nonlocal session
        async with svc as s:
            session = s.session
            assert session is not None
        assert session.closed

    asyncio.run(run())
