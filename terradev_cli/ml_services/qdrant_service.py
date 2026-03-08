#!/usr/bin/env python3
"""
Qdrant Vector Database Service Integration for Terradev
RAG Infrastructure — BYOAPI pattern. License: Apache 2.0.

API notes (verified from api.qdrant.tech):
  - REST on 6333, gRPC on 6334. MCP tools use REST.
  - Auth: api-key header (NOT Authorization: Bearer) for Qdrant Cloud
  - PUT /collections/{name} with {"vectors": {"size": N, "distance": "Cosine"}}
  - PUT /collections/{name}/points with {"points": [...]}
  - POST /collections/{name}/points/search with {"vector": [...], "limit": N}
"""

import logging
import asyncio
import random
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 10.0
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

EMBEDDING_DIMENSIONS: Dict[str, int] = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text-v1.5": 768,
}


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    grpc_port: int = 6334
    prefer_grpc: bool = False
    default_collection: str = "terradev-embeddings"
    vector_size: int = 1024
    distance: str = "Cosine"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    image: str = "qdrant/qdrant:latest"
    replicas: int = 1
    storage_size: str = "100Gi"
    storage_class: Optional[str] = None
    port: int = 6333
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    quantization: Optional[str] = None
    on_disk: bool = False


class QdrantService:
    """Qdrant vector DB integration for RAG infrastructure"""

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["api-key"] = self.config.api_key
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def _request(
        self, method: str, path: str, *,
        params: Optional[Dict] = None, json_body: Optional[Any] = None,
        timeout: float = 30, retries: int = _MAX_RETRIES,
    ) -> Any:
        session = self._ensure_session()
        url = f"{self.config.url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                async with session.request(
                    method, url, params=params, json=json_body,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status in (200, 201):
                        return await resp.json()
                    if resp.status in _RETRYABLE_STATUSES and attempt < retries - 1:
                        wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                        await asyncio.sleep(wait)
                        continue
                    error_text = await resp.text()
                    raise Exception(f"Qdrant API {resp.status}: {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < retries - 1:
                    wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                    await asyncio.sleep(wait)
                    continue
                raise
        raise last_exc or Exception("Request failed after retries")

    # ── Core REST API ────────────────────────────────────────────────

    async def test_connection(self) -> Dict[str, Any]:
        try:
            data = await self._request("GET", "/collections")
            cols = data.get("result", {}).get("collections", [])
            return {"status": "connected", "url": self.config.url, "collections": [c["name"] for c in cols]}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def list_collections(self) -> List[str]:
        data = await self._request("GET", "/collections")
        return [c["name"] for c in data.get("result", {}).get("collections", [])]

    async def get_collection_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("GET", f"/collections/{name or self.config.default_collection}")

    async def create_collection(self, name: Optional[str] = None, *, vector_size: Optional[int] = None, distance: Optional[str] = None) -> Dict[str, Any]:
        n = name or self.config.default_collection
        body: Dict[str, Any] = {
            "vectors": {"size": vector_size or self.config.vector_size, "distance": distance or self.config.distance},
            "hnsw_config": {"m": self.config.hnsw_m, "ef_construct": self.config.hnsw_ef_construct},
        }
        if self.config.on_disk:
            body["vectors"]["on_disk"] = True
        if self.config.quantization == "scalar":
            body["quantization_config"] = {"scalar": {"type": "int8", "always_ram": True}}
        elif self.config.quantization == "binary":
            body["quantization_config"] = {"binary": {"always_ram": True}}
        return await self._request("PUT", f"/collections/{n}", json_body=body)

    async def delete_collection(self, name: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/collections/{name}")

    async def upsert_points(self, points: List[Dict[str, Any]], *, name: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("PUT", f"/collections/{name or self.config.default_collection}/points", json_body={"points": points})

    async def search(self, vector: List[float], *, name: Optional[str] = None, limit: int = 10, score_threshold: Optional[float] = None, filter_conditions: Optional[Dict] = None, with_payload: bool = True) -> Dict[str, Any]:
        body: Dict[str, Any] = {"vector": vector, "limit": limit, "with_payload": with_payload}
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        if filter_conditions:
            body["filter"] = filter_conditions
        return await self._request("POST", f"/collections/{name or self.config.default_collection}/points/search", json_body=body)

    async def count_points(self, name: Optional[str] = None) -> int:
        data = await self._request("POST", f"/collections/{name or self.config.default_collection}/points/count", json_body={"exact": True})
        return data.get("result", {}).get("count", 0)

    async def configure_rag_collection(self, name: Optional[str] = None, *, embedding_model: Optional[str] = None) -> Dict[str, Any]:
        n = name or self.config.default_collection
        model = embedding_model or self.config.embedding_model
        size = EMBEDDING_DIMENSIONS.get(model, self.config.vector_size)
        result = await self.create_collection(name=n, vector_size=size, distance="Cosine")
        return {"collection": n, "embedding_model": model, "vector_size": size, "result": result}

    # ── K8s deployment ───────────────────────────────────────────────

    def generate_k8s_deployment(self, namespace: str = "vector-db") -> str:
        sc = f"\n          storageClassName: {self.config.storage_class}" if self.config.storage_class else ""
        p, gp = self.config.port, self.config.grpc_port
        return (
            f'---\napiVersion: v1\nkind: Namespace\nmetadata:\n  name: {namespace}\n'
            f'---\napiVersion: apps/v1\nkind: StatefulSet\nmetadata:\n  name: qdrant\n  namespace: {namespace}\n'
            f'spec:\n  serviceName: qdrant\n  replicas: {self.config.replicas}\n'
            f'  selector:\n    matchLabels:\n      app: qdrant\n'
            f'  template:\n    metadata:\n      labels:\n        app: qdrant\n'
            f'    spec:\n      containers:\n        - name: qdrant\n          image: {self.config.image}\n'
            f'          ports:\n            - containerPort: {p}\n              name: rest\n'
            f'            - containerPort: {gp}\n              name: grpc\n'
            f'          resources:\n            requests:\n              cpu: "500m"\n              memory: "2Gi"\n'
            f'            limits:\n              cpu: "4"\n              memory: "8Gi"\n'
            f'          volumeMounts:\n            - name: qdrant-storage\n              mountPath: /qdrant/storage\n'
            f'          readinessProbe:\n            httpGet:\n              path: /healthz\n              port: rest\n'
            f'            initialDelaySeconds: 5\n            periodSeconds: 10\n'
            f'  volumeClaimTemplates:\n    - metadata:\n        name: qdrant-storage\n'
            f'      spec:\n        accessModes: ["ReadWriteOnce"]{sc}\n'
            f'        resources:\n          requests:\n            storage: {self.config.storage_size}\n'
            f'---\napiVersion: v1\nkind: Service\nmetadata:\n  name: qdrant-svc\n  namespace: {namespace}\n'
            f'spec:\n  selector:\n    app: qdrant\n  ports:\n'
            f'    - port: {p}\n      targetPort: rest\n      name: rest\n'
            f'    - port: {gp}\n      targetPort: grpc\n      name: grpc\n  type: ClusterIP\n'
        )

    def generate_helm_values(self) -> Dict[str, Any]:
        v: Dict[str, Any] = {
            "qdrant": {
                "image": self.config.image, "replicas": self.config.replicas,
                "ports": {"rest": self.config.port, "grpc": self.config.grpc_port},
                "persistence": {"enabled": True, "size": self.config.storage_size},
                "resources": {"requests": {"cpu": "500m", "memory": "2Gi"}, "limits": {"cpu": "4", "memory": "8Gi"}},
                "hnsw": {"m": self.config.hnsw_m, "efConstruct": self.config.hnsw_ef_construct},
                "defaultCollection": {"name": self.config.default_collection, "vectorSize": self.config.vector_size, "distance": self.config.distance, "embeddingModel": self.config.embedding_model},
            }
        }
        if self.config.storage_class:
            v["qdrant"]["persistence"]["storageClass"] = self.config.storage_class
        if self.config.quantization:
            v["qdrant"]["quantization"] = self.config.quantization
        return v


def create_qdrant_service_from_credentials(credentials: Dict[str, str]) -> QdrantService:
    em = credentials.get("embedding_model", "BAAI/bge-large-en-v1.5")
    config = QdrantConfig(
        url=credentials.get("url", "http://localhost:6333"),
        api_key=credentials.get("api_key"),
        embedding_model=em,
        vector_size=EMBEDDING_DIMENSIONS.get(em, 1024),
        default_collection=credentials.get("default_collection", "terradev-embeddings"),
    )
    return QdrantService(config)


def get_qdrant_setup_instructions() -> str:
    return """
🗄️ Qdrant Setup Instructions:

1. Start Qdrant:
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

2. Configure Terradev:
   terradev configure --provider qdrant --url http://localhost:6333

3. Create a RAG collection:
   terradev qdrant create-collection --embedding-model BAAI/bge-large-en-v1.5

4. Provision full RAG stack:
   terradev provision --template rag --vector-db qdrant --embedding-model BAAI/bge-large-en-v1.5

📋 Credentials:
- url: Qdrant URL (default: http://localhost:6333)
- api_key: API key (Qdrant Cloud only, uses api-key header)
- embedding_model: Model name (auto-sets vector dimensions)

⚠️ Auth: Qdrant Cloud uses 'api-key' header, NOT 'Authorization: Bearer'.
"""
