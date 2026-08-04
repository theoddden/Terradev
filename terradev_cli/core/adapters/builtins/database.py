#!/usr/bin/env python3
"""Built-in database / vector store adapter stubs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..base import (
    Adapter,
    AdapterHealth,
    AdapterSpec,
    DatabaseBackendAdapter,
    VectorStoreBackendAdapter,
)
from ..capabilities import Capability, Capabilities
from ..registry import REGISTRY
from ....ml_services.qdrant_service import (  # reuse existing Qdrant service
    QdrantService,
    QdrantConfig,
    create_qdrant_service_from_credentials,
    EMBEDDING_DIMENSIONS,
)
from ....core.database_connection import (  # reuse existing DB connection manager
    create_sqlite_connection,
    query_database,
    upsert_database,
)

logger = logging.getLogger(__name__)


class SqliteDatabase(DatabaseBackendAdapter):
    """SQLite database backend backed by DatabaseConnectionManager."""

    KIND = "database"
    NAME = "sqlite"
    VERSION = "0.1.0"
    DESCRIPTION = "SQLite database backend"
    CAPABILITIES = Capabilities([Capability.CRUD, Capability.SQL, Capability.TRANSACTIONS])
    CONFIG_SCHEMA = {"required": ["path"]}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._connection_id: Optional[str] = None

    async def _do_initialize(self) -> None:
        self._connection_id = await create_sqlite_connection(
            path=self.config.get("path"),
            table_prefix=self.config.get("table_prefix", "terradev_"),
        )

    async def health(self) -> AdapterHealth:
        return AdapterHealth(
            healthy=self._connection_id is not None,
            latency_ms=0.0,
            message="sqlite connected" if self._connection_id else "sqlite not connected",
        )

    async def connect(self) -> "SqliteDatabase":
        return self

    async def crud(
        self,
        operation: str,
        table: str,
        data: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = operation.lower()
        if operation == "insert":
            success = await upsert_database(self._connection_id, table, data or {})
            return {"operation": operation, "table": table, "status": "ok" if success else "error"}
        if operation == "select":
            try:
                rows = await query_database(
                    self._connection_id,
                    f"SELECT * FROM {table}",
                    filters,
                )
            except Exception as e:  # noqa: BLE001
                return {"operation": operation, "table": table, "rows": [], "rowcount": 0, "error": str(e)}
            return {"operation": operation, "table": table, "rows": rows, "rowcount": len(rows)}
        if operation in ("update", "delete"):
            return {
                "operation": operation,
                "table": table,
                "status": "ok",
                "note": f"Use database sql for {operation} statements",
            }
        return {"operation": operation, "table": table, "status": "ok"}

    async def sql(
        self,
        query: str,
        table: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a raw SQL query through the existing connection manager."""
        if not self._connection_id:
            return {"error": "SQLite adapter not initialized", "status": "error"}
        try:
            rows = await query_database(self._connection_id, query, params)
            return {
                "columns": list(rows[0].keys()) if rows else [],
                "rows": rows,
                "rowcount": len(rows),
                "status": "ok",
            }
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "status": "error"}


class RedisVectorStore(VectorStoreBackendAdapter):
    """Redis vector store backend stub."""

    KIND = "vector_store"
    NAME = "redis"
    VERSION = "0.1.0"
    DESCRIPTION = "Redis vector store backend"
    CAPABILITIES = Capabilities([
        Capability.CRUD,
        Capability.VECTOR_SEARCH,
    ])
    CONFIG_SCHEMA = {"required": ["url"]}

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="redis vector stub")

    async def connect(self) -> "RedisVectorStore":
        return self

    async def crud(
        self,
        operation: str,
        table: str,
        data: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {"operation": operation, "table": table, "status": "ok"}

    async def vector_search(
        self,
        table: str,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return [{"id": "stub", "score": 1.0}]


class QdrantVectorStore(VectorStoreBackendAdapter):
    """Qdrant vector store adapter backed by the existing QdrantService."""

    KIND = "vector_store"
    NAME = "qdrant"
    VERSION = "0.1.0"
    DESCRIPTION = "Qdrant vector store backend"
    CAPABILITIES = Capabilities([
        Capability.CRUD,
        Capability.VECTOR_SEARCH,
    ])
    CONFIG_SCHEMA = {
        "required": ["url"],
        "properties": {
            "url": {"type": "string"},
            "api_key": {"type": "string"},
            "embedding_model": {"type": "string"},
        },
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._service: Optional[QdrantService] = None

    def _get_service(self) -> QdrantService:
        if self._service is None:
            self._service = create_qdrant_service_from_credentials({
                "url": self.config.get("url"),
                "api_key": self.config.get("api_key"),
                "embedding_model": self.config.get("embedding_model"),
            })
        return self._service

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        try:
            svc = self._get_service()
            result = await svc.test_connection()
            if result.get("status") == "connected":
                return AdapterHealth(healthy=True, latency_ms=0.0, message=result["url"])
            return AdapterHealth(healthy=False, latency_ms=0.0, message=result.get("error", "unknown"))
        except Exception as e:  # noqa: BLE001
            return AdapterHealth(healthy=False, latency_ms=0.0, message=str(e))

    async def connect(self) -> "QdrantVectorStore":
        return self

    async def crud(
        self,
        operation: str,
        table: str,
        data: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = operation.lower()
        if operation in ("insert", "upsert"):
            points = (data or {}).get("points", [])
            return await self.upsert(table, points)
        if operation in ("select", "count"):
            count = await self._get_service().count_points(table)
            return {"collection": table, "count": count, "status": "ok"}
        if operation in ("delete", "drop"):
            return await self.delete_collection(table)
        return {"error": f"Unsupported CRUD operation '{operation}' for Qdrant", "status": "error"}

    async def search(
        self,
        collection: str,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            result = await self._get_service().search(
                vector,
                name=collection,
                limit=top_k,
                filter_conditions=filters,
            )
            return result.get("result", [])
        except Exception as e:  # noqa: BLE001
            return [{"error": str(e), "collection": collection}]

    async def scroll(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Dict[str, Any]:
        try:
            result = await self._get_service().scroll_points(
                collection,
                limit=limit,
                filter_conditions=filters,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            return {"collection": collection, "result": result, "status": "ok"}
        except Exception as e:  # noqa: BLE001
            return {"collection": collection, "error": str(e), "status": "error"}

    async def upsert(self, collection: str, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            result = await self._get_service().upsert_points(points, name=collection)
            return {"collection": collection, "result": result, "status": "ok"}
        except Exception as e:  # noqa: BLE001
            return {"collection": collection, "error": str(e), "status": "error"}

    async def create_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> Dict[str, Any]:
        try:
            result = await self._get_service().create_collection(
                name=collection,
                vector_size=vector_size,
                distance=distance,
            )
            return {"collection": collection, "result": result, "status": "created"}
        except Exception as e:  # noqa: BLE001
            return {"collection": collection, "error": str(e), "status": "error"}

    async def delete_collection(self, collection: str) -> Dict[str, Any]:
        try:
            result = await self._get_service().delete_collection(collection)
            return {"collection": collection, "result": result, "status": "deleted"}
        except Exception as e:  # noqa: BLE001
            return {"collection": collection, "error": str(e), "status": "error"}

    async def vector_search(
        self,
        table: str,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return await self.search(table, vector, top_k, filters)

    async def sql(
        self,
        query: str,
        table: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Qdrant does not support SQL queries")


REGISTRY.register(SqliteDatabase.KIND, SqliteDatabase.NAME, SqliteDatabase)
REGISTRY.register(RedisVectorStore.KIND, RedisVectorStore.NAME, RedisVectorStore)
REGISTRY.register(QdrantVectorStore.KIND, QdrantVectorStore.NAME, QdrantVectorStore)
