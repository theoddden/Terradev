#!/usr/bin/env python3
"""Built-in database / vector store adapter stubs."""

from __future__ import annotations

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


class SqliteDatabase(DatabaseBackendAdapter):
    """SQLite database backend stub."""

    KIND = "database"
    NAME = "sqlite"
    VERSION = "0.1.0"
    DESCRIPTION = "SQLite database backend"
    CAPABILITIES = Capabilities([Capability.CRUD, Capability.SQL, Capability.TRANSACTIONS])
    CONFIG_SCHEMA = {"required": ["path"]}

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="sqlite stub")

    async def connect(self) -> "SqliteDatabase":
        return self

    async def crud(
        self,
        operation: str,
        table: str,
        data: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {"operation": operation, "table": table, "status": "ok"}


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


REGISTRY.register(SqliteDatabase.KIND, SqliteDatabase.NAME, SqliteDatabase)
REGISTRY.register(RedisVectorStore.KIND, RedisVectorStore.NAME, RedisVectorStore)
