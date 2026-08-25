#!/usr/bin/env python3
"""Capability taxonomy for universal adapters."""

from __future__ import annotations

from enum import Enum
from typing import AbstractSet, Iterable, Optional


class Capability(str, Enum):
    """A capability exposed by an adapter."""

    # Serving engines
    INFERENCE = "inference"
    STREAMING = "streaming"
    BATCH = "batch"
    EMBEDDINGS = "embeddings"
    TOKENIZE = "tokenize"

    # Compute modules
    SUBPROCESS = "subprocess"
    CONTAINER = "container"
    SERVERLESS = "serverless"
    DISTRIBUTED = "distributed"
    GPU = "gpu"

    # Model / dataset registries
    LOCAL_FS = "local_fs"
    REMOTE_URI = "remote_uri"
    VERSIONED = "versioned"
    CACHING = "caching"

    # Databases / vector stores
    CRUD = "crud"
    SQL = "sql"
    VECTOR_SEARCH = "vector_search"
    TRANSACTIONS = "transactions"

    # Telemetry
    TRACES = "traces"
    METRICS = "metrics"


class Capabilities:
    """Lightweight capability set."""

    def __init__(self, capabilities: Optional[Iterable[Capability]] = None) -> None:
        self._caps: set = set(capabilities or [])

    def add(self, *caps: Capability) -> "Capabilities":
        self._caps.update(caps)
        return self

    def has(self, cap: Capability) -> bool:
        return cap in self._caps

    def has_any(self, *caps: Capability) -> bool:
        return any(self.has(c) for c in caps)

    def has_all(self, *caps: Capability) -> bool:
        return all(self.has(c) for c in caps)

    def to_set(self) -> AbstractSet[Capability]:
        return set(self._caps)

    def to_list(self) -> list:
        return sorted(str(c) for c in self._caps)
