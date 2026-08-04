#!/usr/bin/env python3
"""Abstract adapter protocols for the universal core."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from .capabilities import Capabilities
from .exceptions import AdapterConfigError

logger = logging.getLogger(__name__)


@dataclass
class AdapterHealth:
    """Runtime health snapshot for an adapter."""

    healthy: bool
    latency_ms: float
    message: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class AdapterSpec:
    """Static specification for an adapter."""

    kind: str
    name: str
    version: str
    description: str = ""
    capabilities: Capabilities = None
    config_schema: Dict[str, Any] = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = Capabilities()
        if self.config_schema is None:
            self.config_schema = {}


class Adapter(ABC):
    """Base class for every universal adapter.

    Adapters are hot-swappable, capability-bearing execution primitives.
    They are initialized from a typed config dict and must be disposed.
    """

    def __init__(
        self,
        spec: AdapterSpec,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.spec = spec
        self.config = config or {}
        self._initialized = False
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate ``config`` against ``self.spec.config_schema``."""
        required = set(self.spec.config_schema.get("required", []))
        missing = required - set(self.config.keys())
        if missing:
            raise AdapterConfigError(
                message=f"Missing required config keys: {sorted(missing)}",
                adapter_kind=self.spec.kind,
                adapter_name=self.spec.name,
                context={"missing": sorted(missing)},
            )

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def capabilities(self) -> Capabilities:
        return self.spec.capabilities

    async def initialize(self) -> "Adapter":
        """Initialize the adapter. Idempotent."""
        if not self._initialized:
            await self._do_initialize()
            self._initialized = True
        return self

    @abstractmethod
    async def _do_initialize(self) -> None:
        """Concrete adapters implement one-time setup here."""
        raise NotImplementedError

    async def dispose(self) -> None:
        """Release resources. Idempotent."""
        if self._initialized:
            try:
                await self._do_dispose()
            finally:
                self._initialized = False

    async def _do_dispose(self) -> None:
        """Concrete adapters override this."""
        pass

    @abstractmethod
    async def health(self) -> AdapterHealth:
        """Return a runtime health snapshot."""
        raise NotImplementedError


class ServingEngineAdapter(Adapter):
    """Pluggable serving backend (vLLM, TGI, Ollama, custom remote)."""

    @abstractmethod
    async def load(self, model_uri: str, **options) -> Dict[str, Any]:
        """Load a model into the serving engine."""
        raise NotImplementedError

    @abstractmethod
    async def predict(self, inputs: Any, **options) -> Dict[str, Any]:
        """Run a single prediction."""
        raise NotImplementedError

    async def stream(self, inputs: Any, **options) -> AsyncIterator[Dict[str, Any]]:
        """Stream tokens/chunks if the engine supports streaming."""
        raise NotImplementedError

    async def unload(self, model_uri: str) -> bool:
        """Unload a model."""
        return True


class ComputeModuleAdapter(Adapter):
    """Pluggable compute runtime (local subprocess, container, serverless, distributed)."""

    @abstractmethod
    async def execute(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        **options,
    ) -> Dict[str, Any]:
        """Execute a command and return a structured result."""
        raise NotImplementedError

    @abstractmethod
    async def status(self, job_id: str) -> Dict[str, Any]:
        """Return the status of a running or completed job."""
        raise NotImplementedError

    async def stop(self, job_id: str) -> bool:
        """Stop a running job."""
        return True


class ModelRegistryAdapter(Adapter):
    """Swappable model weight / artifact registry."""

    @abstractmethod
    async def resolve(self, model_uri: str) -> Dict[str, Any]:
        """Resolve a URI to local or remote model metadata."""
        raise NotImplementedError

    @abstractmethod
    async def list_models(self, **filters) -> List[Dict[str, Any]]:
        """List available models matching the filters."""
        raise NotImplementedError

    async def load_metadata(self, model_uri: str) -> Dict[str, Any]:
        """Load lightweight metadata without full weight pull."""
        return await self.resolve(model_uri)


class DatasetRegistryAdapter(Adapter):
    """Swappable dataset pointer / cache registry."""

    @abstractmethod
    async def resolve(self, dataset_uri: str) -> Dict[str, Any]:
        """Resolve a dataset URI to a local or remote pointer."""
        raise NotImplementedError

    @abstractmethod
    async def list_datasets(self, **filters) -> List[Dict[str, Any]]:
        """List available datasets."""
        raise NotImplementedError

    async def cache(self, dataset_uri: str) -> str:
        """Return a local path/cached copy if one exists."""
        resolved = await self.resolve(dataset_uri)
        return resolved.get("local_path", "")


class DatabaseBackendAdapter(Adapter):
    """Unified database / vector store backend."""

    @abstractmethod
    async def connect(self) -> "DatabaseBackendAdapter":
        """Create a connection pool or session."""
        raise NotImplementedError

    @abstractmethod
    async def crud(
        self,
        operation: str,
        table: str,
        data: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a CRUD operation (insert, select, update, delete)."""
        raise NotImplementedError

    async def vector_search(
        self,
        table: str,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search if supported."""
        raise NotImplementedError


class VectorStoreBackendAdapter(DatabaseBackendAdapter):
    """Explicit vector store specialization."""

    async def vector_search(
        self,
        table: str,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Vector stores must implement vector similarity search."""
        raise NotImplementedError
