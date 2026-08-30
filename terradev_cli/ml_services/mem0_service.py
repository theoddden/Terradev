#!/usr/bin/env python3
"""
Mem0 (mem-zero) integration for the Terradev CLI.

Mem0 is an intelligent memory layer for AI agents and assistants.
This service wraps both:
  - the hosted Mem0 Platform client (`MemoryClient` / `AsyncMemoryClient`)
  - the open-source self-hosted `Memory` class

Auth:
  - Hosted: `MEM0_API_KEY` env var or explicit `api_key`.
  - Self-hosted: a `MemoryConfig` dict describing LLM, embedder, vector store,
    and optional graph store providers.

API docs: https://docs.mem0.ai
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    from mem0 import MemoryClient, AsyncMemoryClient, Memory

    MEM0_AVAILABLE = True
except ImportError:  # pragma: no cover
    MemoryClient = None  # type: ignore[assignment,misc]
    AsyncMemoryClient = None  # type: ignore[assignment,misc]
    Memory = None  # type: ignore[assignment,misc]
    MEM0_AVAILABLE = False


@dataclass
class Mem0Config:
    """Configuration for the Mem0 service."""

    mode: str = "hosted"  # "hosted" or "self_hosted"
    api_key: Optional[str] = None
    host: Optional[str] = None  # default https://api.mem0.ai for hosted
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    # Self-hosted configuration (overrides default in-memory / OpenAI config)
    vector_store: Optional[Dict[str, Any]] = field(default=None)
    llm: Optional[Dict[str, Any]] = field(default=None)
    embedder: Optional[Dict[str, Any]] = field(default=None)
    graph_store: Optional[Dict[str, Any]] = field(default=None)
    history_db_path: Optional[str] = None
    version: str = "v1.1"
    custom_instructions: Optional[str] = None
    custom_update_memory_prompt: Optional[str] = None
    # Default entity scoping when not provided on individual calls
    default_user_id: Optional[str] = None
    default_agent_id: Optional[str] = None
    default_app_id: Optional[str] = None
    default_run_id: Optional[str] = None
    # API client settings
    timeout: int = 300


class Mem0Service:
    """Terradev wrapper for the Mem0 memory layer."""

    def __init__(self, config: Mem0Config):
        if not MEM0_AVAILABLE:
            raise ImportError(
                "mem0ai is not installed. Install with: pip install mem0ai"
            )
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for the Mem0 service. Install with: pip install aiohttp"
            )

        self.config = config
        self._client: Optional[Any] = None

    def _resolve_api_key(self) -> str:
        """Resolve the Mem0 API key from config or environment."""
        key = self.config.api_key or os.environ.get("MEM0_API_KEY")
        if not key:
            raise ValueError(
                "Mem0 API key not configured. Set --api-key, MEM0_API_KEY, "
                "or run 'terradev agent mem0 configure'."
            )
        return key

    def _build_memory_config(self) -> Any:
        """Build a mem0 MemoryConfig for self-hosted mode."""
        from mem0.configs.base import MemoryConfig

        kwargs: Dict[str, Any] = {"version": self.config.version}
        if self.config.history_db_path:
            kwargs["history_db_path"] = self.config.history_db_path
        if self.config.custom_instructions:
            kwargs["custom_instructions"] = self.config.custom_instructions
        if self.config.custom_update_memory_prompt:
            kwargs["custom_update_memory_prompt"] = self.config.custom_update_memory_prompt

        if self.config.vector_store:
            from mem0.configs.base import VectorStoreConfig

            kwargs["vector_store"] = VectorStoreConfig(
                provider=self.config.vector_store.get("provider", "qdrant"),
                config=self.config.vector_store.get("config", {}),
            )
        if self.config.llm:
            from mem0.configs.base import LlmConfig

            kwargs["llm"] = LlmConfig(
                provider=self.config.llm.get("provider", "openai"),
                config=self.config.llm.get("config", {}),
            )
        if self.config.embedder:
            from mem0.configs.base import EmbedderConfig

            kwargs["embedder"] = EmbedderConfig(
                provider=self.config.embedder.get("provider", "openai"),
                config=self.config.embedder.get("config", {}),
            )
        if self.config.graph_store:
            from mem0.configs.base import GraphStoreConfig

            kwargs["graph_store"] = GraphStoreConfig(
                provider=self.config.graph_store.get("provider", "neo4j"),
                config=self.config.graph_store.get("config", {}),
            )

        return MemoryConfig(**kwargs)

    def _client_sync(self) -> Any:
        """Return a synchronous Mem0 client, creating it if necessary."""
        if self._client is not None:
            return self._client

        if self.config.mode == "hosted":
            kwargs: Dict[str, Any] = {"api_key": self._resolve_api_key()}
            if self.config.host:
                kwargs["host"] = self.config.host
            if self.config.org_id:
                kwargs["org_id"] = self.config.org_id
            if self.config.project_id:
                kwargs["project_id"] = self.config.project_id
            self._client = MemoryClient(**kwargs)
        else:
            config = self._build_memory_config()
            self._client = Memory.from_config(config)

        return self._client

    def _entity_scope(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in default entity IDs when not provided."""
        for key in ("user_id", "agent_id", "app_id", "run_id"):
            if kwargs.get(key) is None:
                default = getattr(self.config, f"default_{key}")
                if default:
                    kwargs[key] = default
        return kwargs

    # ------------------------------------------------------------------
    # Public synchronous API
    # ------------------------------------------------------------------

    def test_connection(self) -> Dict[str, Any]:
        """Test connectivity to the configured Mem0 backend."""
        try:
            client = self._client_sync()
            if self.config.mode == "hosted":
                # The platform client does not expose a dedicated ping,
                # so we list memories with a small page size.
                result = client.get_all(
                    filters={"user_id": "__terradev_test__"}, page_size=1
                )
                return {
                    "status": "connected",
                    "mode": self.config.mode,
                    "host": self.config.host or "https://api.mem0.ai",
                    "memories": result.get("count", 0),
                }
            else:
                _ = client.config if hasattr(client, "config") else None
                return {
                    "status": "connected",
                    "mode": self.config.mode,
                    "vector_store": self.config.vector_store,
                    "llm": self.config.llm,
                }
        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    def add(
        self,
        messages: Any,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        custom_categories: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add memories from messages."""
        client = self._client_sync()
        kwargs = self._entity_scope(
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "app_id": app_id,
                "run_id": run_id,
                "metadata": metadata,
                "infer": infer,
                "custom_categories": custom_categories,
                "custom_instructions": custom_instructions,
            }
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return client.add(messages, **kwargs)

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        top_k: int = 10,
        rerank: bool = False,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        categories: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search stored memories by semantic similarity."""
        client = self._client_sync()
        search_filters = filters or {}
        for key in ("user_id", "agent_id", "app_id", "run_id"):
            value = locals()[key] or getattr(self.config, f"default_{key}")
            if value and key not in search_filters:
                search_filters[key] = value

        kwargs: Dict[str, Any] = {
            "query": query,
            "filters": search_filters,
            "top_k": top_k,
            "rerank": rerank,
        }
        if threshold is not None:
            kwargs["threshold"] = threshold
        if categories is not None:
            kwargs["categories"] = categories
        if fields is not None:
            kwargs["fields"] = fields
        return client.search(**kwargs)

    def get(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a single memory by ID."""
        client = self._client_sync()
        return client.get(memory_id)

    def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """List memories with optional filtering."""
        client = self._client_sync()
        use_filters = filters or {}
        for key in ("user_id", "agent_id", "app_id", "run_id"):
            value = locals()[key] or getattr(self.config, f"default_{key}")
            if value:
                use_filters[key] = value

        kwargs: Dict[str, Any] = {"filters": use_filters}
        if top_k is not None:
            kwargs["top_k"] = top_k
        if page is not None:
            kwargs["page"] = page
        if page_size is not None:
            kwargs["page_size"] = page_size
        return client.get_all(**kwargs)

    def update(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Update a memory by ID."""
        client = self._client_sync()
        kwargs: Dict[str, Any] = {}
        if text is not None:
            kwargs["text"] = text
        if metadata is not None:
            kwargs["metadata"] = metadata
        if timestamp is not None:
            kwargs["timestamp"] = timestamp
        return client.update(memory_id, **kwargs)

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory by ID."""
        client = self._client_sync()
        return client.delete(memory_id)

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete all memories matching the given entity scope."""
        client = self._client_sync()
        kwargs = self._entity_scope(
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "app_id": app_id,
                "run_id": run_id,
            }
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return client.delete_all(**kwargs)

    def history(self, memory_id: str) -> Dict[str, Any]:
        """Fetch the edit history of a memory."""
        client = self._client_sync()
        return client.history(memory_id)

    # ------------------------------------------------------------------
    # Async API (wraps the sync client for now)
    # ------------------------------------------------------------------

    async def test_connection_async(self) -> Dict[str, Any]:
        """Async test connectivity."""
        return self.test_connection()

    async def add_async(self, *args, **kwargs) -> Dict[str, Any]:
        """Async add memory."""
        return self.add(*args, **kwargs)

    async def search_async(self, *args, **kwargs) -> Dict[str, Any]:
        """Async search memory."""
        return self.search(*args, **kwargs)

    async def get_all_async(self, *args, **kwargs) -> Dict[str, Any]:
        """Async list memories."""
        return self.get_all(*args, **kwargs)


# ------------------------------------------------------------------
# Credential helpers
# ------------------------------------------------------------------


def create_mem0_service_from_credentials(
    credentials: Dict[str, str]
) -> Tuple[Mem0Service, Mem0Config]:
    """Build a Mem0Service from a Terradev provider credential dict."""
    mode = credentials.get("mode", "hosted")
    config = Mem0Config(
        mode=mode,
        api_key=credentials.get("api_key"),
        host=credentials.get("host"),
        org_id=credentials.get("org_id"),
        project_id=credentials.get("project_id"),
        default_user_id=credentials.get("default_user_id"),
        default_agent_id=credentials.get("default_agent_id"),
        default_app_id=credentials.get("default_app_id"),
        default_run_id=credentials.get("default_run_id"),
    )

    if mode == "self_hosted":
        if credentials.get("vector_store"):
            config.vector_store = _safe_load_json(
                credentials["vector_store"], "vector_store"
            )
        if credentials.get("llm"):
            config.llm = _safe_load_json(credentials["llm"], "llm")
        if credentials.get("embedder"):
            config.embedder = _safe_load_json(credentials["embedder"], "embedder")
        if credentials.get("graph_store"):
            config.graph_store = _safe_load_json(
                credentials["graph_store"], "graph_store"
            )
        if credentials.get("custom_instructions"):
            config.custom_instructions = credentials["custom_instructions"]

    return Mem0Service(config), config


def get_mem0_setup_instructions() -> str:
    """Return setup instructions for Mem0."""
    return """
🧠 Mem0 Setup Instructions:

1. Get a Mem0 API key:
   https://app.mem0.ai/dashboard/api-keys

2. Configure Terradev:
   terradev agent mem0 configure --api-key $MEM0_API_KEY

3. Store a memory:
   terradev agent mem0 add --text "I love pizza" --user alice

4. Search memories:
   terradev agent mem0 search --query "food preferences" --user alice

5. (Optional) Self-host with Qdrant backend:
   terradev agent mem0 configure \
     --mode self_hosted \
     --vector-store '{"provider":"qdrant","config":{"host":"localhost","port":6333}}'

📋 Credentials:
- api_key: Mem0 API key (hosted mode)
- host: Mem0 Platform host (default: https://api.mem0.ai)
- mode: hosted or self_hosted
"""


def _safe_load_json(raw: str, label: str) -> Any:
    """Load a JSON string or return it unchanged if already a dict."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {exc}") from exc
