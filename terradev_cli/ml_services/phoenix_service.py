#!/usr/bin/env python3
"""
Arize Phoenix Service Integration for Terradev
LLM Trace Observability — BYOAPI pattern.

License: Elastic License 2.0 (ELv2) — free to self-host, no feature gates.
Terradev provisions/configures Phoenix on user's infrastructure; never hosts it.

API notes (verified from docs):
  - REST API is for READING traces: GET /v1/projects, GET /v1/projects/{id}/spans
  - Trace INGESTION uses OTLP (OpenTelemetry Protocol) on same port (6006)
  - Self-hosted does NOT require auth by default
  - Cursor-based pagination: {"data": [...], "next_cursor": "..."}
  - SpanQuery DSL filter: span_kind == 'RETRIEVER', status_code == 'ERROR'
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


@dataclass
class PhoenixConfig:
    """Arize Phoenix configuration"""
    collector_endpoint: str = "http://localhost:6006"
    api_key: Optional[str] = None
    project_name: str = "default"
    image: str = "arizephoenix/phoenix:latest"
    replicas: int = 1
    storage_size: str = "50Gi"
    storage_class: Optional[str] = None
    auth_enabled: bool = False
    secret_name: Optional[str] = None
    otlp_protocol: str = "grpc"
    otlp_port: int = 6006
    db_backend: str = "sqlite"
    postgres_dsn: Optional[str] = None


class PhoenixService:
    """Arize Phoenix integration for LLM trace observability"""

    def __init__(self, config: PhoenixConfig):
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
            headers: Dict[str, str] = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def _request(
        self, method: str, path: str, *,
        params: Optional[Dict] = None, json_body: Optional[Any] = None,
        timeout: float = 30, retries: int = _MAX_RETRIES,
    ) -> Any:
        session = self._ensure_session()
        url = f"{self.config.collector_endpoint}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                async with session.request(
                    method, url, params=params, json=json_body,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status in _RETRYABLE_STATUSES and attempt < retries - 1:
                        wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                        logger.warning("Phoenix %s %s → %d, retry in %.1fs", method, path, resp.status, wait)
                        await asyncio.sleep(wait)
                        continue
                    error_text = await resp.text()
                    raise Exception(f"Phoenix API {resp.status}: {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < retries - 1:
                    wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                    logger.warning("Phoenix %s %s error: %s, retry in %.1fs", method, path, e, wait)
                    await asyncio.sleep(wait)
                    continue
                raise
        raise last_exc or Exception("Request failed after retries")

    # ── REST API (reading traces) ────────────────────────────────────

    async def test_connection(self) -> Dict[str, Any]:
        try:
            data = await self._request("GET", "/v1/projects", params={"limit": 1})
            return {
                "status": "connected",
                "collector_endpoint": self.config.collector_endpoint,
                "projects_found": len(data.get("data", [])),
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def list_projects(self, limit: int = 50, cursor: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/v1/projects", params=params)

    async def list_spans(
        self, project_identifier: Optional[str] = None, *,
        limit: int = 100, cursor: Optional[str] = None,
        filter_condition: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List spans. filter_condition uses SpanQuery DSL e.g. "span_kind == 'RETRIEVER'" """
        project = project_identifier or self.config.project_name
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if filter_condition:
            params["filter"] = filter_condition
        return await self._request("GET", f"/v1/projects/{project}/spans", params=params)

    async def get_trace(self, trace_id: str, project_identifier: Optional[str] = None) -> Dict[str, Any]:
        """Get all spans for a specific trace_id."""
        project = project_identifier or self.config.project_name
        return await self._request(
            "GET", f"/v1/projects/{project}/spans",
            params={"limit": 500, "filter": f"context.trace_id == '{trace_id}'"},
        )

    # ── OTEL env generation ──────────────────────────────────────────

    def generate_otel_env(self, project_name: Optional[str] = None) -> Dict[str, str]:
        """Env vars for vLLM/LangGraph pods. Traces ingested via OTLP, not REST."""
        proj = project_name or self.config.project_name
        env = {
            "PHOENIX_COLLECTOR_ENDPOINT": self.config.collector_endpoint,
            "PHOENIX_PROJECT_NAME": proj,
            "OTEL_EXPORTER_OTLP_ENDPOINT": self.config.collector_endpoint,
            "OTEL_EXPORTER_OTLP_PROTOCOL": self.config.otlp_protocol,
            "OTEL_SERVICE_NAME": f"terradev-{proj}",
        }
        if self.config.api_key:
            env["PHOENIX_API_KEY"] = self.config.api_key
            env["PHOENIX_CLIENT_HEADERS"] = f"api_key={self.config.api_key}"
        return env

    def generate_instrumentation_snippet(self, project_name: Optional[str] = None) -> str:
        """Python snippet for arize-phoenix-otel auto-instrumentation."""
        proj = project_name or self.config.project_name
        ep = self.config.collector_endpoint
        key_line = f'\nos.environ["PHOENIX_API_KEY"] = "<your-key>"' if self.config.api_key else ""
        return (
            f'# pip install arize-phoenix-otel openinference-instrumentation-openai\n'
            f'import os\n'
            f'os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "{ep}"\n'
            f'os.environ["PHOENIX_PROJECT_NAME"] = "{proj}"{key_line}\n\n'
            f'from phoenix.otel import register\n'
            f'tracer_provider = register(\n'
            f'    project_name="{proj}",\n'
            f'    auto_instrument=True,  # auto-patches OpenAI, LangChain, LlamaIndex\n'
            f')\n'
        )

    # ── K8s deployment ───────────────────────────────────────────────

    def generate_k8s_deployment(self, namespace: str = "observability") -> str:
        auth_env = ""
        if self.config.auth_enabled and self.config.secret_name:
            auth_env = (
                f'\n            - name: PHOENIX_ENABLE_AUTH\n'
                f'              value: "true"\n'
                f'            - name: PHOENIX_SECRET\n'
                f'              valueFrom:\n'
                f'                secretKeyRef:\n'
                f'                  name: {self.config.secret_name}\n'
                f'                  key: phoenix-secret'
            )
        pg_env = ""
        if self.config.db_backend == "postgresql" and self.config.postgres_dsn:
            pg_env = (
                f'\n            - name: PHOENIX_SQL_DATABASE_URL\n'
                f'              value: "{self.config.postgres_dsn}"'
            )
        sc = f"\n          storageClassName: {self.config.storage_class}" if self.config.storage_class else ""
        port = self.config.otlp_port
        return (
            f'---\napiVersion: v1\nkind: Namespace\nmetadata:\n  name: {namespace}\n'
            f'---\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n'
            f'  name: phoenix-server\n  namespace: {namespace}\n  labels:\n    app: phoenix\n'
            f'spec:\n  replicas: {self.config.replicas}\n  selector:\n    matchLabels:\n      app: phoenix\n'
            f'  template:\n    metadata:\n      labels:\n        app: phoenix\n    spec:\n'
            f'      containers:\n        - name: phoenix\n          image: {self.config.image}\n'
            f'          ports:\n            - containerPort: {port}\n              name: http\n'
            f'          env:\n            - name: PHOENIX_WORKING_DIR\n              value: "/data"\n'
            f'            - name: PHOENIX_HOST\n              value: "0.0.0.0"\n'
            f'            - name: PHOENIX_PORT\n              value: "{port}"{auth_env}{pg_env}\n'
            f'          resources:\n            requests:\n              cpu: "500m"\n              memory: "1Gi"\n'
            f'            limits:\n              cpu: "2"\n              memory: "4Gi"\n'
            f'          volumeMounts:\n            - name: phoenix-data\n              mountPath: /data\n'
            f'          readinessProbe:\n            httpGet:\n              path: /v1/projects\n'
            f'              port: http\n            initialDelaySeconds: 10\n            periodSeconds: 15\n'
            f'      volumes:\n        - name: phoenix-data\n          persistentVolumeClaim:\n'
            f'            claimName: phoenix-data-pvc\n'
            f'---\napiVersion: v1\nkind: PersistentVolumeClaim\nmetadata:\n'
            f'  name: phoenix-data-pvc\n  namespace: {namespace}\nspec:\n'
            f'  accessModes:\n    - ReadWriteOnce{sc}\n  resources:\n    requests:\n'
            f'      storage: {self.config.storage_size}\n'
            f'---\napiVersion: v1\nkind: Service\nmetadata:\n'
            f'  name: phoenix-svc\n  namespace: {namespace}\nspec:\n'
            f'  selector:\n    app: phoenix\n  ports:\n    - port: {port}\n'
            f'      targetPort: http\n      name: http\n  type: ClusterIP\n'
        )

    def generate_helm_values(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {
            "phoenix": {
                "image": self.config.image,
                "replicas": self.config.replicas,
                "port": self.config.otlp_port,
                "persistence": {
                    "enabled": True,
                    "size": self.config.storage_size,
                },
                "resources": {
                    "requests": {"cpu": "500m", "memory": "1Gi"},
                    "limits": {"cpu": "2", "memory": "4Gi"},
                },
                "auth": {"enabled": self.config.auth_enabled},
                "database": {"backend": self.config.db_backend},
            }
        }
        if self.config.storage_class:
            values["phoenix"]["persistence"]["storageClass"] = self.config.storage_class
        if self.config.postgres_dsn:
            values["phoenix"]["database"]["dsn"] = self.config.postgres_dsn
        return values


def create_phoenix_service_from_credentials(credentials: Dict[str, str]) -> PhoenixService:
    config = PhoenixConfig(
        collector_endpoint=credentials.get("collector_endpoint", "http://localhost:6006"),
        api_key=credentials.get("api_key"),
        project_name=credentials.get("project_name", "default"),
    )
    return PhoenixService(config)


def get_phoenix_setup_instructions() -> str:
    return """
🔭 Arize Phoenix Setup Instructions:

1. Install Phoenix (self-hosted):
   pip install arize-phoenix
   phoenix serve --port 6006

   # Or via Docker:
   docker run -p 6006:6006 arizephoenix/phoenix:latest

2. Configure Terradev:
   terradev configure --provider phoenix \\
     --collector-endpoint http://localhost:6006 \\
     --project-name my-llm-app

3. Instrument your app (auto-patches OpenAI, LangChain, LlamaIndex):
   pip install arize-phoenix-otel
   # Then in your code:
   from phoenix.otel import register
   register(project_name="my-llm-app", auto_instrument=True)

4. For vLLM: just set env vars (vLLM has built-in OTEL support):
   OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix-svc:6006

📋 Required Credentials:
- collector_endpoint: Phoenix server URL (default: http://localhost:6006)
- api_key: API key (optional, only for Phoenix Cloud or auth-enabled)
- project_name: Project name (default: "default")

🔍 Usage:
terradev phoenix test                    # Test connection
terradev phoenix projects                # List projects
terradev phoenix spans --project myapp   # List spans
terradev phoenix trace --trace-id abc123 # Get full trace tree
"""
