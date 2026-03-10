#!/usr/bin/env python3
"""
Langfuse Service Integration for Terradev
LLM Observability, Scoring, Prompt Management & Dataset Export.

License: MIT (self-hosted), Proprietary (Cloud)
Terradev provisions/configures Langfuse on user's infrastructure; never hosts it.

API notes (verified from docs):
  - Auth: HTTP Basic Auth — public key as username, secret key as password
  - Base URL: https://cloud.langfuse.com (EU) or https://us.cloud.langfuse.com (US)
  - Self-hosted: http://localhost:3000 (default)
  - Trace ingestion: OTLP/HTTP (recommended) or Legacy POST /api/public/ingestion
  - Data retrieval: GET /api/public/traces, /observations, /scores, /datasets
  - Scoring: POST /api/public/scores (attach eval scores to traces)
  - Pagination: page + limit query params, cursor-based on some endpoints
  - All responses are JSON with {"data": [...], "meta": {...}} envelope
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
class LangfuseConfig:
    """Langfuse configuration"""
    base_url: str = "https://cloud.langfuse.com"
    public_key: str = ""          # pk-lf-...
    secret_key: str = ""          # sk-lf-...
    project_name: str = "default"
    # K8s deployment
    image: str = "langfuse/langfuse:latest"
    replicas: int = 1
    storage_size: str = "50Gi"
    storage_class: Optional[str] = None
    postgres_dsn: Optional[str] = None


class LangfuseService:
    """Langfuse integration for LLM observability, scoring, and dataset management"""

    def __init__(self, config: LangfuseConfig):
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
            auth = aiohttp.BasicAuth(self.config.public_key, self.config.secret_key)
            self.session = aiohttp.ClientSession(auth=auth)
        return self.session

    async def _request(
        self, method: str, path: str, *,
        params: Optional[Dict] = None, json_body: Optional[Any] = None,
        timeout: float = 30, retries: int = _MAX_RETRIES,
    ) -> Any:
        session = self._ensure_session()
        url = f"{self.config.base_url}{path}"
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
                        logger.warning("Langfuse %s %s → %d, retry in %.1fs", method, path, resp.status, wait)
                        await asyncio.sleep(wait)
                        continue
                    error_text = await resp.text()
                    raise Exception(f"Langfuse API {resp.status}: {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < retries - 1:
                    wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                    logger.warning("Langfuse %s %s error: %s, retry in %.1fs", method, path, e, wait)
                    await asyncio.sleep(wait)
                    continue
                raise
        raise last_exc or Exception("Request failed after retries")

    # ── Connection ────────────────────────────────────────────────────

    async def test_connection(self) -> Dict[str, Any]:
        try:
            data = await self._request("GET", "/api/public/projects")
            projects = data.get("data", [])
            return {
                "status": "connected",
                "base_url": self.config.base_url,
                "projects": len(projects),
                "project_names": [p.get("name", "?") for p in projects[:5]],
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # ── Traces ────────────────────────────────────────────────────────

    async def list_traces(
        self, *, page: int = 1, limit: int = 50,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        order_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List traces with optional filters."""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        if name:
            params["name"] = name
        if user_id:
            params["userId"] = user_id
        if tags:
            for tag in tags:
                params.setdefault("tags", [])
                # Langfuse accepts repeated query params for tags
        if order_by:
            params["orderBy"] = order_by
        return await self._request("GET", "/api/public/traces", params=params)

    async def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get a single trace with all observations."""
        return await self._request("GET", f"/api/public/traces/{trace_id}")

    # ── Observations (spans, generations, events) ─────────────────────

    async def list_observations(
        self, *, page: int = 1, limit: int = 50,
        trace_id: Optional[str] = None,
        name: Optional[str] = None,
        observation_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List observations. type can be: GENERATION, SPAN, EVENT."""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        if trace_id:
            params["traceId"] = trace_id
        if name:
            params["name"] = name
        if observation_type:
            params["type"] = observation_type
        return await self._request("GET", "/api/public/observations", params=params)

    async def get_observation(self, observation_id: str) -> Dict[str, Any]:
        """Get a single observation."""
        return await self._request("GET", f"/api/public/observations/{observation_id}")

    # ── Scores (evaluation results) ──────────────────────────────────

    async def create_score(
        self, *,
        trace_id: str,
        name: str,
        value: float,
        observation_id: Optional[str] = None,
        comment: Optional[str] = None,
        data_type: str = "NUMERIC",
    ) -> Dict[str, Any]:
        """Create a score attached to a trace or observation.

        Use this to record eval results (accuracy, quality, etc.) from retrain cycles.
        """
        body: Dict[str, Any] = {
            "traceId": trace_id,
            "name": name,
            "value": value,
            "dataType": data_type,
        }
        if observation_id:
            body["observationId"] = observation_id
        if comment:
            body["comment"] = comment
        return await self._request("POST", "/api/public/scores", json_body=body)

    async def list_scores(
        self, *, page: int = 1, limit: int = 50,
        trace_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List scores with optional filters."""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        if trace_id:
            params["traceId"] = trace_id
        if name:
            params["name"] = name
        return await self._request("GET", "/api/public/scores", params=params)

    # ── Datasets ──────────────────────────────────────────────────────

    async def list_datasets(self, *, page: int = 1, limit: int = 50) -> Dict[str, Any]:
        """List datasets."""
        return await self._request("GET", "/api/public/datasets", params={"page": page, "limit": limit})

    async def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset by name."""
        return await self._request("GET", f"/api/public/datasets/{dataset_name}")

    async def create_dataset(self, name: str, description: str = "", metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new dataset."""
        body: Dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        if metadata:
            body["metadata"] = metadata
        return await self._request("POST", "/api/public/datasets", json_body=body)

    async def create_dataset_item(
        self, *,
        dataset_name: str,
        input_data: Any,
        expected_output: Optional[Any] = None,
        metadata: Optional[Dict] = None,
        source_trace_id: Optional[str] = None,
        source_observation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a dataset item (for eval/fine-tuning)."""
        body: Dict[str, Any] = {
            "datasetName": dataset_name,
            "input": input_data,
        }
        if expected_output is not None:
            body["expectedOutput"] = expected_output
        if metadata:
            body["metadata"] = metadata
        if source_trace_id:
            body["sourceTraceId"] = source_trace_id
        if source_observation_id:
            body["sourceObservationId"] = source_observation_id
        return await self._request("POST", "/api/public/dataset-items", json_body=body)

    # ── Prompts ───────────────────────────────────────────────────────

    async def list_prompts(self, *, page: int = 1, limit: int = 50) -> Dict[str, Any]:
        """List prompt templates."""
        return await self._request("GET", "/api/public/v2/prompts", params={"page": page, "limit": limit})

    async def get_prompt(self, name: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Get a specific prompt by name and optional version."""
        params: Dict[str, Any] = {}
        if version is not None:
            params["version"] = version
        return await self._request("GET", f"/api/public/v2/prompts/{name}", params=params)

    # ── Export training data from traces ──────────────────────────────

    async def export_training_data(
        self, *,
        limit: int = 1000,
        name_filter: Optional[str] = None,
        min_score: Optional[float] = None,
        score_name: str = "quality",
    ) -> List[Dict[str, str]]:
        """Export traces as instruction/response pairs for LoRA fine-tuning.

        Optionally filter by minimum quality score to only train on good examples.
        Returns list of {"instruction": ..., "response": ...} dicts.
        """
        pairs: List[Dict[str, str]] = []
        page = 1
        while len(pairs) < limit:
            traces_resp = await self.list_traces(page=page, limit=50, name=name_filter)
            traces = traces_resp.get("data", [])
            if not traces:
                break

            for trace in traces:
                # Check score filter if requested
                if min_score is not None:
                    scores_resp = await self.list_scores(trace_id=trace["id"], name=score_name)
                    scores = scores_resp.get("data", [])
                    if scores:
                        avg = sum(s.get("value", 0) for s in scores) / len(scores)
                        if avg < min_score:
                            continue
                    else:
                        continue  # no score = skip when filtering

                input_text = trace.get("input")
                output_text = trace.get("output")

                # Extract from nested structures
                if isinstance(input_text, dict):
                    input_text = input_text.get("content") or input_text.get("messages", [{}])[-1].get("content", "")
                if isinstance(output_text, dict):
                    output_text = output_text.get("content") or output_text.get("text", "")

                if input_text and output_text:
                    pairs.append({"instruction": str(input_text), "response": str(output_text)})
                    if len(pairs) >= limit:
                        break

            page += 1
            meta = traces_resp.get("meta", {})
            if page > meta.get("totalPages", page):
                break

        return pairs

    # ── Quality metrics for drift detection ───────────────────────────

    async def get_quality_metrics(
        self, *,
        score_name: str = "quality",
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Get average quality score across recent traces.

        Used by drift detection to monitor model degradation.
        """
        scores_resp = await self.list_scores(limit=limit, name=score_name)
        scores = scores_resp.get("data", [])

        if not scores:
            return {"avg_score": 0.0, "samples": 0, "detail": "No scores found"}

        values = [s.get("value", 0) for s in scores if s.get("value") is not None]
        if not values:
            return {"avg_score": 0.0, "samples": 0, "detail": "No numeric scores found"}

        avg = sum(values) / len(values)
        return {
            "avg_score": round(avg, 4),
            "min_score": round(min(values), 4),
            "max_score": round(max(values), 4),
            "samples": len(values),
        }

    # ── OTEL env generation ───────────────────────────────────────────

    def generate_otel_env(self, project_name: Optional[str] = None) -> Dict[str, str]:
        """Env vars for instrumenting LLM apps to send traces to Langfuse via OTLP."""
        proj = project_name or self.config.project_name
        env = {
            "LANGFUSE_PUBLIC_KEY": self.config.public_key,
            "LANGFUSE_SECRET_KEY": self.config.secret_key,
            "LANGFUSE_BASE_URL": self.config.base_url,
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"{self.config.base_url}/api/public/otel",
            "OTEL_SERVICE_NAME": f"terradev-{proj}",
        }
        return env

    # ── K8s deployment ────────────────────────────────────────────────

    def generate_k8s_deployment(self, namespace: str = "observability") -> str:
        pg_env = ""
        if self.config.postgres_dsn:
            pg_env = (
                f'\n            - name: DATABASE_URL\n'
                f'              value: "{self.config.postgres_dsn}"'
            )
        else:
            pg_env = (
                f'\n            - name: DATABASE_URL\n'
                f'              value: "postgresql://langfuse:langfuse@langfuse-postgres:5432/langfuse"'
            )
        sc = f"\n          storageClassName: {self.config.storage_class}" if self.config.storage_class else ""
        return (
            f'---\napiVersion: v1\nkind: Namespace\nmetadata:\n  name: {namespace}\n'
            f'---\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n'
            f'  name: langfuse-server\n  namespace: {namespace}\n  labels:\n    app: langfuse\n'
            f'spec:\n  replicas: {self.config.replicas}\n  selector:\n    matchLabels:\n      app: langfuse\n'
            f'  template:\n    metadata:\n      labels:\n        app: langfuse\n    spec:\n'
            f'      containers:\n        - name: langfuse\n          image: {self.config.image}\n'
            f'          ports:\n            - containerPort: 3000\n              name: http\n'
            f'          env:\n            - name: NEXTAUTH_URL\n              value: "http://langfuse-svc.{namespace}:3000"\n'
            f'            - name: NEXTAUTH_SECRET\n              value: "changeme-generate-a-real-secret"{pg_env}\n'
            f'          resources:\n            requests:\n              cpu: "500m"\n              memory: "512Mi"\n'
            f'            limits:\n              cpu: "2"\n              memory: "2Gi"\n'
            f'          readinessProbe:\n            httpGet:\n              path: /api/public/health\n'
            f'              port: http\n            initialDelaySeconds: 15\n            periodSeconds: 15\n'
            f'---\napiVersion: v1\nkind: Service\nmetadata:\n'
            f'  name: langfuse-svc\n  namespace: {namespace}\nspec:\n'
            f'  selector:\n    app: langfuse\n  ports:\n    - port: 3000\n'
            f'      targetPort: http\n      name: http\n  type: ClusterIP\n'
        )


def create_langfuse_service_from_credentials(credentials: Dict[str, str]) -> LangfuseService:
    config = LangfuseConfig(
        base_url=credentials.get("base_url", credentials.get("host", "https://cloud.langfuse.com")),
        public_key=credentials.get("public_key", ""),
        secret_key=credentials.get("secret_key", ""),
        project_name=credentials.get("project_name", "default"),
    )
    return LangfuseService(config)


def get_langfuse_setup_instructions() -> str:
    return """
Langfuse Setup Instructions:

1. Self-hosted (Docker):
   docker compose up -d  # from langfuse repo

   Or via Helm:
   helm repo add langfuse https://langfuse.github.io/langfuse-k8s
   helm install langfuse langfuse/langfuse

2. Cloud:
   Sign up at https://cloud.langfuse.com
   Create a project → Settings → API Keys

3. Configure Terradev:
   terradev langfuse configure \\
     --public-key pk-lf-... \\
     --secret-key sk-lf-... \\
     --host https://cloud.langfuse.com

4. Instrument your app:
   pip install langfuse
   # Uses LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL env vars

Required Credentials:
- public_key: Langfuse Public Key (pk-lf-...)
- secret_key: Langfuse Secret Key (sk-lf-...)
- host: Langfuse server URL (default: https://cloud.langfuse.com)

Usage:
terradev langfuse test                        # Test connection
terradev langfuse traces                      # List traces
terradev langfuse trace <trace-id>            # Get full trace
terradev langfuse scores                      # List scores
terradev langfuse score --trace-id <id> ...   # Create score
terradev langfuse datasets                    # List datasets
terradev langfuse export-training-data        # Export for LoRA fine-tuning
terradev langfuse k8s                         # K8s deployment manifest
"""
