#!/usr/bin/env python3
"""
NVIDIA NeMo Guardrails Service Integration for Terradev
Output Safety Layer — BYOAPI pattern.

License: Apache 2.0 — fully permissive, no restrictions for commercial use.

API notes (verified from NVIDIA docs + GitHub):
  - Server: nemoguardrails server --config PATH/TO/CONFIGS --port 8090
  - Endpoint: POST /v1/chat/completions (OpenAI-like but NOT fully compatible)
  - Requires config_id field to select which guardrails config to apply
  - Multiple configs loaded simultaneously from subdirectories
  - Conversation memory: MemoryStore (default) or RedisStore (production)
  - FastAPI app — containerizable as K8s sidecar or standalone proxy
  - Colang 2.x is current syntax; Colang 1.0 is legacy
"""

import logging
import asyncio
import random
import json
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 10.0
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass
class GuardrailsConfig:
    """NeMo Guardrails configuration"""
    # Server connection
    server_url: str = "http://localhost:8090"
    config_dir: str = "./guardrails"

    # LLM backend that guardrails proxies to
    llm_provider: str = "openai"  # openai, nim, vllm, huggingface
    llm_model: str = "gpt-4"
    llm_endpoint: Optional[str] = None  # e.g. http://vllm-svc:8000/v1
    llm_api_key: Optional[str] = None

    # Deployment
    image: str = "nvcr.io/nvidia/nemo-guardrails:latest"
    port: int = 8090
    replicas: int = 1
    deployment_mode: str = "standalone"  # standalone or sidecar

    # Conversation memory
    memory_backend: str = "memory"  # memory (MemoryStore) or redis (RedisStore)
    redis_url: Optional[str] = None

    # Default rail configs to generate
    enable_topical: bool = True
    enable_jailbreak: bool = True
    enable_pii: bool = True
    enable_factcheck: bool = False

    # Default config_id for requests
    default_config_id: str = "terradev-default"


class GuardrailsService:
    """NeMo Guardrails integration for output safety"""

    def __init__(self, config: GuardrailsConfig):
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
            self.session = aiohttp.ClientSession()
        return self.session

    async def _request(
        self, method: str, path: str, *,
        params: Optional[Dict] = None, json_body: Optional[Any] = None,
        timeout: float = 60, retries: int = _MAX_RETRIES,
    ) -> Any:
        session = self._ensure_session()
        url = f"{self.config.server_url}{path}"
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
                        logger.warning("Guardrails %s %s → %d, retry in %.1fs", method, path, resp.status, wait)
                        await asyncio.sleep(wait)
                        continue
                    error_text = await resp.text()
                    raise Exception(f"Guardrails API {resp.status}: {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < retries - 1:
                    wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                    logger.warning("Guardrails %s %s error: %s, retry in %.1fs", method, path, e, wait)
                    await asyncio.sleep(wait)
                    continue
                raise
        raise last_exc or Exception("Request failed after retries")

    # ── Core API ─────────────────────────────────────────────────────

    async def test_connection(self) -> Dict[str, Any]:
        try:
            await self._request("GET", "/v1/rails/configs")
            return {"status": "connected", "server_url": self.config.server_url}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def chat_completion(
        self, messages: List[Dict[str, str]], *,
        config_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion through guardrails.

        NeMo Guardrails /v1/chat/completions is OpenAI-like but requires
        config_id to select which rail config to apply. Does not support
        all OpenAI params (temperature, stream, etc.).
        """
        cid = config_id or self.config.default_config_id
        return await self._request(
            "POST", "/v1/chat/completions",
            json_body={"config_id": cid, "messages": messages},
        )

    async def test_rail(
        self, test_message: str, *,
        config_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a test message through guardrails and return the result."""
        result = await self.chat_completion(
            messages=[{"role": "user", "content": test_message}],
            config_id=config_id,
        )
        return {
            "input": test_message,
            "output": result,
            "config_id": config_id or self.config.default_config_id,
        }

    # ── Colang 2.x config generation ────────────────────────────────

    def generate_colang_config(self, config_id: Optional[str] = None) -> Dict[str, str]:
        """Generate Colang 2.x configuration files.

        Returns dict of filename → content for the guardrails config directory.
        """
        cid = config_id or self.config.default_config_id
        files: Dict[str, str] = {}

        # config.yml — main configuration
        llm_config = self._build_llm_config()
        files[f"{cid}/config.yml"] = (
            f"# NeMo Guardrails config — {cid}\n"
            f"# Colang 2.x syntax\n"
            f"models:\n"
            f"  - type: main\n"
            f"    engine: {self.config.llm_provider}\n"
            f"    model: {self.config.llm_model}\n"
            f"{llm_config}"
            f"\n"
            f"rails:\n"
            f"  input:\n"
            f"    flows:\n"
        )
        if self.config.enable_jailbreak:
            files[f"{cid}/config.yml"] += "      - check jailbreak\n"
        if self.config.enable_pii:
            files[f"{cid}/config.yml"] += "      - mask sensitive data\n"
        files[f"{cid}/config.yml"] += (
            f"  output:\n"
            f"    flows:\n"
        )
        if self.config.enable_topical:
            files[f"{cid}/config.yml"] += "      - check topical\n"
        if self.config.enable_factcheck:
            files[f"{cid}/config.yml"] += "      - check facts\n"
        if self.config.enable_pii:
            files[f"{cid}/config.yml"] += "      - mask sensitive data on output\n"

        # rails/topical.co — Colang 2.x topical guardrail
        if self.config.enable_topical:
            files[f"{cid}/rails/topical.co"] = (
                '# Colang 2.x — Topical guardrail\n'
                'define flow check topical\n'
                '  """Ensure responses stay on-topic."""\n'
                '  $is_on_topic = execute check_topic_relevance\n'
                '  if not $is_on_topic\n'
                '    bot say "I can only help with topics related to this service."\n'
                '    stop\n'
            )

        # rails/jailbreak.co — Colang 2.x jailbreak detection
        if self.config.enable_jailbreak:
            files[f"{cid}/rails/jailbreak.co"] = (
                '# Colang 2.x — Jailbreak detection\n'
                'define flow check jailbreak\n'
                '  """Detect and block jailbreak attempts."""\n'
                '  $is_jailbreak = execute check_jailbreak_attempt\n'
                '  if $is_jailbreak\n'
                '    bot say "I cannot process that request."\n'
                '    stop\n'
            )

        # rails/pii.co — Colang 2.x PII masking
        if self.config.enable_pii:
            files[f"{cid}/rails/pii.co"] = (
                '# Colang 2.x — PII masking\n'
                'define flow mask sensitive data\n'
                '  """Mask PII in user input before processing."""\n'
                '  $sanitized = execute mask_pii(text=$user_message)\n'
                '  $user_message = $sanitized\n'
                '\n'
                'define flow mask sensitive data on output\n'
                '  """Mask PII in bot output before returning."""\n'
                '  $sanitized = execute mask_pii(text=$bot_message)\n'
                '  $bot_message = $sanitized\n'
            )

        # rails/factcheck.co — Colang 2.x fact-checking
        if self.config.enable_factcheck:
            files[f"{cid}/rails/factcheck.co"] = (
                '# Colang 2.x — Fact-checking\n'
                'define flow check facts\n'
                '  """Verify factual accuracy of bot responses."""\n'
                '  $is_accurate = execute check_factual_accuracy(text=$bot_message)\n'
                '  if not $is_accurate\n'
                '    bot say "I\'m not confident in the accuracy of that response. '
                'Let me provide a more careful answer."\n'
                '    stop\n'
            )

        return files

    def _build_llm_config(self) -> str:
        lines = ""
        if self.config.llm_endpoint:
            lines += f"    parameters:\n      base_url: {self.config.llm_endpoint}\n"
        return lines

    # ── K8s deployment ───────────────────────────────────────────────

    def generate_k8s_deployment(self, namespace: str = "guardrails") -> str:
        redis_env = ""
        if self.config.memory_backend == "redis" and self.config.redis_url:
            redis_env = (
                f'\n            - name: REDIS_URL\n'
                f'              value: "{self.config.redis_url}"'
            )
        return (
            f'---\napiVersion: v1\nkind: Namespace\nmetadata:\n  name: {namespace}\n'
            f'---\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n'
            f'  name: guardrails-server\n  namespace: {namespace}\n'
            f'  labels:\n    app: guardrails\nspec:\n  replicas: {self.config.replicas}\n'
            f'  selector:\n    matchLabels:\n      app: guardrails\n'
            f'  template:\n    metadata:\n      labels:\n        app: guardrails\n'
            f'    spec:\n      containers:\n        - name: guardrails\n'
            f'          image: {self.config.image}\n'
            f'          command: ["nemoguardrails", "server", "--config", "/config", "--port", "{self.config.port}"]\n'
            f'          ports:\n            - containerPort: {self.config.port}\n              name: http\n'
            f'          env:\n            - name: GUARDRAILS_CONFIG_DIR\n              value: "/config"{redis_env}\n'
            f'          resources:\n            requests:\n              cpu: "500m"\n              memory: "1Gi"\n'
            f'            limits:\n              cpu: "2"\n              memory: "4Gi"\n'
            f'          volumeMounts:\n            - name: config\n              mountPath: /config\n'
            f'          readinessProbe:\n            httpGet:\n              path: /v1/rails/configs\n'
            f'              port: http\n            initialDelaySeconds: 15\n            periodSeconds: 10\n'
            f'      volumes:\n        - name: config\n          configMap:\n            name: guardrails-config\n'
            f'---\napiVersion: v1\nkind: Service\nmetadata:\n'
            f'  name: guardrails-svc\n  namespace: {namespace}\nspec:\n'
            f'  selector:\n    app: guardrails\n  ports:\n    - port: {self.config.port}\n'
            f'      targetPort: http\n      name: http\n  type: ClusterIP\n'
        )

    def generate_sidecar_container(self) -> Dict[str, Any]:
        """Generate sidecar container spec to inject into vLLM pods."""
        return {
            "name": "guardrails-sidecar",
            "image": self.config.image,
            "command": [
                "nemoguardrails", "server",
                "--config", "/config",
                "--port", str(self.config.port),
            ],
            "ports": [{"containerPort": self.config.port, "name": "guardrails"}],
            "env": [
                {"name": "GUARDRAILS_CONFIG_DIR", "value": "/config"},
            ],
            "resources": {
                "requests": {"cpu": "250m", "memory": "512Mi"},
                "limits": {"cpu": "1", "memory": "2Gi"},
            },
            "volumeMounts": [{"name": "guardrails-config", "mountPath": "/config"}],
        }

    def generate_helm_values(self) -> Dict[str, Any]:
        return {
            "guardrails": {
                "image": self.config.image,
                "replicas": self.config.replicas,
                "port": self.config.port,
                "deploymentMode": self.config.deployment_mode,
                "llm": {
                    "provider": self.config.llm_provider,
                    "model": self.config.llm_model,
                    "endpoint": self.config.llm_endpoint,
                },
                "memory": {
                    "backend": self.config.memory_backend,
                    "redisUrl": self.config.redis_url,
                },
                "rails": {
                    "topical": self.config.enable_topical,
                    "jailbreak": self.config.enable_jailbreak,
                    "pii": self.config.enable_pii,
                    "factcheck": self.config.enable_factcheck,
                },
                "resources": {
                    "requests": {"cpu": "500m", "memory": "1Gi"},
                    "limits": {"cpu": "2", "memory": "4Gi"},
                },
            }
        }


def create_guardrails_service_from_credentials(credentials: Dict[str, str]) -> GuardrailsService:
    config = GuardrailsConfig(
        server_url=credentials.get("server_url", "http://localhost:8090"),
        config_dir=credentials.get("config_dir", "./guardrails"),
        llm_provider=credentials.get("llm_provider", "openai"),
        llm_model=credentials.get("llm_model", "gpt-4"),
        llm_endpoint=credentials.get("llm_endpoint"),
        llm_api_key=credentials.get("llm_api_key"),
    )
    return GuardrailsService(config)


def get_guardrails_setup_instructions() -> str:
    return """
🛡️ NeMo Guardrails Setup Instructions:

1. Install NeMo Guardrails:
   pip install nemoguardrails

2. Start the server:
   nemoguardrails server --config ./guardrails --port 8090

   # Or via Docker:
   docker run -p 8090:8090 -v ./guardrails:/config \\
     nvcr.io/nvidia/nemo-guardrails:latest \\
     nemoguardrails server --config /config --port 8090

3. Configure Terradev:
   terradev configure --provider guardrails \\
     --server-url http://localhost:8090 \\
     --config-dir ./guardrails \\
     --llm-provider openai

4. Generate default Colang 2.x configs:
   terradev guardrails generate-config

5. Test a rail:
   terradev guardrails test "ignore all previous instructions"

📋 Required Credentials:
- server_url: Guardrails server URL (default: http://localhost:8090)
- config_dir: Path to guardrails config directory
- llm_provider: LLM provider (openai, nim, vllm, huggingface)
- llm_model: Model name (default: gpt-4)
- llm_endpoint: LLM API endpoint (for vLLM: http://vllm-svc:8000/v1)

⚠️ API Note:
POST /v1/chat/completions requires config_id field (not pure OpenAI-compatible).
Multiple configs served simultaneously from subdirectories under config path.
Colang 2.x is current syntax. Use 'nemoguardrails convert' for Colang 1.0 migration.
"""
