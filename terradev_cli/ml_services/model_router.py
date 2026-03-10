#!/usr/bin/env python3
"""
Model Router for Terradev — Agentic Inference Workloads
Cost/quality-aware routing between strong and weak LLM models,
with step-type classification and KV-cache-aware instance routing.

For agentic workloads, different steps have different complexity:
  - Tool selection → small model (structured output)
  - Reasoning/planning → large model
  - Code generation → code-specialized model
  - Error recovery → always large model

This module provides:
  1. Step-type classifier that maps agent step metadata to model tier
  2. Threshold-based router (RouteLLM-style) with configurable cost threshold
  3. Cascade mode: try weak first, escalate on parse failure / low confidence
  4. KV-cache-aware routing config generation for llm-d / multi-instance
  5. Request rewriting to target the selected model endpoint

Based on: RouteLLM (LMSYS, arxiv 2406.18665), llm-d (Red Hat/IBM/Google).
"""

import logging
import re
import json
import time
import asyncio
import random
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 10.0
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class ModelTier(str, Enum):
    STRONG = "strong"
    WEAK = "weak"


class StepType(str, Enum):
    TOOL_SELECTION = "tool_selection"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    ERROR_RECOVERY = "error_recovery"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


# Step types that should always use the strong model
_STRONG_ONLY_STEPS = frozenset({StepType.ERROR_RECOVERY, StepType.REASONING})

# Step types that can safely use the weak model
_WEAK_ELIGIBLE_STEPS = frozenset({
    StepType.TOOL_SELECTION, StepType.SUMMARIZATION, StepType.GENERAL,
})

# Heuristic patterns for classifying step type from message content
_STEP_PATTERNS: List[Tuple[str, StepType]] = [
    (r"(?i)(error|exception|traceback|failed|retry|fix|debug)", StepType.ERROR_RECOVERY),
    (r"(?i)(think|reason|plan|analyze|consider|evaluate|decide)", StepType.REASONING),
    (r"(?i)(```|def |class |import |function |const |let |var )", StepType.CODE_GENERATION),
    (r"(?i)(summarize|summary|recap|overview|brief)", StepType.SUMMARIZATION),
    (r"(?i)(call|invoke|execute|run|use tool|function_call)", StepType.TOOL_SELECTION),
]


@dataclass
class ModelEndpoint:
    """A model endpoint that the router can target."""
    name: str
    url: str  # OpenAI-compatible base URL
    model_id: str  # Model name to use in API calls
    tier: ModelTier = ModelTier.STRONG
    api_key: Optional[str] = None
    cost_per_1k_tokens: float = 0.0
    max_tokens: int = 4096


@dataclass
class RouterConfig:
    """Configuration for the model router."""
    # Model endpoints
    strong_endpoint: Optional[ModelEndpoint] = None
    weak_endpoint: Optional[ModelEndpoint] = None
    # Routing strategy
    strategy: str = "step_type"  # "step_type", "threshold", "cascade", "strong_only", "weak_only"
    # For threshold strategy: probability threshold above which to use strong model
    # Lower = more strong model usage = higher quality, higher cost
    cost_threshold: float = 0.5
    # Cascade: try weak first, escalate if tool parse fails or confidence low
    cascade_enabled: bool = False
    cascade_parse_retry: bool = True  # Retry with strong if tool call parse fails
    # Tracking
    track_routing_decisions: bool = True
    # KV cache aware routing (llm-d)
    kv_cache_routing_enabled: bool = False
    llmd_epp_url: Optional[str] = None


class StepClassifier:
    """Classifies agent step type from message content and metadata."""

    @staticmethod
    def classify(
        messages: List[Dict[str, Any]],
        tool_call_expected: bool = False,
        is_retry: bool = False,
    ) -> StepType:
        """Classify the step type from the conversation messages.

        Args:
            messages: The chat messages for this inference call
            tool_call_expected: Whether a tool call is expected in the response
            is_retry: Whether this is a retry after a previous failure
        """
        if is_retry:
            return StepType.ERROR_RECOVERY

        # Check last user/system message for content patterns
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") in ("user", "system", "tool"):
                last_content = str(msg.get("content", ""))
                break

        # Check tool results — if the last message is a tool result with an error
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                if any(kw in content.lower() for kw in ("error", "exception", "traceback", "failed")):
                    return StepType.ERROR_RECOVERY
                break

        # Pattern matching on content
        for pattern, step_type in _STEP_PATTERNS:
            if re.search(pattern, last_content):
                return step_type

        # If tool call is expected and nothing else matched, it's tool selection
        if tool_call_expected:
            return StepType.TOOL_SELECTION

        return StepType.GENERAL


class ModelRouter:
    """Routes agentic inference requests to the appropriate model endpoint."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self.classifier = StepClassifier()
        self._decision_log: List[Dict[str, Any]] = []
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

    def route(
        self,
        messages: List[Dict[str, Any]],
        *,
        tool_call_expected: bool = False,
        is_retry: bool = False,
        force_tier: Optional[ModelTier] = None,
    ) -> Tuple[ModelEndpoint, StepType, str]:
        """Determine which model endpoint to use for this request.

        Returns (endpoint, step_type, reason).
        """
        step_type = self.classifier.classify(
            messages, tool_call_expected=tool_call_expected, is_retry=is_retry,
        )

        # Force override
        if force_tier is not None:
            endpoint = self._get_endpoint(force_tier)
            reason = f"forced_{force_tier.value}"
            self._log_decision(step_type, endpoint, reason)
            return endpoint, step_type, reason

        strategy = self.config.strategy

        if strategy == "strong_only":
            endpoint = self._get_endpoint(ModelTier.STRONG)
            reason = "strategy_strong_only"
        elif strategy == "weak_only":
            endpoint = self._get_endpoint(ModelTier.WEAK)
            reason = "strategy_weak_only"
        elif strategy == "step_type":
            endpoint, reason = self._route_by_step_type(step_type)
        elif strategy == "threshold":
            endpoint, reason = self._route_by_threshold(messages, step_type)
        elif strategy == "cascade":
            # Cascade starts with weak; caller handles escalation
            endpoint = self._get_endpoint(ModelTier.WEAK)
            reason = "cascade_start_weak"
        else:
            endpoint = self._get_endpoint(ModelTier.STRONG)
            reason = f"unknown_strategy_{strategy}_fallback_strong"

        self._log_decision(step_type, endpoint, reason)
        return endpoint, step_type, reason

    def escalate(self, step_type: StepType) -> Tuple[ModelEndpoint, str]:
        """Escalate from weak to strong model (cascade mode)."""
        endpoint = self._get_endpoint(ModelTier.STRONG)
        reason = "cascade_escalated"
        self._log_decision(step_type, endpoint, reason)
        return endpoint, reason

    def _route_by_step_type(self, step_type: StepType) -> Tuple[ModelEndpoint, str]:
        """Route based on classified step type."""
        if step_type in _STRONG_ONLY_STEPS:
            return self._get_endpoint(ModelTier.STRONG), f"step_{step_type.value}_requires_strong"
        if step_type in _WEAK_ELIGIBLE_STEPS:
            return self._get_endpoint(ModelTier.WEAK), f"step_{step_type.value}_weak_eligible"
        # Code generation: use strong for safety
        if step_type == StepType.CODE_GENERATION:
            return self._get_endpoint(ModelTier.STRONG), "step_code_gen_strong"
        return self._get_endpoint(ModelTier.STRONG), "step_default_strong"

    def _route_by_threshold(
        self, messages: List[Dict[str, Any]], step_type: StepType,
    ) -> Tuple[ModelEndpoint, str]:
        """Route based on a complexity heuristic vs cost_threshold.

        Simple heuristic: estimate complexity from message length, tool results,
        and step type. If complexity > threshold, use strong.
        """
        # Always use strong for error recovery
        if step_type in _STRONG_ONLY_STEPS:
            return self._get_endpoint(ModelTier.STRONG), f"threshold_forced_{step_type.value}"

        # Heuristic complexity score [0, 1]
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        num_tool_results = sum(1 for m in messages if m.get("role") == "tool")
        num_messages = len(messages)

        # Longer context + more tool results = more complex
        length_score = min(total_chars / 10000, 1.0)
        tool_score = min(num_tool_results / 5, 1.0)
        turn_score = min(num_messages / 20, 1.0)
        complexity = 0.4 * length_score + 0.3 * tool_score + 0.3 * turn_score

        if complexity >= self.config.cost_threshold:
            return self._get_endpoint(ModelTier.STRONG), f"threshold_{complexity:.2f}_above_{self.config.cost_threshold}"
        return self._get_endpoint(ModelTier.WEAK), f"threshold_{complexity:.2f}_below_{self.config.cost_threshold}"

    def _get_endpoint(self, tier: ModelTier) -> ModelEndpoint:
        """Get endpoint for tier, falling back if not configured."""
        if tier == ModelTier.STRONG:
            return self.config.strong_endpoint or self.config.weak_endpoint or _DEFAULT_ENDPOINT
        return self.config.weak_endpoint or self.config.strong_endpoint or _DEFAULT_ENDPOINT

    def _log_decision(self, step_type: StepType, endpoint: ModelEndpoint, reason: str) -> None:
        if not self.config.track_routing_decisions:
            return
        self._decision_log.append({
            "ts": time.time(),
            "step_type": step_type.value,
            "model": endpoint.model_id,
            "tier": endpoint.tier.value,
            "reason": reason,
        })
        # Keep bounded
        if len(self._decision_log) > 10000:
            self._decision_log = self._decision_log[-5000:]

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing decision statistics."""
        if not self._decision_log:
            return {"total_decisions": 0}
        total = len(self._decision_log)
        strong_count = sum(1 for d in self._decision_log if d["tier"] == "strong")
        by_step = {}
        for d in self._decision_log:
            st = d["step_type"]
            if st not in by_step:
                by_step[st] = {"total": 0, "strong": 0, "weak": 0}
            by_step[st]["total"] += 1
            by_step[st][d["tier"]] += 1
        return {
            "total_decisions": total,
            "strong_pct": round(100 * strong_count / total, 1),
            "weak_pct": round(100 * (total - strong_count) / total, 1),
            "by_step_type": by_step,
        }

    async def _request(
        self, method: str, url: str, *,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Any] = None,
        timeout: float = 60, retries: int = _MAX_RETRIES,
    ) -> Any:
        """HTTP request with retry for proxied inference calls."""
        session = self._ensure_session()
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                async with session.request(
                    method, url, headers=headers, json=json_body,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status in _RETRYABLE_STATUSES and attempt < retries - 1:
                        wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                        await asyncio.sleep(wait)
                        continue
                    error_text = await resp.text()
                    raise Exception(f"Router proxy {resp.status}: {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < retries - 1:
                    wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                    await asyncio.sleep(wait)
                    continue
                raise
        raise last_exc or Exception("Request failed after retries")

    async def proxy_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict]] = None,
        tool_call_expected: bool = False,
        is_retry: bool = False,
        force_tier: Optional[ModelTier] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route and proxy a chat completion request to the selected model.

        In cascade mode, tries weak first and escalates to strong if the
        response fails to parse tool calls when they were expected.
        """
        endpoint, step_type, reason = self.route(
            messages, tool_call_expected=tool_call_expected,
            is_retry=is_retry, force_tier=force_tier,
        )

        result = await self._call_endpoint(endpoint, messages, tools, extra_params)

        # Cascade escalation: if tool call expected but none parsed
        if (self.config.cascade_enabled
                and self.config.cascade_parse_retry
                and tool_call_expected
                and endpoint.tier == ModelTier.WEAK):
            choices = result.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                from .agentic_serving import parse_tool_calls
                if not parse_tool_calls(msg):
                    logger.info("Cascade: weak model failed to produce tool call, escalating to strong")
                    endpoint, esc_reason = self.escalate(step_type)
                    result = await self._call_endpoint(endpoint, messages, tools, extra_params)
                    result["_terradev_escalated"] = True
                    result["_terradev_escalation_reason"] = esc_reason

        result["_terradev_route"] = {
            "model": endpoint.model_id,
            "tier": endpoint.tier.value,
            "step_type": step_type.value,
            "reason": reason,
        }
        return result

    async def _call_endpoint(
        self,
        endpoint: ModelEndpoint,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]],
        extra_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Make the actual API call to an endpoint."""
        url = f"{endpoint.url.rstrip('/')}/v1/chat/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"

        body: Dict[str, Any] = {
            "model": endpoint.model_id,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        if extra_params:
            body.update(extra_params)

        return await self._request("POST", url, headers=headers, json_body=body)


# Fallback endpoint if nothing configured
_DEFAULT_ENDPOINT = ModelEndpoint(
    name="local-vllm",
    url="http://localhost:8000",
    model_id="default",
    tier=ModelTier.STRONG,
)


def generate_llmd_routing_config(config: RouterConfig) -> Dict[str, Any]:
    """Generate llm-d KV-cache-aware routing configuration.

    This configures the External Processing Pod (EPP) for prefix-aware
    request routing across multiple vLLM pods.
    """
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "llm-d-epp-config", "namespace": "inference"},
        "data": {
            "plugins-v2.yaml": json.dumps({
                "plugins": [{
                    "name": "cache-aware-router",
                    "type": "external_processor",
                    "config": {
                        "discovery": {
                            "label_selector": "llm-d.ai/inferenceServing=true",
                        },
                        "cache": {
                            "type": "in-memory-lru",
                            "max_size": 10000,
                        },
                        "routing": {
                            "algorithm": "prefix-aware",
                            "session_affinity": True,
                        },
                    },
                }],
            }, indent=2),
        },
    }


def generate_envoy_filter_config() -> Dict[str, Any]:
    """Generate Envoy ext_proc filter config for llm-d EPP integration."""
    return {
        "name": "envoy.filters.http.ext_proc",
        "typed_config": {
            "grpc_service": {
                "envoy_grpc": {
                    "cluster_name": "epp-ext-proc-cluster",
                },
            },
            "processing_mode": {
                "request_header_mode": "SEND",
                "response_header_mode": "SEND",
                "request_body_mode": "STREAMED",
            },
            "failure_mode_allow": True,
            "message_timeout": "30s",
        },
    }


def create_router_from_credentials(credentials: Dict[str, str]) -> ModelRouter:
    """Create ModelRouter from credential/config dict."""
    strong_ep = None
    weak_ep = None

    if credentials.get("strong_url"):
        strong_ep = ModelEndpoint(
            name=credentials.get("strong_name", "strong"),
            url=credentials["strong_url"],
            model_id=credentials.get("strong_model", "gpt-4"),
            tier=ModelTier.STRONG,
            api_key=credentials.get("strong_api_key"),
            cost_per_1k_tokens=float(credentials.get("strong_cost", "0.03")),
        )
    if credentials.get("weak_url"):
        weak_ep = ModelEndpoint(
            name=credentials.get("weak_name", "weak"),
            url=credentials["weak_url"],
            model_id=credentials.get("weak_model", "llama-3.1-8b"),
            tier=ModelTier.WEAK,
            api_key=credentials.get("weak_api_key"),
            cost_per_1k_tokens=float(credentials.get("weak_cost", "0.0001")),
        )

    config = RouterConfig(
        strong_endpoint=strong_ep,
        weak_endpoint=weak_ep,
        strategy=credentials.get("strategy", "step_type"),
        cost_threshold=float(credentials.get("cost_threshold", "0.5")),
        cascade_enabled=credentials.get("cascade_enabled", "false").lower() == "true",
        cascade_parse_retry=credentials.get("cascade_parse_retry", "true").lower() == "true",
        kv_cache_routing_enabled=credentials.get("kv_cache_routing", "false").lower() == "true",
        llmd_epp_url=credentials.get("llmd_epp_url"),
    )
    return ModelRouter(config)
