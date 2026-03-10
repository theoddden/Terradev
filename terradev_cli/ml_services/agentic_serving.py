#!/usr/bin/env python3
"""
Agentic Inference Serving for Terradev
KV Cache TTL retention, tool-call-aware scheduling, LMCache integration,
and prefill-decode disaggregation config for agentic workloads.

Agentic workloads (ReAct loops) interleave LLM inference with tool calls.
Standard engines evict KV cache at end-of-turn, forcing expensive re-prefill
when the tool returns (<2s later). This module:
  1. Tracks agent sessions via program_id
  2. Parses tool calls from LLM output (OpenAI function_call schema)
  3. Computes adaptive TTL for KV cache pinning per tool type
  4. Generates vLLM/SGLang launch configs with prefix caching + LMCache
  5. Generates K8s manifests for agentic-optimized inference deployments
  6. Provides priority queue metadata for LS/BE request classification

Based on: Continuum (arxiv 2511.02230), LMCache, llm-d, QLLM.
"""

import logging
import time
import math
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# Priority levels for agentic request classification
PRIORITY_CRITICAL = 0   # User-facing reactive agent, TTFT < 200ms
PRIORITY_INTERACTIVE = 1 # Tool-calling agent mid-session, TTFT < 1s
PRIORITY_BACKGROUND = 2  # Proactive agents (monitoring, indexing)
PRIORITY_BATCH = 3       # Eval runs, training data generation

PRIORITY_LABELS = {
    PRIORITY_CRITICAL: "critical",
    PRIORITY_INTERACTIVE: "interactive",
    PRIORITY_BACKGROUND: "background",
    PRIORITY_BATCH: "batch",
}

# Default TTL bounds (seconds)
_DEFAULT_TTL_MIN = 1.0
_DEFAULT_TTL_MAX = 60.0
_DEFAULT_TTL_MULTIPLIER = 2.0  # TTL = multiplier * EMA(tool_latency)
_EMA_ALPHA = 0.3  # Exponential moving average decay for tool latency


@dataclass
class AgenticServingConfig:
    """Configuration for agentic inference serving."""
    # vLLM / SGLang engine settings
    engine: str = "vllm"  # "vllm" or "sglang"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    tensor_parallel_size: int = 1
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.85
    block_size: int = 16
    # Agentic KV cache settings
    enable_prefix_caching: bool = True
    ttl_min: float = _DEFAULT_TTL_MIN
    ttl_max: float = _DEFAULT_TTL_MAX
    ttl_multiplier: float = _DEFAULT_TTL_MULTIPLIER
    # LMCache offload settings
    lmcache_enabled: bool = True
    lmcache_backend: str = "cpu"  # "cpu", "disk", "redis"
    lmcache_redis_url: Optional[str] = None
    lmcache_max_cpu_gb: float = 32.0
    lmcache_disk_path: str = "/tmp/lmcache"
    # Prefill-decode disaggregation
    disaggregation_enabled: bool = False
    prefill_replicas: int = 1
    decode_replicas: int = 1
    # Priority scheduling
    enable_priority_scheduling: bool = True
    default_priority: int = PRIORITY_INTERACTIVE
    # K8s settings
    image: str = "vllm/vllm-openai:latest"
    namespace: str = "inference"
    gpu_type: str = "nvidia.com/gpu"
    gpu_count: int = 1


class ToolCallTracker:
    """Tracks per-tool latency history and computes adaptive TTL values.

    Uses exponential moving average of observed tool-call durations to set
    a TTL that covers most tool returns without over-pinning GPU memory.
    """

    def __init__(self, config: AgenticServingConfig):
        self.config = config
        # tool_name -> EMA of latency
        self._tool_ema: Dict[str, float] = {}
        # tool_name -> count of observations
        self._tool_counts: Dict[str, int] = defaultdict(int)
        # program_id -> {turn: int, last_finish_ts: float, last_tool: str}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def register_session(self, program_id: str) -> Dict[str, Any]:
        """Register or retrieve an agent session."""
        if program_id not in self._sessions:
            self._sessions[program_id] = {
                "turn": 0,
                "last_finish_ts": None,
                "last_tool": None,
                "created_ts": time.time(),
            }
        return self._sessions[program_id]

    def record_tool_call_finish(self, program_id: str, tool_name: str) -> None:
        """Record that an LLM turn finished with a tool call."""
        session = self.register_session(program_id)
        session["last_finish_ts"] = time.time()
        session["last_tool"] = tool_name
        session["turn"] += 1

    def record_tool_return(self, program_id: str) -> Optional[float]:
        """Record that a tool returned and the next LLM turn is starting.
        Returns the observed tool latency in seconds, or None if no prior finish."""
        session = self._sessions.get(program_id)
        if not session or session["last_finish_ts"] is None:
            return None
        latency = time.time() - session["last_finish_ts"]
        tool_name = session.get("last_tool", "_unknown")
        # Update EMA
        if tool_name in self._tool_ema:
            self._tool_ema[tool_name] = (
                _EMA_ALPHA * latency + (1 - _EMA_ALPHA) * self._tool_ema[tool_name]
            )
        else:
            self._tool_ema[tool_name] = latency
        self._tool_counts[tool_name] += 1
        session["last_finish_ts"] = None
        return latency

    def compute_ttl(self, tool_name: str) -> float:
        """Compute TTL for pinning KV cache after a tool call.

        TTL = clamp(multiplier * EMA(tool_latency), ttl_min, ttl_max)
        For unseen tools, uses ttl_max as conservative default.
        """
        if tool_name not in self._tool_ema:
            return self.config.ttl_max
        ema = self._tool_ema[tool_name]
        ttl = self.config.ttl_multiplier * ema
        return max(self.config.ttl_min, min(ttl, self.config.ttl_max))

    def get_stats(self) -> Dict[str, Any]:
        """Return tool latency statistics."""
        return {
            "active_sessions": len(self._sessions),
            "tools_tracked": len(self._tool_ema),
            "tool_stats": {
                name: {
                    "ema_latency_s": round(self._tool_ema[name], 3),
                    "ttl_s": round(self.compute_ttl(name), 3),
                    "observations": self._tool_counts[name],
                }
                for name in self._tool_ema
            },
        }

    def end_session(self, program_id: str) -> None:
        """Clean up a finished agent session."""
        self._sessions.pop(program_id, None)


def parse_tool_calls(response_message: Dict[str, Any]) -> List[Dict[str, str]]:
    """Parse tool calls from an OpenAI-compatible chat completion message.

    Handles both:
      - response.choices[0].message.tool_calls (OpenAI/vLLM)
      - response.choices[0].message.function_call (legacy)
    Returns list of {"name": tool_name, "call_id": call_id, "arguments": args_json}.
    """
    calls = []
    # Modern tool_calls array
    tool_calls = response_message.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            calls.append({
                "name": fn.get("name", "_unknown"),
                "call_id": tc.get("id", ""),
                "arguments": fn.get("arguments", "{}"),
            })
        return calls
    # Legacy function_call
    fc = response_message.get("function_call")
    if fc:
        calls.append({
            "name": fc.get("name", "_unknown"),
            "call_id": "",
            "arguments": fc.get("arguments", "{}"),
        })
    return calls


def build_request_metadata(
    program_id: str,
    priority: int = PRIORITY_INTERACTIVE,
    turn_number: int = 0,
    tool_call_expected: bool = True,
    estimated_output_len: Optional[int] = None,
    deadline_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Build agentic request metadata to attach to inference requests.

    This metadata is passed as extra headers or request fields to the
    serving engine so the scheduler can make priority/TTL decisions.
    """
    meta = {
        "x-terradev-program-id": program_id,
        "x-terradev-priority": str(priority),
        "x-terradev-priority-label": PRIORITY_LABELS.get(priority, "unknown"),
        "x-terradev-turn": str(turn_number),
        "x-terradev-tool-expected": "1" if tool_call_expected else "0",
    }
    if estimated_output_len is not None:
        meta["x-terradev-est-output-len"] = str(estimated_output_len)
    if deadline_ms is not None:
        meta["x-terradev-deadline-ms"] = str(deadline_ms)
    return meta


def generate_vllm_args(config: AgenticServingConfig) -> List[str]:
    """Generate vLLM server launch arguments optimized for agentic workloads."""
    args = [
        "--model", config.model,
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--max-model-len", str(config.max_model_len),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--block-size", str(config.block_size),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    if config.enable_prefix_caching:
        args.append("--enable-prefix-caching")
    if config.enable_priority_scheduling:
        args.extend(["--scheduler-policy", "priority"])
    return args


def generate_sglang_args(config: AgenticServingConfig) -> List[str]:
    """Generate SGLang server launch arguments optimized for agentic workloads."""
    args = [
        "--model-path", config.model,
        "--tp", str(config.tensor_parallel_size),
        "--context-length", str(config.max_model_len),
        "--mem-fraction-static", str(config.gpu_memory_utilization),
    ]
    if config.enable_prefix_caching:
        args.extend(["--enable-cache-report", "--chunked-prefill-size", "8192"])
    return args


def generate_lmcache_config(config: AgenticServingConfig) -> Dict[str, Any]:
    """Generate LMCache configuration for KV cache offloading."""
    if not config.lmcache_enabled:
        return {"enabled": False}
    lmc: Dict[str, Any] = {
        "enabled": True,
        "chunk_size": config.block_size,
        "local_device": config.lmcache_backend,
    }
    if config.lmcache_backend == "cpu":
        lmc["max_local_cache_size_gb"] = config.lmcache_max_cpu_gb
    elif config.lmcache_backend == "disk":
        lmc["local_device"] = "file"
        lmc["local_device_path"] = config.lmcache_disk_path
    elif config.lmcache_backend == "redis":
        lmc["remote_url"] = config.lmcache_redis_url or "redis://localhost:6379"
        lmc["remote_serde"] = "naive"
    return lmc


def generate_lmcache_env(config: AgenticServingConfig) -> Dict[str, str]:
    """Generate environment variables for LMCache integration with vLLM."""
    if not config.lmcache_enabled:
        return {}
    env = {
        "LMCACHE_ENABLED": "1",
        "LMCACHE_LOCAL_DEVICE": config.lmcache_backend if config.lmcache_backend != "redis" else "cpu",
        "LMCACHE_CHUNK_SIZE": str(config.block_size),
    }
    if config.lmcache_backend == "cpu":
        env["LMCACHE_MAX_LOCAL_CACHE_SIZE"] = f"{config.lmcache_max_cpu_gb}GB"
    elif config.lmcache_backend == "disk":
        env["LMCACHE_LOCAL_DEVICE"] = "file"
        env["LMCACHE_LOCAL_DEVICE_PATH"] = config.lmcache_disk_path
    if config.lmcache_backend == "redis" or config.lmcache_redis_url:
        env["LMCACHE_REMOTE_URL"] = config.lmcache_redis_url or "redis://localhost:6379"
        env["LMCACHE_REMOTE_SERDE"] = "naive"
    return env


def generate_engine_env(config: AgenticServingConfig) -> Dict[str, str]:
    """Generate all env vars for an agentic-optimized inference pod."""
    env: Dict[str, str] = {}
    if config.engine == "vllm":
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env.update(generate_lmcache_env(config))
    return env


def generate_k8s_deployment(config: AgenticServingConfig, namespace: Optional[str] = None) -> str:
    """Generate K8s manifests for agentic-optimized inference deployment.

    Includes: vLLM/SGLang with prefix caching, LMCache sidecar config,
    priority-aware scheduling annotations, and optional PD disaggregation.
    """
    ns = namespace or config.namespace
    engine_args = generate_vllm_args(config) if config.engine == "vllm" else generate_sglang_args(config)
    args_str = "\n".join(f'            - "{a}"' for a in engine_args)
    env_map = generate_engine_env(config)
    env_str = ""
    for k, v in env_map.items():
        env_str += f'\n            - name: {k}\n              value: "{v}"'
    gpu_res = f"""
            resources:
              requests:
                cpu: "4"
                memory: "16Gi"
                {config.gpu_type}: "{config.gpu_count}"
              limits:
                cpu: "16"
                memory: "64Gi"
                {config.gpu_type}: "{config.gpu_count}" """

    lmc_annotation = ""
    if config.lmcache_enabled:
        lmc_annotation = '\n        lmcache.ai/enabled: "true"'

    # Decode deployment
    decode_manifest = f"""---
apiVersion: v1
kind: Namespace
metadata:
  name: {ns}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-decode
  namespace: {ns}
  labels:
    app: agentic-inference
    terradev.io/workload: agentic
    llm-d.ai/inferenceServing: "true"
spec:
  replicas: {config.decode_replicas if config.disaggregation_enabled else 1}
  selector:
    matchLabels:
      app: agentic-inference
      role: decode
  template:
    metadata:
      labels:
        app: agentic-inference
        role: decode
        terradev.io/workload: agentic
        llm-d.ai/inferenceServing: "true"
      annotations:
        terradev.io/engine: {config.engine}
        terradev.io/prefix-caching: "true"{lmc_annotation}
    spec:
      containers:
        - name: {config.engine}
          image: {config.image}
          args:
{args_str}
          ports:
            - containerPort: 8000
              name: http
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
                  optional: true{env_str}
{gpu_res}
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-inference-svc
  namespace: {ns}
spec:
  selector:
    app: agentic-inference
    role: decode
  ports:
    - port: 8000
      targetPort: http
      name: http
  type: ClusterIP
"""

    if config.disaggregation_enabled:
        prefill_manifest = f"""---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-prefill
  namespace: {ns}
  labels:
    app: agentic-inference
    terradev.io/workload: agentic
spec:
  replicas: {config.prefill_replicas}
  selector:
    matchLabels:
      app: agentic-inference
      role: prefill
  template:
    metadata:
      labels:
        app: agentic-inference
        role: prefill
        terradev.io/workload: agentic
    spec:
      containers:
        - name: {config.engine}
          image: {config.image}
          args:
{args_str}
          ports:
            - containerPort: 8000
              name: http
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
                  optional: true{env_str}
{gpu_res}
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-prefill-svc
  namespace: {ns}
spec:
  selector:
    app: agentic-inference
    role: prefill
  ports:
    - port: 8000
      targetPort: http
      name: http
  type: ClusterIP
"""
        decode_manifest += prefill_manifest

    return decode_manifest


def generate_helm_values(config: AgenticServingConfig) -> Dict[str, Any]:
    """Generate Helm values for agentic inference deployment."""
    engine_args = generate_vllm_args(config) if config.engine == "vllm" else generate_sglang_args(config)
    values: Dict[str, Any] = {
        "agenticInference": {
            "engine": config.engine,
            "model": config.model,
            "args": engine_args,
            "env": generate_engine_env(config),
            "image": config.image,
            "gpu": {"type": config.gpu_type, "count": config.gpu_count},
            "resources": {
                "requests": {"cpu": "4", "memory": "16Gi"},
                "limits": {"cpu": "16", "memory": "64Gi"},
            },
            "prefixCaching": {"enabled": config.enable_prefix_caching, "blockSize": config.block_size},
            "lmcache": generate_lmcache_config(config),
            "priorityScheduling": {"enabled": config.enable_priority_scheduling},
            "disaggregation": {
                "enabled": config.disaggregation_enabled,
                "prefillReplicas": config.prefill_replicas,
                "decodeReplicas": config.decode_replicas,
            },
            "ttl": {
                "min": config.ttl_min,
                "max": config.ttl_max,
                "multiplier": config.ttl_multiplier,
            },
        }
    }
    return values


def create_agentic_serving_from_credentials(credentials: Dict[str, str]) -> Tuple[AgenticServingConfig, ToolCallTracker]:
    """Create AgenticServingConfig and ToolCallTracker from credential/config dict."""
    config = AgenticServingConfig(
        engine=credentials.get("engine", "vllm"),
        model=credentials.get("model", "meta-llama/Llama-3.1-8B-Instruct"),
        tensor_parallel_size=int(credentials.get("tensor_parallel_size", "1")),
        max_model_len=int(credentials.get("max_model_len", "32768")),
        gpu_memory_utilization=float(credentials.get("gpu_memory_utilization", "0.85")),
        enable_prefix_caching=credentials.get("enable_prefix_caching", "true").lower() == "true",
        lmcache_enabled=credentials.get("lmcache_enabled", "true").lower() == "true",
        lmcache_backend=credentials.get("lmcache_backend", "cpu"),
        lmcache_redis_url=credentials.get("lmcache_redis_url"),
        disaggregation_enabled=credentials.get("disaggregation_enabled", "false").lower() == "true",
    )
    tracker = ToolCallTracker(config)
    return config, tracker
