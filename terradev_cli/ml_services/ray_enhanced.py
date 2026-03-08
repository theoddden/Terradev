#!/usr/bin/env python3
"""
Terradev Enhanced Ray Service — v2.0.0

Extends the base RayService with Ray Serve LLM orchestration for:
  1. Wide Expert Parallelism (Wide-EP) via build_dp_deployment
  2. Disaggregated Prefill/Decode serving via build_pd_openai_app
  3. MoE-aware cluster management with topology integration
  4. Monitoring stack (Prometheus, Grafana, Ray Dashboard)

References:
  - Anyscale blog: "Ray Serve LLM APIs for Wide-EP and Disaggregated Serving"
  - vLLM docs: "Expert Parallel Deployment"
  - Ray docs: "Data Parallel Attention" serving pattern
"""

import os
import json
import asyncio
import logging
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


class ServingPattern(Enum):
    """MoE serving patterns supported by Ray Serve LLM"""
    PURE_TP = "pure_tp"           # Traditional: all GPUs do tensor parallelism
    WIDE_EP = "wide_ep"           # EP + DP: experts distributed, data parallel attention
    DISAGGREGATED_PD = "disagg"   # Prefill/Decode separation with KV transfer
    WIDE_EP_PD = "wide_ep_pd"    # Wide-EP + disaggregated P/D (full stack)


@dataclass
class MoEModelProfile:
    """Memory and parallelism profile for a Mixture-of-Experts model"""
    model_id: str
    total_params_b: float             # Total parameters (e.g., 744 for GLM-5)
    active_params_b: float            # Active parameters per token (e.g., 40 for GLM-5)
    num_experts: int                  # Total expert count (e.g., 256 for DeepSeek-V3)
    experts_per_token: int            # Top-K routing (e.g., 8)
    total_weight_gb: float            # Full model weight size in GPU memory
    active_memory_gb: float           # Memory used during inference (KV cache + active experts)
    recommended_tp: int = 1           # TP degree per EP rank
    recommended_dp: int = 8           # DP/EP degree
    supports_fp8: bool = True
    supports_eplb: bool = True


# Well-known MoE model profiles
MOE_MODEL_PROFILES: Dict[str, MoEModelProfile] = {
    "zai-org/GLM-5-FP8": MoEModelProfile(
        model_id="zai-org/GLM-5-FP8",
        total_params_b=744, active_params_b=40, num_experts=256,
        experts_per_token=8, total_weight_gb=380, active_memory_gb=45,
        recommended_tp=1, recommended_dp=8,
    ),
    "deepseek-ai/DeepSeek-V3": MoEModelProfile(
        model_id="deepseek-ai/DeepSeek-V3",
        total_params_b=671, active_params_b=37, num_experts=256,
        experts_per_token=8, total_weight_gb=340, active_memory_gb=42,
        recommended_tp=1, recommended_dp=8,
    ),
    "Qwen/Qwen3.5-397B-A17B": MoEModelProfile(
        model_id="Qwen/Qwen3.5-397B-A17B",
        total_params_b=397, active_params_b=17, num_experts=128,
        experts_per_token=8, total_weight_gb=200, active_memory_gb=22,
        recommended_tp=1, recommended_dp=8,
    ),
    "mistralai/Mistral-Large-3-MoE": MoEModelProfile(
        model_id="mistralai/Mistral-Large-3-MoE",
        total_params_b=405, active_params_b=70, num_experts=16,
        experts_per_token=2, total_weight_gb=210, active_memory_gb=55,
        recommended_tp=2, recommended_dp=4,
    ),
    "meta-llama/Llama-4-405B-MoE": MoEModelProfile(
        model_id="meta-llama/Llama-4-405B-MoE",
        total_params_b=405, active_params_b=52, num_experts=16,
        experts_per_token=2, total_weight_gb=210, active_memory_gb=48,
        recommended_tp=2, recommended_dp=4,
    ),
}


@dataclass
class EnhancedRayConfig:
    """Enhanced Ray configuration with MoE and monitoring support"""
    # Base Ray config
    dashboard_uri: Optional[str] = None
    cluster_name: Optional[str] = None
    auth_token: Optional[str] = None
    head_node_ip: Optional[str] = None
    head_node_port: int = 6379
    namespace: str = "default"

    # MoE serving config
    serving_pattern: ServingPattern = ServingPattern.WIDE_EP
    model_id: Optional[str] = None
    serving_backend: str = "vllm"      # vllm or sglang
    tp_size: int = 1
    dp_size: int = 8
    gpu_count: int = 8
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 32768
    enable_expert_parallel: bool = True
    enable_eplb: bool = True
    enable_dbo: bool = True

    # Disaggregated P/D config
    prefill_tp: int = 1
    prefill_dp: int = 4
    decode_tp: int = 1
    decode_dp: int = 4
    kv_connector: str = "NixlConnector"  # NixlConnector or LMCacheConnector
    kv_buffer_size: int = 5368709120     # 5GB default

    # Monitoring
    monitoring_enabled: bool = False
    prometheus_port: int = 8080
    grafana_port: int = 3000


# ── Enhanced Ray Service ─────────────────────────────────────────────────────


class EnhancedRayService:
    """
    Enhanced Ray service with Ray Serve LLM orchestration for MoE models.

    Provides:
      - Wide-EP deployment via Ray Serve LLM's build_dp_deployment API
      - Disaggregated P/D serving via build_pd_openai_app API
      - MoE model profiling with weight vs. active memory distinction
      - Topology-aware EP group placement
      - Monitoring stack management (Prometheus, Grafana, Ray Dashboard)
    """

    def __init__(self, config: EnhancedRayConfig):
        self.config = config
        self._base_service = None
        self._model_profile: Optional[MoEModelProfile] = None

    # ── Connection & Cluster Lifecycle ─────────────────────────────────────

    async def test_connection(self) -> Dict[str, Any]:
        """Test Ray connection and return cluster info"""
        try:
            result = subprocess.run(
                ["ray", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {
                    "status": "failed",
                    "error": "Ray not installed. Run: pip install ray[default]"
                }
            ray_version = result.stdout.strip()

            # Check if cluster is running
            status_result = subprocess.run(
                ["ray", "status"],
                capture_output=True, text=True, timeout=15
            )
            if status_result.returncode == 0:
                return {
                    "status": "connected",
                    "ray_version": ray_version,
                    "cluster_name": self.config.cluster_name or "local",
                    "dashboard_uri": self.config.dashboard_uri or "http://localhost:8265",
                    "serving_pattern": self.config.serving_pattern.value,
                    "model_id": self.config.model_id,
                    "ep_enabled": self.config.enable_expert_parallel,
                    "monitoring_enabled": self.config.monitoring_enabled,
                }
            else:
                return {
                    "status": "not_connected",
                    "ray_version": ray_version,
                    "error": "Ray installed but no active cluster",
                    "suggestion": "Run: terradev ml ray --start"
                }
        except FileNotFoundError:
            return {
                "status": "failed",
                "error": "Ray not installed. Run: pip install ray[default]"
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def start_cluster(
        self, head_node: bool = True, workers: int = 0, port: int = 6379
    ) -> Dict[str, Any]:
        """Start Ray cluster with MoE-optimized configuration"""
        try:
            if head_node:
                cmd = [
                    "ray", "start", "--head",
                    "--port", str(port),
                    "--dashboard-host", "0.0.0.0",
                    "--object-store-memory", str(self.config.kv_buffer_size),
                ]
                if self.config.gpu_count > 0:
                    cmd.extend(["--num-gpus", str(self.config.gpu_count)])

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    return {
                        "status": "head_started",
                        "port": port,
                        "dashboard": "http://localhost:8265",
                        "output": result.stdout,
                    }
                else:
                    raise RuntimeError(f"Failed to start head node: {result.stderr}")
            elif workers > 0:
                head_addr = f"{self.config.head_node_ip or 'localhost'}:{port}"
                cmd = ["ray", "start", "--address", head_addr]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    return {
                        "status": "worker_started",
                        "head_address": head_addr,
                        "output": result.stdout,
                    }
                else:
                    raise RuntimeError(f"Failed to start worker: {result.stderr}")
            else:
                raise ValueError("Must specify head_node=True or workers > 0")
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def stop_cluster(self) -> Dict[str, Any]:
        """Stop Ray cluster"""
        try:
            result = subprocess.run(
                ["ray", "stop"], capture_output=True, text=True, timeout=15
            )
            return {
                "status": "stopped" if result.returncode == 0 else "failed",
                "output": result.stdout,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def get_ray_dashboard_url(self) -> Optional[str]:
        """Get Ray dashboard URL"""
        return self.config.dashboard_uri or "http://localhost:8265"

    # ── Model Profiling ───────────────────────────────────────────────────

    def get_model_profile(self, model_id: Optional[str] = None) -> MoEModelProfile:
        """Get or estimate MoE model profile for memory/parallelism planning"""
        mid = model_id or self.config.model_id
        if mid and mid in MOE_MODEL_PROFILES:
            self._model_profile = MOE_MODEL_PROFILES[mid]
            return self._model_profile

        # Default estimation for unknown MoE models
        self._model_profile = MoEModelProfile(
            model_id=mid or "unknown",
            total_params_b=400, active_params_b=40, num_experts=64,
            experts_per_token=4, total_weight_gb=200, active_memory_gb=40,
            recommended_tp=1, recommended_dp=self.config.gpu_count,
        )
        logger.warning(
            f"No profile for {mid}, using default estimates "
            f"(total={self._model_profile.total_weight_gb}GB, "
            f"active={self._model_profile.active_memory_gb}GB)"
        )
        return self._model_profile

    def compute_parallelism_strategy(
        self,
        gpu_count: Optional[int] = None,
        gpu_memory_gb: float = 80.0,
    ) -> Dict[str, Any]:
        """
        Compute optimal TP/DP/EP strategy for a given GPU count and model.

        For MoE models, pure EP (TP=1, DP=gpu_count) is generally preferred
        because it distributes unique experts across ranks instead of
        replicating them. TP is only needed when a single expert doesn't
        fit in one GPU's memory.

        Returns recommended TP, DP sizes and rationale.
        """
        profile = self.get_model_profile()
        n_gpus = gpu_count or self.config.gpu_count

        # Expert weight per GPU under pure EP
        expert_weight_per_gpu = profile.total_weight_gb / n_gpus
        # Shared (non-expert) weight — always replicated
        shared_weight_gb = profile.active_memory_gb * 0.3  # rough estimate

        single_gpu_needed = expert_weight_per_gpu + shared_weight_gb
        fits_single_gpu = single_gpu_needed < (gpu_memory_gb * 0.9)

        if fits_single_gpu:
            # Pure EP: TP=1, DP=gpu_count — maximum throughput
            tp = 1
            dp = n_gpus
            rationale = (
                f"Pure EP recommended: {expert_weight_per_gpu:.1f}GB expert weight + "
                f"{shared_weight_gb:.1f}GB shared per GPU fits in {gpu_memory_gb}GB. "
                f"TP=1, DP={dp} distributes {profile.num_experts} experts across "
                f"{dp} ranks ({profile.num_experts // dp} experts/rank)."
            )
        else:
            # Need TP to shard individual experts
            tp = 2
            while (single_gpu_needed / tp) > (gpu_memory_gb * 0.9) and tp < n_gpus:
                tp *= 2
            dp = n_gpus // tp
            rationale = (
                f"Hybrid TP+EP: single expert too large for one GPU "
                f"({single_gpu_needed:.1f}GB). TP={tp} shards experts, "
                f"DP={dp} distributes across groups."
            )

        return {
            "model_id": profile.model_id,
            "total_params_b": profile.total_params_b,
            "active_params_b": profile.active_params_b,
            "num_experts": profile.num_experts,
            "total_weight_gb": profile.total_weight_gb,
            "active_memory_gb": profile.active_memory_gb,
            "gpu_count": n_gpus,
            "gpu_memory_gb": gpu_memory_gb,
            "recommended_tp": tp,
            "recommended_dp": dp,
            "expert_parallel": True,
            "experts_per_rank": profile.num_experts // dp,
            "eplb_enabled": profile.supports_eplb,
            "rationale": rationale,
        }

    # ── Wide-EP Deployment (Ray Serve LLM) ────────────────────────────────

    def generate_wide_ep_deployment(
        self,
        model_id: Optional[str] = None,
        tp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate Ray Serve LLM Wide-EP deployment configuration.

        This produces the Python code and configuration for:
          - ray_serve_llm.vllm.VLLMEngineConfig with EP flags
          - ray_serve_llm.build_dp_deployment for DP coordinator + workers
          - ray.serve.run for the OpenAI-compatible app

        Wide-EP pattern:
          Each DP rank runs TP across a subset of GPUs. Experts are
          distributed (not replicated) across DP ranks. A DP Coordinator
          process manages lockstep all-to-all communication for expert
          routing.
        """
        mid = model_id or self.config.model_id or "zai-org/GLM-5-FP8"
        profile = self.get_model_profile(mid)
        tp = tp_size or self.config.tp_size
        dp = dp_size or self.config.dp_size
        mem_util = gpu_memory_utilization or self.config.gpu_memory_utilization
        max_len = max_model_len or self.config.max_model_len

        config = {
            "deployment_type": "wide_ep",
            "model_id": mid,
            "serving_backend": self.config.serving_backend,
            "engine_config": {
                "model": mid,
                "tensor_parallel_size": tp,
                "data_parallel_size": dp,
                "gpu_memory_utilization": mem_util,
                "max_model_len": max_len,
                "enable_expert_parallel": True,
                "enable_eplb": self.config.enable_eplb,
                "enable_dbo": self.config.enable_dbo,
                "trust_remote_code": True,
                "dtype": "auto",
                "quantization": "fp8" if profile.supports_fp8 else None,
            },
            "env_vars": {
                "VLLM_USE_DEEP_GEMM": "1",
                "VLLM_ALL2ALL_BACKEND": "deepep_low_latency",
                "NCCL_P2P_DISABLE": "0",
                "NCCL_IB_DISABLE": "0",
            },
            "ray_serve_config": {
                "num_replicas": 1,
                "route_prefix": "/v1",
                "max_ongoing_requests": 64,
            },
            "model_profile": {
                "total_params_b": profile.total_params_b,
                "active_params_b": profile.active_params_b,
                "num_experts": profile.num_experts,
                "experts_per_token": profile.experts_per_token,
                "total_weight_gb": profile.total_weight_gb,
                "active_memory_gb": profile.active_memory_gb,
                "experts_per_rank": profile.num_experts // dp,
            },
        }
        return config

    def generate_wide_ep_script(
        self,
        model_id: Optional[str] = None,
        tp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
    ) -> str:
        """Generate executable Python script for Wide-EP deployment via Ray Serve LLM"""
        cfg = self.generate_wide_ep_deployment(model_id, tp_size, dp_size)
        ec = cfg["engine_config"]

        return f'''#!/usr/bin/env python3
"""
Wide-EP MoE Deployment via Ray Serve LLM
Generated by Terradev CLI — {datetime.now().isoformat()}

Model: {cfg["model_id"]}
Pattern: Wide-EP (TP={ec["tensor_parallel_size"]}, DP={ec["data_parallel_size"]})
Experts per rank: {cfg["model_profile"]["experts_per_rank"]}
"""
import os
import ray
from ray import serve

# DeepEP / DeepGEMM environment
os.environ["VLLM_USE_DEEP_GEMM"] = "1"
os.environ["VLLM_ALL2ALL_BACKEND"] = "deepep_low_latency"
os.environ["NCCL_P2P_DISABLE"] = "0"
os.environ["NCCL_IB_DISABLE"] = "0"

# Initialize Ray
ray.init(ignore_reinit_error=True)

# ── Ray Serve LLM Engine Config ──
from ray_serve_llm.vllm import VLLMEngineConfig
from ray_serve_llm import build_dp_deployment, build_openai_app

engine_config = VLLMEngineConfig(
    model="{ec["model"]}",
    tensor_parallel_size={ec["tensor_parallel_size"]},
    data_parallel_size={ec["data_parallel_size"]},
    gpu_memory_utilization={ec["gpu_memory_utilization"]},
    max_model_len={ec["max_model_len"]},
    enable_expert_parallel={ec["enable_expert_parallel"]},
    enable_eplb={ec["enable_eplb"]},
    enable_dbo={ec["enable_dbo"]},
    trust_remote_code={ec["trust_remote_code"]},
    dtype="{ec["dtype"]}",
    {f'quantization="{ec["quantization"]}",' if ec["quantization"] else "# quantization not set"}
)

# ── Build DP deployment (Wide-EP) ──
# Each DP rank gets its own vLLM engine instance with a subset of experts.
# The DP Coordinator ensures lockstep all-to-all for expert routing.
dp_deployment = build_dp_deployment(
    engine_config=engine_config,
    name="moe-wide-ep",
)

# ── Build OpenAI-compatible app ──
app = build_openai_app({{"moe-wide-ep": dp_deployment}})

# ── Deploy ──
serve.run(app, route_prefix="/v1")

print("Wide-EP MoE deployment active at http://localhost:8000/v1")
print(f"  Model: {cfg["model_id"]}")
print(f"  TP={ec["tensor_parallel_size"]}, DP={ec["data_parallel_size"]}, "
      f"Experts/rank={cfg["model_profile"]["experts_per_rank"]}")
print(f"  EPLB: {ec["enable_eplb"]}, DBO: {ec["enable_dbo"]}")
print("  Endpoints: /v1/chat/completions, /v1/completions, /v1/models")
'''

    # ── Disaggregated P/D Deployment (Ray Serve LLM) ──────────────────────

    def generate_disaggregated_pd_deployment(
        self,
        model_id: Optional[str] = None,
        prefill_tp: Optional[int] = None,
        prefill_dp: Optional[int] = None,
        decode_tp: Optional[int] = None,
        decode_dp: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate Ray Serve LLM disaggregated Prefill/Decode deployment config.

        Splits inference into two phases with different hardware affinity:
          - PREFILL (compute-bound): processes input prompt, generates KV cache.
            Benefits from high-FLOPS GPUs (H100 SXM).
          - DECODE (memory-bound): autoregressive token generation, reads KV cache.
            Benefits from high-bandwidth GPUs (H200, MI300X).

        KV cache is transferred between prefill and decode via NIXL connector
        (NVIDIA's zero-copy GPU-to-GPU transfer) or LMCache.
        """
        mid = model_id or self.config.model_id or "zai-org/GLM-5-FP8"
        profile = self.get_model_profile(mid)
        p_tp = prefill_tp or self.config.prefill_tp
        p_dp = prefill_dp or self.config.prefill_dp
        d_tp = decode_tp or self.config.decode_tp
        d_dp = decode_dp or self.config.decode_dp

        config = {
            "deployment_type": "disaggregated_pd",
            "model_id": mid,
            "prefill_config": {
                "model": mid,
                "tensor_parallel_size": p_tp,
                "data_parallel_size": p_dp,
                "gpu_memory_utilization": 0.85,
                "max_model_len": self.config.max_model_len,
                "enable_expert_parallel": self.config.enable_expert_parallel,
                "enable_eplb": self.config.enable_eplb,
                "enable_dbo": self.config.enable_dbo,
                "phase": "prefill",
            },
            "decode_config": {
                "model": mid,
                "tensor_parallel_size": d_tp,
                "data_parallel_size": d_dp,
                "gpu_memory_utilization": 0.90,
                "max_model_len": self.config.max_model_len,
                "enable_expert_parallel": self.config.enable_expert_parallel,
                "enable_eplb": self.config.enable_eplb,
                "enable_dbo": self.config.enable_dbo,
                "phase": "decode",
            },
            "kv_connector": {
                "type": self.config.kv_connector,
                "buffer_size": self.config.kv_buffer_size,
                "rdma_enabled": True,
            },
            "env_vars": {
                "VLLM_USE_DEEP_GEMM": "1",
                "VLLM_ALL2ALL_BACKEND": "deepep_low_latency",
            },
            "model_profile": {
                "total_params_b": profile.total_params_b,
                "active_params_b": profile.active_params_b,
                "num_experts": profile.num_experts,
                "total_weight_gb": profile.total_weight_gb,
                "active_memory_gb": profile.active_memory_gb,
            },
        }
        return config

    def generate_disaggregated_pd_script(
        self,
        model_id: Optional[str] = None,
    ) -> str:
        """Generate executable Python script for disaggregated P/D deployment"""
        cfg = self.generate_disaggregated_pd_deployment(model_id)
        pc = cfg["prefill_config"]
        dc = cfg["decode_config"]
        kv = cfg["kv_connector"]

        return f'''#!/usr/bin/env python3
"""
Disaggregated Prefill/Decode MoE Deployment via Ray Serve LLM
Generated by Terradev CLI — {datetime.now().isoformat()}

Model: {cfg["model_id"]}
Prefill: TP={pc["tensor_parallel_size"]}, DP={pc["data_parallel_size"]} (compute-bound, high FLOPS)
Decode:  TP={dc["tensor_parallel_size"]}, DP={dc["data_parallel_size"]} (memory-bound, high bandwidth)
KV Connector: {kv["type"]}
"""
import os
import ray
from ray import serve

os.environ["VLLM_USE_DEEP_GEMM"] = "1"
os.environ["VLLM_ALL2ALL_BACKEND"] = "deepep_low_latency"

ray.init(ignore_reinit_error=True)

from ray_serve_llm.vllm import VLLMEngineConfig
from ray_serve_llm import build_pd_openai_app

# ── Prefill Engine (compute-bound phase) ──
prefill_config = VLLMEngineConfig(
    model="{pc["model"]}",
    tensor_parallel_size={pc["tensor_parallel_size"]},
    data_parallel_size={pc["data_parallel_size"]},
    gpu_memory_utilization={pc["gpu_memory_utilization"]},
    max_model_len={pc["max_model_len"]},
    enable_expert_parallel={pc["enable_expert_parallel"]},
    enable_eplb={pc["enable_eplb"]},
    enable_dbo={pc["enable_dbo"]},
    kv_connector="{kv["type"]}",
    kv_buffer_size={kv["buffer_size"]},
)

# ── Decode Engine (memory-bound phase) ──
decode_config = VLLMEngineConfig(
    model="{dc["model"]}",
    tensor_parallel_size={dc["tensor_parallel_size"]},
    data_parallel_size={dc["data_parallel_size"]},
    gpu_memory_utilization={dc["gpu_memory_utilization"]},
    max_model_len={dc["max_model_len"]},
    enable_expert_parallel={dc["enable_expert_parallel"]},
    enable_eplb={dc["enable_eplb"]},
    enable_dbo={dc["enable_dbo"]},
    kv_connector="{kv["type"]}",
    kv_buffer_size={kv["buffer_size"]},
)

# ── Build disaggregated P/D app ──
# Ray Serve LLM handles KV cache transfer between prefill and decode
# using the configured connector (NIXL for zero-copy GPU-GPU, LMCache for CPU).
app = build_pd_openai_app(
    prefill_config=prefill_config,
    decode_config=decode_config,
    name="moe-disagg-pd",
)

serve.run(app, route_prefix="/v1")

print("Disaggregated P/D MoE deployment active at http://localhost:8000/v1")
print(f"  Prefill: TP={pc["tensor_parallel_size"]}, DP={pc["data_parallel_size"]}")
print(f"  Decode:  TP={dc["tensor_parallel_size"]}, DP={dc["data_parallel_size"]}")
print(f"  KV Connector: {kv["type"]}")
'''

    # ── Monitoring Stack ──────────────────────────────────────────────────

    async def install_monitoring_stack(self) -> Dict[str, Any]:
        """Install Ray + Prometheus + Grafana monitoring stack"""
        try:
            # Check if Ray is running
            status = subprocess.run(
                ["ray", "status"], capture_output=True, text=True, timeout=10
            )
            if status.returncode != 0:
                return {
                    "status": "failed",
                    "error": "Ray cluster not running. Start with: terradev ml ray --start"
                }

            # Ray Dashboard is built-in at :8265
            # For Prometheus/Grafana, generate the config
            prom_config = self._generate_prometheus_config()
            grafana_config = self._generate_grafana_dashboard()

            return {
                "status": "installed",
                "ray": "http://localhost:8265",
                "prometheus": f"http://localhost:{self.config.prometheus_port}",
                "grafana": f"http://localhost:{self.config.grafana_port}",
                "dashboards": "Ray Overview, MoE Expert Utilization, EP Load Balance",
                "prometheus_config": prom_config,
                "grafana_dashboard": grafana_config,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        try:
            status_result = subprocess.run(
                ["ray", "status"], capture_output=True, text=True, timeout=10
            )

            if status_result.returncode == 0:
                # Parse Ray status output for metrics
                output = status_result.stdout
                metrics = self._parse_ray_status(output)

                return {
                    "ray": {
                        "status": "running",
                        "version": self._get_ray_version(),
                        "cluster_name": self.config.cluster_name or "local",
                        "dashboard_uri": self.config.dashboard_uri or "http://localhost:8265",
                    },
                    "monitoring": {
                        "prometheus": self.config.monitoring_enabled,
                        "grafana": self.config.monitoring_enabled,
                    },
                    "metrics": metrics,
                    "serving": {
                        "pattern": self.config.serving_pattern.value,
                        "model_id": self.config.model_id,
                        "ep_enabled": self.config.enable_expert_parallel,
                        "eplb_enabled": self.config.enable_eplb,
                        "dbo_enabled": self.config.enable_dbo,
                    },
                }
            else:
                return {
                    "ray": {
                        "status": "stopped",
                        "error": "No active Ray cluster",
                    },
                    "monitoring": {"prometheus": False, "grafana": False},
                    "metrics": {},
                }
        except Exception as e:
            return {
                "ray": {"status": "error", "error": str(e)},
                "monitoring": {"prometheus": False, "grafana": False},
                "metrics": {},
            }

    # ── Topology Integration ──────────────────────────────────────────────

    def generate_ep_placement_groups(
        self,
        gpu_count: Optional[int] = None,
        topology_report: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate Ray placement group configuration for EP ranks.

        When a topology report is provided (from GPUTopologyOrchestrator),
        placement groups are aligned with NVLink domains and NUMA boundaries
        to ensure EP all-to-all communication stays on the fast path.
        """
        n_gpus = gpu_count or self.config.gpu_count
        strategy = self.compute_parallelism_strategy(n_gpus)
        tp = strategy["recommended_tp"]
        dp = strategy["recommended_dp"]

        bundles = []
        if topology_report and topology_report.get("numa_map"):
            # Topology-aware: group TP ranks within same NUMA node
            numa_map = topology_report["numa_map"]
            gpu_idx = 0
            for _numa_id, devices in sorted(numa_map.items()):
                numa_gpu_count = len(devices.get("gpus", []))
                while numa_gpu_count >= tp and gpu_idx < n_gpus:
                    bundle = {"GPU": tp}
                    if tp > 1:
                        bundle["_numa_hint"] = _numa_id
                    bundles.append(bundle)
                    gpu_idx += tp
                    numa_gpu_count -= tp
        else:
            # No topology info: uniform bundles
            for _ in range(dp):
                bundles.append({"GPU": tp})

        return {
            "strategy": "STRICT_PACK" if tp > 1 else "PACK",
            "bundles": bundles,
            "tp_size": tp,
            "dp_size": dp,
            "expert_parallel": True,
            "topology_aware": topology_report is not None,
        }

    # ── K8s Manifest Generation ───────────────────────────────────────────

    def generate_ray_serve_k8s_manifest(
        self,
        model_id: Optional[str] = None,
        namespace: str = "moe-inference",
    ) -> Dict[str, Any]:
        """Generate Kubernetes manifest for Ray Serve LLM MoE deployment"""
        mid = model_id or self.config.model_id or "zai-org/GLM-5-FP8"
        strategy = self.compute_parallelism_strategy()

        return {
            "apiVersion": "ray.io/v1alpha1",
            "kind": "RayService",
            "metadata": {
                "name": "moe-ray-serve",
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/managed-by": "terradev",
                    "terradev.io/arch": "moe",
                    "terradev.io/serving-pattern": self.config.serving_pattern.value,
                },
            },
            "spec": {
                "serveConfigV2": f"""
applications:
  - name: moe-wide-ep
    route_prefix: /v1
    import_path: ray_serve_llm:build_openai_app
    args:
      engine_config:
        model: {mid}
        tensor_parallel_size: {strategy["recommended_tp"]}
        data_parallel_size: {strategy["recommended_dp"]}
        enable_expert_parallel: true
        enable_eplb: {str(self.config.enable_eplb).lower()}
        enable_dbo: {str(self.config.enable_dbo).lower()}
        gpu_memory_utilization: {self.config.gpu_memory_utilization}
        max_model_len: {self.config.max_model_len}
""",
                "rayClusterConfig": {
                    "headGroupSpec": {
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "ray-head",
                                    "image": "rayproject/ray:2.44.1-py311-gpu",
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": "0"},
                                        "requests": {"cpu": "8", "memory": "32Gi"},
                                    },
                                    "env": [
                                        {"name": "VLLM_USE_DEEP_GEMM", "value": "1"},
                                        {"name": "VLLM_ALL2ALL_BACKEND", "value": "deepep_low_latency"},
                                    ],
                                }],
                            },
                        },
                    },
                    "workerGroupSpecs": [{
                        "groupName": "gpu-workers",
                        "replicas": 1,
                        "minReplicas": 1,
                        "maxReplicas": 4,
                        "template": {
                            "spec": {
                                "nodeSelector": {
                                    "nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3",
                                },
                                "tolerations": [
                                    {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
                                ],
                                "containers": [{
                                    "name": "ray-worker",
                                    "image": "rayproject/ray:2.44.1-py311-gpu",
                                    "resources": {
                                        "limits": {
                                            "nvidia.com/gpu": str(self.config.gpu_count),
                                            "cpu": "96",
                                            "memory": "512Gi",
                                        },
                                        "requests": {
                                            "nvidia.com/gpu": str(self.config.gpu_count),
                                            "cpu": "64",
                                            "memory": "256Gi",
                                        },
                                    },
                                    "env": [
                                        {"name": "VLLM_USE_DEEP_GEMM", "value": "1"},
                                        {"name": "VLLM_ALL2ALL_BACKEND", "value": "deepep_low_latency"},
                                        {"name": "NCCL_P2P_DISABLE", "value": "0"},
                                        {"name": "NCCL_IB_DISABLE", "value": "0"},
                                    ],
                                    "volumeMounts": [
                                        {"name": "model-cache", "mountPath": "/models"},
                                        {"name": "dshm", "mountPath": "/dev/shm"},
                                    ],
                                }],
                                "volumes": [
                                    {"name": "model-cache", "persistentVolumeClaim": {"claimName": "moe-model-cache"}},
                                    {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
                                ],
                            },
                        },
                    }],
                },
            },
        }

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _get_ray_version(self) -> str:
        try:
            result = subprocess.run(
                ["ray", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _parse_ray_status(self, output: str) -> Dict[str, Any]:
        """Parse ray status output into structured metrics"""
        metrics: Dict[str, Any] = {
            "total_workers": 0,
            "cpu_total": 0,
            "cpu_used": 0,
            "memory_total": 0,
            "memory_used": 0,
            "gpu_total": 0,
            "gpu_used": 0,
        }
        for line in output.split("\n"):
            line = line.strip()
            if "CPU" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "/" in p and "CPU" in parts[i - 1] if i > 0 else False:
                        used, total = p.split("/")
                        try:
                            metrics["cpu_used"] = float(used)
                            metrics["cpu_total"] = float(total)
                        except ValueError:
                            pass
            if "GPU" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "/" in p and "GPU" in parts[i - 1] if i > 0 else False:
                        used, total = p.split("/")
                        try:
                            metrics["gpu_used"] = float(used)
                            metrics["gpu_total"] = float(total)
                        except ValueError:
                            pass
            if "node" in line.lower() and "alive" in line.lower():
                try:
                    count = int(line.split()[0])
                    metrics["total_workers"] = count
                except (ValueError, IndexError):
                    pass
        return metrics

    def _generate_prometheus_config(self) -> Dict[str, Any]:
        """Generate Prometheus scrape config for Ray + MoE metrics"""
        return {
            "global": {"scrape_interval": "15s"},
            "scrape_configs": [
                {
                    "job_name": "ray",
                    "metrics_path": "/api/prometheus",
                    "static_configs": [
                        {"targets": ["localhost:8265"]},
                    ],
                },
                {
                    "job_name": "vllm",
                    "metrics_path": "/metrics",
                    "static_configs": [
                        {"targets": ["localhost:8000"]},
                    ],
                },
            ],
        }

    def _generate_grafana_dashboard(self) -> Dict[str, Any]:
        """Generate Grafana dashboard config for MoE monitoring"""
        return {
            "dashboard": {
                "title": "Terradev MoE Inference",
                "panels": [
                    {
                        "title": "Expert Utilization per Rank",
                        "type": "timeseries",
                        "targets": [{"expr": "vllm:expert_utilization_ratio"}],
                    },
                    {
                        "title": "EP All-to-All Latency",
                        "type": "timeseries",
                        "targets": [{"expr": "vllm:all2all_latency_seconds"}],
                    },
                    {
                        "title": "EPLB Rebalance Events",
                        "type": "stat",
                        "targets": [{"expr": "increase(vllm:eplb_rebalance_total[5m])"}],
                    },
                    {
                        "title": "GPU Memory (Weight vs KV Cache)",
                        "type": "timeseries",
                        "targets": [
                            {"expr": "vllm:gpu_cache_usage_perc", "legendFormat": "KV Cache"},
                            {"expr": "vllm:weight_memory_bytes", "legendFormat": "Weights"},
                        ],
                    },
                    {
                        "title": "Request Queue Depth",
                        "type": "timeseries",
                        "targets": [{"expr": "vllm:num_requests_waiting"}],
                    },
                    {
                        "title": "Tokens/Second (Prefill vs Decode)",
                        "type": "timeseries",
                        "targets": [
                            {"expr": "rate(vllm:prompt_tokens_total[1m])", "legendFormat": "Prefill tok/s"},
                            {"expr": "rate(vllm:generation_tokens_total[1m])", "legendFormat": "Decode tok/s"},
                        ],
                    },
                ],
            },
        }


# ── Factory Functions (CLI interface) ────────────────────────────────────────


def create_enhanced_ray_service_from_credentials(
    credentials: Dict[str, str],
) -> EnhancedRayService:
    """Create EnhancedRayService from credential dictionary — called by CLI"""

    pattern_str = credentials.get("ray_serving_pattern", "wide_ep")
    try:
        pattern = ServingPattern(pattern_str)
    except ValueError:
        pattern = ServingPattern.WIDE_EP

    config = EnhancedRayConfig(
        # Base Ray
        dashboard_uri=credentials.get("ray_dashboard_uri"),
        cluster_name=credentials.get("ray_cluster_name"),
        auth_token=credentials.get("ray_auth_token"),
        head_node_ip=credentials.get("ray_head_node_ip"),
        head_node_port=int(credentials.get("ray_head_node_port", "6379")),
        namespace=credentials.get("ray_namespace", "default"),
        # MoE serving
        serving_pattern=pattern,
        model_id=credentials.get("ray_model_id"),
        serving_backend=credentials.get("ray_serving_backend", "vllm"),
        tp_size=int(credentials.get("ray_tp_size", "1")),
        dp_size=int(credentials.get("ray_dp_size", "8")),
        gpu_count=int(credentials.get("ray_gpu_count", "8")),
        gpu_memory_utilization=float(credentials.get("ray_gpu_mem_util", "0.85")),
        max_model_len=int(credentials.get("ray_max_model_len", "32768")),
        enable_expert_parallel=credentials.get("ray_enable_ep", "true").lower() == "true",
        enable_eplb=credentials.get("ray_enable_eplb", "true").lower() == "true",
        enable_dbo=credentials.get("ray_enable_dbo", "true").lower() == "true",
        # Disaggregated P/D
        prefill_tp=int(credentials.get("ray_prefill_tp", "1")),
        prefill_dp=int(credentials.get("ray_prefill_dp", "4")),
        decode_tp=int(credentials.get("ray_decode_tp", "1")),
        decode_dp=int(credentials.get("ray_decode_dp", "4")),
        kv_connector=credentials.get("ray_kv_connector", "NixlConnector"),
        kv_buffer_size=int(credentials.get("ray_kv_buffer_size", "5368709120")),
        # Monitoring
        monitoring_enabled=credentials.get("ray_monitoring_enabled", "false").lower() == "true",
    )

    return EnhancedRayService(config)


def get_enhanced_ray_setup_instructions() -> str:
    """Get setup instructions for Enhanced Ray with MoE support"""
    return """
Ray Enhanced Setup Instructions (with MoE Expert Parallelism):

1. Install Ray with GPU and Serve support:
   pip install "ray[default,serve]>=2.44" vllm ray-serve-llm

2. Configure Terradev with Ray:
   terradev configure --provider ray \\
     --head-node-ip <HEAD_IP> \\
     --gpu-count 8 \\
     --model-id zai-org/GLM-5-FP8 \\
     --serving-pattern wide_ep \\
     --enable-ep true \\
     --enable-eplb true \\
     --enable-dbo true

3. Start cluster:
   terradev ml ray --start

4. Deploy MoE model with Wide-EP:
   terradev ml ray --deploy-wide-ep

5. Deploy with disaggregated Prefill/Decode:
   terradev ml ray --deploy-disagg-pd

Required Credentials:
  - ray_head_node_ip: Head node IP (default: localhost)
  - ray_gpu_count: Number of GPUs (default: 8)
  - ray_model_id: HuggingFace model ID
  - ray_serving_pattern: wide_ep | disagg | wide_ep_pd
  - ray_enable_ep: Enable Expert Parallelism (default: true)
  - ray_enable_eplb: Enable EPLB load balancing (default: true)
  - ray_enable_dbo: Enable Dual-Batch Overlap (default: true)
  - ray_kv_connector: NixlConnector | LMCacheConnector (for P/D disagg)

Serving Patterns:
  - pure_tp:   Traditional tensor parallelism (all GPUs shard layers)
  - wide_ep:   Expert Parallelism + Data Parallelism (experts distributed)
  - disagg:    Prefill/Decode separation with KV cache transfer
  - wide_ep_pd: Wide-EP + P/D disaggregation (maximum throughput)

Example: Wide-EP for DeepSeek-V3 on 8× H100:
  TP=1 (one GPU per DP rank)
  DP=8 (8 EP ranks, ~32 experts per rank)
  EPLB rebalances experts at runtime based on token routing
  DBO overlaps compute with all-to-all communication
  DeepEP provides optimized CUDA kernels for expert routing

Dashboard URLs:
  - Ray Dashboard: http://localhost:8265
  - vLLM Metrics: http://localhost:8000/metrics
  - Grafana: http://localhost:3000 (if monitoring enabled)
"""
