#!/usr/bin/env python3
"""
Agentic Topology Planner — Research-grounded fleet sizing for multi-agent LLM workloads.

Grounded in arXiv:2605.26297 "Agentic AI Workload Characteristics" (2026) which
characterises ReAct-style agents running Claude Code, Codex, and OpenClaw through
ADE-Bench, DABStep, GAIA, SWE-bench Pro, and Terminal-Bench 2.0.

Key empirical findings that drive every sizing decision here:

  1. DECODE DOMINATES: 91.0–98.6% of LLM time is decode, not prefill.
     → Bottleneck is memory bandwidth, not compute throughput.
     → A100 80GB (2 TB/s HBM2e) is optimal for worker tier.

  2. KV CACHE HIT RATES: 84.6–99.5% empirical hit ratio.
     → KV cache MUST remain resident. Eviction turns decode into expensive recompute.
     → Reasoning tier needs max VRAM (H100 NVLink 80GB × 2 for 70B+ models).
     → This is the single biggest lever on cost.

  3. CONTEXT FOOTPRINT: avg 37K–80K tokens, max up to 166K tokens.
     → SWE-bench Pro average: 68.7K–80.1K tokens.
     → At 70B fp16: KV cache ≈ 2 × layers × d_head × n_heads × seq_len × 2 bytes
        ≈ ~40 MB per 1K tokens → 80K context ≈ 3.2 GB per active agent.
     → Multiply by concurrency to get total KV pressure.

  4. TOOL CALLS: 2–29% of runtime. NOT negligible for retrieval workloads (GAIA: ~25%).
     → Tool tier needs high vCPU, fast disk I/O, outbound network.
     → Bash (code execution), WebFetch (HTTP), Read/Edit/Grep (file ops).

  5. TURN COUNTS: avg 12–62 per task, max 786 (pathological failure loops).
     → Each turn = one re-entry into the model. Scheduling overhead multiplies.
     → Failed agents accumulate MORE context (error messages → history growth).

  6. PREFILL IS SMALL: append-to-output ratio only 1.5×–7.3×, not 53×–560×.
     → Don't overprovision prefill capacity. Each turn only prefills the delta.
     → Disaggregated prefill/decode (P/D) is LESS critical here than for RAG.

See also: arXiv:2507.19635 "Efficient and Scalable Agentic AI with Heterogeneous Systems"
  → Heterogeneous H100 + Gaudi3 combo matches pure B200 TCO for some agentic workloads.
  → Directed graphs of compute+IO operations; MLIR-based decomposition for multi-vendor.
"""

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Any

if TYPE_CHECKING:
    from terradev_cli.core.kv_sharing import KVSharingPlan

from terradev_cli.core.dag_executor import DAGExecutor

logger = logging.getLogger(__name__)


# ── Research-grounded constants ───────────────────────────────────────────────

# KV cache bytes per token per layer (fp16 K + V tensors)
# For 70B Llama-class: 80 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes
KV_BYTES_PER_TOKEN_PER_LAYER = 512  # bytes
KV_LAYERS = {
    "7b": 32, "8b": 32,
    "13b": 40,
    "30b": 60, "34b": 60,
    "70b": 80, "72b": 80,
    "405b": 128,
}
MODEL_WEIGHTS_GB_FP16 = {
    "7b": 14, "8b": 16,
    "13b": 26,
    "30b": 60, "34b": 68,
    "70b": 140, "72b": 144,
    "405b": 810,
}

# Per-tier GPU VRAM available (GB)
GPU_VRAM_GB = {
    "H100_NVL": 94,     # H100 NVL 94GB (PCIe NVLink bridge variant)
    "H100_SXM": 80,     # H100 SXM 80GB
    "H100_PCIe": 80,    # H100 PCIe 80GB
    "A100_SXM_80": 80,  # A100 SXM 80GB
    "A100_PCIe_80": 80, # A100 PCIe 80GB
    "A100_SXM_40": 40,  # A100 SXM 40GB
    "A100_PCIe_40": 40, # A100 PCIe 40GB
    "RTX4090": 24,
    "L40S": 48,
    "H200_SXM": 141,    # H200 SXM 141GB HBM3e
}

# Memory bandwidth TB/s (key for decode throughput)
GPU_BANDWIDTH_TBS = {
    "H200_SXM": 4.8,
    "H100_SXM": 3.35,
    "H100_NVL": 3.35,
    "H100_PCIe": 2.0,
    "A100_SXM_80": 2.0,
    "A100_PCIe_80": 1.9,
    "A100_SXM_40": 2.0,
    "A100_PCIe_40": 1.6,
    "L40S": 0.864,
    "RTX4090": 1.008,
}

# Decode throughput tokens/sec at batch=8 (empirical estimates)
GPU_DECODE_TPS_70B = {
    "H200_SXM": 900,
    "H100_SXM": 600,
    "H100_NVL": 580,
    "H100_PCIe": 350,
    "A100_SXM_80": 320,
    "A100_PCIe_80": 280,
    "A100_SXM_40": 180,
}

# Approximate $/hr spot pricing (guidance only — use price_intelligence.py for live quotes)
GPU_SPOT_PRICE_HR = {
    "H100_SXM": 2.49,
    "H100_NVL": 2.20,
    "H100_PCIe": 1.89,
    "A100_SXM_80": 1.49,
    "A100_PCIe_80": 1.29,
    "A100_SXM_40": 0.89,
    "A100_PCIe_40": 0.79,
    "L40S": 0.69,
    "RTX4090": 0.44,
}


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class AgentRole:
    """A single compute tier within an agent fleet."""
    name: str                         # "reasoning" | "decode" | "cpu_tools"
    count: int                        # number of instances in this tier
    gpu_type: Optional[str]           # None for cpu_tools tier
    gpu_count_per_instance: int       # GPUs per VM (1 for most, 2+ for 70B reasoning)
    vcpu_count: int                   # vCPU per instance (CPU tier primary metric)
    concurrency_per_instance: int     # simultaneous agent slots this instance can hold
    role_profile: str                 # "kv_preservation" | "decode_throughput" | "cpu_io"
    tensor_parallel: int              # TP degree for vLLM serving
    warm_slots: int                   # pre-warmed model slots (feeds WarmPoolManager)
    context_budget_k_tokens: int      # max KV context per slot (K tokens)


@dataclass
class AgentScalingSpec:
    """Per-tier autoscaling rules. Grounded in research finding that GPU util
    is the WRONG signal for bursty agentic workloads."""
    reasoning_scale_metric: str = "ttft_p95_ms"
    reasoning_scale_out_threshold: float = 2000.0   # ms — reasoning turn is expensive
    reasoning_scale_in_cooldown_s: int = 300         # long cooldown; KV eviction costly

    decode_scale_metric: str = "decode_queue_depth"
    decode_scale_out_threshold: float = 6.0          # queued decode requests
    decode_scale_in_cooldown_s: int = 90

    cpu_scale_metric: str = "tool_latency_p95_ms"
    cpu_scale_out_threshold: float = 400.0           # ms tool execution latency
    cpu_scale_in_cooldown_s: int = 30

    max_reasoning_instances: int = 8
    max_decode_instances: int = 32
    max_cpu_instances: int = 16


@dataclass
class NetworkSpec:
    """Inter-tier networking requirements."""
    placement_group: str = "cluster"         # co-locate in same AZ
    vpc_peering: bool = True
    target_inter_tier_latency_ms: float = 2.0
    bandwidth_gbps: int = 25
    enable_rdma: bool = False                # RDMA only needed for decode↔reasoning KV transfer


@dataclass
class AgentFleetSpec:
    """Complete specification for an agent fleet across all tiers."""
    fleet_id: str
    model: str
    model_size_b: int                     # parameter count in billions
    n_agents: int                         # target concurrent agent loops
    tiers: Dict[str, AgentRole]
    networking: NetworkSpec
    autoscaling: AgentScalingSpec
    total_cost_hr_estimate: float         # sum of spot prices across all tiers
    kv_cache_budget_gb_total: float       # total KV headroom across reasoning tier
    reasoning: str                        # "thinking" | "instant" — affects token composition
    context_k_tokens: int = 120            # context window per agent in K tokens
    kv_sharing_plan: Optional[Any] = None  # KVSharingPlan if computed, else None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "model": self.model,
            "model_size_b": self.model_size_b,
            "n_agents": self.n_agents,
            "reasoning": self.reasoning,
            "tiers": {
                name: {
                    "count": role.count,
                    "gpu_type": role.gpu_type,
                    "gpu_count_per_instance": role.gpu_count_per_instance,
                    "vcpu_count": role.vcpu_count,
                    "concurrency_per_instance": role.concurrency_per_instance,
                    "role_profile": role.role_profile,
                    "tensor_parallel": role.tensor_parallel,
                    "warm_slots": role.warm_slots,
                    "context_budget_k_tokens": role.context_budget_k_tokens,
                }
                for name, role in self.tiers.items()
            },
            "networking": {
                "placement_group": self.networking.placement_group,
                "vpc_peering": self.networking.vpc_peering,
                "target_inter_tier_latency_ms": self.networking.target_inter_tier_latency_ms,
                "bandwidth_gbps": self.networking.bandwidth_gbps,
            },
            "total_cost_hr_estimate": round(self.total_cost_hr_estimate, 2),
            "kv_cache_budget_gb_total": round(self.kv_cache_budget_gb_total, 1),
            "context_k_tokens": self.context_k_tokens,
            "kv_sharing_plan": self.kv_sharing_plan.to_dict() if self.kv_sharing_plan else None,
        }


@dataclass
class CostBreakdown:
    """Per-tier cost breakdown."""
    reasoning_hr: float
    decode_hr: float
    cpu_hr: float
    total_hr: float
    daily: float
    monthly: float
    cost_per_agent_hr: float


# ── Sizing logic ──────────────────────────────────────────────────────────────


class AgentTopologyPlanner:
    """
    Maps an agent count + model to a hardware fleet spec.

    All sizing rules are derived from arXiv:2605.26297 empirical measurements.
    The planner deliberately avoids over-engineering: it produces a starting
    configuration that is safe to run, not a theoretically optimal minimum.
    """

    # Context budget: 80th percentile across benchmarks (SWE-bench Pro: 80K avg)
    # We provision for the 95th pct tail to avoid KV thrashing.
    CONTEXT_P95_TOKENS = 120_000

    # From research: decode accounts for 91-98% of LLM time at ideal cache hit rate.
    # We target 85% decode efficiency (5% below ideal to account for some thrashing).
    TARGET_DECODE_EFFICIENCY = 0.85

    # Default context budget in K tokens (P95 from arXiv:2605.26297).
    # Override with context_k parameter for known workloads.
    DEFAULT_CONTEXT_K = 120

    # Reasoning tier handles orchestration and planning. Research shows 1 planner
    # can serve multiple workers sequentially since planners issue batch dispatches.
    AGENTS_PER_REASONING_INSTANCE = 10

    # Decode tier: each A100 80GB can hold ~12 concurrent 70B agent slots with 80K context.
    # (80GB - 140GB model) → negative; need TP=2. For 13B: (80GB - 26GB = 54GB free)
    # 54GB / (120K tokens × 0.04 GB/1K tokens at 13B) = 54 / 4.8 ≈ 11 slots
    # We use conservative 8 for headroom.
    AGENTS_PER_DECODE_INSTANCE_SMALL_MODEL = 8   # ≤13B
    AGENTS_PER_DECODE_INSTANCE_LARGE_MODEL = 4   # 14B–70B (needs more VRAM per slot)

    # Tool/CPU tier: research shows tools = 2-29% of runtime. GAIA (retrieval-heavy)
    # hits 25-29%. We provision 1 CPU instance per 8 agents as baseline.
    AGENTS_PER_CPU_INSTANCE = 8

    def _parse_model_size(self, model: str) -> int:
        """Extract parameter count in billions from model name string."""
        import re
        model_lower = model.lower()
        # Match patterns: 70b, 7b, 405b, 8x7b (MoE → 56b effective), etc.
        moe = re.search(r'(\d+)x(\d+)b', model_lower)
        if moe:
            return int(moe.group(1)) * int(moe.group(2))
        match = re.search(r'(\d+(?:\.\d+)?)b', model_lower)
        if match:
            return int(float(match.group(1)))
        # Default to 70B if unknown — conservative sizing
        logger.warning(f"Could not parse model size from '{model}', defaulting to 70B sizing")
        return 70

    def _select_gpu_type(
        self, model_size_b: int, tier: str, require_nvlink: bool = False
    ) -> tuple[str, int]:
        """
        Select GPU type and TP degree for a tier.
        Returns (gpu_type, tensor_parallel_degree).

        Reasoning:
          - Reasoning tier prioritises VRAM (KV cache preservation).
          - Decode tier prioritises memory bandwidth (token throughput).
          - Both tiers need enough VRAM to hold model + KV cache.
        """
        weights_gb = MODEL_WEIGHTS_GB_FP16.get(f"{model_size_b}b", model_size_b * 2)
        kv_per_slot_gb = (
            self.CONTEXT_P95_TOKENS
            * KV_BYTES_PER_TOKEN_PER_LAYER
            * KV_LAYERS.get(f"{model_size_b}b", 80)
            / 1e9
        )

        if tier == "reasoning":
            # Need: model weights + N_slots × kv_per_slot. Prioritise NVLink for large models.
            # Aim for 4 concurrent reasoning slots per instance.
            slots = 4
            required_vram = weights_gb + slots * kv_per_slot_gb
            if required_vram <= 80:
                return ("H100_SXM", 1)
            elif required_vram <= 160:
                return ("H100_SXM", 2)   # TP=2 across two H100 SXMs
            else:
                return ("H100_SXM", 4)   # TP=4

        elif tier == "decode":
            # Prioritise bandwidth for decode throughput.
            # A100 SXM 80GB = 2TB/s bandwidth, great price/perf.
            if model_size_b <= 13:
                return ("A100_SXM_80", 1) if weights_gb <= 70 else ("H100_SXM", 1)
            elif model_size_b <= 34:
                return ("A100_SXM_80", 1)
            elif model_size_b <= 70:
                return ("A100_SXM_80", 2)   # TP=2: each A100 holds half the model
            else:
                return ("H100_SXM", 4)

        else:
            return (None, 1)   # cpu_tools tier

    def _compute_kv_budget(
        self, gpu_type: str, gpu_count: int, model_size_b: int, tp: int
    ) -> float:
        """Total KV cache headroom in GB per instance."""
        if gpu_type is None:
            return 0.0
        total_vram = GPU_VRAM_GB.get(gpu_type, 80) * gpu_count
        weights_per_gpu = MODEL_WEIGHTS_GB_FP16.get(f"{model_size_b}b", model_size_b * 2) / tp
        # vLLM reserves 5% for CUDA context, page table, etc.
        available = total_vram - (weights_per_gpu * tp) - (total_vram * 0.05)
        return max(0.0, available)

    def infer_from_agent_count(
        self,
        n_agents: int,
        model: str = "meta-llama/Llama-3.1-70B-Instruct",
        reasoning: str = "instant",
        providers: Optional[List[str]] = None,
        context_k: int = 120,
        sharing_topology: str = "broadcast",
        shared_fraction: Optional[float] = None,
        dtype: str = "fp16",
    ) -> AgentFleetSpec:
        """
        Produce a complete AgentFleetSpec from a single agent count.

        This is the 'magic' entry point: terradev provision --agents N --context K.
        All sizing is grounded in the arXiv empirical data above.

        Parameters
        ----------
        context_k : int
            Context window per agent in K tokens (default 120K = P95 from research).
            Use the actual value from your workload: 8 for simple tasks, 32–128 for
            code agents, up to 166K for the research P99 tail.
        sharing_topology : str
            "broadcast" | "star" | "chain" | "none"
            How agents share prefill context. broadcast = all share system prompt + task.
        shared_fraction : float, optional
            Override the default sharing fraction for the topology.
        dtype : str
            KV cache dtype. "fp8" halves KV memory vs "fp16".
        """
        fleet_id = f"ag_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        model_size_b = self._parse_model_size(model)

        # Compute KV sharing plan to inform VRAM sizing
        kv_plan = None
        try:
            from terradev_cli.core.kv_sharing import MultiAgentVRAMPlanner, SharingTopology
            topology_map = {
                "broadcast": SharingTopology.BROADCAST,
                "star": SharingTopology.STAR,
                "chain": SharingTopology.CHAIN,
                "none": SharingTopology.NONE,
            }
            topo = topology_map.get(sharing_topology, SharingTopology.BROADCAST)
            kv_plan = MultiAgentVRAMPlanner().compute(
                n_agents=n_agents,
                context_k=context_k,
                model=model,
                topology=topo,
                shared_fraction=shared_fraction,
                dtype=dtype,
            )
        except Exception as e:
            logger.debug("KV sharing plan skipped: %s", e)

        # Use context_k from KV plan if available, else P95 default
        effective_context_k = context_k if context_k else self.DEFAULT_CONTEXT_K
        # Update CONTEXT_P95_TOKENS to reflect the requested context
        actual_context_tokens = effective_context_k * 1000

        # ── Reasoning tier ────────────────────────────────────────────────────
        reasoning_instances = max(1, math.ceil(n_agents / self.AGENTS_PER_REASONING_INSTANCE))
        r_gpu, r_tp = self._select_gpu_type(model_size_b, "reasoning")
        r_gpu_count = r_tp   # one GPU per TP rank
        r_kv_budget = self._compute_kv_budget(r_gpu, r_gpu_count, model_size_b, r_tp)
        r_concurrency = max(1, int(r_kv_budget / (
            actual_context_tokens
            * KV_BYTES_PER_TOKEN_PER_LAYER
            * KV_LAYERS.get(f"{model_size_b}b", 80)
            / 1e9
        ))) if r_kv_budget > 0 else 4

        reasoning_role = AgentRole(
            name="reasoning",
            count=reasoning_instances,
            gpu_type=r_gpu,
            gpu_count_per_instance=r_gpu_count,
            vcpu_count=16,
            concurrency_per_instance=min(r_concurrency, 8),
            role_profile="kv_preservation",
            tensor_parallel=r_tp,
            warm_slots=min(r_concurrency, 4),
            context_budget_k_tokens=effective_context_k,
        )

        # ── Decode tier ───────────────────────────────────────────────────────
        agents_per_decode = (
            self.AGENTS_PER_DECODE_INSTANCE_SMALL_MODEL
            if model_size_b <= 13
            else self.AGENTS_PER_DECODE_INSTANCE_LARGE_MODEL
        )
        decode_instances = max(1, math.ceil(n_agents / agents_per_decode))
        d_gpu, d_tp = self._select_gpu_type(model_size_b, "decode")
        d_gpu_count = d_tp
        d_kv_budget = self._compute_kv_budget(d_gpu, d_gpu_count, model_size_b, d_tp)
        d_concurrency = max(1, int(d_kv_budget / (
            actual_context_tokens
            * KV_BYTES_PER_TOKEN_PER_LAYER
            * KV_LAYERS.get(f"{model_size_b}b", 80)
            / 1e9
        ))) if d_kv_budget > 0 else agents_per_decode

        decode_role = AgentRole(
            name="decode",
            count=decode_instances,
            gpu_type=d_gpu,
            gpu_count_per_instance=d_gpu_count,
            vcpu_count=16,
            concurrency_per_instance=min(d_concurrency, agents_per_decode),
            role_profile="decode_throughput",
            tensor_parallel=d_tp,
            warm_slots=min(d_concurrency, agents_per_decode),
            context_budget_k_tokens=effective_context_k,
        )

        # ── CPU tools tier ────────────────────────────────────────────────────
        # Research: tools = 2-29% of runtime. For retrieval-heavy workloads (GAIA),
        # WebFetch/WebSearch can hit 25%. We provision generously since CPU is cheap.
        cpu_instances = max(1, math.ceil(n_agents / self.AGENTS_PER_CPU_INSTANCE))
        cpu_role = AgentRole(
            name="cpu_tools",
            count=cpu_instances,
            gpu_type=None,
            gpu_count_per_instance=0,
            vcpu_count=48,          # c5.12xlarge-class: high core count for parallel tool calls
            concurrency_per_instance=n_agents // cpu_instances + 4,
            role_profile="cpu_io",
            tensor_parallel=1,
            warm_slots=0,
            context_budget_k_tokens=0,
        )

        # ── Cost estimate ─────────────────────────────────────────────────────
        r_cost = reasoning_instances * r_gpu_count * GPU_SPOT_PRICE_HR.get(r_gpu, 2.0)
        d_cost = decode_instances * d_gpu_count * GPU_SPOT_PRICE_HR.get(d_gpu or "", 1.5)
        cpu_cost = cpu_instances * 0.60   # ~$0.60/hr for 48-vCPU spot instance
        total_cost = r_cost + d_cost + cpu_cost

        # ── Networking ────────────────────────────────────────────────────────
        # Reasoning → decode KV transfer not needed (disaggregated P/D not used here;
        # agents share context within their tier). Standard 25Gbps Ethernet is fine.
        networking = NetworkSpec(
            placement_group="cluster",
            vpc_peering=True,
            target_inter_tier_latency_ms=2.0,
            bandwidth_gbps=25,
            enable_rdma=model_size_b >= 70,   # RDMA only if cross-GPU KV needed for 70B+
        )

        autoscaling = AgentScalingSpec(
            max_reasoning_instances=max(reasoning_instances * 3, 4),
            max_decode_instances=max(decode_instances * 3, 8),
            max_cpu_instances=max(cpu_instances * 4, 8),
        )

        kv_total = (
            reasoning_instances * r_kv_budget
            + decode_instances * d_kv_budget
        )

        return AgentFleetSpec(
            fleet_id=fleet_id,
            model=model,
            model_size_b=model_size_b,
            n_agents=n_agents,
            tiers={
                "reasoning": reasoning_role,
                "decode": decode_role,
                "cpu_tools": cpu_role,
            },
            networking=networking,
            autoscaling=autoscaling,
            total_cost_hr_estimate=total_cost,
            kv_cache_budget_gb_total=kv_total,
            reasoning=reasoning,
            context_k_tokens=effective_context_k,
            kv_sharing_plan=kv_plan,
        )

    def from_explicit(
        self,
        n_agents: int,
        model: str,
        planner_gpu: str,
        planner_count: int,
        worker_gpu: str,
        worker_count: int,
        cpu_cores: int = 48,
        cpu_count: int = 1,
        reasoning: str = "instant",
    ) -> AgentFleetSpec:
        """Build a spec from explicit tier parameters (--planner-gpu, etc.)."""
        fleet_id = f"ag_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        model_size_b = self._parse_model_size(model)

        gpu_norm = planner_gpu.upper().replace("-", "_")
        worker_gpu_norm = worker_gpu.upper().replace("-", "_")

        # Best-effort TP selection
        weights = MODEL_WEIGHTS_GB_FP16.get(f"{model_size_b}b", model_size_b * 2)
        planner_vram = GPU_VRAM_GB.get(gpu_norm, 80)
        worker_vram = GPU_VRAM_GB.get(worker_gpu_norm, 80)
        r_tp = max(1, math.ceil(weights / planner_vram))
        d_tp = max(1, math.ceil(weights / worker_vram))

        r_kv = self._compute_kv_budget(gpu_norm, r_tp, model_size_b, r_tp)
        d_kv = self._compute_kv_budget(worker_gpu_norm, d_tp, model_size_b, d_tp)

        r_conc = max(1, int(r_kv / max(0.1, (
            self.CONTEXT_P95_TOKENS * KV_BYTES_PER_TOKEN_PER_LAYER
            * KV_LAYERS.get(f"{model_size_b}b", 80) / 1e9
        ))))
        d_conc = max(1, int(d_kv / max(0.1, (
            self.CONTEXT_P95_TOKENS * KV_BYTES_PER_TOKEN_PER_LAYER
            * KV_LAYERS.get(f"{model_size_b}b", 80) / 1e9
        ))))

        reasoning_role = AgentRole(
            name="reasoning", count=planner_count, gpu_type=gpu_norm,
            gpu_count_per_instance=r_tp, vcpu_count=16,
            concurrency_per_instance=min(r_conc, 8), role_profile="kv_preservation",
            tensor_parallel=r_tp, warm_slots=min(r_conc, 4),
            context_budget_k_tokens=self.CONTEXT_P95_TOKENS // 1000,
        )
        decode_role = AgentRole(
            name="decode", count=worker_count, gpu_type=worker_gpu_norm,
            gpu_count_per_instance=d_tp, vcpu_count=16,
            concurrency_per_instance=min(d_conc, 8), role_profile="decode_throughput",
            tensor_parallel=d_tp, warm_slots=min(d_conc, 4),
            context_budget_k_tokens=self.CONTEXT_P95_TOKENS // 1000,
        )
        cpu_role = AgentRole(
            name="cpu_tools", count=cpu_count, gpu_type=None,
            gpu_count_per_instance=0, vcpu_count=cpu_cores,
            concurrency_per_instance=cpu_cores // 2, role_profile="cpu_io",
            tensor_parallel=1, warm_slots=0, context_budget_k_tokens=0,
        )

        r_cost = planner_count * r_tp * GPU_SPOT_PRICE_HR.get(gpu_norm, 2.0)
        d_cost = worker_count * d_tp * GPU_SPOT_PRICE_HR.get(worker_gpu_norm, 1.5)
        cpu_cost = cpu_count * 0.60

        return AgentFleetSpec(
            fleet_id=fleet_id, model=model, model_size_b=model_size_b,
            n_agents=n_agents,
            tiers={"reasoning": reasoning_role, "decode": decode_role, "cpu_tools": cpu_role},
            networking=NetworkSpec(enable_rdma=model_size_b >= 70),
            autoscaling=AgentScalingSpec(),
            total_cost_hr_estimate=r_cost + d_cost + cpu_cost,
            kv_cache_budget_gb_total=(planner_count * r_kv) + (worker_count * d_kv),
            reasoning=reasoning,
        )

    def estimate_cost(self, spec: AgentFleetSpec) -> CostBreakdown:
        """Produce a per-tier cost breakdown from a fleet spec."""
        r = spec.tiers.get("reasoning")
        d = spec.tiers.get("decode")
        c = spec.tiers.get("cpu_tools")

        r_hr = (r.count * r.gpu_count_per_instance * GPU_SPOT_PRICE_HR.get(r.gpu_type or "", 2.0)) if r else 0
        d_hr = (d.count * d.gpu_count_per_instance * GPU_SPOT_PRICE_HR.get(d.gpu_type or "", 1.5)) if d else 0
        c_hr = (c.count * 0.60) if c else 0
        total_hr = r_hr + d_hr + c_hr

        return CostBreakdown(
            reasoning_hr=round(r_hr, 2),
            decode_hr=round(d_hr, 2),
            cpu_hr=round(c_hr, 2),
            total_hr=round(total_hr, 2),
            daily=round(total_hr * 24, 2),
            monthly=round(total_hr * 24 * 30, 2),
            cost_per_agent_hr=round(total_hr / max(1, spec.n_agents), 4),
        )

    def build_dag(self, spec: AgentFleetSpec) -> DAGExecutor:
        """
        Build a 5-wave DAG for provisioning the fleet.

        Wave 0: quote all tiers in parallel (no deps)
        Wave 1: provision all tiers in parallel (no deps on each other, only on quotes)
        Wave 2: configure cross-tier networking (depends on all tier provisions)
        Wave 3: deploy inference stack per GPU tier (depends on networking)
        Wave 4: register fleet + warm pools (depends on all deployments)
        """
        dag = DAGExecutor(name=f"fleet_{spec.fleet_id}", reuse_pool=True)

        # Wave 0: parallel quotes
        dag.add_node("quote_reasoning", lambda ctx: {
            "tier": "reasoning", "gpu": spec.tiers["reasoning"].gpu_type,
            "count": spec.tiers["reasoning"].count,
        })
        dag.add_node("quote_decode", lambda ctx: {
            "tier": "decode", "gpu": spec.tiers["decode"].gpu_type,
            "count": spec.tiers["decode"].count,
        })
        dag.add_node("quote_cpu", lambda ctx: {
            "tier": "cpu_tools", "vcpu": spec.tiers["cpu_tools"].vcpu_count,
            "count": spec.tiers["cpu_tools"].count,
        })

        # Wave 1: parallel provisions (each depends on its quote)
        dag.add_node(
            "provision_reasoning",
            lambda ctx: {
                "status": "planned",
                "tier": "reasoning",
                "instances": spec.tiers["reasoning"].count,
                "gpu": spec.tiers["reasoning"].gpu_type,
                "tp": spec.tiers["reasoning"].tensor_parallel,
            },
            depends_on={"quote_reasoning"},
        )
        dag.add_node(
            "provision_decode",
            lambda ctx: {
                "status": "planned",
                "tier": "decode",
                "instances": spec.tiers["decode"].count,
                "gpu": spec.tiers["decode"].gpu_type,
                "tp": spec.tiers["decode"].tensor_parallel,
            },
            depends_on={"quote_decode"},
        )
        dag.add_node(
            "provision_cpu",
            lambda ctx: {
                "status": "planned",
                "tier": "cpu_tools",
                "instances": spec.tiers["cpu_tools"].count,
                "vcpu": spec.tiers["cpu_tools"].vcpu_count,
            },
            depends_on={"quote_cpu"},
        )

        # Wave 2: networking (depends on all provisions)
        dag.add_node(
            "configure_networking",
            lambda ctx: {
                "placement_group": spec.networking.placement_group,
                "inter_tier_latency_target_ms": spec.networking.target_inter_tier_latency_ms,
                "rdma": spec.networking.enable_rdma,
                "status": "planned",
            },
            depends_on={"provision_reasoning", "provision_decode", "provision_cpu"},
        )

        # Wave 3: deploy inference (depends on networking)
        dag.add_node(
            "deploy_reasoning_inference",
            lambda ctx: {
                "vllm_args": _build_vllm_args(spec, "reasoning"),
                "status": "planned",
            },
            depends_on={"configure_networking"},
        )
        dag.add_node(
            "deploy_decode_inference",
            lambda ctx: {
                "vllm_args": _build_vllm_args(spec, "decode"),
                "status": "planned",
            },
            depends_on={"configure_networking"},
        )

        # Wave 4: fleet registration (depends on both deployments)
        dag.add_node(
            "register_fleet",
            lambda ctx: {
                "fleet_id": spec.fleet_id,
                "model": spec.model,
                "n_agents": spec.n_agents,
                "tiers_ready": ["reasoning", "decode", "cpu_tools"],
                "status": "planned",
                "cost_hr": spec.total_cost_hr_estimate,
            },
            depends_on={"deploy_reasoning_inference", "deploy_decode_inference"},
        )

        return dag

    def print_plan(self, spec: AgentFleetSpec) -> None:
        """Print a human-readable fleet plan."""
        cost = self.estimate_cost(spec)
        print(f"\nAGENT FLEET PLAN  [{spec.fleet_id}]")
        print(f"Model: {spec.model}  ({spec.model_size_b}B params)")
        print(f"Target: {spec.n_agents} concurrent agent loops")
        print(f"Reasoning mode: {spec.reasoning}")
        print()
        print(f"{'TIER':<16} {'INSTANCES':>9} {'GPU':>14} {'TP':>4} {'CONC':>6} {'CONTEXT':>8}  {'$/HR':>7}")
        print("-" * 76)
        for tier_name, role in spec.tiers.items():
            gpu_str = role.gpu_type or "CPU"
            ctx_str = f"{role.context_budget_k_tokens}K" if role.context_budget_k_tokens > 0 else "n/a"
            tier_cost = (
                role.count * role.gpu_count_per_instance
                * GPU_SPOT_PRICE_HR.get(role.gpu_type or "", 0.60)
                if role.gpu_type else role.count * 0.60
            )
            print(
                f"{tier_name:<16} {role.count:>9} {gpu_str:>14} "
                f"{role.tensor_parallel:>4} {role.concurrency_per_instance:>6} "
                f"{ctx_str:>8}  ${tier_cost:>6.2f}"
            )
        print("-" * 76)
        print(f"{'TOTAL':<16} {'':>9} {'':>14} {'':>4} {'':>6} {'':>8}  ${cost.total_hr:>6.2f}/hr")
        print()
        print(f"KV cache headroom:  {spec.kv_cache_budget_gb_total:.0f} GB total across GPU tiers")
        print(f"Daily estimate:     ${cost.daily:.2f}")
        print(f"Monthly estimate:   ${cost.monthly:.2f}")
        print(f"Cost per agent/hr:  ${cost.cost_per_agent_hr:.4f}")
        print()
        print("Research basis: arXiv:2605.26297 — decode dominates (91-98% of LLM time)")
        print("KV cache hit rate target: >85% (eviction = expensive recompute)")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_vllm_args(spec: AgentFleetSpec, tier: str) -> Dict[str, Any]:
    """Build vLLM server arguments for a tier. Integrates with vllm_service.py."""
    role = spec.tiers[tier]
    args: Dict[str, Any] = {
        "model": spec.model,
        "tensor-parallel-size": role.tensor_parallel,
        "max-model-len": role.context_budget_k_tokens * 1000,
        "gpu-memory-utilization": 0.90,
        "enable-prefix-caching": True,       # CRITICAL: 84-99% hit rate requires this on
        "kv-connector": "offloading",         # auto-offload KV to CPU when under pressure
        "speculative-config": {"method": "mtp"},  # decode speedup
        "enable-sleep-mode": True,            # fast cold-start
    }
    if tier == "reasoning" and spec.reasoning == "thinking":
        # Reasoning models generate thinking tokens (45-67% of output per research).
        # Allow larger max tokens for extended thinking output.
        args["max-tokens"] = 32768
    if role.role_profile == "decode_throughput":
        # Decode tier: continuous batching for high-throughput streaming.
        args["max-num-seqs"] = role.concurrency_per_instance * 2
        args["disable-log-requests"] = False
    return args
