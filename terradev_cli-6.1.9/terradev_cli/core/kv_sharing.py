#!/usr/bin/env python3
"""
Multi-Agent KV Cache Sharing — VRAM Arithmetic and Sharing Topology Planner.

── The Problem ──────────────────────────────────────────────────────────────────

In multi-agent systems each agent independently maintains a KV cache. When
N agents run the same model with a shared system prompt or shared task context,
every agent re-prefills and stores its own copy of that shared prefix in GPU HBM.

This redundancy compounds at scale:
  - 20 agents × 32K context × 70B fp16 = 20 × 1.25 GB = 25 GB of KV in VRAM
  - With 80% context sharing (16K shared prefix): shared portion = 16 × 0.625 GB = 10 GB
    → Currently stored 20× (once per agent) = 200 GB wasted
    → With KV sharing: stored 1× = 10 GB total + 20 × (16K unique context × 0.625 GB) = 10 + 12.5 = 22.5 GB
    → Saving: ~177 GB (88% VRAM reduction for shared KV)

── Empirical grounding (arXiv:2605.26297) ──────────────────────────────────────

  - KV cache hit rate: 84.6–99.5% in ReAct-style agents
  - Context footprint: avg 37K–80K tokens, P95 up to 166K tokens
  - On Apple M4 Pro (10.2 GB cache budget): only 3 agents fit at 8K context FP16
  - Re-prefill cost at 4K context on M4 Pro: 15.7 seconds per eviction
  - Cloud GPU (H100 SXM): re-prefill at 4K ≈ 0.13s (30K tokens/sec prefill throughput)
    → Still significant at 32K context: 32K/30K ≈ 1.07s lost per eviction

── Sharing topologies ───────────────────────────────────────────────────────────

  BROADCAST: All agents share the same system prompt + task description (prefix).
    The shared portion can be stored ONCE, with each agent's individual history
    stored separately. This is the common case for agentic frameworks (LangGraph,
    AutoGen, Claude Code).
    Example: 20 agents working on the same codebase — repo context is shared.

  STAR: One orchestrator agent + N worker agents. Orchestrator's context may be
    partially shared with workers. Common in planner-executor topologies.
    Example: Planner generates a plan (512 tokens) → all workers have plan in context.

  CHAIN: Agent N depends on Agent N-1's output. Each agent has unique context.
    Sharing is low (only initial system prompt shared).
    Example: Multi-step pipeline with sequential refinement.

  NONE: All contexts are unique. No sharing possible.
    This is the worst case — KV VRAM scales linearly with agent count.

── Infrastructure implications ─────────────────────────────────────────────────

  Nobody's CLI does this math today. Terradev targets:

    terradev provision --agents 20 --context 32k --model llama-70b

  To output:

    Fleet plan: 20 agents × Llama-3.1-70B, 32K context, BROADCAST topology
    ┌──────────────┬──────────┬───────────┬──────────┬──────────┐
    │ Tier         │ GPU      │ Instances │ VRAM     │ KV alloc │
    ├──────────────┼──────────┼───────────┼──────────┼──────────┤
    │ shared_kv    │ H100 SXM │ 1         │ 80 GB    │ 15.6 GB  │
    │ reasoning    │ H100 SXM │ 2         │ 160 GB   │ 6.25 GB  │
    │ decode       │ A100 80G │ 5         │ 400 GB   │ remaining│
    │ cpu_tools    │ 48-vCPU  │ 3         │ —        │ —        │
    └──────────────┴──────────┴───────────┴──────────┴──────────┘
    Total VRAM: 640 GB | KV budget: 167 GB | Cost: $14.25/hr
    Without sharing: would need 2,560 GB (4× more VRAM)
    KV sharing saves: $42.75/hr (75% cost reduction)

── CXL implication ─────────────────────────────────────────────────────────────

  When CXL 3.0 becomes the KV transport (see pd_transport.py), shared KV lives
  in the CXL pool as a single memory-mapped region. All agents map the same
  physical pages. No copy, no transfer. The VRAM arithmetic here becomes:
    required_vram = model_weights_only
    required_cxl_dram = shared_kv_gb + per_agent_unique_kv_gb × n_agents
  This reduces GPU VRAM requirements by 60–80% for broadcast topologies.
"""

from __future__ import annotations

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── KV cache byte arithmetic ──────────────────────────────────────────────────

# From agentic_topology.py constants (reproduced here to avoid circular import)
_KV_BYTES_PER_TOKEN_PER_LAYER = 512   # fp16, K+V tensors
_KV_LAYERS: Dict[str, int] = {
    "7b": 32, "8b": 32, "13b": 40,
    "30b": 60, "34b": 60,
    "70b": 80, "72b": 80,
    "405b": 128,
}
_MODEL_WEIGHTS_GB_FP16: Dict[str, float] = {
    "7b": 14, "8b": 16, "13b": 26,
    "30b": 60, "34b": 68,
    "70b": 140, "72b": 144,
    "405b": 810,
}
_GPU_VRAM_GB: Dict[str, float] = {
    "H100_SXM": 80, "H100_NVL": 94, "H100_PCIe": 80,
    "H200_SXM": 141,
    "A100_SXM_80": 80, "A100_PCIe_80": 80,
    "A100_SXM_40": 40, "A100_PCIe_40": 40,
    "L40S": 48, "RTX4090": 24,
}

# H100 SXM prefill throughput (tokens/sec, prompt processing)
_PREFILL_TPS: Dict[str, float] = {
    "H100_SXM": 30_000,
    "H100_NVL": 28_000,
    "H100_PCIe": 18_000,
    "A100_SXM_80": 15_000,
    "A100_PCIe_80": 12_000,
    "edge_m4_pro": 640,   # Apple M4 Pro (arXiv reference: 15.7s / 4K tokens ≈ 254 t/s prefill)
}


# ── Topology enum ─────────────────────────────────────────────────────────────


class SharingTopology(Enum):
    """
    Describes how agents share context with each other.

    The topology determines how much KV cache can be deduplicated
    across the fleet, which directly drives the VRAM arithmetic.
    """
    BROADCAST = "broadcast"   # All agents share a common prefix (most savings)
    STAR = "star"             # One orchestrator + N workers share partial context
    CHAIN = "chain"           # Sequential dependency, low sharing
    NONE = "none"             # No sharing (worst case, linear VRAM scaling)


# ── Eviction cost model ───────────────────────────────────────────────────────


@dataclass
class EvictionCostModel:
    """
    Quantifies the latency penalty when a KV cache entry is evicted.

    When an agent's KV cache is evicted (due to VRAM pressure), the next
    time that agent runs it must re-prefill its full context from scratch.
    This is the 'KV eviction tax' that makes under-provisioned fleets slow.

    Empirical reference (arXiv:2605.26297):
      Apple M4 Pro, 4K tokens, fp16: re-prefill = 15.7 seconds
      → prefill throughput ≈ 4096 / 15.7 ≈ 261 tokens/sec

    Cloud reference (H100 SXM):
      H100 SXM, 30K tokens/sec prefill throughput
      4K tokens → 4096 / 30000 ≈ 0.14s
      32K tokens → 32768 / 30000 ≈ 1.09s
      128K tokens → 131072 / 30000 ≈ 4.37s
    """

    gpu_type: str = "H100_SXM"

    def reprefill_seconds(self, context_tokens: int) -> float:
        """Time in seconds to re-prefill context_tokens on this GPU."""
        tps = _PREFILL_TPS.get(self.gpu_type, 15_000)
        return context_tokens / tps

    def eviction_cost_ms(self, context_tokens: int) -> float:
        return self.reprefill_seconds(context_tokens) * 1000

    def agents_evicted_per_hour(
        self, n_agents: int, vram_fit: int, turn_duration_s: float = 30.0
    ) -> float:
        """
        Estimate eviction frequency when n_agents > vram_fit (VRAM pressure).

        If 20 agents but only 12 fit, the LRU eviction rate is proportional
        to how fast the working set cycles through available slots.
        """
        if n_agents <= vram_fit:
            return 0.0
        overflow = n_agents - vram_fit
        # Each turn cycle, overflow agents must evict someone.
        # Turns per hour: 3600 / avg_turn_duration_s
        turns_per_hour = 3600 / turn_duration_s
        return overflow * turns_per_hour

    def throughput_penalty_fraction(
        self, n_agents: int, vram_fit: int, context_k: int,
        turn_duration_s: float = 30.0,
    ) -> float:
        """
        Fraction of agent time wasted on re-prefills due to VRAM pressure.

        Returns 0.0 (no penalty) when n_agents <= vram_fit.
        Returns up to 1.0 (100% of time spent re-prefilling) in pathological cases.
        """
        if n_agents <= vram_fit:
            return 0.0
        evictions_hr = self.agents_evicted_per_hour(n_agents, vram_fit, turn_duration_s)
        reprefill_s = self.reprefill_seconds(context_k * 1000)
        total_reprefill_s_hr = evictions_hr * reprefill_s
        total_agent_compute_s_hr = n_agents * 3600
        return min(1.0, total_reprefill_s_hr / total_agent_compute_s_hr)


# ── Per-agent KV budget ───────────────────────────────────────────────────────


@dataclass
class AgentKVBudget:
    """
    VRAM arithmetic for a single agent's KV cache at a given context length.

    All fields are computed by MultiAgentVRAMPlanner — do not instantiate directly.
    """
    model: str
    model_size_b: int
    context_k_tokens: int               # context window in K tokens
    dtype: str                          # "fp16" | "fp8"
    n_layers: int
    kv_bytes_per_token: int             # per-layer K+V bytes per token
    kv_gb_per_agent: float              # total KV GB for one agent at full context
    model_weights_gb: float             # model weights (shared across all agents on same GPU)
    gpu_type: str
    gpu_vram_gb: float
    vram_for_model_gb: float            # model weights portion consumed from VRAM
    vram_available_for_kv_gb: float     # VRAM left after model weights (5% CUDA overhead)
    agents_per_gpu: int                 # how many agent KV caches fit on this GPU
    fits: bool                          # True if at least 1 agent fits
    eviction_cost: EvictionCostModel

    def kv_utilisation(self, n_agents: int) -> float:
        """KV VRAM utilisation for n_agents on this GPU."""
        return min(1.0, n_agents * self.kv_gb_per_agent / self.vram_available_for_kv_gb)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "context_k": self.context_k_tokens,
            "dtype": self.dtype,
            "kv_gb_per_agent": round(self.kv_gb_per_agent, 3),
            "model_weights_gb": round(self.model_weights_gb, 1),
            "vram_available_for_kv_gb": round(self.vram_available_for_kv_gb, 1),
            "agents_per_gpu": self.agents_per_gpu,
            "reprefill_cost_s": round(self.eviction_cost.reprefill_seconds(self.context_k_tokens * 1000), 2),
        }


# ── Sharing plan ──────────────────────────────────────────────────────────────


@dataclass
class KVSharingPlan:
    """
    Fleet-level KV sharing plan: shared prefix + per-agent unique context.

    The plan is the output of MultiAgentVRAMPlanner.compute() and is embedded
    into AgentFleetSpec.kv_sharing_plan for use by the provisioner.
    """
    fleet_id: str
    n_agents: int
    model: str
    model_size_b: int
    context_k_tokens: int               # total context per agent
    shared_prefix_k_tokens: int         # shared by all agents (e.g. system prompt + task)
    per_agent_unique_k_tokens: int      # unique history per agent
    topology: SharingTopology
    dtype: str

    # Budget results
    shared_kv_gb: float                 # KV for the shared prefix (stored ONCE)
    per_agent_unique_kv_gb: float       # KV for per-agent unique portion
    total_kv_naive_gb: float            # Without sharing: n_agents × full context KV
    total_kv_with_sharing_gb: float     # With sharing: shared_once + unique×n
    vram_saving_gb: float               # = total_kv_naive_gb - total_kv_with_sharing_gb
    sharing_efficiency: float           # vram_saving_gb / total_kv_naive_gb

    # Per-GPU fit analysis
    agents_per_gpu_without_sharing: int
    agents_per_gpu_with_sharing: int
    recommended_gpu_type: str
    recommended_gpu_count: int          # total GPUs needed for n_agents with sharing
    recommended_gpu_count_naive: int    # total GPUs needed without sharing

    # Cost impact
    cost_per_hr_with_sharing: float
    cost_per_hr_naive: float
    hourly_savings: float

    # Eviction penalty
    eviction_cost: EvictionCostModel
    throughput_penalty_no_sharing: float  # fraction of time wasted on re-prefills
    throughput_penalty_with_sharing: float

    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "n_agents": self.n_agents,
            "model": self.model,
            "context_k": self.context_k_tokens,
            "shared_prefix_k": self.shared_prefix_k_tokens,
            "per_agent_unique_k": self.per_agent_unique_k_tokens,
            "topology": self.topology.value,
            "dtype": self.dtype,
            "shared_kv_gb": round(self.shared_kv_gb, 2),
            "per_agent_unique_kv_gb": round(self.per_agent_unique_kv_gb, 3),
            "total_kv_naive_gb": round(self.total_kv_naive_gb, 1),
            "total_kv_with_sharing_gb": round(self.total_kv_with_sharing_gb, 1),
            "vram_saving_gb": round(self.vram_saving_gb, 1),
            "sharing_efficiency": round(self.sharing_efficiency, 3),
            "agents_per_gpu_without_sharing": self.agents_per_gpu_without_sharing,
            "agents_per_gpu_with_sharing": self.agents_per_gpu_with_sharing,
            "recommended_gpu_type": self.recommended_gpu_type,
            "recommended_gpu_count": self.recommended_gpu_count,
            "recommended_gpu_count_naive": self.recommended_gpu_count_naive,
            "cost_per_hr_with_sharing": round(self.cost_per_hr_with_sharing, 2),
            "cost_per_hr_naive": round(self.cost_per_hr_naive, 2),
            "hourly_savings": round(self.hourly_savings, 2),
            "throughput_penalty_no_sharing": round(self.throughput_penalty_no_sharing, 3),
            "throughput_penalty_with_sharing": round(self.throughput_penalty_with_sharing, 3),
        }

    def summary_lines(self) -> List[str]:
        """Multi-line human-readable summary for CLI output."""
        lines = [
            f"KV Sharing Plan — {self.n_agents} agents × {self.model} @ {self.context_k_tokens}K ctx",
            f"Topology: {self.topology.value}  |  Shared prefix: {self.shared_prefix_k_tokens}K tokens",
            "",
            f"  VRAM without sharing: {self.total_kv_naive_gb:.1f} GB "
            f"({self.agents_per_gpu_without_sharing} agents/GPU → "
            f"{self.recommended_gpu_count_naive} GPUs needed)",
            f"  VRAM with sharing:    {self.total_kv_with_sharing_gb:.1f} GB "
            f"({self.agents_per_gpu_with_sharing} agents/GPU → "
            f"{self.recommended_gpu_count} GPUs needed)",
            f"  Saving: {self.vram_saving_gb:.1f} GB ({self.sharing_efficiency:.0%} reduction)",
            "",
            f"  Cost/hr without sharing: ${self.cost_per_hr_naive:.2f}",
            f"  Cost/hr with sharing:    ${self.cost_per_hr_with_sharing:.2f}  "
            f"(saves ${self.hourly_savings:.2f}/hr = ${self.hourly_savings*24:.0f}/day)",
            "",
        ]
        if self.throughput_penalty_no_sharing > 0.05:
            pct = self.throughput_penalty_no_sharing * 100
            reprefill_s = self.eviction_cost.reprefill_seconds(self.context_k_tokens * 1000)
            lines.append(
                f"  ⚠  Without sharing: {pct:.0f}% of compute wasted on re-prefills "
                f"({reprefill_s:.2f}s per eviction on {self.eviction_cost.gpu_type})"
            )
        if self.throughput_penalty_with_sharing > 0.01:
            pct = self.throughput_penalty_with_sharing * 100
            lines.append(f"  ✓  With sharing: {pct:.0f}% re-prefill overhead (acceptable)")
        else:
            lines.append(f"  ✓  With sharing: re-prefill overhead negligible (<1%)")
        return lines


# ── Fleet VRAM planner ────────────────────────────────────────────────────────

# Spot pricing guidance (USD/hr) — matches agentic_topology.py
_GPU_SPOT_PRICE_HR: Dict[str, float] = {
    "H100_SXM": 2.49, "H100_NVL": 2.20, "H100_PCIe": 1.89,
    "H200_SXM": 3.99,
    "A100_SXM_80": 1.49, "A100_PCIe_80": 1.29,
    "A100_SXM_40": 0.89, "L40S": 0.69, "RTX4090": 0.44,
}


class MultiAgentVRAMPlanner:
    """
    Computes fleet-level VRAM requirements for N agents with KV cache sharing.

    This is the math that nobody's CLI does today. Given:
      - n_agents: number of concurrent agent loops
      - context_k: context window per agent in K tokens
      - model: model name (for weight + KV sizing)
      - sharing_topology: how agents share context
      - shared_fraction: fraction of context that is shared (0.0–1.0)

    Returns a KVSharingPlan with:
      - Exact VRAM requirements with and without sharing
      - GPU count recommendations
      - Cost savings
      - Eviction penalty estimates
    """

    # Default sharing fractions per topology
    # (overridden by explicit shared_fraction parameter)
    _DEFAULT_SHARING_FRACTION: Dict[SharingTopology, float] = {
        SharingTopology.BROADCAST: 0.70,  # 70% shared (system prompt + task context)
        SharingTopology.STAR: 0.40,       # 40% shared (plan shared by orchestrator)
        SharingTopology.CHAIN: 0.10,      # 10% shared (only system prompt)
        SharingTopology.NONE: 0.0,
    }

    def _parse_model_size(self, model: str) -> int:
        import re
        model_lower = model.lower()
        moe = re.search(r'(\d+)x(\d+)b', model_lower)
        if moe:
            return int(moe.group(1)) * int(moe.group(2))
        m = re.search(r'(\d+(?:\.\d+)?)b', model_lower)
        return int(float(m.group(1))) if m else 70

    def _kv_gb_per_token_k(self, model_size_b: int, dtype: str = "fp16") -> float:
        """KV cache GB per 1K tokens for a given model and dtype."""
        bytes_per_elem = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}.get(dtype, 2)
        n_layers = _KV_LAYERS.get(f"{model_size_b}b", 80)
        # K+V tensors: 2 tensors per layer
        return (n_layers * _KV_BYTES_PER_TOKEN_PER_LAYER * bytes_per_elem) / (2 * 1e9)

    def _select_gpu(self, model_size_b: int, need_nvlink: bool = False) -> Tuple[str, int]:
        """Select GPU type and TP degree for this model size."""
        weights = _MODEL_WEIGHTS_GB_FP16.get(f"{model_size_b}b", model_size_b * 2.0)
        for gpu, vram in [
            ("H200_SXM", 141), ("H100_SXM", 80), ("A100_SXM_80", 80),
            ("A100_PCIe_80", 80), ("L40S", 48),
        ]:
            if vram * 0.90 >= weights:   # 90% of VRAM for weights + 10% CUDA overhead
                tp = 1
                return (gpu, tp)
        # Model doesn't fit on one GPU — use TP=2 on H100
        return ("H100_SXM", 2)

    def compute_per_agent_budget(
        self,
        model: str,
        context_k: int,
        gpu_type: str = "H100_SXM",
        tp: int = 1,
        dtype: str = "fp16",
    ) -> AgentKVBudget:
        """Compute per-agent KV budget for a specific GPU and model."""
        model_size_b = self._parse_model_size(model)
        n_layers = _KV_LAYERS.get(f"{model_size_b}b", 80)
        bytes_per_elem = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}.get(dtype, 2)

        kv_bytes_per_token = n_layers * _KV_BYTES_PER_TOKEN_PER_LAYER * bytes_per_elem // 2
        kv_gb_per_agent = (context_k * 1000 * kv_bytes_per_token) / 1e9

        total_vram = _GPU_VRAM_GB.get(gpu_type, 80) * tp
        weights_gb = _MODEL_WEIGHTS_GB_FP16.get(f"{model_size_b}b", model_size_b * 2.0)
        cuda_overhead_gb = total_vram * 0.05
        vram_for_kv = max(0.0, total_vram - weights_gb - cuda_overhead_gb)

        agents_per_gpu = max(0, int(vram_for_kv / kv_gb_per_agent)) if kv_gb_per_agent > 0 else 0

        return AgentKVBudget(
            model=model,
            model_size_b=model_size_b,
            context_k_tokens=context_k,
            dtype=dtype,
            n_layers=n_layers,
            kv_bytes_per_token=kv_bytes_per_token,
            kv_gb_per_agent=kv_gb_per_agent,
            model_weights_gb=weights_gb,
            gpu_type=gpu_type,
            gpu_vram_gb=total_vram,
            vram_for_model_gb=weights_gb,
            vram_available_for_kv_gb=vram_for_kv,
            agents_per_gpu=agents_per_gpu,
            fits=agents_per_gpu >= 1,
            eviction_cost=EvictionCostModel(gpu_type=gpu_type),
        )

    def compute(
        self,
        n_agents: int,
        context_k: int,
        model: str,
        topology: SharingTopology = SharingTopology.BROADCAST,
        shared_fraction: Optional[float] = None,
        dtype: str = "fp16",
        gpu_type: Optional[str] = None,
    ) -> KVSharingPlan:
        """
        Full fleet VRAM plan for n_agents with KV sharing.

        Parameters
        ----------
        n_agents : int
            Number of concurrently active agent loops.
        context_k : int
            Context window per agent in K tokens (e.g. 32 for 32K).
        model : str
            Model name (e.g. "meta-llama/Llama-3.1-70B-Instruct").
        topology : SharingTopology
            Sharing topology between agents.
        shared_fraction : float, optional
            Override default sharing fraction (0.0–1.0).
            Defaults to topology-based heuristic.
        dtype : str
            KV cache dtype — "fp16" | "fp8". fp8 halves KV memory.
        gpu_type : str, optional
            Override GPU selection.
        """
        fleet_id = f"kvplan_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        model_size_b = self._parse_model_size(model)

        if shared_fraction is None:
            shared_fraction = self._DEFAULT_SHARING_FRACTION.get(topology, 0.0)
        shared_fraction = max(0.0, min(1.0, shared_fraction))

        shared_k = int(context_k * shared_fraction)
        unique_k = context_k - shared_k

        kv_gb_per_k = self._kv_gb_per_token_k(model_size_b, dtype)

        shared_kv_gb = shared_k * kv_gb_per_k
        per_agent_unique_kv_gb = unique_k * kv_gb_per_k
        total_kv_naive_gb = context_k * kv_gb_per_k * n_agents
        total_kv_with_sharing_gb = shared_kv_gb + per_agent_unique_kv_gb * n_agents

        vram_saving_gb = total_kv_naive_gb - total_kv_with_sharing_gb
        sharing_efficiency = vram_saving_gb / total_kv_naive_gb if total_kv_naive_gb > 0 else 0.0

        # Select GPU
        if gpu_type is None:
            gpu_type, tp = self._select_gpu(model_size_b)
        else:
            tp = 1

        budget = self.compute_per_agent_budget(model, context_k, gpu_type, tp, dtype)
        budget_unique = self.compute_per_agent_budget(model, unique_k, gpu_type, tp, dtype)

        agents_per_gpu_no_share = budget.agents_per_gpu
        agents_per_gpu_share = budget_unique.agents_per_gpu if unique_k > 0 else n_agents

        gpu_count_naive = max(1, math.ceil(n_agents / agents_per_gpu_no_share)) if agents_per_gpu_no_share > 0 else n_agents
        gpu_count_sharing = max(1, math.ceil(n_agents / agents_per_gpu_share)) if agents_per_gpu_share > 0 else n_agents

        spot_price = _GPU_SPOT_PRICE_HR.get(gpu_type, 2.49)
        cost_naive = gpu_count_naive * spot_price
        cost_sharing = gpu_count_sharing * spot_price

        eviction = EvictionCostModel(gpu_type=gpu_type)
        penalty_no_share = eviction.throughput_penalty_fraction(
            n_agents, agents_per_gpu_no_share * gpu_count_naive, context_k
        )
        penalty_share = eviction.throughput_penalty_fraction(
            n_agents, agents_per_gpu_share * gpu_count_sharing, unique_k
        )

        return KVSharingPlan(
            fleet_id=fleet_id,
            n_agents=n_agents,
            model=model,
            model_size_b=model_size_b,
            context_k_tokens=context_k,
            shared_prefix_k_tokens=shared_k,
            per_agent_unique_k_tokens=unique_k,
            topology=topology,
            dtype=dtype,
            shared_kv_gb=shared_kv_gb,
            per_agent_unique_kv_gb=per_agent_unique_kv_gb,
            total_kv_naive_gb=total_kv_naive_gb,
            total_kv_with_sharing_gb=total_kv_with_sharing_gb,
            vram_saving_gb=vram_saving_gb,
            sharing_efficiency=sharing_efficiency,
            agents_per_gpu_without_sharing=agents_per_gpu_no_share,
            agents_per_gpu_with_sharing=agents_per_gpu_share,
            recommended_gpu_type=gpu_type,
            recommended_gpu_count=gpu_count_sharing,
            recommended_gpu_count_naive=gpu_count_naive,
            cost_per_hr_with_sharing=cost_sharing,
            cost_per_hr_naive=cost_naive,
            hourly_savings=max(0.0, cost_naive - cost_sharing),
            eviction_cost=eviction,
            throughput_penalty_no_sharing=penalty_no_share,
            throughput_penalty_with_sharing=penalty_share,
        )

    def compute_multi_topology(
        self,
        n_agents: int,
        context_k: int,
        model: str,
        dtype: str = "fp16",
    ) -> Dict[str, KVSharingPlan]:
        """Compute plans for all topologies — useful for CLI comparison table."""
        return {
            t.value: self.compute(n_agents, context_k, model, t, dtype=dtype)
            for t in SharingTopology
        }
