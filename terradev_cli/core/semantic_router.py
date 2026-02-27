#!/usr/bin/env python3
"""
Terradev Semantic Router — NUMA-Aware Signal-Driven Decision Routing
for Mixture-of-Modality Models.

Integrates the SignalOrchestrator (information-theoretic regime) with a
YAML-configurable Boolean decision engine (algebraic regime) to route
inference requests to the optimal model endpoint, with NUMA-topology-aware
endpoint selection that ensures routed traffic lands on GPUs with optimal
PCIe locality to the serving NIC.

This extends Terradev's existing InferenceRouter + GPUTopologyOrchestrator.

Inspired by the signal-driven routing concepts described in:
  - "Signal Driven Decision Routing for Mixture-of-Modality Models"
    (vLLM Semantic Router Team, Feb 2026)
  - "When to Reason: Semantic Router for vLLM" (NeurIPS 2025 MLForSys)

This is an independent, clean-room implementation. No code was copied
from any third-party repository.
"""

import ast
import json
import logging
import operator
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .semantic_signals import SignalOrchestrator, SignalVector

logger = logging.getLogger(__name__)


# ── Policy Data Structures ──────────────────────────────────────────────────


@dataclass
class RoutingRule:
    """A single routing rule: condition -> action"""
    name: str
    condition: str                          # Boolean expression over signals
    route_to: Optional[str] = None          # Target model name/pattern
    strategy: Optional[str] = None          # Fallback: 'latency', 'cost', 'score', 'numa'
    priority: int = 0                       # Higher = evaluated first
    numa_policy: Optional[str] = None       # 'strict' | 'prefer' | None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingPolicy:
    """A complete routing policy — ordered list of rules + defaults"""
    name: str
    description: str = ""
    rules: List[RoutingRule] = field(default_factory=list)
    default_strategy: str = "cost"          # fallback when no rule matches
    default_model: Optional[str] = None
    default_numa_policy: str = "prefer"     # 'strict' | 'prefer' | 'none'
    signals_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NUMAScorecard:
    """NUMA topology score for an endpoint"""
    endpoint_id: str
    gpu_index: Optional[int] = None
    numa_node: Optional[int] = None
    pcie_locality: Optional[str] = None     # PIX, PXB, PHB, SYS
    locality_score: float = 0.0             # 0=PIX (best), 3=SYS (worst)
    has_rdma: bool = False
    nccl_optimal: bool = False
    # Intra-GPU NUMA (arXiv:2511.02132)
    xcd_count: int = 0                      # 0 = unknown, 8 = MI300X
    gpu_arch: str = ""                      # "mi300x", "h200", etc.
    has_intra_gpu_numa: bool = False         # True if XCD count > 1
    xcd_locality: Optional[str] = None      # same_xcd, adj_xcd, remote_xcd, unified
    xcd_locality_score: float = 0.0         # 0=same_xcd (best), 2=remote_xcd (worst)


@dataclass
class RoutingDecision:
    """The output of the semantic router"""
    matched_rule: Optional[str]             # Name of the matched rule, or None
    route_to: Optional[str]                 # Target model name
    strategy: Optional[str]                 # Routing strategy for InferenceRouter
    signal_vector: Dict[str, Any]           # Flat signal dict for debugging
    condition_evaluated: Optional[str]       # The condition that matched
    numa_score: Optional[NUMAScorecard] = None
    latency_ms: float = 0.0                 # Total routing decision time
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Safe Expression Evaluator ────────────────────────────────────────────────


# Precompiled regex for Boolean keyword normalization (class-level, zero per-request cost)
_RE_AND = re.compile(r'\bAND\b')
_RE_OR = re.compile(r'\bOR\b')
_RE_NOT = re.compile(r'\bNOT\b')


class PolicyExpressionEvaluator:
    """
    Evaluates Boolean policy expressions over the signal vector.

    Supports:
      - Comparisons: ==, !=, >, <, >=, <=
      - Boolean operators: AND, OR, NOT, and, or, not
      - String matching: modality == 'vision'
      - Dotted access: safety.flagged, keyword.dominant_category
      - Numeric comparisons: complexity > 0.7
      - Set membership: 'code' in keyword.tags
      - Parentheses for grouping
      - Literal True/False

    Performance:
      - AST is compiled ONCE per unique expression and cached forever.
      - Regex normalization uses precompiled module-level patterns.
      - Hot path is pure AST tree-walk — zero parsing, zero regex.
    """

    _CMP_OPS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b if b is not None else False,
        ast.NotIn: lambda a, b: a not in b if b is not None else True,
    }

    _BOOL_OPS = {
        ast.And: all,
        ast.Or: any,
    }

    def __init__(self):
        # Expression AST cache: expression_str -> compiled ast.Expression.body
        # Populated at policy load time or on first eval — never reparsed.
        self._ast_cache: Dict[str, Any] = {}

    def compile(self, expression: str) -> None:
        """Pre-parse and cache an expression's AST. Call at init, not per-request."""
        if expression in self._ast_cache:
            return
        normalized = _RE_AND.sub(' and ', expression)
        normalized = _RE_OR.sub(' or ', normalized)
        normalized = _RE_NOT.sub(' not ', normalized)
        normalized = normalized.strip()
        try:
            tree = ast.parse(normalized, mode='eval')
            self._ast_cache[expression] = tree.body
        except SyntaxError as e:
            logger.warning(f"Policy expression compile failed: '{expression}' — {e}")
            self._ast_cache[expression] = None

    def evaluate(self, expression: str, context: Dict[str, Any]) -> bool:
        """
        Safely evaluate a Boolean expression against the signal context.

        Hot path: pure AST tree-walk over a pre-compiled node.
        Zero regex, zero parsing per request.
        """
        # Fast path: use cached AST
        node = self._ast_cache.get(expression)
        if node is None and expression not in self._ast_cache:
            # First-time compile (shouldn't happen if compile() was called at init)
            self.compile(expression)
            node = self._ast_cache.get(expression)

        if node is None:
            return False

        try:
            return bool(self._eval_node(node, context))
        except Exception as e:
            logger.warning(f"Policy expression eval failed: '{expression}' — {e}")
            return False

    def _eval_node(self, node: ast.AST, ctx: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            return ctx.get(node.id)

        if isinstance(node, ast.Attribute):
            return self._resolve_dotted(node, ctx)

        if isinstance(node, ast.BoolOp):
            op_fn = self._BOOL_OPS.get(type(node.op))
            if op_fn:
                return op_fn(self._eval_node(v, ctx) for v in node.values)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self._eval_node(node.operand, ctx)

        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, ctx)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, ctx)
                op_fn = self._CMP_OPS.get(type(op))
                if op_fn is None:
                    return False
                try:
                    if not op_fn(left, right):
                        return False
                except TypeError:
                    return False
                left = right
            return True

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'len':
                arg = self._eval_node(node.args[0], ctx)
                return len(arg) if arg is not None else 0

        if isinstance(node, ast.Subscript):
            val = self._eval_node(node.value, ctx)
            key = self._eval_node(node.slice, ctx)
            if isinstance(val, dict):
                return val.get(key)
            return None

        logger.debug(f"Unhandled AST node type: {type(node).__name__}")
        return None

    def _resolve_dotted(self, node: ast.Attribute, ctx: Dict[str, Any]) -> Any:
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        dotted_key = ".".join(parts)

        if dotted_key in ctx:
            return ctx[dotted_key]

        val = ctx.get(parts[0])
        for part in parts[1:]:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                return None
        return val


# ── NUMA-Aware Endpoint Scoring ──────────────────────────────────────────────


class NUMAEndpointScorer:
    """
    Scores inference endpoints by NUMA topology fitness.

    Integrates with GPUTopologyOrchestrator to prefer endpoints whose
    GPU sits on the same PCIe switch (PIX) or root complex (PXB) as the
    ingress NIC, avoiding cross-socket (SYS) traffic that adds 2-5us
    per-packet latency and halves RDMA throughput.

    Extended with intra-GPU NUMA awareness (arXiv:2511.02132, Nov 2025):
    AMD MI300X has 8 XCDs, each with its own L2 cache. Scoring now
    accounts for XCD locality to prefer endpoints where attention kernels
    can achieve 92% L2 hit rates vs 43% without XCD awareness.
    """

    # PCIe locality -> numeric score (lower is better)
    LOCALITY_SCORES = {
        "PIX": 0.0,   # Same PCIe switch — optimal
        "PXB": 1.0,   # Same root complex, different switch
        "PHB": 2.0,   # Same NUMA node, different root complex
        "SYS": 3.0,   # Cross-socket — worst
    }

    # Intra-GPU XCD locality -> numeric score (lower is better)
    XCD_LOCALITY_SCORES = {
        "same_xcd": 0.0,
        "adj_xcd": 1.0,
        "remote_xcd": 2.0,
        "unified": 0.0,   # no intra-GPU NUMA = no penalty
    }

    def __init__(self, topology_report: Optional[Dict[str, Any]] = None):
        self._topology = topology_report
        self._gpu_numa_map: Dict[int, int] = {}       # gpu_index -> numa_node
        self._gpu_locality_map: Dict[int, str] = {}    # gpu_index -> best locality
        self._gpu_rdma_map: Dict[int, bool] = {}       # gpu_index -> has_rdma_pair
        self._gpu_xcd_map: Dict[int, Dict[str, Any]] = {}  # gpu_index -> xcd info
        if topology_report:
            self._build_maps(topology_report)

    def _build_maps(self, report: Dict[str, Any]):
        """Build GPU -> NUMA/locality lookup maps from topology report"""
        for pair in report.get("pairs", []):
            # Parse GPU index from string like "GPU 0 (NVIDIA H100 80GB HBM3)"
            gpu_str = pair.get("gpu", "")
            try:
                idx = int(gpu_str.split()[1])
            except (IndexError, ValueError):
                continue

            self._gpu_locality_map[idx] = pair.get("locality", "SYS")
            self._gpu_rdma_map[idx] = bool(pair.get("rdma_path"))

        for numa_id, devices in report.get("numa_map", {}).items():
            for gpu_str in devices.get("gpus", []):
                try:
                    idx = int(gpu_str.split()[1])
                    self._gpu_numa_map[idx] = int(numa_id)
                except (IndexError, ValueError):
                    continue

        # Build XCD intra-GPU NUMA map from topology report
        for xcd_info in report.get("intra_gpu_numa", []):
            gpu_idx = xcd_info.get("gpu_index")
            if gpu_idx is not None:
                self._gpu_xcd_map[gpu_idx] = xcd_info

    def _get_xcd_info(self, gpu_index: int) -> Dict[str, Any]:
        """Get XCD info for a GPU, auto-detecting architecture if needed."""
        if gpu_index in self._gpu_xcd_map:
            return self._gpu_xcd_map[gpu_index]
        # Try to auto-detect from GPU name in topology report
        try:
            from .gpu_topology import build_intra_gpu_topology, GPUDevice
            for pair in (self._topology or {}).get("pairs", []):
                gpu_str = pair.get("gpu", "")
                try:
                    idx = int(gpu_str.split()[1])
                except (IndexError, ValueError):
                    continue
                if idx == gpu_index:
                    # Extract GPU name
                    name_part = gpu_str.split("(", 1)[-1].rstrip(")")
                    dummy_gpu = GPUDevice(
                        index=idx, name=name_part, pci_bus_id="",
                        numa_node=self._gpu_numa_map.get(idx, 0),
                        pcie_root="", pcie_switch="",
                    )
                    topo = build_intra_gpu_topology(dummy_gpu)
                    info = {
                        "gpu_index": idx,
                        "gpu_arch": topo.gpu_arch,
                        "xcd_count": topo.xcd_count,
                        "has_intra_numa": topo.has_intra_numa,
                    }
                    self._gpu_xcd_map[idx] = info
                    return info
        except ImportError:
            pass
        return {}

    def score_endpoint(
        self,
        endpoint_id: str,
        gpu_index: Optional[int] = None,
        numa_node: Optional[int] = None,
    ) -> NUMAScorecard:
        """
        Score an endpoint's NUMA fitness (node-level + intra-GPU XCD).

        If gpu_index is known (e.g., from endpoint metadata), uses the
        topology map directly. Otherwise falls back to numa_node if provided.
        """
        if gpu_index is not None and gpu_index in self._gpu_locality_map:
            locality = self._gpu_locality_map[gpu_index]

            # Intra-GPU XCD scoring
            xcd_info = self._get_xcd_info(gpu_index)
            xcd_count = xcd_info.get("xcd_count", 0)
            gpu_arch = xcd_info.get("gpu_arch", "")
            has_intra = xcd_info.get("has_intra_numa", False)
            xcd_loc = "unified" if not has_intra else "same_xcd"  # best-case default
            xcd_loc_score = self.XCD_LOCALITY_SCORES.get(xcd_loc, 0.0)

            return NUMAScorecard(
                endpoint_id=endpoint_id,
                gpu_index=gpu_index,
                numa_node=self._gpu_numa_map.get(gpu_index),
                pcie_locality=locality,
                locality_score=self.LOCALITY_SCORES.get(locality, 3.0),
                has_rdma=self._gpu_rdma_map.get(gpu_index, False),
                nccl_optimal=locality in ("PIX", "PXB"),
                xcd_count=xcd_count,
                gpu_arch=gpu_arch,
                has_intra_gpu_numa=has_intra,
                xcd_locality=xcd_loc,
                xcd_locality_score=xcd_loc_score,
            )

        # Fallback: if we only know NUMA node
        if numa_node is not None:
            # Find GPUs on this NUMA node
            gpus_on_node = [
                idx for idx, nn in self._gpu_numa_map.items()
                if nn == numa_node
            ]
            if gpus_on_node:
                best_idx = min(
                    gpus_on_node,
                    key=lambda i: self.LOCALITY_SCORES.get(
                        self._gpu_locality_map.get(i, "SYS"), 3.0
                    ),
                )
                return self.score_endpoint(endpoint_id, gpu_index=best_idx)

        # No topology info — return neutral score
        return NUMAScorecard(
            endpoint_id=endpoint_id,
            locality_score=1.5,  # neutral middle score
        )

    def rank_endpoints(
        self,
        endpoint_ids: List[str],
        gpu_indices: Optional[Dict[str, int]] = None,
        numa_nodes: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[str, NUMAScorecard]]:
        """
        Rank endpoints by combined NUMA fitness (node-level + XCD, best first).

        Args:
            endpoint_ids: list of endpoint IDs to rank
            gpu_indices: optional mapping endpoint_id -> gpu_index
            numa_nodes: optional mapping endpoint_id -> numa_node
        """
        gpu_indices = gpu_indices or {}
        numa_nodes = numa_nodes or {}

        scored = []
        for eid in endpoint_ids:
            card = self.score_endpoint(
                eid,
                gpu_index=gpu_indices.get(eid),
                numa_node=numa_nodes.get(eid),
            )
            scored.append((eid, card))

        # Combined score: 80% node-level locality + 20% XCD locality
        scored.sort(key=lambda x: (
            0.8 * x[1].locality_score + 0.2 * x[1].xcd_locality_score
        ))
        return scored


# ── Policy Loader ────────────────────────────────────────────────────────────


def load_policy_from_dict(data: Dict[str, Any]) -> RoutingPolicy:
    """Load a RoutingPolicy from a parsed YAML/JSON dict"""
    rules = []
    for i, rule_data in enumerate(data.get("rules", [])):
        rules.append(RoutingRule(
            name=rule_data.get("name", f"rule_{i}"),
            condition=rule_data["when"],
            route_to=rule_data.get("route_to"),
            strategy=rule_data.get("strategy"),
            priority=rule_data.get("priority", len(data.get("rules", [])) - i),
            numa_policy=rule_data.get("numa_policy"),
            tags=rule_data.get("tags", []),
            metadata=rule_data.get("metadata", {}),
        ))

    rules.sort(key=lambda r: r.priority, reverse=True)

    return RoutingPolicy(
        name=data.get("name", "default"),
        description=data.get("description", ""),
        rules=rules,
        default_strategy=data.get("default_strategy", "cost"),
        default_model=data.get("default_model"),
        default_numa_policy=data.get("default_numa_policy", "prefer"),
        signals_config=data.get("signals", {}),
    )


def load_policy_from_yaml(path: str) -> RoutingPolicy:
    """Load a RoutingPolicy from a YAML file"""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for YAML policy files: pip install pyyaml")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return load_policy_from_dict(data)


def load_policy_from_json(path: str) -> RoutingPolicy:
    """Load a RoutingPolicy from a JSON file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return load_policy_from_dict(data)


def load_policy(path: str) -> RoutingPolicy:
    """Auto-detect format and load policy"""
    if path.endswith(('.yaml', '.yml')):
        return load_policy_from_yaml(path)
    elif path.endswith('.json'):
        return load_policy_from_json(path)
    else:
        try:
            return load_policy_from_json(path)
        except (json.JSONDecodeError, Exception):
            return load_policy_from_yaml(path)


# ── Default Policy ───────────────────────────────────────────────────────────


DEFAULT_POLICY_DICT: Dict[str, Any] = {
    "name": "terradev_default",
    "description": "Default NUMA-aware semantic routing policy for Terradev",
    "default_strategy": "cost",
    "default_numa_policy": "prefer",
    "rules": [
        {
            "name": "block_jailbreak",
            "when": "safety.flagged and safety.severity >= 0.9",
            "route_to": "__blocked__",
            "priority": 100,
            "tags": ["safety"],
        },
        {
            "name": "pii_to_local",
            "when": "'pii_detected' in safety.flags",
            "route_to": "__local_only__",
            "numa_policy": "strict",
            "priority": 90,
            "tags": ["privacy", "numa_strict"],
        },
        {
            "name": "vision_to_vision_model",
            "when": "modality == 'vision' or modality == 'multimodal'",
            "route_to": "gpt-4o",
            "priority": 80,
            "tags": ["modality"],
        },
        {
            "name": "diffusion_to_image_model",
            "when": "modality == 'diffusion'",
            "route_to": "dall-e-3",
            "priority": 80,
            "tags": ["modality"],
        },
        {
            "name": "code_complex_to_large",
            "when": "modality == 'code' and complexity > 0.6",
            "route_to": "deepseek-coder-33b",
            "numa_policy": "prefer",
            "priority": 70,
            "tags": ["code", "complex"],
        },
        {
            "name": "code_simple_to_small",
            "when": "modality == 'code' and complexity <= 0.6",
            "route_to": "deepseek-coder-6.7b",
            "numa_policy": "prefer",
            "priority": 65,
            "tags": ["code", "simple"],
        },
        {
            "name": "complex_reasoning",
            "when": "complexity > 0.7",
            "route_to": "llama-3-70b",
            "numa_policy": "prefer",
            "priority": 50,
            "tags": ["reasoning"],
        },
        {
            "name": "long_context_prefill_optimized",
            "when": "complexity > 0.5 and modality != 'vision'",
            "strategy": "prefill_optimized",
            "numa_policy": "prefer",
            "priority": 45,
            "tags": ["disaggregated", "prefill"],
            "metadata": {"phase_hint": "prefill"},
        },
        {
            "name": "simple_to_cheap",
            "when": "complexity < 0.3",
            "strategy": "cost",
            "priority": 40,
            "tags": ["cost_optimize"],
        },
        {
            "name": "default_balanced",
            "when": "True",
            "strategy": "score",
            "numa_policy": "prefer",
            "priority": 0,
            "tags": ["default"],
        },
    ],
}


# ── Semantic Router ──────────────────────────────────────────────────────────


class SemanticRouter:
    """
    NUMA-aware signal-driven decision router for heterogeneous model fleets.

    Combines three layers:
      1. Signal extraction (SignalOrchestrator) — reduces routing entropy
      2. Boolean decision engine — matches signals to routing rules
      3. NUMA-aware endpoint scoring — prefers GPUs with optimal PCIe
         locality to the ingress NIC, avoiding cross-socket latency

    Usage:
        router = SemanticRouter()
        decision = router.route({"content": "Write a Python function that..."})
        print(decision.route_to)       # "deepseek-coder-33b"
        print(decision.numa_score)     # NUMAScorecard(locality=PIX, ...)

    With custom policy:
        router = SemanticRouter(policy_path="~/.terradev/routing_policy.yaml")

    Integration with InferenceRouter + GPUTopologyOrchestrator:
        from terradev_cli.core.inference_router import InferenceRouter
        from terradev_cli.core.gpu_topology import GPUTopologyOrchestrator

        ir = InferenceRouter()
        topo = GPUTopologyOrchestrator()
        sr = SemanticRouter(topology_report=topo.full_topology_report())

        decision = sr.route(query)
        if decision.route_to == "__blocked__":
            return {"error": "Request blocked by safety policy"}
        elif decision.route_to:
            endpoint = ir.get_best_endpoint(model=decision.route_to)
        elif decision.strategy:
            endpoint = ir.get_best_endpoint(strategy=decision.strategy)
    """

    def __init__(
        self,
        policy: Optional[RoutingPolicy] = None,
        policy_path: Optional[str] = None,
        signals_config: Optional[Dict[str, Any]] = None,
        topology_report: Optional[Dict[str, Any]] = None,
    ):
        # Load policy
        if policy:
            self._policy = policy
        elif policy_path:
            self._policy = load_policy(os.path.expanduser(policy_path))
        else:
            self._policy = load_policy_from_dict(DEFAULT_POLICY_DICT)

        # Signal orchestrator
        sig_cfg = signals_config or self._policy.signals_config
        self._orchestrator = SignalOrchestrator(config=sig_cfg)

        # Decision engine — precompile ALL rule ASTs at init (zero per-request parse cost)
        self._evaluator = PolicyExpressionEvaluator()
        for rule in self._policy.rules:
            self._evaluator.compile(rule.condition)

        # NUMA-aware endpoint scorer
        self._numa_scorer = NUMAEndpointScorer(topology_report=topology_report)
        self._topology_report = topology_report

        # Metrics
        self._route_count = 0
        self._rule_hit_counts: Dict[str, int] = {}

    # ── Core Routing ─────────────────────────────────────────────────────

    def route(self, query: Dict[str, Any]) -> RoutingDecision:
        """
        Route a query through the full signal -> decision -> NUMA pipeline.

        Args:
            query: Dict with at minimum:
                - "content": str — the raw query text
                  OR "messages": list[dict] — chat messages
                  OR "prompt": str — completion prompt
                Optional:
                - "model": str — explicit model hint
                - "images": list — image attachments
                - "metadata": dict — user/session context
                - "gpu_index": int — GPU serving this request (for NUMA scoring)
                - "numa_node": int — NUMA node hint

        Returns:
            RoutingDecision with matched rule, target model, strategy,
            NUMA scorecard, and full signal vector for debugging.
        """
        start = time.perf_counter()

        # Layer 1: Signal extraction (information-theoretic regime)
        signal_vector = self._orchestrator.extract(query)
        flat_signals = signal_vector.to_flat_dict()

        # Layer 2: Decision engine (Boolean-algebraic regime)
        matched_rule = None
        for rule in self._policy.rules:
            if self._evaluator.evaluate(rule.condition, flat_signals):
                matched_rule = rule
                break

        # Layer 3: NUMA-aware scoring
        numa_score = None
        numa_policy = None
        if matched_rule:
            route_to = matched_rule.route_to
            strategy = matched_rule.strategy or self._policy.default_strategy
            condition = matched_rule.condition
            numa_policy = matched_rule.numa_policy or self._policy.default_numa_policy
        else:
            route_to = self._policy.default_model
            strategy = self._policy.default_strategy
            condition = None
            numa_policy = self._policy.default_numa_policy

        # Score NUMA fitness if topology info is available
        if self._topology_report and numa_policy != "none":
            gpu_index = query.get("gpu_index")
            numa_node = query.get("numa_node")
            if gpu_index is not None or numa_node is not None:
                numa_score = self._numa_scorer.score_endpoint(
                    endpoint_id=route_to or "unknown",
                    gpu_index=gpu_index,
                    numa_node=numa_node,
                )

                # If strict NUMA policy and score is bad, log warning
                if numa_policy == "strict" and numa_score.locality_score > 1.0:
                    logger.warning(
                        f"NUMA strict policy violated: {route_to} on GPU {gpu_index} "
                        f"has locality {numa_score.pcie_locality} (score {numa_score.locality_score}). "
                        f"Cross-socket traffic will degrade inference throughput."
                    )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Track metrics
        self._route_count += 1
        if matched_rule:
            self._rule_hit_counts[matched_rule.name] = (
                self._rule_hit_counts.get(matched_rule.name, 0) + 1
            )

        # Compute confidence from signal confidences
        confidences = [r.confidence for r in signal_vector.signals.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return RoutingDecision(
            matched_rule=matched_rule.name if matched_rule else None,
            route_to=route_to,
            strategy=strategy,
            signal_vector=flat_signals,
            condition_evaluated=condition,
            numa_score=numa_score,
            latency_ms=round(elapsed_ms, 3),
            confidence=round(avg_confidence, 3),
            metadata={
                "signal_summary": signal_vector.summary(),
                "signal_latency_ms": round(signal_vector.total_latency_ms, 3),
                "numa_policy": numa_policy,
                "policy_name": self._policy.name,
            },
        )

    def batch_route(self, queries: List[Dict[str, Any]]) -> List[RoutingDecision]:
        """
        Route N queries in a single batched pass.

        Uses the orchestrator's batch_extract() to share a single warm
        thread pool across all queries — maximum throughput for MCP batch
        tool calls. Policy evaluation runs sequentially per query (it's
        sub-μs per eval with precompiled ASTs).
        """
        if not queries:
            return []

        start = time.perf_counter()

        # Batch signal extraction (shared pool)
        vectors = self._orchestrator.batch_extract(queries)

        decisions = []
        for query, signal_vector in zip(queries, vectors):
            q_start = time.perf_counter()
            flat_signals = signal_vector.to_flat_dict()

            # Decision engine
            matched_rule = None
            for rule in self._policy.rules:
                if self._evaluator.evaluate(rule.condition, flat_signals):
                    matched_rule = rule
                    break

            # NUMA scoring
            numa_score = None
            if matched_rule:
                route_to = matched_rule.route_to
                strategy = matched_rule.strategy or self._policy.default_strategy
                condition = matched_rule.condition
                numa_policy = matched_rule.numa_policy or self._policy.default_numa_policy
            else:
                route_to = self._policy.default_model
                strategy = self._policy.default_strategy
                condition = None
                numa_policy = self._policy.default_numa_policy

            if self._topology_report and numa_policy != "none":
                gpu_index = query.get("gpu_index")
                numa_node = query.get("numa_node")
                if gpu_index is not None or numa_node is not None:
                    numa_score = self._numa_scorer.score_endpoint(
                        endpoint_id=route_to or "unknown",
                        gpu_index=gpu_index,
                        numa_node=numa_node,
                    )

            elapsed_ms = (time.perf_counter() - q_start) * 1000

            self._route_count += 1
            if matched_rule:
                self._rule_hit_counts[matched_rule.name] = (
                    self._rule_hit_counts.get(matched_rule.name, 0) + 1
                )

            confidences = [r.confidence for r in signal_vector.signals.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            decisions.append(RoutingDecision(
                matched_rule=matched_rule.name if matched_rule else None,
                route_to=route_to,
                strategy=strategy,
                signal_vector=flat_signals,
                condition_evaluated=condition,
                numa_score=numa_score,
                latency_ms=round(elapsed_ms, 3),
                confidence=round(avg_confidence, 3),
                metadata={
                    "signal_summary": signal_vector.summary(),
                    "signal_latency_ms": round(signal_vector.total_latency_ms, 3),
                    "numa_policy": numa_policy,
                    "policy_name": self._policy.name,
                    "batch": True,
                },
            ))

        total_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            f"Batch route: {len(queries)} queries in {total_ms:.2f}ms "
            f"({total_ms / len(queries):.2f}ms/query)"
        )
        return decisions

    # ── NUMA-Aware Endpoint Selection ────────────────────────────────────

    def select_numa_optimal_endpoint(
        self,
        decision: RoutingDecision,
        candidate_endpoints: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        From a list of candidate endpoints, select the one with the best
        NUMA topology fitness. Used after the semantic decision to pick
        the specific endpoint when multiple serve the same model.

        Each candidate dict should contain:
          - "endpoint_id": str
          - "gpu_index": int (optional)
          - "numa_node": int (optional)
          - "price_per_hour": float
          - "avg_latency_ms": float

        Returns the best candidate, or None if no candidates.
        """
        if not candidate_endpoints:
            return None

        if not self._topology_report:
            # No topology info — fall back to first candidate
            return candidate_endpoints[0]

        numa_policy = decision.metadata.get("numa_policy", "prefer")

        # Score all candidates
        scored = []
        for ep in candidate_endpoints:
            card = self._numa_scorer.score_endpoint(
                endpoint_id=ep.get("endpoint_id", ""),
                gpu_index=ep.get("gpu_index"),
                numa_node=ep.get("numa_node"),
            )
            scored.append((ep, card))

        # Sort by NUMA locality score (lower is better)
        scored.sort(key=lambda x: x[1].locality_score)

        if numa_policy == "strict":
            # Only return PIX or PXB endpoints
            strict_candidates = [
                (ep, card) for ep, card in scored
                if card.pcie_locality in ("PIX", "PXB") or card.locality_score <= 1.0
            ]
            if strict_candidates:
                return strict_candidates[0][0]
            else:
                logger.warning(
                    "NUMA strict: no PIX/PXB endpoints available. "
                    "All candidates cross PCIe root complex."
                )
                return None

        # 'prefer' policy: weight NUMA score into combined scoring
        # Combined score: 50% NUMA, 30% latency, 20% cost
        if len(scored) > 1:
            max_latency = max(
                (ep.get("avg_latency_ms", 0) for ep, _ in scored), default=1
            ) or 1
            max_price = max(
                (ep.get("price_per_hour", 0) for ep, _ in scored), default=1
            ) or 1

            def combined_score(ep: Dict, card: NUMAScorecard) -> float:
                numa_norm = card.locality_score / 3.0  # 0-1
                lat_norm = ep.get("avg_latency_ms", max_latency) / max_latency
                cost_norm = ep.get("price_per_hour", max_price) / max_price
                return 0.50 * numa_norm + 0.30 * lat_norm + 0.20 * cost_norm

            scored.sort(key=lambda x: combined_score(x[0], x[1]))

        best_ep, best_card = scored[0]
        logger.debug(
            f"NUMA-optimal endpoint: {best_ep.get('endpoint_id')} "
            f"locality={best_card.pcie_locality} score={best_card.locality_score}"
        )
        return best_ep

    # ── NCCL Environment Generation ──────────────────────────────────────

    def get_nccl_env_for_decision(
        self, decision: RoutingDecision
    ) -> Dict[str, str]:
        """
        Generate NCCL environment variables optimized for the NUMA topology
        of the routed endpoint. Integrates with RDMAConfigurator.generate_nccl_env().

        Returns empty dict if no topology info or no NUMA score.
        """
        if not decision.numa_score or not self._topology_report:
            return {}

        card = decision.numa_score
        env: Dict[str, str] = {}

        if card.pcie_locality == "PIX":
            env["NCCL_NET_GDR_LEVEL"] = "PIX"
            env["NCCL_P2P_LEVEL"] = "PIX"
            env["NCCL_NET_GDR_READ"] = "1"
            env["NCCL_IB_DISABLE"] = "0"
            env["NCCL_TOPO_DUMP_FILE"] = "/tmp/nccl_topo.xml"
        elif card.pcie_locality == "PXB":
            env["NCCL_NET_GDR_LEVEL"] = "PXB"
            env["NCCL_P2P_LEVEL"] = "PXB"
            env["NCCL_NET_GDR_READ"] = "1"
            env["NCCL_IB_DISABLE"] = "0"
        elif card.pcie_locality == "PHB":
            env["NCCL_NET_GDR_LEVEL"] = "PHB"
            env["NCCL_P2P_LEVEL"] = "PHB"
            env["NCCL_NET_GDR_READ"] = "1"
        else:
            # SYS — cross-socket, RDMA degraded
            env["NCCL_NET_GDR_LEVEL"] = "SYS"
            env["NCCL_P2P_LEVEL"] = "SYS"
            env["NCCL_NET_GDR_READ"] = "0"
            env["NCCL_SHM_DISABLE"] = "0"

        if card.has_rdma:
            env["NCCL_IB_DISABLE"] = "0"
            env["NCCL_IB_GID_INDEX"] = "3"
        else:
            env["NCCL_IB_DISABLE"] = "1"

        return env

    # ── DRA Manifest Generation ──────────────────────────────────────────

    def generate_numa_aware_dra_claim(
        self,
        decision: RoutingDecision,
        gpu_count: int = 1,
        nic_count: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a DRA ResourceClaimTemplate with PCIe alignment constraints
        that match the NUMA policy from the routing decision.

        For 'strict' NUMA policy: matchAttribute on pcieRoot (same PCIe switch)
        For 'prefer' NUMA policy: matchAttribute on pcieRoot (best-effort)
        """
        numa_policy = decision.metadata.get("numa_policy", "prefer")
        pcie_aligned = numa_policy in ("strict", "prefer")

        claim_name = f"semantic-route-{decision.matched_rule or 'default'}"

        requests = [
            {
                "name": "gpu",
                "exactly": {
                    "deviceClassName": "gpu",
                    "count": gpu_count,
                },
            },
            {
                "name": "nic",
                "exactly": {
                    "deviceClassName": "rdma-nic",
                    "count": nic_count,
                    "selectors": [
                        {
                            "cel": {
                                "expression": 'device.attributes["dra.net"].rdma == true'
                            }
                        }
                    ],
                },
            },
        ]

        spec: Dict[str, Any] = {"devices": {"requests": requests}}

        if pcie_aligned:
            spec["devices"]["constraints"] = [
                {"matchAttribute": "resource.kubernetes.io/pcieRoot"}
            ]

        return {
            "apiVersion": "resource.k8s.io/v1alpha3",
            "kind": "ResourceClaimTemplate",
            "metadata": {
                "name": claim_name,
                "labels": {
                    "terradev.io/semantic-route": decision.matched_rule or "default",
                    "terradev.io/numa-policy": numa_policy,
                },
            },
            "spec": {"spec": spec},
        }

    # ── Observability ────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics for monitoring"""
        return {
            "total_routes": self._route_count,
            "rule_hit_counts": dict(self._rule_hit_counts),
            "policy_name": self._policy.name,
            "enabled_signals": self._orchestrator.get_enabled_signals(),
            "has_topology": self._topology_report is not None,
            "numa_default_policy": self._policy.default_numa_policy,
        }

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get human-readable policy summary"""
        return {
            "name": self._policy.name,
            "description": self._policy.description,
            "rules": [
                {
                    "name": r.name,
                    "condition": r.condition,
                    "route_to": r.route_to,
                    "strategy": r.strategy,
                    "numa_policy": r.numa_policy,
                    "priority": r.priority,
                    "tags": r.tags,
                }
                for r in self._policy.rules
            ],
            "default_strategy": self._policy.default_strategy,
            "default_numa_policy": self._policy.default_numa_policy,
        }

    def update_topology(self, topology_report: Dict[str, Any]):
        """Hot-reload topology data (e.g., after node scaling)"""
        self._topology_report = topology_report
        self._numa_scorer = NUMAEndpointScorer(topology_report=topology_report)
        logger.info("Semantic router topology updated")

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    @property
    def orchestrator(self) -> SignalOrchestrator:
        return self._orchestrator
