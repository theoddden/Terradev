#!/usr/bin/env python3
"""
Terradev DAG Executor — Terraform-Style Parallel Execution Engine

Models computation as a directed acyclic graph (DAG) and executes
independent nodes in parallel, exactly like Terraform's resource graph walker.

Used by:
  - SignalOrchestrator: run 6 signal extractors in parallel (all independent)
  - GPUTopologyOrchestrator: run NUMA detection, GPU enumeration, NIC enumeration
    in parallel, then GPU-NIC pairing (depends on both), then scoring
  - SemanticRouter: signal extraction ∥ topology refresh → policy eval → NUMA scoring

The executor supports both thread-pool and process-pool backends.
Thread-pool is default since our workloads are CPU-bound sub-ms heuristics
where the GIL overhead is negligible compared to the parallelism benefit
of overlapping I/O-bound topology probes.

Architecture mirrors Terraform's internal DAG walker:
  1. Build graph: add_node(), add_edge()
  2. Plan: topological sort into execution waves (nodes at same depth run in parallel)
  3. Apply: execute each wave concurrently, passing outputs downstream
  4. Result: collect all node outputs into a single result dict
"""

import logging
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import os
import threading

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class DAGNode:
    """A single unit of work in the execution graph.

    Equivalent to a Terraform resource node. The `execute` callable
    receives a context dict containing outputs from all upstream
    dependencies, keyed by their node name.
    """
    name: str
    execute: Callable[[Dict[str, Any]], Any]
    dependencies: Set[str] = field(default_factory=set)
    # Populated after execution
    output: Any = None
    latency_ms: float = 0.0
    status: str = "pending"  # pending | running | done | failed
    error: Optional[str] = None


@dataclass
class ExecutionWave:
    """A set of nodes that can execute in parallel (same topological depth).

    Terraform calls these "walk groups" — all nodes in a wave have their
    dependencies satisfied by prior waves.
    """
    depth: int
    nodes: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """The full execution plan — analogous to `terraform plan` output.

    Contains the topologically sorted wave schedule and the total
    estimated parallelism factor.
    """
    waves: List[ExecutionWave] = field(default_factory=list)
    total_nodes: int = 0
    max_parallelism: int = 0  # widest wave
    critical_path_depth: int = 0  # number of sequential waves


@dataclass
class ExecutionResult:
    """Result of executing the full DAG — analogous to `terraform apply` output."""
    outputs: Dict[str, Any] = field(default_factory=dict)
    node_latencies: Dict[str, float] = field(default_factory=dict)
    node_statuses: Dict[str, str] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    wall_clock_ms: float = 0.0
    parallelism_achieved: float = 0.0  # sum(node_latencies) / wall_clock
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ── DAG Executor ──────────────────────────────────────────────────────────────


class DAGExecutor:
    """
    Execute DAGs with topological wave parallelism.
    
    Enhanced with scalable thread pool management:
    - Adaptive worker count based on system resources
    - Thread-safe pool sharing across executions
    - Automatic scaling based on load
    """
    
    def __init__(self, name: str = "dag_executor", max_workers: Optional[int] = None, reuse_pool: bool = True):
        self._nodes: Dict[str, DAGNode] = {}
        self._edges: Dict[str, Set[str]] = defaultdict(set)
        self._max_workers = max_workers or self._calculate_optimal_workers()
        self._name = name
        self._reuse_pool = reuse_pool
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()
        if reuse_pool:
            self._pool = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix=f"dag-{name}")
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal worker count based on system resources."""
        cpu_count = os.cpu_count() or 4
        # Scale between 4 and 32 workers based on CPU count
        return min(32, max(4, cpu_count * 2))

    def shutdown(self):
        """Shutdown the persistent pool if reuse_pool was set."""
        if self._pool:
            self._pool.shutdown(wait=False)
            self._pool = None

    # ── Graph Construction (terraform init) ───────────────────────────────

    def add_node(
        self,
        name: str,
        execute: Callable[[Dict[str, Any]], Any],
        depends_on: Optional[Set[str]] = None,
    ) -> "DAGExecutor":
        """Register a node in the execution graph. Returns self for chaining."""
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists in DAG '{self._name}'")
        self._nodes[name] = DAGNode(
            name=name,
            execute=execute,
            dependencies=depends_on or set(),
        )
        return self

    def add_edge(self, from_node: str, to_node: str) -> "DAGExecutor":
        """Add a dependency edge: to_node depends on from_node."""
        if to_node not in self._nodes:
            raise ValueError(f"Target node '{to_node}' not in DAG")
        self._nodes[to_node].dependencies.add(from_node)
        return self

    # ── Validation ────────────────────────────────────────────────────────

    def _validate(self):
        """Check for missing dependencies and cycles."""
        # Missing nodes
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise ValueError(
                        f"Node '{node.name}' depends on '{dep}' which does not exist"
                    )

        # Cycle detection via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self._nodes}

        def dfs(name: str, path: List[str]):
            color[name] = GRAY
            path.append(name)
            for dep_name, dep_node in self._nodes.items():
                # Check reverse: who depends on `name`
                pass
            # Check forward: what does `name` depend on
            for dep in self._nodes[name].dependencies:
                if color[dep] == GRAY:
                    cycle_start = path.index(dep)
                    cycle = " → ".join(path[cycle_start:] + [dep])
                    raise ValueError(f"Cycle detected in DAG '{self._name}': {cycle}")
                if color[dep] == WHITE:
                    dfs(dep, path)
            path.pop()
            color[name] = BLACK

        for name in self._nodes:
            if color[name] == WHITE:
                dfs(name, [])

    # ── Planning (terraform plan) ─────────────────────────────────────────

    def plan(self) -> ExecutionPlan:
        """
        Topological sort into execution waves.

        Kahn's algorithm: nodes with in-degree 0 go into wave 0,
        then decrement neighbors and repeat.
        """
        self._validate()

        # Build in-degree map and reverse adjacency
        in_degree: Dict[str, int] = {name: 0 for name in self._nodes}
        # Forward edges: if A depends on B, then B → A (B must complete before A)
        forward: Dict[str, Set[str]] = defaultdict(set)

        for name, node in self._nodes.items():
            in_degree[name] = len(node.dependencies)
            for dep in node.dependencies:
                forward[dep].add(name)

        waves: List[ExecutionWave] = []
        queue = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        depth = 0
        processed = 0

        while queue:
            wave = ExecutionWave(depth=depth, nodes=list(queue))
            waves.append(wave)
            next_queue: deque = deque()

            for name in queue:
                processed += 1
                for successor in forward[name]:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        next_queue.append(successor)

            queue = next_queue
            depth += 1

        if processed != len(self._nodes):
            raise ValueError(
                f"DAG '{self._name}' has unreachable nodes — likely a hidden cycle"
            )

        max_par = max(len(w.nodes) for w in waves) if waves else 0

        plan = ExecutionPlan(
            waves=waves,
            total_nodes=len(self._nodes),
            max_parallelism=max_par,
            critical_path_depth=len(waves),
        )

        logger.debug(
            f"DAG '{self._name}' plan: {plan.total_nodes} nodes, "
            f"{plan.critical_path_depth} waves, max parallelism={plan.max_parallelism}"
        )

        return plan

    # ── Execution (terraform apply) ───────────────────────────────────────

    def _get_pool(self, workers: Optional[int] = None) -> ThreadPoolExecutor:
        """Get a thread pool — reuse persistent one or create ephemeral with scaling."""
        if self._pool:
            return self._pool
        
        # Scale workers based on request and system capacity
        worker_count = workers or self._max_workers
        optimal_workers = min(worker_count, self._calculate_optimal_workers())
        
        with self._pool_lock:
            if not self._pool and self._reuse_pool:
                self._pool = ThreadPoolExecutor(max_workers=optimal_workers, thread_name_prefix=f"dag-{self._name}")
                return self._pool
        
        return ThreadPoolExecutor(max_workers=optimal_workers)

    def apply(
        self,
        initial_context: Optional[Dict[str, Any]] = None,
        fail_fast: bool = True,
    ) -> ExecutionResult:
        """
        Execute the DAG in topological wave order.

        Each wave's nodes run in parallel via ThreadPoolExecutor.
        Outputs from completed nodes are merged into the shared context
        dict before the next wave begins.

        If reuse_pool=True was set at init, the pool stays warm across calls.
        """
        plan = self.plan()
        context: Dict[str, Any] = dict(initial_context or {})
        wall_start = time.perf_counter()

        result = ExecutionResult()

        workers = min(self._max_workers, plan.max_parallelism or 1)
        pool = self._get_pool(workers)
        owns_pool = pool is not self._pool  # True if we created an ephemeral pool

        try:
            for wave in plan.waves:
                futures: Dict[Future, str] = {}

                for node_name in wave.nodes:
                    node = self._nodes[node_name]
                    node.status = "running"

                    # Build dependency context for this node
                    dep_ctx = {
                        dep: context.get(dep)
                        for dep in node.dependencies
                    }
                    # Merge full context so nodes can access anything
                    merged_ctx = {**context, **dep_ctx, "__deps__": dep_ctx}

                    future = pool.submit(self._execute_node, node, merged_ctx)
                    futures[future] = node_name

                # Collect wave results
                for future in as_completed(futures):
                    node_name = futures[future]
                    node = self._nodes[node_name]

                    try:
                        output, latency = future.result()
                        node.output = output
                        node.latency_ms = latency
                        node.status = "done"
                        context[node_name] = output
                        result.outputs[node_name] = output
                        result.node_latencies[node_name] = latency
                        result.node_statuses[node_name] = "done"
                    except Exception as e:
                        node.status = "failed"
                        node.error = str(e)
                        result.errors[node_name] = str(e)
                        result.node_statuses[node_name] = "failed"
                        logger.error(
                            f"DAG '{self._name}' node '{node_name}' failed: {e}"
                        )
                        if fail_fast:
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break

                if fail_fast and result.errors:
                    break
        finally:
            if owns_pool:
                pool.shutdown(wait=False)

        wall_ms = (time.perf_counter() - wall_start) * 1000
        total_node_ms = sum(result.node_latencies.values())

        result.total_latency_ms = total_node_ms
        result.wall_clock_ms = wall_ms
        result.parallelism_achieved = (
            total_node_ms / wall_ms if wall_ms > 0 else 1.0
        )

        logger.debug(
            f"DAG '{self._name}' apply: wall={wall_ms:.2f}ms, "
            f"sum={total_node_ms:.2f}ms, parallelism={result.parallelism_achieved:.1f}x"
        )

        return result

    def batch_apply(
        self,
        contexts: List[Dict[str, Any]],
        fail_fast: bool = True,
    ) -> List[ExecutionResult]:
        """
        Execute the same DAG shape for N independent contexts in parallel.

        All contexts share a single warm thread pool. This is the fastest
        path for batch routing — e.g., routing 100 MCP tool calls at once.
        """
        if not contexts:
            return []

        plan = self.plan()
        workers = min(self._max_workers, max(plan.max_parallelism, len(contexts)))
        pool = self._get_pool(workers)
        owns_pool = pool is not self._pool

        results: List[ExecutionResult] = []
        try:
            # Submit all contexts as independent DAG runs on the shared pool
            batch_futures: Dict[Future, int] = {}
            for i, ctx in enumerate(contexts):
                future = pool.submit(self._apply_single, plan, ctx, fail_fast, pool)
                batch_futures[future] = i

            # Collect in completion order, store in original order
            results = [None] * len(contexts)
            for future in as_completed(batch_futures):
                idx = batch_futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    err_result = ExecutionResult()
                    err_result.errors["batch"] = str(e)
                    results[idx] = err_result
        finally:
            if owns_pool:
                pool.shutdown(wait=False)

        return results

    def _apply_single(
        self,
        plan: ExecutionPlan,
        initial_context: Dict[str, Any],
        fail_fast: bool,
        pool: ThreadPoolExecutor,
    ) -> ExecutionResult:
        """Execute a single DAG run using a pre-existing pool (no pool creation cost)."""
        context = dict(initial_context)
        wall_start = time.perf_counter()
        result = ExecutionResult()

        for wave in plan.waves:
            futures: Dict[Future, str] = {}
            for node_name in wave.nodes:
                node = self._nodes[node_name]
                dep_ctx = {dep: context.get(dep) for dep in node.dependencies}
                merged_ctx = {**context, **dep_ctx, "__deps__": dep_ctx}
                future = pool.submit(self._execute_node, node, merged_ctx)
                futures[future] = node_name

            for future in as_completed(futures):
                node_name = futures[future]
                try:
                    output, latency = future.result()
                    context[node_name] = output
                    result.outputs[node_name] = output
                    result.node_latencies[node_name] = latency
                    result.node_statuses[node_name] = "done"
                except Exception as e:
                    result.errors[node_name] = str(e)
                    result.node_statuses[node_name] = "failed"
                    if fail_fast:
                        for f in futures:
                            f.cancel()
                        break
            if fail_fast and result.errors:
                break

        wall_ms = (time.perf_counter() - wall_start) * 1000
        total_node_ms = sum(result.node_latencies.values())
        result.total_latency_ms = total_node_ms
        result.wall_clock_ms = wall_ms
        result.parallelism_achieved = total_node_ms / wall_ms if wall_ms > 0 else 1.0
        return result

    @staticmethod
    def _execute_node(node: DAGNode, context: Dict[str, Any]) -> Tuple[Any, float]:
        """Execute a single node with timing."""
        t0 = time.perf_counter()
        output = node.execute(context)
        latency = (time.perf_counter() - t0) * 1000
        return output, latency

    # ── Introspection ─────────────────────────────────────────────────────

    def describe(self) -> Dict[str, Any]:
        """Return a human-readable description of the DAG."""
        plan = self.plan()
        return {
            "name": self._name,
            "total_nodes": plan.total_nodes,
            "waves": [
                {
                    "depth": w.depth,
                    "nodes": w.nodes,
                    "parallelism": len(w.nodes),
                }
                for w in plan.waves
            ],
            "max_parallelism": plan.max_parallelism,
            "critical_path_depth": plan.critical_path_depth,
            "nodes": {
                name: {
                    "dependencies": sorted(node.dependencies),
                    "status": node.status,
                }
                for name, node in self._nodes.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"DAGExecutor(name={self._name!r}, "
            f"nodes={len(self._nodes)}, "
            f"max_workers={self._max_workers})"
        )


# ── Convenience Builders ──────────────────────────────────────────────────────


def build_signal_dag(orchestrator) -> DAGExecutor:
    """
    Build a DAG for parallel signal extraction.

    All 6 signals are independent → single wave, full parallelism.

        ┌─────────┐ ┌──────────┐ ┌────────────┐
        │ keyword  │ │ modality │ │ complexity │
        └────┬─────┘ └────┬─────┘ └─────┬──────┘
             │            │              │
        ┌────┴─────┐ ┌────┴─────┐ ┌─────┴──────┐
        │  domain  │ │ language │ │   safety   │
        └──────────┘ └──────────┘ └────────────┘

    All nodes at depth 0 — one wave, 6-way parallel.
    """
    dag = DAGExecutor(max_workers=6, name="signal_extraction")

    for name, signal in orchestrator.signals.items():
        if not signal.enabled:
            continue

        def make_fn(sig):
            """Capture signal in closure"""
            def fn(ctx: Dict[str, Any]) -> Any:
                query = ctx.get("__query__", {})
                return sig.run(query)
            return fn

        dag.add_node(name, make_fn(signal))

    return dag


def build_topology_dag(
    enumerate_gpus_fn: Callable,
    enumerate_nics_fn: Callable,
    compute_pairs_fn: Callable,
    score_endpoints_fn: Optional[Callable] = None,
    generate_nccl_fn: Optional[Callable] = None,
) -> DAGExecutor:
    """
    Build a DAG for NUMA topology detection and optimization.

        ┌────────────┐   ┌────────────┐
        │ gpu_enum   │   │ nic_enum   │
        └─────┬──────┘   └─────┬──────┘
              │                │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  gpu_nic_pair  │
              └───────┬────────┘
                      │
           ┌──────────┼──────────┐
           │          │          │
    ┌──────▼───┐ ┌────▼─────┐ ┌─▼──────────┐
    │ scoring  │ │ nccl_env │ │ dra_claims │
    └──────────┘ └──────────┘ └────────────┘

    Waves:
      0: gpu_enum ∥ nic_enum           (2-way parallel)
      1: gpu_nic_pair                  (depends on both)
      2: scoring ∥ nccl_env ∥ dra      (3-way parallel, if present)
    """
    dag = DAGExecutor(max_workers=4, name="topology_discovery")

    dag.add_node("gpu_enum", enumerate_gpus_fn)
    dag.add_node("nic_enum", enumerate_nics_fn)
    dag.add_node(
        "gpu_nic_pair",
        compute_pairs_fn,
        depends_on={"gpu_enum", "nic_enum"},
    )

    if score_endpoints_fn:
        dag.add_node(
            "endpoint_scoring",
            score_endpoints_fn,
            depends_on={"gpu_nic_pair"},
        )

    if generate_nccl_fn:
        dag.add_node(
            "nccl_env",
            generate_nccl_fn,
            depends_on={"gpu_nic_pair"},
        )

    return dag


def build_routing_dag(
    signal_extract_fn: Callable,
    topology_refresh_fn: Optional[Callable] = None,
    policy_eval_fn: Optional[Callable] = None,
    numa_score_fn: Optional[Callable] = None,
) -> DAGExecutor:
    """
    Build the full semantic routing DAG.

        ┌──────────────────┐   ┌───────────────────┐
        │ signal_extraction│   │ topology_refresh   │
        └────────┬─────────┘   └────────┬───────────┘
                 │                      │
                 └──────────┬───────────┘
                            │
                   ┌────────▼─────────┐
                   │   policy_eval    │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │   numa_scoring   │
                   └──────────────────┘

    Waves:
      0: signal_extraction ∥ topology_refresh    (overlap I/O with CPU)
      1: policy_eval                             (needs signals)
      2: numa_scoring                            (needs policy + topology)
    """
    dag = DAGExecutor(max_workers=4, name="semantic_routing")

    dag.add_node("signal_extraction", signal_extract_fn)

    deps_for_policy = {"signal_extraction"}

    if topology_refresh_fn:
        dag.add_node("topology_refresh", topology_refresh_fn)
        deps_for_numa = {"policy_eval", "topology_refresh"}
    else:
        deps_for_numa = {"policy_eval"}

    if policy_eval_fn:
        dag.add_node("policy_eval", policy_eval_fn, depends_on=deps_for_policy)

        if numa_score_fn:
            dag.add_node("numa_scoring", numa_score_fn, depends_on=deps_for_numa)

    return dag
