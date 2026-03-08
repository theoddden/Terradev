#!/usr/bin/env python3
"""
Tests for the Terraform-style DAG Parallel Executor.

Covers:
  1. Graph construction and validation
  2. Cycle detection
  3. Topological sort into execution waves
  4. Parallel execution with dependency passing
  5. Fail-fast behavior
  6. Signal extraction DAG builder
  7. Topology DAG builder
  8. Routing DAG builder
  9. Parallelism measurement
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.core.dag_executor import (
    DAGExecutor,
    DAGNode,
    ExecutionPlan,
    ExecutionResult,
    build_signal_dag,
    build_topology_dag,
    build_routing_dag,
)


# ── Graph Construction ────────────────────────────────────────────────────────


class TestGraphConstruction:
    def test_add_node(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1)
        assert "a" in dag._nodes

    def test_add_edge(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1)
        dag.add_node("b", lambda ctx: 2)
        dag.add_edge("a", "b")
        assert "a" in dag._nodes["b"].dependencies

    def test_chaining(self):
        dag = (
            DAGExecutor(name="test")
            .add_node("a", lambda ctx: 1)
            .add_node("b", lambda ctx: 2)
            .add_edge("a", "b")
        )
        assert len(dag._nodes) == 2

    def test_duplicate_node_raises(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1)
        with pytest.raises(ValueError, match="already exists"):
            dag.add_node("a", lambda ctx: 2)

    def test_edge_to_missing_node_raises(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1)
        with pytest.raises(ValueError, match="not in DAG"):
            dag.add_edge("a", "nonexistent")


# ── Cycle Detection ──────────────────────────────────────────────────────────


class TestCycleDetection:
    def test_no_cycle(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1)
        dag.add_node("b", lambda ctx: 2, depends_on={"a"})
        dag.add_node("c", lambda ctx: 3, depends_on={"b"})
        plan = dag.plan()  # Should not raise
        assert plan.total_nodes == 3

    def test_direct_cycle_raises(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1, depends_on={"b"})
        dag.add_node("b", lambda ctx: 2, depends_on={"a"})
        with pytest.raises(ValueError, match="Cycle detected"):
            dag.plan()

    def test_indirect_cycle_raises(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1, depends_on={"c"})
        dag.add_node("b", lambda ctx: 2, depends_on={"a"})
        dag.add_node("c", lambda ctx: 3, depends_on={"b"})
        with pytest.raises(ValueError, match="Cycle detected"):
            dag.plan()

    def test_missing_dependency_raises(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1, depends_on={"nonexistent"})
        with pytest.raises(ValueError, match="does not exist"):
            dag.plan()


# ── Planning (Topological Sort) ──────────────────────────────────────────────


class TestPlanning:
    def test_single_wave_independent_nodes(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1)
        dag.add_node("b", lambda ctx: 2)
        dag.add_node("c", lambda ctx: 3)
        plan = dag.plan()
        assert plan.critical_path_depth == 1
        assert plan.max_parallelism == 3
        assert len(plan.waves) == 1
        assert set(plan.waves[0].nodes) == {"a", "b", "c"}

    def test_linear_chain(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1)
        dag.add_node("b", lambda ctx: 2, depends_on={"a"})
        dag.add_node("c", lambda ctx: 3, depends_on={"b"})
        plan = dag.plan()
        assert plan.critical_path_depth == 3
        assert plan.max_parallelism == 1

    def test_diamond_pattern(self):
        """
            A
           / \
          B   C
           \ /
            D
        """
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1)
        dag.add_node("b", lambda ctx: 2, depends_on={"a"})
        dag.add_node("c", lambda ctx: 3, depends_on={"a"})
        dag.add_node("d", lambda ctx: 4, depends_on={"b", "c"})
        plan = dag.plan()
        assert plan.critical_path_depth == 3  # A → B|C → D
        assert plan.max_parallelism == 2  # B and C in parallel
        # Wave 0: A, Wave 1: B+C, Wave 2: D
        assert plan.waves[0].nodes == ["a"]
        assert set(plan.waves[1].nodes) == {"b", "c"}
        assert plan.waves[2].nodes == ["d"]

    def test_topology_shaped_dag(self):
        """Matches the topology discovery shape:
        gpu_enum ∥ nic_enum → pairing → scoring ∥ nccl
        """
        dag = DAGExecutor(name="test")
        dag.add_node("gpu_enum", lambda ctx: [])
        dag.add_node("nic_enum", lambda ctx: [])
        dag.add_node("pairing", lambda ctx: {}, depends_on={"gpu_enum", "nic_enum"})
        dag.add_node("scoring", lambda ctx: {}, depends_on={"pairing"})
        dag.add_node("nccl", lambda ctx: {}, depends_on={"pairing"})
        plan = dag.plan()
        assert plan.critical_path_depth == 3
        assert plan.max_parallelism == 2  # gpu+nic or scoring+nccl


# ── Execution (Apply) ────────────────────────────────────────────────────────


class TestExecution:
    def test_simple_execution(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 42)
        result = dag.apply()
        assert result.success
        assert result.outputs["a"] == 42

    def test_dependency_passing(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 10)
        dag.add_node("b", lambda ctx: ctx["a"] * 2, depends_on={"a"})
        dag.add_node("c", lambda ctx: ctx["b"] + 5, depends_on={"b"})
        result = dag.apply()
        assert result.success
        assert result.outputs["a"] == 10
        assert result.outputs["b"] == 20
        assert result.outputs["c"] == 25

    def test_diamond_dependency_passing(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 10)
        dag.add_node("b", lambda ctx: ctx["a"] + 1, depends_on={"a"})
        dag.add_node("c", lambda ctx: ctx["a"] + 2, depends_on={"a"})
        dag.add_node("d", lambda ctx: ctx["b"] + ctx["c"], depends_on={"b", "c"})
        result = dag.apply()
        assert result.success
        assert result.outputs["d"] == 23  # (10+1) + (10+2)

    def test_initial_context(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: ctx["seed"] * 2)
        result = dag.apply(initial_context={"seed": 5})
        assert result.outputs["a"] == 10

    def test_parallel_independent_nodes(self):
        """6 independent nodes should run in parallel."""
        def slow_node(n):
            def fn(ctx):
                time.sleep(0.05)
                return n
            return fn

        dag = DAGExecutor(max_workers=6, name="test")
        for i in range(6):
            dag.add_node(f"n{i}", slow_node(i))

        result = dag.apply()
        assert result.success
        # Serial would be ~300ms, parallel should be ~50-80ms
        assert result.wall_clock_ms < 200
        # Parallelism factor should be > 2x
        assert result.parallelism_achieved > 2.0

    def test_fail_fast(self):
        dag = DAGExecutor(name="test")
        dag.add_node("good", lambda ctx: 1)
        dag.add_node("bad", lambda ctx: 1 / 0)
        dag.add_node("after", lambda ctx: ctx["bad"], depends_on={"bad"})
        result = dag.apply(fail_fast=True)
        assert not result.success
        assert "bad" in result.errors

    def test_fail_no_fast(self):
        dag = DAGExecutor(name="test")
        dag.add_node("good1", lambda ctx: 1)
        dag.add_node("good2", lambda ctx: 2)
        result = dag.apply(fail_fast=False)
        assert result.success
        assert len(result.outputs) == 2

    def test_latency_tracking(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: time.sleep(0.01) or 1)
        result = dag.apply()
        assert result.node_latencies["a"] > 5  # at least 5ms


# ── Introspection ─────────────────────────────────────────────────────────────


class TestIntrospection:
    def test_describe(self):
        dag = DAGExecutor(name="test")
        dag.add_node("a", lambda ctx: 1)
        dag.add_node("b", lambda ctx: 2, depends_on={"a"})
        desc = dag.describe()
        assert desc["name"] == "test"
        assert desc["total_nodes"] == 2
        assert desc["critical_path_depth"] == 2
        assert desc["max_parallelism"] == 1
        assert "a" in desc["nodes"]
        assert desc["nodes"]["b"]["dependencies"] == ["a"]

    def test_repr(self):
        dag = DAGExecutor(max_workers=4, name="my_dag")
        dag.add_node("a", lambda ctx: 1)
        s = repr(dag)
        assert "my_dag" in s
        assert "nodes=1" in s


# ── Builder: Signal Extraction ────────────────────────────────────────────────


class TestBuildSignalDag:
    def test_signal_dag_all_independent(self):
        from terradev_cli.core.semantic_signals.orchestrator import SignalOrchestrator
        orch = SignalOrchestrator(parallel=False)
        dag = build_signal_dag(orch)
        plan = dag.plan()
        # All 6 signals should be in wave 0
        assert plan.critical_path_depth == 1
        assert plan.max_parallelism == 6
        assert plan.total_nodes == 6

    def test_signal_dag_executes(self):
        from terradev_cli.core.semantic_signals.orchestrator import SignalOrchestrator
        orch = SignalOrchestrator(parallel=False)
        dag = build_signal_dag(orch)
        result = dag.apply(initial_context={"__query__": {"content": "Write Python code"}})
        assert result.success
        assert "keyword" in result.outputs
        assert "modality" in result.outputs
        assert "complexity" in result.outputs

    def test_signal_dag_disabled_signal_excluded(self):
        from terradev_cli.core.semantic_signals.orchestrator import SignalOrchestrator
        orch = SignalOrchestrator(config={"safety_enabled": False}, parallel=False)
        dag = build_signal_dag(orch)
        plan = dag.plan()
        assert plan.total_nodes == 5
        assert "safety" not in [n for w in plan.waves for n in w.nodes]


# ── Builder: Topology Discovery ───────────────────────────────────────────────


class TestBuildTopologyDag:
    def test_topology_dag_shape(self):
        dag = build_topology_dag(
            enumerate_gpus_fn=lambda ctx: ["gpu0", "gpu1"],
            enumerate_nics_fn=lambda ctx: ["nic0"],
            compute_pairs_fn=lambda ctx: {"gpu0": "nic0", "gpu1": "nic0"},
            score_endpoints_fn=lambda ctx: {"ep0": 0.9},
            generate_nccl_fn=lambda ctx: {"NCCL_NET_GDR_LEVEL": "PIX"},
        )
        plan = dag.plan()
        # Wave 0: gpu_enum ∥ nic_enum
        # Wave 1: gpu_nic_pair
        # Wave 2: endpoint_scoring ∥ nccl_env
        assert plan.critical_path_depth == 3
        assert plan.max_parallelism == 2  # wave 0 or wave 2
        assert plan.total_nodes == 5

    def test_topology_dag_executes(self):
        dag = build_topology_dag(
            enumerate_gpus_fn=lambda ctx: ["gpu0", "gpu1"],
            enumerate_nics_fn=lambda ctx: ["nic0"],
            compute_pairs_fn=lambda ctx: {
                "pairs": [(ctx["gpu_enum"][0], ctx["nic_enum"][0])]
            },
        )
        result = dag.apply()
        assert result.success
        assert result.outputs["gpu_enum"] == ["gpu0", "gpu1"]
        assert result.outputs["nic_enum"] == ["nic0"]
        assert "pairs" in result.outputs["gpu_nic_pair"]

    def test_topology_dag_minimal(self):
        """Without optional scoring/nccl nodes."""
        dag = build_topology_dag(
            enumerate_gpus_fn=lambda ctx: ["gpu0"],
            enumerate_nics_fn=lambda ctx: ["nic0"],
            compute_pairs_fn=lambda ctx: {},
        )
        plan = dag.plan()
        assert plan.total_nodes == 3
        assert plan.critical_path_depth == 2  # enum wave → pairing wave


# ── Builder: Routing DAG ─────────────────────────────────────────────────────


class TestBuildRoutingDag:
    def test_routing_dag_full(self):
        dag = build_routing_dag(
            signal_extract_fn=lambda ctx: {"modality": "code"},
            topology_refresh_fn=lambda ctx: {"gpus": 4},
            policy_eval_fn=lambda ctx: {"route_to": "deepseek"},
            numa_score_fn=lambda ctx: {"score": 0.95},
        )
        plan = dag.plan()
        # Wave 0: signals ∥ topology
        # Wave 1: policy_eval
        # Wave 2: numa_scoring
        assert plan.critical_path_depth == 3
        assert plan.max_parallelism == 2

    def test_routing_dag_executes(self):
        dag = build_routing_dag(
            signal_extract_fn=lambda ctx: {"modality": "code", "complexity": 0.8},
            topology_refresh_fn=lambda ctx: {"numa_nodes": 2},
            policy_eval_fn=lambda ctx: {
                "route_to": "deepseek",
                "signals": ctx["signal_extraction"],
            },
            numa_score_fn=lambda ctx: {
                "best_gpu": 0,
                "topology": ctx["topology_refresh"],
                "decision": ctx["policy_eval"],
            },
        )
        result = dag.apply()
        assert result.success
        assert result.outputs["numa_scoring"]["best_gpu"] == 0
        assert result.outputs["numa_scoring"]["topology"]["numa_nodes"] == 2
        assert result.outputs["policy_eval"]["route_to"] == "deepseek"

    def test_routing_dag_no_topology(self):
        dag = build_routing_dag(
            signal_extract_fn=lambda ctx: {"modality": "text"},
            policy_eval_fn=lambda ctx: {"route_to": "llama-8b"},
        )
        plan = dag.plan()
        assert plan.total_nodes == 2
        assert plan.critical_path_depth == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
