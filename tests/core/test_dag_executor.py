"""Tests for terradev_cli.core.dag_executor.

The DAG executor is the backbone of parallel work in Terradev. These tests
cover graph construction, topological planning, dependency propagation, and
failure handling.
"""

import pytest

from terradev_cli.core.dag_executor import DAGExecutor


def test_plan_for_independent_nodes():
    """Independent nodes are scheduled in a single wave."""
    dag = DAGExecutor(reuse_pool=False, max_workers=2, name="independent")
    dag.add_node("a", lambda ctx: 1)
    dag.add_node("b", lambda ctx: 2)

    plan = dag.plan()
    assert plan.total_nodes == 2
    assert plan.critical_path_depth == 1
    assert plan.max_parallelism == 2


def test_plan_waves_for_chained_dependencies():
    """A chain creates one node per wave."""
    dag = DAGExecutor(reuse_pool=False, max_workers=2, name="chain")
    dag.add_node("first", lambda ctx: 1)
    dag.add_node("second", lambda ctx: ctx["first"] + 1, depends_on={"first"})
    dag.add_node("third", lambda ctx: ctx["second"] + 1, depends_on={"second"})

    plan = dag.plan()
    assert plan.critical_path_depth == 3
    assert plan.max_parallelism == 1
    assert [w.nodes for w in plan.waves] == [["first"], ["second"], ["third"]]


def test_plan_detects_cycle():
    """Cyclic dependencies raise a clear ValueError."""
    dag = DAGExecutor(reuse_pool=False, max_workers=2, name="cycle")
    dag.add_node("a", lambda ctx: 1, depends_on={"b"})
    dag.add_node("b", lambda ctx: 2, depends_on={"a"})

    with pytest.raises(ValueError, match="Cycle detected"):
        dag.plan()


def test_plan_rejects_missing_dependency():
    """A node depending on a non-existent node raises a ValueError."""
    dag = DAGExecutor(reuse_pool=False, max_workers=2, name="missing_dep")
    dag.add_node("a", lambda ctx: 1, depends_on={"ghost"})

    with pytest.raises(ValueError, match="does not exist"):
        dag.plan()


def test_apply_passes_dependency_outputs():
    """A node receives the outputs of all its dependencies in context."""
    dag = DAGExecutor(reuse_pool=False, max_workers=2, name="deps")
    dag.add_node("left", lambda ctx: "L")
    dag.add_node("right", lambda ctx: "R")
    dag.add_node(
        "merge",
        lambda ctx: f"{ctx['left']}{ctx['right']}",
        depends_on={"left", "right"},
    )

    result = dag.apply()
    assert result.success is True
    assert result.outputs["merge"] == "LR"


def test_apply_carries_initial_context():
    """Initial context is available to every node."""
    dag = DAGExecutor(reuse_pool=False, max_workers=2, name="context")
    dag.add_node("greet", lambda ctx: f"hello {ctx['name']}")

    result = dag.apply(initial_context={"name": "world"})
    assert result.outputs["greet"] == "hello world"


def test_apply_fail_fast_on_error():
    """A failing node stops further execution when fail_fast is True."""
    dag = DAGExecutor(reuse_pool=False, max_workers=2, name="failing")
    dag.add_node("ok", lambda ctx: "ok")
    dag.add_node("bad", lambda ctx: (_ for _ in ()).throw(ValueError("boom")), depends_on={"ok"})
    dag.add_node("never", lambda ctx: "never", depends_on={"bad"})

    result = dag.apply()
    assert result.success is False
    assert "bad" in result.errors
    assert "never" not in result.outputs


def test_apply_respects_dependencies():
    """Diamond dependency shape resolves in two waves and returns correct order."""
    order = []

    def make(n):
        def fn(ctx):
            order.append(n)
            return n
        return fn

    dag = DAGExecutor(reuse_pool=False, max_workers=2, name="diamond")
    dag.add_node("root", make("root"))
    dag.add_node("left", make("left"), depends_on={"root"})
    dag.add_node("right", make("right"), depends_on={"root"})
    dag.add_node("join", make("join"), depends_on={"left", "right"})

    result = dag.apply()
    assert result.success is True
    assert order.index("root") < order.index("left")
    assert order.index("root") < order.index("right")
    assert order.index("join") == len(order) - 1


