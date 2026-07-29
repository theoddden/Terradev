"""Tests for terradev_cli.core.kv_sharing.

KV sharing is a key VRAM/cost moat for multi-agent serving. These tests cover
the planner and the resulting budgets/plans.
"""

import pytest

from terradev_cli.core.kv_sharing import (
    AgentKVBudget,
    KVSharingPlan,
    MultiAgentVRAMPlanner,
    SharingTopology,
)


@pytest.fixture
def planner():
    return MultiAgentVRAMPlanner()


def test_sharing_topology_enum_values():
    """Sharing topology values are stable strings."""
    assert SharingTopology.BROADCAST.value == "broadcast"
    assert SharingTopology.STAR.value == "star"
    assert SharingTopology.CHAIN.value == "chain"
    assert SharingTopology.NONE.value == "none"


def test_compute_per_agent_budget(planner):
    """Per-agent budget captures model weights and KV requirements."""
    budget = planner.compute_per_agent_budget("70b", 8000)
    assert budget.model == "70b"
    assert budget.context_k_tokens == 8000
    assert budget.model_size_b == 70
    assert budget.kv_gb_per_agent > 0
    assert budget.model_weights_gb > 0


def test_compute_per_agent_budget_fits_7b(planner):
    """A 7B model with a small context fits on an H100."""
    budget = planner.compute_per_agent_budget("7b", 100, gpu_type="H100_SXM")
    assert budget.fits is True
    assert budget.agents_per_gpu >= 1


def test_compute_broadcast_plan(planner):
    """A broadcast plan shows large VRAM savings for shared prefixes."""
    plan = planner.compute(
        n_agents=10,
        context_k=32000,
        model="70b",
        topology=SharingTopology.BROADCAST,
        shared_fraction=0.8,
    )
    assert plan.n_agents == 10
    assert plan.topology == SharingTopology.BROADCAST
    assert plan.shared_kv_gb >= 0
    assert plan.total_kv_naive_gb > plan.total_kv_with_sharing_gb
    assert plan.vram_saving_gb > 0
    assert plan.recommended_gpu_type is not None
    assert plan.recommended_gpu_count > 0


def test_compute_multi_topology(planner):
    """compute_multi_topology returns a plan for each topology."""
    plans = planner.compute_multi_topology(
        n_agents=10,
        context_k=16000,
        model="70b",
    )
    assert set(plans.keys()) == {
        "broadcast",
        "star",
        "chain",
        "none",
    }
    for plan in plans.values():
        assert isinstance(plan, KVSharingPlan)
        assert plan.n_agents == 10


def test_plan_summary_and_to_dict(planner):
    """A plan can produce human-readable lines and a serializable dict."""
    plan = planner.compute(
        n_agents=5,
        context_k=8000,
        model="8b",
        topology=SharingTopology.BROADCAST,
    )
    lines = plan.summary_lines()
    assert isinstance(lines, list)
    assert any("VRAM" in line for line in lines)

    d = plan.to_dict()
    assert d["n_agents"] == 5
    assert d["model"] == "8b"
    assert "total_kv_with_sharing_gb" in d
