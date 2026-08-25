"""Tests for terradev_cli.core.semantic_router.

The semantic router combines signal extraction, Boolean policy evaluation,
and NUMA-aware endpoint scoring. Tests use fake signals to avoid running
heavy NLP.
"""

from unittest.mock import MagicMock

import pytest

from terradev_cli.core.semantic_router import (
    NUMAEndpointScorer,
    NUMAScorecard,
    PolicyExpressionEvaluator,
    RoutingDecision,
    RoutingPolicy,
    RoutingRule,
    SemanticRouter,
    load_policy_from_dict,
    load_policy_from_json,
    load_policy_from_yaml,
)
from terradev_cli.core.semantic_signals import SignalResult, SignalVector, SignalType


def _make_signal_vector(**flat):
    """Build a SignalVector with the requested flat keys and default confidence."""
    results = {}
    for name, value in flat.items():
        results[name] = SignalResult(
            signal_type=SignalType.CUSTOM,
            name=name,
            value=value,
            confidence=0.95,
        )
    return SignalVector(signals=results, total_latency_ms=1.0)


def test_routing_rule_defaults():
    """RoutingRule has expected defaults."""
    rule = RoutingRule(name="test", condition="True")
    assert rule.route_to is None
    assert rule.strategy is None
    assert rule.priority == 0
    assert rule.numa_policy is None
    assert rule.tags == []


def test_policy_expression_evaluator_literal():
    """Evaluator accepts literal True/False."""
    evaluator = PolicyExpressionEvaluator()
    assert evaluator.evaluate("True", {}) is True
    assert evaluator.evaluate("False", {}) is False


def test_policy_expression_evaluator_comparisons():
    """Evaluator handles numeric and string comparisons."""
    evaluator = PolicyExpressionEvaluator()
    ctx = {"complexity": 0.8, "modality": "code"}
    assert evaluator.evaluate("complexity > 0.7", ctx) is True
    assert evaluator.evaluate("complexity < 0.3", ctx) is False
    assert evaluator.evaluate("modality == 'code'", ctx) is True
    assert evaluator.evaluate("modality != 'vision'", ctx) is True


def test_policy_expression_evaluator_boolean_ops():
    """Evaluator handles AND, OR, and NOT."""
    evaluator = PolicyExpressionEvaluator()
    ctx = {"modality": "code", "complexity": 0.8}
    assert evaluator.evaluate("modality == 'code' AND complexity > 0.7", ctx) is True
    assert evaluator.evaluate("modality == 'vision' OR complexity > 0.7", ctx) is True
    assert evaluator.evaluate("NOT modality == 'vision'", ctx) is True


def test_policy_expression_evaluator_dotted():
    """Evaluator supports dotted access and set membership."""
    evaluator = PolicyExpressionEvaluator()
    ctx = {
        "safety": {"flagged": True, "severity": 0.95, "flags": {"pii_detected"}},
        "keyword": {"dominant_category": "code", "tags": {"code"}},
    }
    assert evaluator.evaluate("safety.flagged and safety.severity >= 0.9", ctx) is True
    assert evaluator.evaluate("'pii_detected' in safety.flags", ctx) is True
    assert evaluator.evaluate("'code' in keyword.tags", ctx) is True
    assert evaluator.evaluate("keyword.dominant_category == 'code'", ctx) is True


def test_policy_expression_evaluator_malformed():
    """Malformed expressions compile to False."""
    evaluator = PolicyExpressionEvaluator()
    assert evaluator.evaluate("complexity @ 5", {}) is False


def test_policy_expression_compile_caches():
    """compile() pre-caches AST and subsequent evals are fast."""
    evaluator = PolicyExpressionEvaluator()
    evaluator.compile("complexity > 0.5")
    assert "complexity > 0.5" in evaluator._ast_cache
    assert evaluator.evaluate("complexity > 0.5", {"complexity": 0.9}) is True


def test_load_policy_from_dict():
    """load_policy_from_dict builds a sorted policy."""
    data = {
        "name": "test-policy",
        "default_strategy": "cost",
        "rules": [
            {"name": "cheap", "when": "complexity < 0.3", "priority": 10},
            {"name": "expensive", "when": "complexity > 0.7", "priority": 20},
        ],
    }
    policy = load_policy_from_dict(data)
    assert policy.name == "test-policy"
    assert policy.rules[0].name == "expensive"  # sorted by priority desc
    assert policy.rules[1].name == "cheap"


def test_load_policy_from_json(tmp_path):
    """load_policy_from_json reads a policy from disk."""
    path = tmp_path / "policy.json"
    path.write_text(
        '{"name": "json-policy", "rules": [{"name": "r1", "when": "True"}]}'
    )
    policy = load_policy_from_json(str(path))
    assert policy.name == "json-policy"
    assert policy.rules[0].name == "r1"


def test_load_policy_from_yaml(tmp_path):
    """load_policy_from_yaml reads a policy from disk."""
    yaml = pytest.importorskip("yaml", reason="PyYAML not installed")
    path = tmp_path / "policy.yaml"
    path.write_text(
        "name: yaml-policy\nrules:\n  - name: r1\n    when: 'True'\n"
    )
    policy = load_policy_from_yaml(str(path))
    assert policy.name == "yaml-policy"


def test_numa_endpoint_scorer_build_maps():
    """NUMAEndpointScorer parses topology report data."""
    topology = {
        "pairs": [
            {"gpu": "GPU 0 (NVIDIA H100 80GB HBM3)", "locality": "PIX", "rdma_path": True},
            {"gpu": "GPU 1 (NVIDIA H100 80GB HBM3)", "locality": "SYS", "rdma_path": False},
        ],
        "numa_map": {"0": {"gpus": ["GPU 0"]}, "1": {"gpus": ["GPU 1"]}},
    }
    scorer = NUMAEndpointScorer(topology)
    assert scorer._gpu_locality_map[0] == "PIX"
    assert scorer._gpu_locality_map[1] == "SYS"
    assert scorer._gpu_numa_map[0] == 0


def test_numa_endpoint_scorer_score_endpoint():
    """score_endpoint returns a populated NUMAScorecard."""
    topology = {
        "pairs": [
            {"gpu": "GPU 0 (NVIDIA H100 80GB HBM3)", "locality": "PIX", "rdma_path": True},
        ],
        "numa_map": {"0": {"gpus": ["GPU 0"]}},
    }
    scorer = NUMAEndpointScorer(topology)
    card = scorer.score_endpoint("ep-1", gpu_index=0, model_type="transformer")
    assert card.endpoint_id == "ep-1"
    assert card.pcie_locality == "PIX"
    assert card.locality_score == 0.0
    assert card.metadata["cuda_graph_recommended"] is True
    assert card.metadata["model_type"] == "transformer"


def test_numa_endpoint_scorer_fallback():
    """Unknown GPU index returns a neutral fallback scorecard."""
    scorer = NUMAEndpointScorer({})
    card = scorer.score_endpoint("ep-1", gpu_index=99)
    assert card.locality_score == 1.5
    assert card.pcie_locality == "SYS"


def test_numa_endpoint_scorer_rank_endpoints():
    """rank_endpoints sorts by combined NUMA score."""
    topology = {
        "pairs": [
            {"gpu": "GPU 0 (NVIDIA H100)", "locality": "SYS"},
            {"gpu": "GPU 1 (NVIDIA H100)", "locality": "PIX"},
        ],
        "numa_map": {"0": {"gpus": ["GPU 0", "GPU 1"]}},
    }
    scorer = NUMAEndpointScorer(topology)
    ranked = scorer.rank_endpoints(["ep-a", "ep-b"], gpu_indices={"ep-a": 0, "ep-b": 1})
    assert ranked[0][0] == "ep-b"  # PIX is better
    assert ranked[1][0] == "ep-a"


def test_numa_cuda_graph_score_penalties():
    """MoE and CNN model types receive CUDA graph penalties."""
    topology = {
        "pairs": [{"gpu": "GPU 0 (NVIDIA H100)", "locality": "PIX"}],
        "numa_map": {"0": {"gpus": ["GPU 0"]}},
    }
    scorer = NUMAEndpointScorer(topology)
    moe = scorer.score_endpoint("ep", gpu_index=0, model_type="moe")
    cnn = scorer.score_endpoint("ep", gpu_index=0, model_type="cnn")
    llm = scorer.score_endpoint("ep", gpu_index=0, model_type="llm")
    assert moe.metadata["cuda_graph_score"] < llm.metadata["cuda_graph_score"]
    assert cnn.metadata["cuda_graph_score"] < llm.metadata["cuda_graph_score"]


@pytest.fixture
def fake_orchestrator_class(monkeypatch):
    """Patch SignalOrchestrator used by SemanticRouter with a fake."""

    class FakeSignalOrchestrator:
        def __init__(self, config=None):
            self.config = config or {}

        def extract(self, query):
            flat = {}
            text = query.get("content", "").lower()
            if "code" in text or "python" in text or "function" in text:
                flat = {
                    "modality": "code",
                    "complexity": 0.8,
                    "safety": {"flagged": False, "severity": 0.0, "flags": set()},
                }
            elif "image" in text:
                flat = {"modality": "vision", "complexity": 0.4}
            else:
                flat = {"modality": "text", "complexity": 0.4}
            return _make_signal_vector(**flat)

        def batch_extract(self, queries):
            return [self.extract(q) for q in queries]

    monkeypatch.setattr(
        "terradev_cli.core.semantic_router.SignalOrchestrator", FakeSignalOrchestrator
    )
    return FakeSignalOrchestrator


def test_semantic_router_routes_code_query(fake_orchestrator_class):
    """A code query matches the default code/complexity rule."""
    router = SemanticRouter()
    decision = router.route({"content": "Write a Python function"})
    assert isinstance(decision, RoutingDecision)
    assert decision.route_to == "deepseek-coder-33b"
    assert decision.strategy == "cost"
    assert decision.matched_rule == "code_complex_to_large"
    assert decision.confidence > 0


def test_semantic_router_routes_vision_query(fake_orchestrator_class):
    """A vision query matches the vision model rule."""
    router = SemanticRouter()
    decision = router.route({"content": "describe this image", "images": []})
    assert decision.route_to == "gpt-4o"
    assert decision.matched_rule == "vision_to_vision_model"


def test_semantic_router_routes_simple_query(fake_orchestrator_class):
    """A simple text query falls through to the default rule."""
    router = SemanticRouter()
    decision = router.route({"content": "hello"})
    assert decision.route_to is None
    assert decision.strategy == "score"
    assert decision.matched_rule == "default_balanced"


def test_semantic_router_numa_scoring(fake_orchestrator_class):
    """When topology and gpu_index are supplied, NUMA score is attached."""
    topology = {
        "pairs": [{"gpu": "GPU 0 (NVIDIA H100)", "locality": "PIX"}],
        "numa_map": {"0": {"gpus": ["GPU 0"]}},
    }
    router = SemanticRouter(topology_report=topology)
    decision = router.route({"content": "code", "gpu_index": 0})
    assert decision.numa_score is not None
    assert decision.numa_score.pcie_locality == "PIX"
    assert decision.metadata["numa_policy"] == "prefer"


def test_semantic_router_batch_route(fake_orchestrator_class):
    """batch_route returns a decision per query."""
    router = SemanticRouter()
    queries = [
        {"content": "Write a Python function"},
        {"content": "hello"},
    ]
    decisions = router.batch_route(queries)
    assert len(decisions) == 2
    assert decisions[0].route_to == "deepseek-coder-33b"
    assert decisions[1].matched_rule == "default_balanced"
    assert decisions[1].metadata.get("batch") is True


def test_semantic_router_select_numa_optimal_endpoint(fake_orchestrator_class):
    """select_numa_optimal_endpoint picks the best-scored candidate."""
    topology = {
        "pairs": [
            {"gpu": "GPU 0 (NVIDIA H100)", "locality": "SYS"},
            {"gpu": "GPU 1 (NVIDIA H100)", "locality": "PIX"},
        ],
        "numa_map": {"0": {"gpus": ["GPU 0", "GPU 1"]}},
    }
    router = SemanticRouter(topology_report=topology)
    decision = router.route({"content": "code"})
    candidates = [
        {"endpoint_id": "slow", "gpu_index": 0, "avg_latency_ms": 10, "price_per_hour": 2.0},
        {"endpoint_id": "fast", "gpu_index": 1, "avg_latency_ms": 5, "price_per_hour": 2.0},
    ]
    best = router.select_numa_optimal_endpoint(decision, candidates)
    assert best["endpoint_id"] == "fast"


def test_semantic_router_select_numa_strict_no_match(fake_orchestrator_class):
    """Strict NUMA policy returns None when no PIX/PXB candidates exist."""
    topology = {
        "pairs": [{"gpu": "GPU 0 (NVIDIA H100)", "locality": "SYS"}],
        "numa_map": {"0": {"gpus": ["GPU 0"]}},
    }
    policy = RoutingPolicy(
        name="strict",
        rules=[RoutingRule(name="r1", condition="True", numa_policy="strict")],
        default_numa_policy="strict",
    )
    router = SemanticRouter(policy=policy, topology_report=topology)
    decision = router.route({"content": "code"})
    candidates = [{"endpoint_id": "only", "gpu_index": 0}]
    assert router.select_numa_optimal_endpoint(decision, candidates) is None


def test_semantic_router_no_topology_select_returns_first(fake_orchestrator_class):
    """Without topology, select_numa_optimal_endpoint falls back to first candidate."""
    router = SemanticRouter()
    decision = router.route({"content": "code"})
    candidates = [{"endpoint_id": "first"}, {"endpoint_id": "second"}]
    assert router.select_numa_optimal_endpoint(decision, candidates)["endpoint_id"] == "first"


def test_semantic_router_get_nccl_env(fake_orchestrator_class):
    """get_nccl_env_for_decision returns NCCL tuning variables."""
    router = SemanticRouter()
    decision = router.route({"content": "code"})
    env = router.get_nccl_env_for_decision(decision)
    assert isinstance(env, dict)
