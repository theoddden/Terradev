"""Tests for terradev_cli.core.hf_smart_templates."""

import pytest

from terradev_cli.core.hf_smart_templates import HardwareOptimizer, SmartTemplateGenerator


@pytest.fixture
def generator():
    return SmartTemplateGenerator()


@pytest.fixture
def optimizer():
    return HardwareOptimizer()


def test_analyze_model_known(generator):
    spec = generator.analyze_model("meta-llama/Llama-3-8B-Instruct")
    assert spec is not None
    assert spec.model_type == "llm"
    assert spec.recommended_hardware == "a10g-large"


def test_analyze_model_estimated(generator):
    spec = generator.analyze_model("org/Some-Llama-70b-hf")
    assert spec is not None
    assert spec.parameters == "70B"
    assert spec.recommended_hardware == "a100-80gb"


def test_analyze_model_unknown(generator):
    spec = generator.analyze_model("unknown/weird-model")
    assert spec is None


def test_optimize_hardware(generator):
    spec = generator.analyze_model("meta-llama/Llama-3-8B-Instruct")
    tiers = generator.optimize_hardware(spec)
    assert tiers
    assert all(t.memory_gb >= spec.min_memory_gb for t in tiers)
    assert tiers == sorted(tiers, key=lambda t: t.performance_score, reverse=True)


def test_optimize_hardware_budget(generator):
    spec = generator.analyze_model("meta-llama/Llama-3-70B-Instruct")
    tiers = generator.optimize_hardware(spec, budget_constraint=2.0)
    assert all(t.hourly_cost <= 2.0 for t in tiers)


def test_generate_cost_breakdown(generator):
    spec = generator.analyze_model("sentence-transformers/all-MiniLM-L6-v2")
    tier = generator.optimize_hardware(spec)[0]
    breakdown = generator.generate_cost_breakdown(spec, tier)
    assert breakdown["model_id"] == spec.model_id
    assert "monthly_24_7" in breakdown["cost_breakdown"]


def test_generate_smart_template(generator):
    template = generator.generate_smart_template(
        "meta-llama/Llama-3-8B-Instruct", space_name="my-space"
    )
    assert "error" not in template
    assert template["name"] == "my-space"
    assert template["sdk"] == "gradio"
    assert "cost_breakdown" in template
    assert "alternative_hardware" in template


def test_generate_chat_template(generator):
    chat = generator.generate_chat_template(
        "meta-llama/Llama-3-8B-Instruct", "chat-space"
    )
    assert "meta-llama/Llama-3-8B-Instruct" in chat
    assert "import gradio" in chat


def test_generate_chat_template_unsuitable(generator):
    chat = generator.generate_chat_template(
        "sentence-transformers/all-MiniLM-L6-v2", "emb-space"
    )
    assert "Error" in chat


def test_hardware_optimizer_recommendation(optimizer):
    rec = optimizer.get_hardware_recommendation("meta-llama/Llama-3-8B-Instruct")
    assert "error" not in rec
    # optimize_hardware sorts by performance_score descending, so a100-80gb wins
    assert rec["recommended_hardware"] == "a100-80gb"
    assert "cost_breakdown" in rec
    assert "alternative_options" in rec


def test_hardware_optimizer_budget(optimizer):
    rec = optimizer.get_hardware_recommendation(
        "meta-llama/Llama-3-8B-Instruct", budget_constraint=2.0
    )
    assert rec["recommended_hardware"] == "a10g-xlarge"


def test_hardware_optimizer_unknown(optimizer):
    rec = optimizer.get_hardware_recommendation("unknown/model")
    assert "error" in rec


def test_hardware_optimizer_compare(optimizer):
    result = optimizer.compare_hardware_options("meta-llama/Llama-3-8B-Instruct")
    assert "error" not in result
    assert len(result["hardware_comparison"]) == 7
    assert any(tier["hardware"] == "a10g-large" and tier["suitable"] for tier in result["hardware_comparison"])
