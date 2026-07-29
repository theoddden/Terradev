"""Tests for terradev_cli.core.mla_vram_estimator.

These protect the P0 MLA-aware VRAM feature — the claim that Terradev can
right-size GPU counts for DeepSeek-V3/R1 and Kimi K2 by applying MLA
compression ratios.
"""

import pytest

from terradev_cli.core.mla_vram_estimator import (
    AttentionType,
    MLA_VRAMEstimator,
    ModelArchitecture,
    VRAMBreakdown,
)


@pytest.fixture
def estimator():
    return MLA_VRAMEstimator()


def test_estimate_vram_for_deepseek_v3(estimator):
    """DeepSeek V3 uses MLA and should produce a substantially smaller KV cache."""
    result = estimator.estimate_vram(
        model_id="deepseek-v3",
        context_tokens=32768,
        batch_size=1,
        target_gpu_vram_gb=80.0,
        precision="bf16",
    )

    assert isinstance(result, VRAMBreakdown)
    assert result.architecture.attention_type == AttentionType.MULTI_HEAD_LATENT
    assert estimator.is_mla_model("deepseek-v3")
    assert result.kv_cache_gb < result.model_weights_gb
    assert result.gpu_count >= 1


def test_estimate_vram_for_llama70b_standard_mha(estimator):
    """Standard Llama-3 70B uses MHA and a larger KV cache."""
    result = estimator.estimate_vram(
        model_id="llama-3-70b",
        context_tokens=8192,
        batch_size=1,
        target_gpu_vram_gb=80.0,
        precision="bf16",
    )

    assert result.architecture.attention_type == AttentionType.STANDARD_MHA
    assert not estimator.is_mla_model("llama-3-70b")


def test_mla_compression_saves_vram(estimator):
    """The same param count with MLA should use less KV cache VRAM than MHA."""
    mha = ModelArchitecture(
        model_id="mha-test",
        total_params_b=7.0,
        num_layers=32,
        num_heads=32,
        head_dim=128,
        hidden_size=4096,
        intermediate_size=14336,
        attention_type=AttentionType.STANDARD_MHA,
        mla_compression_ratio=1.0,
        context_window=4096,
        max_context=4096,
    )

    mla = ModelArchitecture(
        model_id="mla-test",
        total_params_b=7.0,
        num_layers=32,
        num_heads=32,
        head_dim=128,
        hidden_size=4096,
        intermediate_size=14336,
        attention_type=AttentionType.MULTI_HEAD_LATENT,
        mla_compression_ratio=0.1,
        context_window=4096,
        max_context=4096,
    )

    mha_kv = estimator._calculate_kv_cache(mha, 4096, 1)
    mla_kv = estimator._calculate_kv_cache(mla, 4096, 1)

    assert mla_kv < mha_kv
    assert mla_kv == pytest.approx(mha_kv * 0.1, rel=0.05)


def test_auto_select_quantization_respects_accuracy_budget(estimator):
    """High accuracy budget should keep bf16; low budget should quantize."""
    high = estimator.auto_select_quantization(
        "deepseek-v3", "H200", accuracy_budget="high"
    )
    assert high["selected_quantization"] == "bf16"

    medium = estimator.auto_select_quantization(
        "deepseek-v3", "H200", accuracy_budget="medium"
    )
    assert medium["selected_quantization"] == "fp8"

    low = estimator.auto_select_quantization(
        "deepseek-v3", "H200", accuracy_budget="low"
    )
    assert low["selected_quantization"] in ("nvfp4", "mx-fp4")


def test_auto_select_quantization_fallback_on_unsupported_gpu(estimator):
    """A GPU with no special support falls back to bf16."""
    result = estimator.auto_select_quantization(
        "deepseek-v3", "Unknown-GPU", accuracy_budget="low"
    )
    assert result["selected_quantization"] == "bf16"


def test_unknown_model_raises_value_error(estimator):
    """Estimating an unregistered model fails fast with a clear message."""
    with pytest.raises(ValueError, match="Unknown model"):
        estimator.estimate_vram("not-a-real-model")


def test_register_and_estimate_custom_model(estimator):
    """Users can register a custom model and estimate its VRAM."""
    arch = ModelArchitecture(
        model_id="custom-mla",
        total_params_b=13.0,
        num_layers=40,
        num_heads=32,
        head_dim=128,
        hidden_size=5120,
        intermediate_size=13824,
        attention_type=AttentionType.MULTI_HEAD_LATENT,
        mla_compression_ratio=0.12,
        context_window=4096,
        max_context=16384,
    )
    estimator.register_model(arch)

    result = estimator.estimate_vram("custom-mla", context_tokens=4096)
    assert result.architecture.model_id == "custom-mla"
    assert estimator.is_mla_model("custom-mla")


def test_get_supported_models(estimator):
    """The model registry exposes known DeepSeek and Kimi models."""
    models = estimator.get_supported_models()
    assert "deepseek-v3" in models
    assert "kimi-k2" in models
    assert "llama-3-70b" in models


def test_compare_standard_vs_mla_shows_savings(estimator):
    """The comparison helper quantifies MLA savings for a client."""
    result = estimator.compare_standard_vs_mla("deepseek-v3", 32768)
    assert "standard_mha_estimate" in result
    assert "mla_estimate" in result
    assert result["mla_estimate"]["kv_cache_gb"] < result["standard_mha_estimate"]["kv_cache_gb"]
    assert "kv_cache_compression_ratio" in result["savings"]
    assert result["savings"]["kv_cache_compression_ratio"] > 1
