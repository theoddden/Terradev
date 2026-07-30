"""Tests for terradev_cli.core.evaluation_orchestrator.

Evaluation orchestrator runs lightweight model and endpoint benchmarks.
"""

from pathlib import Path

import pytest

from terradev_cli.core.evaluation_orchestrator import (
    EvaluationConfig,
    EvaluationOrchestrator,
    EvaluationResult,
)


def test_evaluation_config_defaults():
    """EvaluationConfig provides sensible defaults."""
    config = EvaluationConfig(model_path="m.pt", dataset_path="d.json")
    assert config.metrics == ["accuracy", "latency"]
    assert config.sample_size == 1000


def test_evaluation_result_dataclass():
    """EvaluationResult stores structured evaluation output."""
    result = EvaluationResult(
        evaluation_id="eval-1",
        model_path="m.pt",
        endpoint_url=None,
        workload_type="general",
        metrics={"accuracy": 0.9},
        baseline_comparison=None,
        timestamp=None,
        duration_seconds=1.0,
    )
    assert result.metrics["accuracy"] == 0.9


def test_evaluate_model():
    """evaluate_model runs metrics and returns an EvaluationResult."""
    orch = EvaluationOrchestrator()
    config = EvaluationConfig(
        model_path="/models/m.pt",
        dataset_path="/data/d.json",
        metrics=["accuracy", "perplexity"],
    )

    result = orch.evaluate_model(config)
    assert isinstance(result, EvaluationResult)
    assert result.model_path == "/models/m.pt"
    assert "accuracy" in result.metrics
    assert 0.7 <= result.metrics["accuracy"] <= 0.95
    assert 10.0 <= result.metrics["perplexity"] <= 25.0
    assert result.metadata["dataset_size"] == 100


def test_evaluate_model_missing_paths():
    """evaluate_model raises ValueError when required paths are missing."""
    orch = EvaluationOrchestrator()
    with pytest.raises(ValueError, match="Model path and dataset path"):
        orch.evaluate_model(EvaluationConfig())


def test_compare_models():
    """compare_models returns winner and difference metrics."""
    orch = EvaluationOrchestrator()
    comparison = orch.compare_models(
        model_a_path="/models/a.pt",
        model_b_path="/models/b.pt",
        dataset_path="/data/d.json",
        metrics=["accuracy"],
    )
    assert "model_a" in comparison
    assert "model_b" in comparison
    assert "winner" in comparison
    assert comparison["winner"]["accuracy"] in ("model_a", "model_b")
    assert "differences" in comparison
    assert "absolute" in comparison["differences"]["accuracy"]


def test_save_result(tmp_path):
    """Evaluation results can be saved to disk as JSON."""
    orch = EvaluationOrchestrator()
    result = orch.evaluate_model(
        EvaluationConfig(
            model_path="/models/m.pt",
            dataset_path="/data/d.json",
            metrics=["accuracy"],
        )
    )

    out = tmp_path / "result.json"
    orch.save_result(result, str(out))
    assert out.exists()
    assert "evaluation_id" in out.read_text()
