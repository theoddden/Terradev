"""Tests for terradev_cli.core.preflight_validator data models and helpers."""

from terradev_cli.core.preflight_validator import (
    CheckResult,
    CheckStatus,
    PreflightReport,
    _check_adversarial_config,
    _run_on,
    _safe_float,
)


def test_check_result_passed():
    r = CheckResult("gpu", CheckStatus.PASS, "ok")
    assert r.passed is True

    r2 = CheckResult("gpu", CheckStatus.FAIL, "broken")
    assert r2.passed is False


def test_preflight_report_summary():
    report = PreflightReport(
        checks=[
            CheckResult("a", CheckStatus.PASS, "ok"),
            CheckResult("b", CheckStatus.WARN, "meh"),
            CheckResult("c", CheckStatus.FAIL, "bad"),
        ]
    )
    assert report.passed is False
    assert len(report.failures) == 1
    assert len(report.warnings) == 1
    summary = report.summary()
    assert summary["counts"]["pass"] == 1
    assert summary["counts"]["warn"] == 1
    assert summary["counts"]["fail"] == 1


def test_preflight_report_to_dict():
    report = PreflightReport(checks=[CheckResult("a", CheckStatus.PASS, "ok")])
    d = report.to_dict()
    assert d["passed"] is True
    assert len(d["checks"]) == 1


def test_safe_float_parsing():
    assert _safe_float("3.14") == 3.14
    assert _safe_float("N/A") == 0.0
    assert _safe_float("abc") == 0.0


def test_run_on_localhost():
    rc, out, err = _run_on("localhost", "echo hello")
    assert rc == 0
    assert out == "hello"


def test_run_on_unsafe_command():
    rc, out, err = _run_on("localhost", "echo a; echo b")
    assert rc == -1
    assert "Unsafe" in err


def test_check_adversarial_config():
    results = _check_adversarial_config(
        {
            "host": "node-1",
            "tensor_parallel_size": 8,
            "expected_gpus_per_node": 4,
        }
    )
    assert any(r.name == "adv_v1_tp_gpu" and r.status == CheckStatus.FAIL for r in results)

    results = _check_adversarial_config({})
    assert any(r.name == "adversarial_config" and r.status == CheckStatus.PASS for r in results)
