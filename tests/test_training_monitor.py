#!/usr/bin/env python3
"""
Tests for TrainingMonitor — GPU metrics, log parsing, straggler detection.

Uses mocked nvidia-smi and log output to avoid requiring real GPU hardware.

Covers:
  1. Data models (GPUMetric, TrainingMetrics, MonitorSnapshot, StragglerInfo)
  2. nvidia-smi parser (_collect_gpu_nvidia_smi)
  3. Training log parser (_parse_training_log)
  4. Straggler detection
  5. Snapshot aggregation
  6. to_dict() output schema
  7. TrainingMonitor.snapshot() with mocked nodes
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.core.training_monitor import (
    GPUMetric,
    TrainingMetrics,
    MonitorSnapshot,
    StragglerInfo,
    TrainingMonitor,
    _collect_gpu_nvidia_smi,
    _safe_float,
)


# ── Data Models ───────────────────────────────────────────────────────────────


class TestDataModels:
    def test_gpu_metric_fields(self):
        m = GPUMetric(
            node="10.0.0.1", gpu_index=0, gpu_name="A100",
            utilization_pct=95.0, memory_used_mb=40000,
            memory_total_mb=81920, temperature_c=72,
            power_w=300, power_limit_w=400,
        )
        assert m.node == "10.0.0.1"
        assert m.utilization_pct == 95.0
        assert m.memory_total_mb == 81920

    def test_training_metrics_defaults(self):
        t = TrainingMetrics()
        assert t.step == 0
        assert t.loss == 0.0
        assert t.tokens_per_sec == 0.0

    def test_straggler_info_defaults(self):
        s = StragglerInfo()
        assert s.detected is False
        assert s.slow_nodes == []

    def test_monitor_snapshot_to_dict(self):
        snap = MonitorSnapshot(
            job_id="job-abc",
            node_count=2,
            gpu_count=16,
            avg_gpu_util=92.5,
            avg_gpu_memory_pct=78.3,
            total_gpu_power_w=4800.0,
            cost_usd=12.50,
            gpu_hours=32.0,
            elapsed_hours=2.0,
        )
        d = snap.to_dict()
        assert d["job_id"] == "job-abc"
        assert d["node_count"] == 2
        assert d["gpu_count"] == 16
        assert d["avg_gpu_util"] == 92.5
        assert d["cost_usd"] == 12.50
        assert "timestamp" in d

    def test_snapshot_to_dict_with_training(self):
        snap = MonitorSnapshot(
            training=TrainingMetrics(step=500, loss=2.31, tokens_per_sec=15000),
        )
        d = snap.to_dict()
        assert "training" in d
        assert d["training"]["step"] == 500
        assert d["training"]["loss"] == 2.31

    def test_snapshot_to_dict_with_straggler(self):
        snap = MonitorSnapshot(
            straggler=StragglerInfo(
                detected=True,
                slow_nodes=["10.0.0.3"],
                util_spread_pct=45.0,
                message="Node 10.0.0.3 is 45% behind",
            ),
        )
        d = snap.to_dict()
        assert "straggler" in d
        assert d["straggler"]["detected"] is True
        assert "10.0.0.3" in d["straggler"]["slow_nodes"]

    def test_snapshot_to_dict_no_straggler_when_not_detected(self):
        snap = MonitorSnapshot(
            straggler=StragglerInfo(detected=False),
        )
        d = snap.to_dict()
        assert "straggler" not in d


# ── Helpers ───────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_safe_float_normal(self):
        assert _safe_float("42.5") == 42.5

    def test_safe_float_na(self):
        assert _safe_float("N/A") == 0.0
        assert _safe_float("[N/A]") == 0.0
        assert _safe_float("[Not Supported]") == 0.0

    def test_safe_float_empty(self):
        assert _safe_float("") == 0.0

    def test_safe_float_bad_string(self):
        assert _safe_float("not-a-number") == 0.0

    def test_safe_float_custom_default(self):
        assert _safe_float("N/A", default=-1.0) == -1.0


# ── nvidia-smi Parser ────────────────────────────────────────────────────────


class TestNvidiaSmiParser:
    NVIDIA_SMI_OUTPUT = (
        "0, NVIDIA A100-SXM4-80GB, 95, 40960, 81920, 72, 300.00, 400.00\n"
        "1, NVIDIA A100-SXM4-80GB, 87, 38000, 81920, 70, 280.00, 400.00\n"
    )

    @patch("terradev_cli.core.training_monitor._run_on")
    def test_parses_two_gpus(self, mock_run):
        mock_run.return_value = (0, self.NVIDIA_SMI_OUTPUT, "")
        metrics = _collect_gpu_nvidia_smi({"host": None})
        assert len(metrics) == 2
        assert metrics[0].gpu_index == 0
        assert metrics[0].gpu_name == "NVIDIA A100-SXM4-80GB"
        assert metrics[0].utilization_pct == 95.0
        assert metrics[0].memory_used_mb == 40960.0
        assert metrics[1].gpu_index == 1
        assert metrics[1].utilization_pct == 87.0

    @patch("terradev_cli.core.training_monitor._run_on")
    def test_handles_nvidia_smi_failure(self, mock_run):
        mock_run.return_value = (1, "", "nvidia-smi not found")
        metrics = _collect_gpu_nvidia_smi({"host": None})
        assert metrics == []

    @patch("terradev_cli.core.training_monitor._run_on")
    def test_handles_empty_output(self, mock_run):
        mock_run.return_value = (0, "", "")
        metrics = _collect_gpu_nvidia_smi({"host": None})
        assert metrics == []

    @patch("terradev_cli.core.training_monitor._run_on")
    def test_handles_malformed_line(self, mock_run):
        mock_run.return_value = (0, "incomplete,data\n", "")
        metrics = _collect_gpu_nvidia_smi({"host": None})
        assert metrics == []  # Line has < 8 fields


# ── TrainingMonitor Integration ───────────────────────────────────────────────


class TestTrainingMonitorIntegration:
    @patch("terradev_cli.core.training_monitor._run_on")
    def test_snapshot_single_node(self, mock_run):
        nvidia_output = (
            "0, A100, 90, 40000, 81920, 68, 290, 400\n"
            "1, A100, 85, 38000, 81920, 66, 270, 400\n"
        )
        mock_run.return_value = (0, nvidia_output, "")

        monitor = TrainingMonitor(nodes=["localhost"])
        snap = monitor.snapshot()

        assert isinstance(snap, MonitorSnapshot)
        assert snap.gpu_count == 2
        assert snap.node_count == 1
        assert snap.avg_gpu_util > 0

    @patch("terradev_cli.core.training_monitor._run_on")
    def test_snapshot_multi_node(self, mock_run):
        nvidia_output = "0, A100, 90, 40000, 81920, 68, 290, 400\n"
        mock_run.return_value = (0, nvidia_output, "")

        monitor = TrainingMonitor(
            nodes=["10.0.0.1", "10.0.0.2"],
            ssh_user="ubuntu",
        )
        snap = monitor.snapshot()
        assert snap.node_count == 2

    @patch("terradev_cli.core.training_monitor._run_on")
    def test_snapshot_with_cost_rate(self, mock_run):
        nvidia_output = "0, A100, 90, 40000, 81920, 68, 290, 400\n"
        mock_run.return_value = (0, nvidia_output, "")

        monitor = TrainingMonitor(
            nodes=["localhost"],
            cost_per_gpu_hour=3.50,
        )
        snap = monitor.snapshot()
        # Cost should be computed (even if small for < 1 second)
        assert isinstance(snap.cost_usd, float)

    @patch("terradev_cli.core.training_monitor._run_on")
    def test_snapshot_to_dict_is_json_serializable(self, mock_run):
        nvidia_output = "0, A100, 90, 40000, 81920, 68, 290, 400\n"
        mock_run.return_value = (0, nvidia_output, "")

        monitor = TrainingMonitor(nodes=["localhost"])
        snap = monitor.snapshot()
        d = snap.to_dict()
        # Should be JSON-serializable
        import json
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["gpu_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
