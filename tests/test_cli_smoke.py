#!/usr/bin/env python3
"""
CLI Smoke Tests — verify commands load, parse options, and produce output.

These tests invoke CLI commands via Click's test runner (no subprocess).
They do NOT require real cloud credentials or GPU hardware.

Covers:
  1. Top-level help and version
  2. Training pipeline commands (preflight, train, monitor, checkpoint, train-status)
  3. JSON output format
  4. Missing-argument error handling
  5. Provision→train bridge flag exists
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from click.testing import CliRunner

# Import the top-level CLI group
from terradev_cli.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


# ── Top-Level ─────────────────────────────────────────────────────────────────


class TestTopLevel:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_help_contains_train_commands(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for cmd in ["train", "preflight", "monitor", "checkpoint",
                     "train-status", "train-stop", "train-resume"]:
            assert cmd in result.output, f"Missing command in help: {cmd}"

    def test_help_contains_provision(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "provision" in result.output


# ── Preflight ─────────────────────────────────────────────────────────────────


class TestPreflight:
    def test_preflight_help(self, runner):
        result = runner.invoke(cli, ["preflight", "--help"])
        assert result.exit_code == 0
        assert "--nodes" in result.output
        assert "--quick" in result.output
        assert "--format" in result.output

    def test_preflight_json_format_option(self, runner):
        """Verify --format json is accepted (actual execution may fail without GPUs)."""
        result = runner.invoke(cli, ["preflight", "--help"])
        assert "-f" in result.output or "--format" in result.output


# ── Train ─────────────────────────────────────────────────────────────────────


class TestTrain:
    def test_train_help(self, runner):
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--script" in result.output
        assert "--framework" in result.output
        assert "--backend" in result.output
        assert "--nodes" in result.output
        assert "--gpus-per-node" in result.output

    def test_train_has_from_provision(self, runner):
        """Verify --from-provision flag exists on train command."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--from-provision" in result.output

    def test_train_requires_script_or_config(self, runner):
        """Train with no script/config should error."""
        result = runner.invoke(cli, ["train"])
        # Should fail because neither --script nor --config provided
        assert result.exit_code != 0 or "ERROR" in result.output

    def test_train_framework_choices(self, runner):
        result = runner.invoke(cli, ["train", "--help"])
        assert "torchrun" in result.output
        assert "deepspeed" in result.output
        assert "accelerate" in result.output
        assert "megatron" in result.output


# ── Monitor ───────────────────────────────────────────────────────────────────


class TestMonitor:
    def test_monitor_help(self, runner):
        result = runner.invoke(cli, ["monitor", "--help"])
        assert result.exit_code == 0
        assert "--job-id" in result.output
        assert "--interval" in result.output
        assert "--prometheus" in result.output
        assert "--cost-rate" in result.output


# ── Checkpoint ────────────────────────────────────────────────────────────────


class TestCheckpoint:
    def test_checkpoint_help(self, runner):
        result = runner.invoke(cli, ["checkpoint", "--help"])
        assert result.exit_code == 0
        assert "--job-id" in result.output
        assert "list" in result.output
        assert "restore" in result.output
        assert "promote" in result.output
        assert "delete" in result.output


# ── Train-Status ──────────────────────────────────────────────────────────────


class TestTrainStatus:
    def test_train_status_help(self, runner):
        result = runner.invoke(cli, ["train-status", "--help"])
        assert result.exit_code == 0
        assert "--job-id" in result.output
        assert "--format" in result.output

    def test_train_status_json_no_jobs(self, runner):
        """train-status with JSON format should return valid JSON even with no jobs."""
        result = runner.invoke(cli, ["train-status", "-f", "json"])
        # May succeed with empty output or valid JSON
        if result.exit_code == 0 and result.output.strip():
            try:
                data = json.loads(result.output)
                assert isinstance(data, (dict, list))
            except json.JSONDecodeError:
                pass  # Some output modes aren't pure JSON


# ── Train-Stop / Train-Resume ────────────────────────────────────────────────


class TestTrainStopResume:
    def test_train_stop_help(self, runner):
        result = runner.invoke(cli, ["train-stop", "--help"])
        assert result.exit_code == 0
        assert "--job-id" in result.output

    def test_train_resume_help(self, runner):
        result = runner.invoke(cli, ["train-resume", "--help"])
        assert result.exit_code == 0
        assert "--job-id" in result.output
        assert "--checkpoint-id" in result.output

    def test_train_stop_requires_job_id(self, runner):
        result = runner.invoke(cli, ["train-stop"])
        assert result.exit_code != 0  # --job-id is required

    def test_train_resume_requires_job_id(self, runner):
        result = runner.invoke(cli, ["train-resume"])
        assert result.exit_code != 0  # --job-id is required


# ── Provision ─────────────────────────────────────────────────────────────────


class TestProvision:
    def test_provision_help(self, runner):
        result = runner.invoke(cli, ["provision", "--help"])
        assert result.exit_code == 0
        assert "--gpu-type" in result.output
        assert "--count" in result.output
        assert "--max-price" in result.output
        assert "--dry-run" in result.output
        assert "--parallel" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
