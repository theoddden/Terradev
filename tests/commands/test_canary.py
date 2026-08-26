"""Tests for the canary reporting CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from terradev_cli.commands import cli


def _write_results(tmp_path, records):
    results = tmp_path / "canary-results.jsonl"
    with open(results, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(results)


class TestCanaryReport:
    def test_canary_help(self, runner):
        result = runner.invoke(cli, ["canary", "--help"])
        assert result.exit_code == 0
        assert "canary" in result.output

    def test_canary_report_help(self, runner):
        result = runner.invoke(cli, ["canary", "report", "--help"])
        assert result.exit_code == 0

    def test_canary_tail_help(self, runner):
        result = runner.invoke(cli, ["canary", "tail", "--help"])
        assert result.exit_code == 0

    def test_report_no_records(self, runner):
        result = runner.invoke(cli, ["canary", "report"])
        assert result.exit_code == 0
        assert "No canary records found" in result.output

    def test_report_text_summary(self, runner, tmp_path):
        recs = [
            {"provider": "runpod", "region": "us-east-1", "gpu_type": "A100", "status": "passed", "duration_ms": 100, "timestamp": "2026-08-01T00:00:00"},
            {"provider": "runpod", "region": "us-east-1", "gpu_type": "A100", "status": "passed", "duration_ms": 200, "timestamp": "2026-08-01T00:01:00"},
            {"provider": "vastai", "region": "us-west-1", "gpu_type": "RTX4090", "status": "failed", "duration_ms": 50, "timestamp": "2026-08-01T00:02:00"},
        ]
        path = _write_results(tmp_path, recs)
        result = runner.invoke(cli, ["canary", "report", "--file", path])
        assert result.exit_code == 0
        assert "Total runs:     3" in result.output
        assert "Passed:         2" in result.output
        assert "Failed:         1" in result.output
        assert "runpod" in result.output
        assert "vastai" in result.output

    def test_report_json_output(self, runner, tmp_path):
        recs = [
            {"provider": "runpod", "status": "passed", "duration_ms": 100},
        ]
        path = _write_results(tmp_path, recs)
        result = runner.invoke(cli, ["canary", "report", "--file", path, "--output", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 1
        assert data["passed"] == 1

    def test_report_filter_by_provider(self, runner, tmp_path):
        recs = [
            {"provider": "runpod", "status": "passed"},
            {"provider": "vastai", "status": "failed"},
        ]
        path = _write_results(tmp_path, recs)
        result = runner.invoke(cli, ["canary", "report", "--file", path, "--provider", "runpod"])
        assert result.exit_code == 0
        assert "Total runs:     1" in result.output

    def test_report_filter_by_gpu(self, runner, tmp_path):
        recs = [
            {"provider": "runpod", "gpu_type": "A100", "status": "passed"},
            {"provider": "runpod", "gpu_type": "H100", "status": "passed"},
        ]
        path = _write_results(tmp_path, recs)
        result = runner.invoke(cli, ["canary", "report", "--file", path, "--gpu", "A100"])
        assert result.exit_code == 0
        assert "Total runs:     1" in result.output

    def test_report_ignores_malformed_lines(self, runner, tmp_path):
        path = tmp_path / "canary-results.jsonl"
        with open(path, "w") as f:
            f.write('{"provider": "runpod", "status": "passed"}\n')
            f.write("not-json\n")
            f.write('{"provider": "runpod", "status": "passed"}\n')
        result = runner.invoke(cli, ["canary", "report", "--file", str(path)])
        assert result.exit_code == 0
        assert "Total runs:     2" in result.output


class TestCanaryTail:
    def test_tail_recent_records(self, runner, tmp_path):
        recs = [{"i": i, "status": "passed"} for i in range(5)]
        path = _write_results(tmp_path, recs)
        result = runner.invoke(cli, ["canary", "tail", "--file", path, "--limit", "2"])
        assert result.exit_code == 0
        data = result.output.strip().split("\n")
        # Each record is pretty-printed across multiple lines; the last object ends with a closing brace
        assert len(data) >= 2

    def test_tail_no_records(self, runner, tmp_path):
        path = _write_results(tmp_path, [])
        result = runner.invoke(cli, ["canary", "tail", "--file", path])
        assert result.exit_code == 0
        assert "No canary records found" in result.output


class TestCanaryDrift:
    def test_drift_help(self, runner):
        result = runner.invoke(cli, ["canary", "drift", "--help"])
        assert result.exit_code == 0
        assert "--all" in result.output
        assert "--provider" in result.output

    def _contract(self, expected=None, method="POST"):
        return f"""
provider: test
base_url: https://api.example.com/graphql
auth_type: Bearer
auth_header: Authorization
endpoints:
  - name: list_gpus
    method: {method}
    required_fields: [query]
    expected_response_fields: [data, gpuTypes]
    smoke_test_query: "query {{ gpuTypes {{ id }} }}"
"""

    def _make_fake_requests(self, body, status=200):
        def post(url, **kwargs):
            r = MagicMock()
            r.status_code = status
            r.ok = (status == 200)
            r.json.return_value = body
            return r

        def get(url, **kwargs):
            return post(url, **kwargs)

        return post, get

    def test_drift_all_healthy_json(self, runner, tmp_path, monkeypatch):
        contracts = tmp_path / "contracts"
        contracts.mkdir()
        contract = contracts / "test.yaml"
        contract.write_text(self._contract())

        post, get = self._make_fake_requests({"data": {"gpuTypes": [{"id": "A100"}]}})
        monkeypatch.setattr("terradev_cli.drift_monitor.agent.requests.post", post)
        monkeypatch.setattr("terradev_cli.drift_monitor.agent.requests.get", get)
        monkeypatch.setenv("TERRADEV_TEST_KEY", "fake-key")

        result = runner.invoke(
            cli,
            [
                "--format",
                "json",
                "canary",
                "drift",
                "--all",
                "--contracts-dir",
                str(contracts),
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_providers"] == 1
        assert data["healthy"] == 1
        assert data["drifted"] == 0

    def test_drift_missing_field(self, runner, tmp_path, monkeypatch):
        contracts = tmp_path / "contracts"
        contracts.mkdir()
        contract = contracts / "test.yaml"
        contract.write_text(self._contract())

        post, get = self._make_fake_requests({"data": {}})
        monkeypatch.setattr("terradev_cli.drift_monitor.agent.requests.post", post)
        monkeypatch.setattr("terradev_cli.drift_monitor.agent.requests.get", get)
        monkeypatch.setenv("TERRADEV_TEST_KEY", "fake-key")

        result = runner.invoke(
            cli,
            [
                "--format",
                "json",
                "canary",
                "drift",
                "--all",
                "--contracts-dir",
                str(contracts),
            ],
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["drifted"] == 1
        assert data["drift_providers"] == ["test"]

    def test_drift_missing_credentials(self, runner, tmp_path):
        contracts = tmp_path / "contracts"
        contracts.mkdir()
        contract = contracts / "test.yaml"
        contract.write_text(self._contract())

        result = runner.invoke(
            cli,
            ["canary", "drift", "--all", "--contracts-dir", str(contracts)],
            env={"TERRADEV_SKIP_ONBOARDING": "1"},
        )
        assert result.exit_code == 0
        assert "skipped" in result.output

    def _no_auth_contract(self, expected=None):
        return f"""
provider: public
base_url: https://api.example.com
auth_required: false
endpoints:
  - name: list_gpus
    path: /gpus
    method: GET
    expected_response_fields: [data, gpuTypes]
"""

    def test_drift_no_auth(self, runner, tmp_path, monkeypatch):
        contracts = tmp_path / "contracts"
        contracts.mkdir()
        contract = contracts / "public.yaml"
        contract.write_text(self._no_auth_contract())

        post, get = self._make_fake_requests({"data": {"gpuTypes": [{"id": "A100"}]}})
        monkeypatch.setattr("terradev_cli.drift_monitor.agent.requests.post", post)
        monkeypatch.setattr("terradev_cli.drift_monitor.agent.requests.get", get)

        result = runner.invoke(
            cli,
            [
                "--format",
                "json",
                "canary",
                "drift",
                "--all",
                "--contracts-dir",
                str(contracts),
            ],
            env={"TERRADEV_SKIP_ONBOARDING": "1"},
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["healthy"] == 1
        assert data["skipped"] == 0
        assert data["providers"][0]["status"] == "healthy"



class TestCanaryMlDrift:
    def test_canary_ml_drift_help(self, runner):
        result = runner.invoke(cli, ["canary", "ml-drift", "--help"])
        assert result.exit_code == 0
        assert "ml-drift" in result.output
