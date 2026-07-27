"""Golden snapshots for stable CLI/MCP surfaces."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _load_snapshot(name):
    with open(SNAPSHOT_DIR / name) as f:
        return json.load(f)


class TestMCPListToolsSnapshot:
    """Snapshot the full tools/list output to catch schema drift."""

    def test_tool_list_matches_snapshot(self):
        from terradev_cli.mcp import server

        server._build_all_tools()
        names = sorted(t.name for t in server._ALL_TOOLS)

        snapshot = _load_snapshot("mcp_tools.json")
        assert snapshot["tool_count"] == len(names), (
            f"Tool count changed: expected {snapshot['tool_count']}, got {len(names)}. "
            "Update tests/snapshots/mcp_tools.json if this is intentional."
        )
        assert snapshot["tool_names"] == names, (
            "Tool list changed. Update tests/snapshots/mcp_tools.json if intentional."
        )


class TestCLIVersionSnapshot:
    """Snapshot terradev --version to catch release/version drift."""

    def test_version_matches_package_version(self):
        from terradev_cli import __version__

        env = os.environ.copy()
        env["TERRADEV_SKIP_ONBOARDING"] = "1"
        env["PYTHONPATH"] = str(Path(__file__).parent.parent)

        result = subprocess.run(
            [sys.executable, "-m", "terradev_cli", "--version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        assert result.returncode == 0
        assert __version__ in result.stdout
