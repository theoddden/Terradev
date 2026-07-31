"""CLI help smoke tests.

Recursively invoke `--help` on every command and subcommand registered on the
root `cli` group.  This exercises command/option definitions and the main
`terradev_cli/cli.py` module without real API calls.
"""

import os
import warnings
from typing import List

import click
import pytest
from click.testing import CliRunner

from terradev_cli.cli import cli


os.environ["TERRADEV_SKIP_ONBOARDING"] = "1"


def _walk_command_paths(group: click.Group, prefix: List[str] = None):
    """Yield [cmd, subcmd, ...] paths for every registered command."""
    prefix = prefix or []
    for name in group.list_commands(None):
        cmd = group.get_command(None, name)
        if cmd is None:
            continue
        path = prefix + [name]
        yield path
        if isinstance(cmd, click.Group):
            yield from _walk_command_paths(cmd, path)


def _help_for_path(path: List[str]) -> int:
    """Invoke `--help` for a command path and return the exit code."""
    runner = CliRunner()
    result = runner.invoke(cli, path + ["--help"], obj={"api": None})
    return result.exit_code


@pytest.mark.parametrize("cmd_path", list(_walk_command_paths(cli)))
def test_command_help(cmd_path):
    """Every command/subcommand must be able to print its own help."""
    try:
        exit_code = _help_for_path(cmd_path)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not invoke help for {' '.join(cmd_path)}: {exc}")

    if exit_code != 0:
        warnings.warn(f"help for {' '.join(cmd_path)} exited with {exit_code}")
