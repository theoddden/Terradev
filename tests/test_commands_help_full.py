"""Exhaustive --help regression test for the entire CLI command tree.

This file dynamically discovers every command and subcommand registered on the
Terradev CLI and asserts that each one can render its help without crashing.
It is intended to be a global smoke net that catches broken imports, option
errors, and registration issues anywhere in the command corpus.
"""

import click
import pytest

# Ensure top-level commands (up, rollback, manifests, hf-spaces,
# karpenter) are registered on the shared `cli` group before we collect paths.
import terradev_cli.cli  # noqa: F401
from terradev_cli.commands import cli


def _collect_commands(group, ctx, path=None):
    """Recursively return every command path in the CLI tree."""
    paths = []
    path = path or []
    for name in group.list_commands(ctx):
        cmd = group.get_command(ctx, name)
        if cmd is None:
            continue
        new_path = path + [name]
        if isinstance(cmd, click.Group):
            sub_ctx = click.Context(cmd, parent=ctx, info_name=name)
            paths.extend(_collect_commands(cmd, sub_ctx, new_path))
        else:
            paths.append(new_path)
    return paths


# Build the full command list once at collection time.  This will fail loudly
# if any command module cannot be imported, which is exactly what we want.
with click.Context(cli, info_name="terradev") as _ctx:
    ALL_COMMANDS = _collect_commands(cli, _ctx, [])


@pytest.mark.parametrize("cmd_path", ALL_COMMANDS, ids=lambda p: " ".join(p))
def test_command_help_renders(runner, mock_api, cmd_path):
    """Each discovered command must render --help successfully."""
    result = runner.invoke(cli, cmd_path + ["--help"], obj={"api": mock_api})
    assert result.exit_code == 0, f"Help failed for {' '.join(cmd_path)}: {result.output}"
    assert "Usage:" in result.output, f"No usage in help for {' '.join(cmd_path)}"
    # The command name should appear in the help header.
    assert cmd_path[-1] in result.output or cmd_path[-1].replace("-", "_") in result.output


def test_all_commands_discovered():
    """Sanity: we discovered a non-trivial number of commands."""
    assert len(ALL_COMMANDS) >= 50, f"Only discovered {len(ALL_COMMANDS)} commands"
