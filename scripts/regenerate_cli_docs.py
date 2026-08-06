#!/usr/bin/env python3
"""Regenerate Terradev CLI reference docs from live --help output."""
import os
from pathlib import Path

import click
from click.testing import CliRunner

# Import the CLI inside the Terradev workspace
os.environ["TERRADEV_SKIP_ONBOARDING"] = "1"
os.environ["TERRADEV_OUTPUT"] = "human"
from terradev_cli.commands import cli

cli.name = "terradev"

TERRADEV_ROOT = Path(__file__).resolve().parents[1]
TERRADEV_CLOUD_ROOT = Path("/Users/theowolfenden/CascadeProjects/terradev-cloud")
VERSION = "v6.0.3"


def _invoke_help(path):
    runner = CliRunner()
    result = runner.invoke(cli, path + ["--help"], obj={"api": None})
    if result.exit_code != 0:
        return None
    return result.output


def _collect(group, ctx, path=None):
    items = []
    path = path or []
    for name in group.list_commands(ctx):
        cmd = group.get_command(ctx, name)
        if cmd is None:
            continue
        new_path = path + [name]
        help_text = _invoke_help(new_path)
        if help_text is None:
            continue
        items.append((new_path, help_text))
        if isinstance(cmd, click.Group):
            sub_ctx = click.Context(cmd, parent=ctx, info_name=name)
            items.extend(_collect(cmd, sub_ctx, new_path))
    return items


def _heading(level, title):
    return f"{'#' * level} {title}"


def _cmd_line(path):
    return " ".join(["terradev"] + path)


def generate_cli_md():
    with click.Context(cli, info_name="terradev") as ctx:
        all_items = _collect(cli, ctx)

    lines = [
        f"# Terradev CLI Command Reference ({VERSION})",
        "",
        "Generated from `terradev --help` and subcommand `--help` output.",
        "",
    ]

    for path, help_text in all_items:
        depth = len(path) + 1  # root at depth 2
        lines.append(_heading(depth, f"`{_cmd_line(path)}`"))
        lines.append("")
        lines.append("```text")
        lines.append(help_text.rstrip())
        lines.append("```")
        lines.append("")

    (TERRADEV_CLOUD_ROOT / "web/docs/CLI.md").write_text("\n".join(lines))
    print(f"Wrote CLI.md with {len(all_items)} commands")


def generate_complete_ref():
    with click.Context(cli, info_name="terradev") as ctx:
        all_items = _collect(cli, ctx)

    lines = [
        "# Complete Terradev CLI Command Reference",
        "",
        f"**All commands and subcommands for Terradev CLI {VERSION}**",
        "",
        "---",
        "",
    ]

    # Bucket commands by top-level group name
    top_buckets = {}
    leaves = []
    for path, help_text in all_items:
        if len(path) == 1:
            top_buckets.setdefault(path[0], []).append((path, help_text))
        else:
            leaves.append((path, help_text))

    lines.append("## Main Commands")
    lines.append("")
    for path, help_text in sorted(top_buckets.get("provision", []) + top_buckets.get("status", [])):
        cmd = _cmd_line(path)
        first_line = help_text.strip().splitlines()[0].lstrip("Usage: ").strip()
        lines.append(_heading(3, f"**{path[-1]}**"))
        lines.append("")
        lines.append("```bash")
        lines.append(first_line)
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")
    for path, help_text in sorted(leaves, key=lambda x: _cmd_line(x[0])):
        cmd = _cmd_line(path)
        first_line = help_text.strip().splitlines()[0].lstrip("Usage: ").strip()
        lines.append(_heading(3, f"**{path[-1]}** - `{cmd}`"))
        lines.append("")
        lines.append("```bash")
        lines.append(first_line)
        lines.append("```")
        lines.append("")

    (TERRADEV_ROOT / "terradev_cli/COMPLETE_COMMAND_REFERENCE.md").write_text("\n".join(lines))
    print(f"Wrote COMPLETE_COMMAND_REFERENCE.md with {len(all_items)} commands")


if __name__ == "__main__":
    generate_cli_md()
    generate_complete_ref()
