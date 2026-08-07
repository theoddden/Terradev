#!/usr/bin/env python3
"""Regenerate Terradev CLI reference docs from live --help output."""
import os
import re
import subprocess
from pathlib import Path

import click
from click.testing import CliRunner

# Import the CLI inside the Terradev workspace
os.environ["TERRADEV_SKIP_ONBOARDING"] = "1"
os.environ["TERRADEV_OUTPUT"] = "human"
from terradev_cli.commands import cli

cli.name = "terradev"

import tomllib

TERRADEV_ROOT = Path(__file__).resolve().parents[1]
TERRADEV_CLOUD_ROOT = Path("/Users/theowolfenden/CascadeProjects/terradev-cloud")

with open(TERRADEV_ROOT / "pyproject.toml", "rb") as f:
    _pyproject = tomllib.load(f)
VERSION = f"v{_pyproject['project']['version']}"


def _invoke_help(path):
    runner = CliRunner()
    result = runner.invoke(cli, path + ["--help"], obj={"api": None})
    if result.exit_code != 0:
        return None
    return result.output


def _collect(group, ctx, path=None):
    items = []
    path = path or []
    # Capture the root `terradev` group help; nested groups are added by the parent loop.
    if not path:
        help_text = _invoke_help(path)
        if help_text is not None:
            items.append((path, help_text))
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
        "",
    ]

    for path, help_text in all_items:
        depth = len(path) + 2  # title #, root ##, top-level ###
        lines.append(_heading(depth, f"`{_cmd_line(path)}`"))
        lines.append("")
        lines.append("```text")
        lines.append(help_text.rstrip())
        lines.append("```")
        lines.append("")

    (TERRADEV_CLOUD_ROOT / "web/docs/CLI.md").write_text("\n".join(lines))
    print(f"Wrote CLI.md with {len(all_items)} commands")


def _first_desc(help_text: str) -> str:
    """Return the first sentence/line of a command's docstring from --help."""
    for line in help_text.strip().splitlines()[1:]:
        s = line.strip()
        if s and not s.endswith(":"):
            return s
    return ""


def _code_block(help_text: str) -> str:
    """Build a concise bash code block for the quick reference."""
    lines = help_text.strip().splitlines()
    usage = lines[0].lstrip("Usage: ").strip()
    start = None
    for i, l in enumerate(lines):
        if l.strip() in ("Commands:", "Options:"):
            start = i
            break
    if start is None:
        return f"```bash\n{usage}\n```\n"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip() and not lines[j].startswith("  "):
            if lines[j].strip().endswith(":"):
                end = j
                break
    body = "\n".join([usage] + lines[start:end])
    return f"```bash\n{body}\n```\n"


def _is_command_heading(line):
    m = re.match(r'^(#{2,}) \*\*([^*]+?)\*\*(?: - (.*))?$', line)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def generate_complete_ref():
    with click.Context(cli, info_name="terradev") as ctx:
        all_items = _collect(cli, ctx)

    help_by_path = {" ".join(path): ht for path, ht in all_items}
    path_by_path = {" ".join(path): path for path, _ in all_items}
    current_names = set(help_by_path)

    # Load the v6.0.2 quick-reference as the formatting template.
    res = subprocess.run(
        ["git", "show", "v6.0.2:terradev_cli/COMPLETE_COMMAND_REFERENCE.md"],
        cwd=TERRADEV_ROOT,
        capture_output=True,
        text=True,
    )
    base = res.stdout if res.returncode == 0 else ""
    if not base:
        raise RuntimeError("Could not load v6.0.2 COMPLETE_COMMAND_REFERENCE.md from git")

    # Update version string occurrences.
    base = re.sub(r"v6\.0\.\d+", VERSION, base)

    old_names = set(re.findall(r'^#{2,} \*\*([^*]+?)\*\*(?: - .*)?$', base, re.MULTILINE))

    out = []
    i = 0
    state = "text"
    current_name = None
    base_lines = base.splitlines()
    while i < len(base_lines):
        line = base_lines[i]
        if state == "text":
            parsed = _is_command_heading(line)
            if parsed:
                level, name, desc = parsed
                if name in current_names:
                    current_name = name
                    if not desc:
                        desc = _first_desc(help_by_path[name])
                    out.append(f"{level} **{name}** - {desc}" if desc else f"{level} **{name}**")
                    out.append("")
                    state = "heading"
                    i += 1
                    continue
            out.append(line)
            i += 1
        elif state == "heading":
            if line.startswith("```bash"):
                state = "in_block"
                i += 1
            else:
                out.append(line)
                i += 1
        elif state == "in_block":
            if line.startswith("```"):
                ht = help_by_path[current_name]
                out.append(_code_block(ht))
                out.append("")
                state = "text"
                current_name = None
                i += 1
            else:
                i += 1

    # Determine genuinely new/updated command paths and append them.
    new_section_names = [
        n for n in current_names
        if n not in old_names and (n.startswith("agent") or n.startswith("ml vllm lora"))
    ]
    if new_section_names:
        new_section = ["", "---", "", f"##  **New & Updated in {VERSION}**", ""]
        for name in sorted(new_section_names):
            ht = help_by_path[name]
            desc = _first_desc(ht)
            new_section.append(f"### **{name}** - {desc}")
            new_section.append("")
            new_section.append(_code_block(ht))
        # Insert before the "Complete Command Summary" if present.
        summary_idx = None
        for idx, line in enumerate(out):
            if "Complete Command Summary" in line:
                summary_idx = idx
                break
        if summary_idx is not None:
            out = out[:summary_idx] + new_section + out[summary_idx:]
        else:
            out.extend(new_section)

    (TERRADEV_ROOT / "terradev_cli/COMPLETE_COMMAND_REFERENCE.md").write_text("\n".join(out))
    print(f"Wrote COMPLETE_COMMAND_REFERENCE.md from v6.0.2 template with {len(old_names & current_names)} updated and {len(new_section_names)} new commands")


if __name__ == "__main__":
    generate_cli_md()
    generate_complete_ref()
