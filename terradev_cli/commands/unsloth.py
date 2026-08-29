#!/usr/bin/env python3
"""Unsloth integration for the Terradev CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click


SUPPORTED_AGENTS = ["claude", "codex", "hermes", "openclaw", "opencode"]


@click.group()
def unsloth():
    """Unsloth optimized local LLM training, serving, and coding agents.

    Unsloth slashes VRAM usage by up to 70% and doubles training speeds
    using optimized Triton kernels. Use `run` to serve a local model and
    `start` to attach a coding agent to it.
    """
    pass


def _find_unsloth() -> str:
    """Return the unsloth executable path or raise a clear error."""
    exe = shutil.which("unsloth")
    if not exe:
        click.echo(
            "ERROR: unsloth CLI not found. Install with:\n"
            "  pip install unsloth\n"
            "or see https://unsloth.ai/docs",
            err=True,
        )
        raise SystemExit(1)
    return exe


def _build_run_command(
    unsloth: str,
    model: str,
    host: str,
    port: int,
    enable_tools: bool,
    no_cloudflare: bool,
    gguf_variant: Optional[str],
    context_length: Optional[int],
    no_load_in_4bit: bool,
    tensor_parallel: Optional[int],
) -> List[str]:
    """Build the `unsloth run` command."""
    cmd = [unsloth, "run", "-H", host, "-p", str(port), "--model", model]
    cmd.append("--enable-tools" if enable_tools else "--disable-tools")
    if no_cloudflare:
        cmd.append("--no-cloudflare")
    if gguf_variant:
        cmd.extend(["--gguf-variant", gguf_variant])
    if context_length:
        cmd.extend(["--context-length", str(context_length)])
    if no_load_in_4bit:
        cmd.append("--no-load-in-4bit")
    if tensor_parallel:
        cmd.extend(["--tensor-parallel", str(tensor_parallel)])
    return cmd


def _health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


@unsloth.command("start")
@click.argument("agent", type=click.Choice(SUPPORTED_AGENTS))
@click.option("--model", "-m", help="Model to load and serve (e.g. unsloth/Llama-3.1-8B)")
@click.option("--host", "-H", default="127.0.0.1", help="Server host")
@click.option("--port", "-p", default=8888, type=click.IntRange(1, 65535), help="Server port")
@click.option("--enable-tools/--disable-tools", default=True, help="Enable/disable tool use")
@click.option("--no-cloudflare", is_flag=True, help="Do not use Cloudflare tunnel")
@click.option("--gguf-variant", help="Preferred GGUF quantization variant")
@click.option("--context-length", type=click.IntRange(1, 10000000), help="Maximum context length")
@click.option("--no-load-in-4bit", is_flag=True, help="Disable 4-bit loading")
@click.option("--tensor-parallel", type=click.IntRange(1, 128), help="Tensor parallel size")
@click.option("--project", "-C", default=".", help="Project directory")
@click.option("--background", is_flag=True, help="Run in background instead of foreground")
def unsloth_start(
    agent,
    model,
    host,
    port,
    enable_tools,
    no_cloudflare,
    gguf_variant,
    context_length,
    no_load_in_4bit,
    tensor_parallel,
    project,
    background,
):
    """Start a coding agent backed by Unsloth's local model server.

    Supported agents: claude, codex, hermes, openclaw, opencode.

    Examples:
      terradev train unsloth start claude --model unsloth/Llama-3.1-8B
      terradev train unsloth start codex --model unsloth/Qwen3.6-7B --port 9999
    """
    exe = _find_unsloth()
    cmd = [exe, "start", agent]

    if model:
        cmd.extend(["--model", model])
    if host:
        cmd.extend(["-H", host])
    if port != 8888:
        cmd.extend(["-p", str(port)])
    if enable_tools:
        cmd.append("--enable-tools")
    else:
        cmd.append("--disable-tools")
    if no_cloudflare:
        cmd.append("--no-cloudflare")
    if gguf_variant:
        cmd.extend(["--gguf-variant", gguf_variant])
    if context_length:
        cmd.extend(["--context-length", str(context_length)])
    if no_load_in_4bit:
        cmd.append("--no-load-in-4bit")
    if tensor_parallel:
        cmd.extend(["--tensor-parallel", str(tensor_parallel)])

    cwd = Path(project) if project else Path.cwd()
    if not cwd.exists():
        click.echo(f"ERROR: project directory {cwd} does not exist", err=True)
        raise SystemExit(1)

    click.echo(f"Starting Unsloth agent '{agent}' in {cwd.resolve()}")
    click.echo(f"  Command: {' '.join(cmd)}")

    if background:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        click.echo(f"  PID: {proc.pid}")
        click.echo(f"  Server URL: {_health_url(host, port)}")
        return

    # Foreground: run and stream output.
    try:
        subprocess.run(cmd, cwd=str(cwd))
    except KeyboardInterrupt:
        click.echo("\nUnsloth agent interrupted.")


@unsloth.command("run")
@click.option("--model", "-m", required=True, help="Model to serve (e.g. unsloth/Llama-3.1-8B)")
@click.option("--host", "-H", default="127.0.0.1", help="Server host")
@click.option("--port", "-p", default=8888, type=click.IntRange(1, 65535), help="Server port")
@click.option("--enable-tools/--disable-tools", default=False, help="Enable/disable tool use")
@click.option("--no-cloudflare", is_flag=True, help="Do not use Cloudflare tunnel")
@click.option("--gguf-variant", help="Preferred GGUF quantization variant")
@click.option("--context-length", type=click.IntRange(1, 10000000), help="Maximum context length")
@click.option("--no-load-in-4bit", is_flag=True, help="Disable 4-bit loading")
@click.option("--tensor-parallel", type=click.IntRange(1, 128), help="Tensor parallel size")
@click.option(
    "--pid-file",
    default=".unsloth-run.pid",
    help="File to store the server PID",
)
def unsloth_run(
    model,
    host,
    port,
    enable_tools,
    no_cloudflare,
    gguf_variant,
    context_length,
    no_load_in_4bit,
    tensor_parallel,
    pid_file,
):
    """Run an Unsloth local model server.

    Examples:
      terradev train unsloth run --model unsloth/Llama-3.1-8B
      terradev train unsloth run --model unsloth/Qwen3.6-7B-GGUF:Q4_K_M --port 8080
    """
    exe = _find_unsloth()
    cmd = _build_run_command(
        exe,
        model,
        host,
        port,
        enable_tools,
        no_cloudflare,
        gguf_variant,
        context_length,
        no_load_in_4bit,
        tensor_parallel,
    )

    click.echo(f"Starting Unsloth server on {_health_url(host, port)}")
    click.echo(f"  Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    pid_path = Path(pid_file)
    pid_path.write_text(str(proc.pid))
    click.echo(f"  PID: {proc.pid} (written to {pid_path})")

    # Stream log output until interrupted.
    try:
        if proc.stdout:
            for line in proc.stdout:
                click.echo(line.rstrip())
    except KeyboardInterrupt:
        click.echo("\nStopping Unsloth server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if pid_path.exists():
            pid_path.unlink()


@unsloth.command("stop")
@click.option("--pid-file", default=".unsloth-run.pid", help="PID file written by unsloth run")
@click.option("--signal", default="SIGTERM", help="Signal to send")
def unsloth_stop(pid_file, signal):
    """Stop a running Unsloth server started with `unsloth run`."""
    pid_path = Path(pid_file)
    if not pid_path.exists():
        click.echo(f"ERROR: no PID file {pid_path}. Is the server running?", err=True)
        raise SystemExit(1)

    pid = int(pid_path.read_text().strip())
    import signal as _signal

    sig_num = getattr(_signal, signal, None)
    if sig_num is None:
        click.echo(f"ERROR: unknown signal {signal}", err=True)
        raise SystemExit(1)

    try:
        os.kill(pid, sig_num)
        click.echo(f"Sent {signal} to Unsloth server PID {pid}")
        pid_path.unlink()
    except ProcessLookupError:
        click.echo(f"WARNING: process {pid} not found")
        pid_path.unlink()
    except PermissionError:
        click.echo(f"ERROR: permission denied to signal PID {pid}", err=True)
        raise SystemExit(1)
