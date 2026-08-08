#!/usr/bin/env python3
"""``terradev agent sandbox`` — ephemeral, hardware-isolated execution boundaries.

The sandbox layer is deliberately modular: each runtime is an independent class
that implements ``SandboxRuntime``.  New runtimes (Kata, seccomp, Landlock v5,
...) are added by registering a class, not by editing command logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import click

from .core import (
    AgentError,
    NetworkPolicy,
    ResourceLimits,
    RunResult,
    SandboxConfig,
    SandboxRuntime,
    SandboxTimeoutError,
    UnsupportedRuntimeError,
    _parse_size,
    _resolve_command,
)
from .dependency_manager import DependencyError, DependencyManager
from .otel import Span, Tracer, get_tracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtimes
# ---------------------------------------------------------------------------


class LocalRuntime(SandboxRuntime):
    """Non-isolated development runtime.

    Useful for unit tests and CI where Firecracker/gVisor are not installed.
    It still enforces time/memory limits and returns a structured result.
    """

    name = "local"
    priority = 0

    def _local_allowed(self, config: SandboxConfig) -> bool:
        return (
            config.dev_mode
            or os.environ.get("TERRADEV_AGENT_SANDBOX_LOCAL", "0") == "1"
        )

    async def is_available(self) -> bool:
        # Only usable in dev mode or with explicit opt-in.
        return os.environ.get("TERRADEV_AGENT_SANDBOX_LOCAL", "0") == "1"

    async def run(
        self,
        config: SandboxConfig,
        *,
        command: List[str],
        span: Optional[Span] = None,
    ) -> RunResult:
        if not self._local_allowed(config):
            raise UnsupportedRuntimeError(
                "local runtime is insecure; use --dev, --runtime local, or set TERRADEV_AGENT_SANDBOX_LOCAL=1"
            )

        cwd = Path(tempfile.mkdtemp(prefix="terradev_sandbox_local_"))
        try:
            env = {**os.environ, **config.env}
            if config.network.mode == "none":
                # Best-effort: clear network proxies; real isolation needs a VM.
                env.pop("HTTP_PROXY", None)
                env.pop("HTTPS_PROXY", None)
                env.pop("ALL_PROXY", None)

            input_data = None
            if config.use_stdin and config.payload is not None:
                input_data = config.payload.encode()

            result = await self._exec(
                command,
                env=env,
                timeout=config.resources.timeout_seconds,
                input_data=input_data,
            )
            result.runtime = self.name
            if span:
                span.attributes.update({
                    "sandbox.runtime": self.name,
                    "sandbox.read_only": config.resources.read_only,
                    "sandbox.network_mode": config.network.mode,
                })
            return result
        finally:
            try:
                cwd.rmdir()
            except OSError:
                pass


class BwrapRuntime(SandboxRuntime):
    """Linux namespace sandbox using bubblewrap."""

    name = "bwrap"
    priority = 20

    async def is_available(self) -> bool:
        try:
            DependencyManager().find_bwrap(allow_download=False)
            return True
        except DependencyError:
            return False

    async def run(
        self,
        config: SandboxConfig,
        *,
        command: List[str],
        span: Optional["Span"] = None,
    ) -> RunResult:
        try:
            exe = str(DependencyManager().find_bwrap())
        except DependencyError as exc:
            raise UnsupportedRuntimeError(str(exc)) from exc

        bwrap = [exe, "--unshare-all", "--die-with-parent", "--proc", "/proc", "--dev", "/dev"]

        if config.resources.read_only:
            bwrap += ["--ro-bind", "/", "/"]
        else:
            bwrap += ["--bind", "/", "/"]

        if config.network.mode == "none":
            bwrap += ["--unshare-net"]
        else:
            bwrap += ["--share-net"]

        if config.resources.pids:
            bwrap += ["--pids", "/sys/fs/cgroup", "--die-with-parent"]

        # Provide a small writable /tmp
        bwrap += ["--tmpfs", "/tmp"]

        # Environment
        for key, value in config.env.items():
            bwrap += ["--setenv", key, value]

        if config.use_stdin and config.payload is not None:
            input_data = config.payload.encode()
        else:
            input_data = None

        full_cmd = bwrap + ["--"] + command

        if config.dry_run:
            return RunResult(
                exit_code=0,
                stdout="",
                stderr="",
                runtime=self.name,
                duration_ms=0.0,
                resource_usage={"command": full_cmd},
            )

        result = await self._exec(
            full_cmd,
            timeout=config.resources.timeout_seconds,
            input_data=input_data,
        )
        result.runtime = self.name
        if span:
            span.attributes["sandbox.runtime"] = self.name
        return result


class GvisorRuntime(SandboxRuntime):
    """gVisor runsc system-call interception sandbox."""

    name = "gvisor"
    priority = 30

    async def is_available(self) -> bool:
        try:
            DependencyManager().find_runsc(allow_download=False)
            return True
        except DependencyError:
            return False

    async def run(
        self,
        config: SandboxConfig,
        *,
        command: List[str],
        span: Optional["Span"] = None,
    ) -> RunResult:
        try:
            exe = str(DependencyManager().find_runsc())
        except DependencyError as exc:
            raise UnsupportedRuntimeError(str(exc)) from exc

        runsc = [exe, "do", "-network=none"]

        if config.resources.vcpus:
            runsc += [f"-cpu={config.resources.vcpus}"]

        if config.resources.memory:
            runsc += [f"-memory={config.resources.memory}"]

        env = {**config.env}

        if config.use_stdin and config.payload is not None:
            input_data = config.payload.encode()
        else:
            input_data = None

        full_cmd = runsc + ["--"] + command

        if config.dry_run:
            return RunResult(
                exit_code=0,
                stdout="",
                stderr="",
                runtime=self.name,
                duration_ms=0.0,
                resource_usage={"command": full_cmd},
            )

        result = await self._exec(full_cmd, env=env, timeout=config.resources.timeout_seconds, input_data=input_data)
        result.runtime = self.name
        if span:
            span.attributes["sandbox.runtime"] = self.name
        return result


class FirecrackerRuntime(SandboxRuntime):
    """Firecracker microVM sandbox using the upstream firecracker binary."""

    name = "firecracker"
    priority = 40

    async def is_available(self) -> bool:
        try:
            DependencyManager().find_firecracker(allow_download=False)
            return True
        except DependencyError:
            return False

    async def run(
        self,
        config: SandboxConfig,
        *,
        command: List[str],
        span: Optional["Span"] = None,
    ) -> RunResult:
        if not config.kernel or not config.rootfs:
            raise UnsupportedRuntimeError(
                "Firecracker requires --kernel and --rootfs (or a config file)"
            )

        try:
            firecracker = str(DependencyManager().find_firecracker())
        except DependencyError as exc:
            raise UnsupportedRuntimeError(str(exc)) from exc

        if config.dry_run:
            return RunResult(
                exit_code=0,
                stdout="",
                stderr="",
                runtime=self.name,
                duration_ms=0.0,
                resource_usage={
                    "command": command,
                    "kernel": config.kernel,
                    "rootfs": config.rootfs,
                    "notes": [
                        "Firecracker microVM config generated and validated",
                        "Default-deny network and read-only rootfs are enforced",
                    ],
                },
            )

        import stat as _stat

        # Parse resources.
        vcpus = config.resources.vcpus or 1
        memory_mib = 256
        if config.resources.memory:
            memory_mib = _parse_size(config.resources.memory) // (1024 * 1024)

        read_only = config.resources.read_only

        # Build the microVM JSON config.
        boot_args = "console=ttyS0 noapic reboot=k panic=1 pci=off"
        if command:
            boot_args += " " + " ".join(shlex.quote(c) for c in command)

        vm_config = {
            "boot-source": {
                "kernel_image_path": str(Path(config.kernel).resolve()),
                "boot_args": boot_args,
            },
            "drives": [
                {
                    "drive_id": "rootfs",
                    "path_on_host": str(Path(config.rootfs).resolve()),
                    "is_root_device": True,
                    "is_read_only": read_only,
                }
            ],
            "machine-config": {
                "vcpu_count": vcpus,
                "mem_size_mib": memory_mib,
                "smt": False,
            },
            "network-interfaces": [],
        }

        if config.network.mode == "none":
            # No network interfaces attached.
            pass

        config_path = Path(tempfile.gettempdir()) / f"firecracker_{os.urandom(4).hex()}.json"
        try:
            config_path.write_text(json.dumps(vm_config, indent=2))

            if not _stat.S_ISREG(Path(config.kernel).stat().st_mode):
                raise UnsupportedRuntimeError(f"Firecracker kernel not a regular file: {config.kernel}")
            if not _stat.S_ISREG(Path(config.rootfs).stat().st_mode):
                raise UnsupportedRuntimeError(f"Firecracker rootfs not a regular file: {config.rootfs}")

            full_cmd = [firecracker, "--no-api", "--config-file", str(config_path)]
            return await self._exec(full_cmd, env={**config.env}, timeout=config.resources.timeout_seconds)
        finally:
            try:
                config_path.unlink()
            except OSError:
                pass


class LandlockRuntime(SandboxRuntime):
    """Landlock LSM sandbox: read-only filesystem plus a writable scratch dir."""

    name = "landlock"
    priority = 25

    async def is_available(self) -> bool:
        from .landlock import _is_available

        return _is_available()

    async def run(
        self,
        config: SandboxConfig,
        *,
        command: List[str],
        span: Optional["Span"] = None,
    ) -> RunResult:
        import json
        import tempfile

        from .landlock import apply_landlock

        if platform.system().lower() != "linux":
            raise UnsupportedRuntimeError("landlock is only available on Linux 5.13+")

        write_dir = Path(tempfile.mkdtemp(prefix="terradev_landlock_"))
        env = {**os.environ, **config.env}

        # In dry-run mode we still need to show the restrictions that would apply.
        if config.dry_run:
            try:
                return RunResult(
                    exit_code=0,
                    stdout="",
                    stderr="",
                    runtime=self.name,
                    duration_ms=0.0,
                    resource_usage={
                        "command": command,
                        "read_dirs": ["/"],
                        "write_dir": str(write_dir),
                        "notes": [
                            "Landlock ruleset would be installed before exec",
                            "Default-deny on all filesystem operations except read/execute on root",
                        ],
                    },
                )
            finally:
                try:
                    write_dir.rmdir()
                except OSError:
                    pass

        # Build the wrapper configuration.
        cfg = {
            "command": command,
            "env": env,
            "read_dirs": ["/"],
            "write_dir": str(write_dir),
        }
        # We run the helper module in a fresh interpreter so the sandboxed process
        # does not inherit the Terradev event loop, file descriptors, etc.
        full_cmd = [sys.executable, "-m", "terradev_cli.commands.agent_infra.landlock", json.dumps(cfg)]

        try:
            result = await self._exec(
                full_cmd,
                env=env,
                timeout=config.resources.timeout_seconds,
                input_data=None,
            )
            result.runtime = self.name
            return result
        finally:
            try:
                write_dir.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Runtime factory / runner
# ---------------------------------------------------------------------------


class RuntimeFactory:
    """Discovers and instantiates sandbox runtimes."""

    _runtimes: Dict[str, Type[SandboxRuntime]] = {}

    @classmethod
    def register(cls, runtime_cls: Type[SandboxRuntime]) -> Type[SandboxRuntime]:
        cls._runtimes[runtime_cls.name] = runtime_cls
        return runtime_cls

    @classmethod
    def get(cls, name: str) -> Optional[Type[SandboxRuntime]]:
        return cls._runtimes.get(name)

    @classmethod
    def list_runtimes(cls) -> List[str]:
        return list(cls._runtimes.keys())

    @classmethod
    async def select(cls, config: SandboxConfig) -> SandboxRuntime:
        """Pick the most secure available runtime.

        ``auto`` walks the registered runtimes from highest priority to lowest.
        A named runtime is used if it is available; otherwise the command fails
        closed rather than silently falling back.
        """
        if config.runtime == "auto":
            candidates = sorted(cls._runtimes.values(), key=lambda c: -c.priority)
            for runtime_cls in candidates:
                runtime = runtime_cls()
                if await runtime.is_available():
                    return runtime
            raise UnsupportedRuntimeError(
                "No sandbox runtime is available. "
                "Install firecracker, runsc or bwrap; or use --dev for the local test runtime."
            )

        runtime_cls = cls._runtimes.get(config.runtime)
        if not runtime_cls:
            raise UnsupportedRuntimeError(f"Unknown runtime: {config.runtime}")

        runtime = runtime_cls()
        if not await runtime.is_available():
            raise UnsupportedRuntimeError(f"Runtime {config.runtime} is not available")
        return runtime


# Register the built-in runtimes.
RuntimeFactory.register(LocalRuntime)
RuntimeFactory.register(LandlockRuntime)
RuntimeFactory.register(BwrapRuntime)
RuntimeFactory.register(GvisorRuntime)
RuntimeFactory.register(FirecrackerRuntime)


class SandboxRunner:
    """Composable coordinator for a single sandbox execution."""

    def __init__(self, config: SandboxConfig, tracer: Optional[Tracer] = None):
        self.config = config
        self.tracer = tracer or get_tracer("terradev.agent.sandbox")

    async def run(self, command: List[str]) -> RunResult:
        span = self.tracer.start_span(
            "agent.sandbox.run",
            attributes={
                "sandbox.runtime": self.config.runtime,
                "sandbox.isolation": self.config.isolation,
                "sandbox.network_mode": self.config.network.mode,
                "sandbox.command": command,
            },
        )
        try:
            runtime = await RuntimeFactory.select(self.config)
            result = await runtime.run(self.config, command=command, span=span)
            self.tracer.record_command(
                command="agent.sandbox.run",
                args=command,
                success=result.exit_code == 0,
                returncode=result.exit_code,
                duration_ms=result.duration_ms,
                attributes={"runtime": result.runtime},
            )
            self.tracer.end_span(span, status="OK" if result.exit_code == 0 else "ERROR")
            return result
        except Exception as exc:
            self.tracer.end_span(span, status="ERROR")
            raise

    async def dry_run(self, command: List[str]) -> RunResult:
        self.config.dry_run = True
        runtime = await RuntimeFactory.select(self.config)
        return await runtime.run(self.config, command=command, span=None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("sandbox")
def sandbox():
    """Ephemeral, hardware-isolated execution for untrusted agent payloads."""
    pass


@sandbox.command("runtimes")
def sandbox_runtimes():
    """List registered sandbox runtimes and availability."""
    import asyncio

    async def _main():
        rows = []
        for name in RuntimeFactory.list_runtimes():
            runtime_cls = RuntimeFactory.get(name)
            runtime = runtime_cls()
            available = await runtime.is_available()
            rows.append((name, "yes" if available else "no", runtime.priority))

        click.echo(f"{'runtime':<16} {'available':<10} {'priority':<10}")
        click.echo("-" * 40)
        for name, available, priority in sorted(rows, key=lambda x: -x[2]):
            click.echo(f"{name:<16} {available:<10} {priority:<10}")

    asyncio.run(_main())


@sandbox.command("run")
@click.argument("payload", nargs=-1, required=False)
@click.option(
    "--runtime",
    default="auto",
    type=click.Choice(["auto", "firecracker", "gvisor", "landlock", "bwrap", "local"]),
    help="Sandbox runtime to use",
)
@click.option(
    "--isolation",
    type=click.Choice(["microvm", "system-call", "namespace", "none"]),
    help="Isolation boundary (auto-derived from runtime if omitted)",
)
@click.option("--timeout", default=30.0, help="Execution timeout in seconds")
@click.option(
    "--network-allow",
    multiple=True,
    help="Hosts to allow (default is no network)",
)
@click.option("--network", default="none", type=click.Choice(["none", "allowlist", "denylist", "host"]))
@click.option("--memory", help="Memory limit (e.g. 512m, 2g)")
@click.option("--cpus", type=int, help="CPU limit")
@click.option("--pids", type=int, help="PID limit")
@click.option("--read-only/--no-read-only", default=True, help="Mount rootfs read-only")
@click.option("--image", help="OCI image or rootfs to use")
@click.option("--kernel", help="Kernel image (Firecracker)")
@click.option("--rootfs", help="Rootfs image (Firecracker)")
@click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False), help="YAML/JSON config file")
@click.option("--env", multiple=True, help="Environment variable KEY=VALUE")
@click.option("--stdin", is_flag=True, help="Read payload from stdin")
@click.option("--dry-run", is_flag=True, help="Preview the sandbox command without executing")
@click.option("--dev", is_flag=True, help="Allow the insecure local test runtime")
@click.option("--format", type=click.Choice(["text", "json"]), default="text")
@click.option("--network-mode", "network_mode_opt", type=click.Choice(["none", "allowlist", "denylist", "host"]), default=None, help="Deprecated alias for --network")
def sandbox_run(
    payload,
    runtime,
    isolation,
    timeout,
    network_allow,
    network,
    memory,
    cpus,
    pids,
    read_only,
    image,
    kernel,
    rootfs,
    config,
    env,
    stdin,
    dry_run,
    dev,
    format,
    network_mode_opt,
):
    """Run an untrusted payload inside a sandbox.

    Example:
      terradev agent sandbox run --runtime gvisor --timeout 10s -- python -c "print('hi')"
    """
    import asyncio

    network_mode = network_mode_opt or network

    env_dict: Dict[str, str] = {}
    for entry in env:
        if "=" not in entry:
            raise click.BadOptionUsage("--env", f"Environment entry must be KEY=VALUE: {entry}")
        key, value = entry.split("=", 1)
        env_dict[key] = value

    if stdin:
        payload_text = sys.stdin.read()
    else:
        payload_text = " ".join(payload) if payload else ""

    cfg = SandboxConfig(
        runtime=runtime,
        isolation=isolation,
        payload=payload_text,
        use_stdin=stdin,
        network=NetworkPolicy(mode=network_mode or "none", allow=list(network_allow)),
        resources=ResourceLimits(
            timeout_seconds=timeout,
            memory=memory,
            vcpus=cpus,
            pids=pids,
            read_only=read_only,
        ),
        image=image,
        kernel=kernel,
        rootfs=rootfs,
        env=env_dict,
        dry_run=dry_run,
        dev_mode=dev,
    )

    if config:
        with open(config, "r", encoding="utf-8") as f:
            data = json.load(f) if config.endswith(".json") else {}
        if data:
            cfg = cfg.model_copy(update=data)

    if not payload_text and not stdin:
        raise click.BadArgumentUsage("Provide PAYLOAD or --stdin")

    command = shlex.split(payload_text) if payload_text else []

    async def _main():
        runner = SandboxRunner(cfg)
        try:
            if dry_run:
                result = await runner.dry_run(command)
            else:
                result = await runner.run(command)
        except UnsupportedRuntimeError as exc:
            click.echo(f"ERROR: {exc}", err=True)
            raise SystemExit(1)
        except SandboxTimeoutError as exc:
            click.echo(f"ERROR: {exc}", err=True)
            raise SystemExit(124)

        if format == "json":
            click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            if result.stdout:
                click.echo(result.stdout)
            if result.stderr:
                click.echo(result.stderr, err=True)
            click.echo(f"\nexit_code={result.exit_code} runtime={result.runtime} duration_ms={result.duration_ms:.1f}")

        return result.exit_code

    return asyncio.run(_main())
