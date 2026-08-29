#!/usr/bin/env python3
"""Shared models, protocols and abstractions for the ``terradev agent`` namespace.

The goal is a small, composable toolkit: every runtime/transport is a
self-contained object that can be instantiated and tested independently.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

import click
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .otel import Span

T = TypeVar("T")


class AgentError(Exception):
    """Base error for all ``terradev agent`` subcommands."""


class UnsupportedRuntimeError(AgentError):
    """A requested sandbox runtime is not available on the host."""


class SandboxTimeoutError(AgentError):
    """A sandboxed payload exceeded its configured timeout."""


class MeshError(AgentError):
    """A mesh operation failed."""


class McpError(AgentError):
    """A Model Context Protocol bridge operation failed."""


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def _parse_size(value: Optional[str]) -> Optional[int]:
    """Convert human-friendly sizes (``512m``, ``2g``) to integer bytes."""
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(b|k|m|g|t)?$", value)
    if not match:
        raise ValueError(f"Invalid size value: {value!r}")
    number = float(match.group(1))
    unit = match.group(2) or "b"
    multipliers = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    return int(number * multipliers[unit])


def _resolve_command(commands: List[str]) -> Optional[str]:
    """Return the first executable found in PATH, or None."""
    for cmd in commands:
        path = shutil.which(cmd)
        if path:
            return path
    return None


# ---------------------------------------------------------------------------
# Sandbox models and protocols
# ---------------------------------------------------------------------------


class NetworkPolicy(BaseModel):
    """Network access rules for a sandboxed payload."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "none"  # none, allowlist, denylist
    allow: List[str] = Field(default_factory=list)
    deny: List[str] = Field(default_factory=list)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in {"none", "allowlist", "denylist", "host"}:
            raise ValueError(f"Unsupported network mode: {v}")
        return v


class ResourceLimits(BaseModel):
    """Resource caps for a sandboxed payload."""

    model_config = ConfigDict(extra="forbid")

    vcpus: Optional[int] = None
    memory: Optional[str] = None  # e.g. 512m, 2g
    pids: Optional[int] = None
    timeout_seconds: float = 30.0
    read_only: bool = True


class SandboxConfig(BaseModel):
    """Configuration for a single sandbox invocation."""

    model_config = ConfigDict(extra="forbid")

    runtime: str = "auto"
    isolation: Optional[str] = None
    payload: Optional[str] = None
    use_stdin: bool = False
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    image: Optional[str] = None
    kernel: Optional[str] = None
    rootfs: Optional[str] = None
    extra_args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    otel_endpoint: Optional[str] = None
    dry_run: bool = False
    dev_mode: bool = False

    @model_validator(mode="after")
    def _set_isolation(self) -> SandboxConfig:
        if not self.isolation:
            mapping = {
                "firecracker": "microvm",
                "gvisor": "system-call",
                "landlock": "namespace",
                "bwrap": "namespace",
                "local": "none",
            }
            self.isolation = mapping.get(self.runtime, "none")
        return self

    @model_validator(mode="after")
    def _validate_runtime(self) -> SandboxConfig:
        allowed = {"auto", "firecracker", "gvisor", "landlock", "bwrap", "local"}
        if self.runtime not in allowed:
            raise ValueError(f"Unknown runtime: {self.runtime}")
        return self


@dataclass
class RunResult:
    """Structured result of a sandboxed execution."""

    exit_code: int
    stdout: str
    stderr: str
    runtime: str
    duration_ms: float
    span: Optional[Span] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "runtime": self.runtime,
            "duration_ms": self.duration_ms,
            "span": self.span.to_dict() if self.span else None,
            "resource_usage": self.resource_usage,
        }


class SandboxRuntime(ABC):
    """Abstract async runtime that can execute an untrusted payload."""

    name: str = "abstract"
    priority: int = 0

    @abstractmethod
    async def is_available(self, config: Optional[SandboxConfig] = None) -> bool:
        """Return True if this runtime can be used on the current host."""

    @abstractmethod
    async def run(
        self,
        config: SandboxConfig,
        *,
        command: List[str],
        span: Optional[Span] = None,
    ) -> RunResult:
        """Run ``command`` inside the sandbox and return the result."""

    def _build_timeout_args(self, config: SandboxConfig) -> List[str]:
        return ["timeout", "-s", "KILL", f"{int(config.resources.timeout_seconds)}"]

    async def _exec(
        self,
        cmd: List[str],
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        input_data: Optional[bytes] = None,
    ) -> RunResult:
        """Execute a process asynchronously and capture both streams."""
        if not cmd:
            return RunResult(
                exit_code=127,
                stdout="",
                stderr="No command provided to sandbox runtime",
                runtime=self.name,
                duration_ms=0.0,
            )

        start = asyncio.get_event_loop().time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if input_data is not None else None,
                env=env,
            )
        except (FileNotFoundError, PermissionError, NotADirectoryError, ValueError) as exc:
            exit_code = 127 if isinstance(exc, (FileNotFoundError, ValueError)) else 126
            return RunResult(
                exit_code=exit_code,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                runtime=self.name,
                duration_ms=0.0,
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input_data),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout, stderr = await proc.communicate()
            raise SandboxTimeoutError(
                f"Sandbox payload exceeded {timeout}s timeout"
            ) from None
        duration = (asyncio.get_event_loop().time() - start) * 1000
        return RunResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            runtime=self.name,
            duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Mesh models and protocols
# ---------------------------------------------------------------------------


class MeshTopology(str, Enum):
    """Supported mesh topologies."""

    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"


class MeshRoutingStrategy(str, Enum):
    """Routing objectives."""

    LATENCY = "latency"
    COST = "cost"
    THROUGHPUT = "throughput"


class AgentCard(BaseModel):
    """A2A-style capability advertisement."""

    model_config = ConfigDict(extra="forbid")

    name: str
    endpoint: str
    skills: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
    authentication: Optional[Dict[str, str]] = None

    @field_validator("endpoint")
    @classmethod
    def _has_endpoint(cls, v: str) -> str:
        if not v.startswith(("http://", "https://", "/ip")):
            raise ValueError("endpoint must be an HTTP URL or a libp2p multiaddr")
        return v


class ArtifactPart(BaseModel):
    """A2A-style content part."""

    model_config = ConfigDict(extra="forbid")

    content_type: str
    content: str


class Artifact(BaseModel):
    """A2A-style artifact returned by a remote agent."""

    model_config = ConfigDict(extra="forbid")

    name: str
    parts: List[ArtifactPart] = Field(default_factory=list)


class Task(BaseModel):
    """A2A-style task delegated to a remote agent."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "pending"
    input: str
    skills: List[str] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_artifact(self, name: str, content_type: str, content: str) -> Artifact:
        artifact = Artifact(
            name=name,
            parts=[ArtifactPart(content_type=content_type, content=content)],
        )
        self.artifacts.append(artifact)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return artifact


class MeshConfig(BaseModel):
    """Configuration for a mesh node."""

    model_config = ConfigDict(extra="forbid")

    protocol: str = "a2a"
    transport: str = "http"  # http, libp2p, wireguard
    listen: str = "127.0.0.1:4222"
    bootstrap: List[str] = Field(default_factory=list)
    topology: MeshTopology = MeshTopology.DECENTRALIZED
    routing: MeshRoutingStrategy = MeshRoutingStrategy.LATENCY
    identity: Optional[str] = None


class MeshTransport(ABC):
    """Abstract transport for agent-to-agent communication."""

    name: str = "abstract"

    async def is_available(self) -> bool:
        """Return True if the transport can be started on this host."""
        return True

    @abstractmethod
    async def start(self, config: MeshConfig) -> None:
        """Start listening for peer traffic."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening and tear down connections."""

    @abstractmethod
    async def publish_card(self, card: AgentCard) -> None:
        """Publish this node's Agent Card to the mesh."""

    @abstractmethod
    async def discover_cards(self, skills: Optional[List[str]] = None) -> List[AgentCard]:
        """Return matching Agent Cards known to the mesh."""

    @abstractmethod
    async def delegate_task(self, card: AgentCard, task: Task) -> Task:
        """Send a task to a remote agent and await the completed result."""


# ---------------------------------------------------------------------------
# MCP models and protocols
# ---------------------------------------------------------------------------


class JsonRpcMessage(BaseModel):
    """A minimal JSON-RPC 2.0 message."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    method: Optional[str] = None
    params: Optional[Any] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_message(self) -> JsonRpcMessage:
        if self.method is not None and self.result is not None:
            raise ValueError("A JSON-RPC message cannot contain both method and result")
        return self

    def is_request(self) -> bool:
        return self.method is not None and self.id is not None

    def is_notification(self) -> bool:
        return self.method is not None and self.id is None

    def is_response(self) -> bool:
        return self.id is not None and (self.result is not None or self.error is not None)


class McpTool(BaseModel):
    """A tool exposed by an MCP server."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Any] = Field(default_factory=dict)


class McpPrompt(BaseModel):
    """A prompt exposed by an MCP server."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    arguments: Optional[List[Dict[str, Any]]] = None


class McpResource(BaseModel):
    """A resource exposed by an MCP server."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    name: str
    mimeType: Optional[str] = None
    description: Optional[str] = None


class McpServerDefinition(BaseModel):
    """A single MCP server definition in the registry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    transport: str = "stdio"  # stdio, http
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = None
    image: Optional[str] = None
    isolation: Optional[str] = None
    network_allow: List[str] = Field(default_factory=list)
    timeout: Optional[float] = None

    @model_validator(mode="after")
    def _validate(self) -> McpServerDefinition:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio transport requires a command")
        if self.transport == "http" and not self.url:
            raise ValueError("http transport requires a url")
        return self


class McpTransport(ABC):
    """Abstract async transport for a single MCP server session."""

    name: str = "abstract"

    @abstractmethod
    async def send(self, message: JsonRpcMessage) -> JsonRpcMessage:
        """Send a JSON-RPC request and return the response."""

    @abstractmethod
    async def send_notification(self, message: JsonRpcMessage) -> None:
        """Send a JSON-RPC notification (no response expected)."""

    @abstractmethod
    async def close(self) -> None:
        """Close the transport and any underlying process/connection."""


# ---------------------------------------------------------------------------
# Async click helper
# ---------------------------------------------------------------------------


def _run_with_timeout(coro, timeout=300):
    """Run an async coroutine with a timeout to prevent hangs."""
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
    except asyncio.TimeoutError:
        click.echo(f"ERROR: Agent operation timed out after {timeout}s", err=True)
        raise SystemExit(1)


def async_command(coro: Callable[..., Any]) -> click.Command:
    """Wrap an async coroutine so it can be used as a Click command callback."""

    @click.pass_context
    def wrapper(ctx: click.Context, *args, **kwargs):
        # Ensure each CLI invocation gets its own fresh event loop.
        return _run_with_timeout(coro(*args, **kwargs))

    # Preserve function signature metadata for Click help generation.
    wrapper.__name__ = coro.__name__
    wrapper.__doc__ = coro.__doc__
    return wrapper
