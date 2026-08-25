#!/usr/bin/env python3
"""OpenTelemetry integration for ``terradev agent``.

Agent subcommands do not invent a second tracing system.  They hook into the
existing Terradev telemetry layer and NodeSpanStream (Redis) that is already
used by ``provision``, ``train`` and ``infer``.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

from terradev_cli.core.node_span_stream import NodeSpanStream
from terradev_cli.core.telemetry import Span, get_telemetry

logger = logging.getLogger(__name__)


@dataclass
class AgentStream:
    """A small container for the per-command node span stream."""

    stream: Optional[NodeSpanStream] = None

    @classmethod
    def start(cls, **extra: Any) -> "AgentStream":
        stream = NodeSpanStream(
            job="terradev-agent",
            version="1",
            instance_id=f"agent-{uuid.uuid4().hex[:12]}",
            provider="local",
            region="local",
        )
        try:
            stream.start(extra)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to start NodeSpanStream: {exc}")
        return cls(stream=stream)

    def emit(
        self,
        name: str,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        if self.stream is None:
            return None
        try:
            return self.stream.emit(name, status, attributes)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to emit span to NodeSpanStream: {exc}")
            return None

    def record_command(
        self,
        command: str,
        args: Optional[Any] = None,
        success: bool = True,
        returncode: int = 0,
        duration_ms: float = 0.0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        if self.stream is None:
            return None
        try:
            return self.stream.record_command(
                command=command,
                args=args,
                success=success,
                returncode=returncode,
                duration_ms=duration_ms,
                attributes=attributes,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to record command to NodeSpanStream: {exc}")
            return None

    def end(self, status: str = "OK", attributes: Optional[Dict[str, Any]] = None) -> None:
        if self.stream is None:
            return
        try:
            self.stream.end(status=status, attributes=attributes)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to end NodeSpanStream: {exc}")


class Tracer:
    """Tracer that writes to the existing Terradev Redis/JSONL span pipeline."""

    def __init__(
        self,
        name: str = "terradev.agent",
        stream: Optional[AgentStream] = None,
        force_telemetry: bool = True,
    ):
        self.name = name
        self.stream = stream or AgentStream.start()
        self._telemetry = get_telemetry()

        # Agent telemetry is always recorded locally; global TERRADEV_TELEMETRY
        # still controls whether local JSONL is appended.
        if force_telemetry and not self._telemetry.enabled:
            os.environ["TERRADEV_TELEMETRY"] = "1"
            self._telemetry = get_telemetry(enabled=True)

        # Attach the stream so telemetry.spans are mirrored into Redis.
        if self.stream.stream is not None:
            self._telemetry.attach_stream(self.stream.stream)

    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[Span] = None,
    ) -> Optional[Span]:
        """Start a span and mirror it to the active node stream."""
        span = self._telemetry.start_span(
            name,
            trace_id=parent.trace_id if parent else None,
            parent_id=parent.span_id if parent else None,
            attributes=attributes,
        )
        # Always emit an event span to the Redis stream as well.
        self.stream.emit(name, "OK", attributes or {})
        return span

    def end_span(self, span: Optional[Span], status: str = "OK") -> None:
        """End a span, flushing it to local JSONL and active streams."""
        if span is None:
            return
        self._telemetry.end_span(span, status=status)

    def record_command(
        self,
        command: str,
        args: Optional[Any] = None,
        success: bool = True,
        returncode: int = 0,
        duration_ms: float = 0.0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """Record a CLI command in both the stream and local telemetry."""
        self._telemetry.record_command_to_active_streams(
            command=command,
            args=args,
            success=success,
            returncode=returncode,
            duration_ms=duration_ms,
            attributes=attributes,
        )
        return self.stream.record_command(
            command=command,
            args=args,
            success=success,
            returncode=returncode,
            duration_ms=duration_ms,
            attributes=attributes,
        )

    def log_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self._telemetry.log_event(name, attributes=attributes)

    @contextmanager
    def trace(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        span = self.start_span(name, attributes)
        try:
            yield span
        except Exception:
            self.end_span(span, "ERROR")
            raise
        else:
            self.end_span(span, "OK")

    def close(self, status: str = "OK") -> None:
        self._telemetry.flush()
        self.stream.end(status=status)


def get_tracer(name: str = "terradev.agent", stream: Optional[AgentStream] = None) -> Tracer:
    """Return a tracer wired into the existing Terradev span pipeline."""
    return Tracer(name=name, stream=stream)


def current_span() -> Optional[Span]:
    """Return the currently active span, if any."""
    telemetry = get_telemetry()
    return telemetry.get_current_span()
