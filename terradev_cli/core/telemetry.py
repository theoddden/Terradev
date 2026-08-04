#!/usr/bin/env python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026 Terradev
#
"""Terradev telemetry — lightweight, local-first span and event logger.

This is the open-source telemetry implementation. It does not phone home. It:

- Collects spans and events in memory.
- Optionally flushes them to a local append-only JSONL trace file under
  ``~/.terradev/traces/`` (or ``TERRADEV_TRACE_DIR``).
- Exports spans as OpenTelemetry-compatible dictionaries for optional upstream
  ingestion by Phoenix, W&B, Langfuse, or any OTel collector.
- Remains a no-op unless ``TERRADEV_TELEMETRY=1`` is set, preserving the
  open-source compliance default.

Future agent protocols in 2031+ can consume these spans directly as a stable
machine-readable history of every Terradev operation.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result import TerradevError

logger = logging.getLogger(__name__)


def _default_trace_dir() -> Path:
    return Path.home() / ".terradev" / "traces"


@dataclass
class Span:
    """A single OpenTelemetry-compatible span."""

    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    name: str = ""
    kind: str = "INTERNAL"
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    status: str = "UNSET"  # UNSET, OK, ERROR
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    resource: Dict[str, Any] = field(default_factory=lambda: {"service.name": "terradev"})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": {
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "parent_id": self.parent_id,
            },
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": {"status_code": self.status},
            "attributes": self.attributes,
            "events": self.events,
            "links": self.links,
            "resource": self.resource,
        }


class TerradevTelemetry:
    """Lightweight telemetry client. Safe to use from async and sync code."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        trace_dir: Optional[Path] = None,
    ):
        if enabled is None:
            enabled = os.environ.get("TERRADEV_TELEMETRY", "").strip().lower() in ("1", "true", "yes")
        self.enabled = enabled
        self.trace_dir = trace_dir or _default_trace_dir()
        self._spans: List[Span] = []
        self._trace_id = str(uuid.uuid4())
        self._span_stack: List[Span] = []
        self._pending_events: List[Dict[str, Any]] = []
        self._flush_count = 0
        self._active_streams: List[Any] = []

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        if not self.enabled:
            return None
        parent = self._span_stack[-1] if self._span_stack else None
        span = Span(
            trace_id=trace_id or (parent.trace_id if parent else self._trace_id),
            parent_id=parent_id or (parent.span_id if parent else None),
            name=name,
            attributes=attributes or {},
        )
        span.start_time = datetime.now(timezone.utc).isoformat()
        self._spans.append(span)
        self._span_stack.append(span)
        return span

    def end_span(
        self,
        span: Optional[Span] = None,
        status: str = "OK",
    ) -> None:
        if not self.enabled:
            return
        if span is None:
            if not self._span_stack:
                return
            span = self._span_stack.pop()
        elif span in self._span_stack:
            self._span_stack.remove(span)
        span.end_time = datetime.now(timezone.utc).isoformat()
        span.status = status
        for event in self._pending_events:
            if event.get("trace_id") == span.trace_id and event.get("span_id") == span.span_id:
                span.events.append(event)
                event["stored"] = True
        self._pending_events = [e for e in self._pending_events if not e.get("stored")]
        self._emit_to_streams(span)

    def log_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
            "trace_id": trace_id or (self._span_stack[-1].trace_id if self._span_stack else self._trace_id),
            "span_id": span_id or (self._span_stack[-1].span_id if self._span_stack else None),
        }
        if self._span_stack and self._span_stack[-1].span_id == event["span_id"]:
            self._span_stack[-1].events.append(event)
        else:
            self._pending_events.append(event)

        # Mirror event as a span to any active node streams.
        span = Span(
            trace_id=event["trace_id"],
            parent_id=event["span_id"],
            name=name,
            start_time=event["timestamp"],
            end_time=event["timestamp"],
            status="OK",
            attributes=attributes or {},
        )
        self._emit_to_streams(span)

    def log_error(
        self,
        error: TerradevError,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "terradev.error",
            attributes=error.to_dict(),
            trace_id=trace_id,
            span_id=span_id,
        )

    def set_attribute(self, key: str, value: Any) -> None:
        if not self.enabled or not self._span_stack:
            return
        self._span_stack[-1].attributes[key] = value

    def get_current_span(self) -> Optional[Span]:
        if not self._span_stack:
            return None
        return self._span_stack[-1]

    def attach_stream(self, stream: Any) -> None:
        """Attach a node span stream to receive all telemetry spans."""
        if stream not in self._active_streams:
            self._active_streams.append(stream)

    def detach_stream(self, stream: Any) -> None:
        """Detach a node span stream."""
        if stream in self._active_streams:
            self._active_streams.remove(stream)

    def _emit_to_streams(self, span: Span) -> None:
        """Send a completed span to every active node stream."""
        if not self._active_streams:
            return
        for stream in self._active_streams:
            try:
                if hasattr(stream, "append_span"):
                    stream.append_span(span)
                elif hasattr(stream, "exporter") and hasattr(stream, "stream_key"):
                    stream.exporter.append_span(stream.stream_key, span)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to emit span to stream {stream}: {e}")

    def record_command_to_active_streams(
        self,
        command: str,
        args: List[str],
        success: bool,
        returncode: int,
        duration_ms: float,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a command in every active node stream's command chain."""
        if not self._active_streams:
            return
        for stream in self._active_streams:
            try:
                if hasattr(stream, "record_command"):
                    stream.record_command(
                        command=command,
                        args=args,
                        success=success,
                        returncode=returncode,
                        duration_ms=duration_ms,
                        attributes=attributes,
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to record command to stream {stream}: {e}")

    def flush(self, force: bool = False) -> Optional[Path]:
        if not self.enabled and not force:
            return None
        if not self._spans:
            return None

        self.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.trace_dir / f"{self._trace_id}.jsonl"

        with open(trace_path, "a", encoding="utf-8") as f:
            for span in self._spans:
                f.write(json.dumps(span.to_dict(), default=str, sort_keys=True) + "\n")

        self._flush_count += 1
        self._spans = []
        self._pending_events = []
        return trace_path

    def get_trace_summary(self) -> Dict[str, Any]:
        return {
            "trace_id": self._trace_id,
            "spans_in_memory": len(self._spans),
            "span_stack_depth": len(self._span_stack),
            "pending_events": len(self._pending_events),
            "enabled": self.enabled,
            "trace_dir": str(self.trace_dir),
            "flush_count": self._flush_count,
        }


# Global telemetry instance for compatibility with the old TelemetryClient API.
_telemetry: Optional[TerradevTelemetry] = None


def get_telemetry(enabled: Optional[bool] = None) -> TerradevTelemetry:
    """Get the global Terradev telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = TerradevTelemetry(enabled=enabled)
    return _telemetry


class TelemetryClient:
    """Backwards-compatible telemetry client wrapper."""

    def __init__(self, enabled: Optional[bool] = None):
        self._telemetry = get_telemetry(enabled=enabled)

    def log_action(self, action: str, details: Dict[str, Any] = None):
        """Log an action as a telemetry event."""
        self._telemetry.log_event(action, attributes=details or {})

    def check_license(self, action: str = "provision") -> Dict[str, Any]:
        """No-op license check for open source compatibility."""
        return {
            "allowed": True,
            "tier": "open-source",
            "limit": float("inf"),
            "usage": 0,
            "reason": "Open source - no restrictions",
        }

    def start_span(self, name: str, **kwargs) -> Optional[Span]:
        return self._telemetry.start_span(name, **kwargs)

    def end_span(self, span: Optional[Span] = None, **kwargs) -> None:
        return self._telemetry.end_span(span, **kwargs)


MandatoryTelemetryClient = TelemetryClient


# Stable singleton for the backwards-compatible client.
_mandatory_telemetry_client: Optional[TelemetryClient] = None


def get_mandatory_telemetry() -> TelemetryClient:
    """Return the backwards-compatible ``TelemetryClient`` singleton."""
    global _mandatory_telemetry_client
    if _mandatory_telemetry_client is None:
        _mandatory_telemetry_client = TelemetryClient()
    return _mandatory_telemetry_client
