#!/usr/bin/env python3
"""Node span stream sidecar.

A ``NodeSpanStream`` is the per-node Redis stream that follows a Terradev
instance from provision to teardown. It is created automatically when a node
is spun up and destroyed when the node is torn down. While active, any
``TerradevTelemetry`` span or event is mirrored to the stream, so preflight,
train, and inference spans appear in real time without explicit code at every
command step.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .redis_span_exporter import RedisSpanExporter
from .result import ErrorCode, ErrorCategory, Severity, TerradevError
from .telemetry import Span, TerradevTelemetry, get_telemetry
from .telinea.connector import maybe_attach_telinea

logger = logging.getLogger(__name__)


@dataclass
class NodeSpanStream:
    """Per-node sidecar stream.

    - ``start()`` creates the Redis stream and attaches it to the global
      telemetry client so subsequent spans are mirrored.
    - ``end()`` appends a final span and detaches.
    - ``destroy()`` appends a final span and deletes the Redis stream.
    """

    job: str
    version: str
    instance_id: str
    provider: str = ""
    region: str = ""
    gpu_type: str = ""
    gpu_count: int = 0
    ttl: str = "1h"
    parent_trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    command_chain: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.exporter = RedisSpanExporter()
        self.trace_id = self.parent_trace_id or str(uuid.uuid4())
        self.span_id = self.parent_span_id or str(uuid.uuid4())
        self.stream_key = self.exporter.stream_key(
            self.job, self.version, self.instance_id
        )
        self._active = False
        self._closed = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        self._lock = threading.RLock()
        self._telinea_connector: Any = None

    @property
    def active(self) -> bool:
        return self._active and not self._closed

    def _node_span(self, name: str, status: str, attributes: Optional[Dict[str, Any]] = None) -> Span:
        return Span(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            parent_id=self.span_id,
            name=name,
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            status=status,
            attributes={
                "job": self.job,
                "version": self.version,
                "instance_id": self.instance_id,
                "provider": self.provider,
                "region": self.region,
                "gpu_type": self.gpu_type,
                "gpu_count": self.gpu_count,
                "ttl": self.ttl,
                "stream_key": self.stream_key,
                "trace_id": self.trace_id,
                "parent_span_id": self.span_id,
                **(attributes or {}),
            },
            resource={
                "service.name": "terradev",
                "service.instance.id": self.instance_id,
                "host.name": self.instance_id,
            },
        )

    def _attach(self) -> None:
        if self._active:
            return
        telemetry = get_telemetry()
        telemetry.attach_stream(self)
        self._active = True

    def _detach(self) -> None:
        if not self._active:
            return
        telemetry = get_telemetry()
        telemetry.detach_stream(self)
        self._active = False

    def start(self, attributes: Optional[Dict[str, Any]] = None) -> "NodeSpanStream":
        """Start the sidecar: create stream and attach to telemetry.

        Idempotent: if the stream is already active, this is a no-op.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("NodeSpanStream has already been closed")
            if self._active:
                logger.debug(f"NodeSpanStream {self.stream_key} already active")
                return self

            telemetry = get_telemetry()
            span = self._node_span("provision.start", "OK", attributes)

            # Always write to the local telemetry/JSONL trace first.
            telemetry.log_event("terradev.node_stream.start", attributes=span.to_dict())

            # Write to Redis if available.
            if self.exporter.is_available():
                self.exporter.start_stream(self.stream_key, span)
                logger.info(f"Started Redis span stream {self.stream_key}")

            self._attach()

            # Attach Telinea cloud connector if the user has configured an API key.
            # This is a no-op when TELINEA_API_KEY is not set, keeping the local
            # Redis stream air-gapped by default.
            if self._telinea_connector is None:
                self._telinea_connector = maybe_attach_telinea(self)

            # If requested, register an atexit handler so the stream is
            # gracefully closed when the sidecar process exits.
            if os.environ.get("TERRADEV_STREAM_END_ON_EXIT"):
                atexit.register(self.end)

        return self

    def emit(
        self,
        name: str,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """Emit a span while the node is alive."""
        if self._closed:
            logger.warning(f"Cannot emit to closed stream {self.stream_key}")
            return None
        span = self._node_span(name, status, attributes)

        if self.exporter.is_available():
            self.exporter.append_span(self.stream_key, span)

        return span

    def record_command(
        self,
        command: str,
        args: Optional[List[str]] = None,
        success: bool = True,
        returncode: int = 0,
        duration_ms: float = 0.0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """Record one command in the node's command chain.

        This is the primary integration point for imperative command chains.
        Any command that runs while the node stream is active is appended both
        to the in-memory command chain and to the Redis stream.
        """
        entry = {
            "command": command,
            "args": args or [],
            "success": success,
            "returncode": returncode,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(attributes or {}),
        }
        with self._lock:
            self.command_chain.append(entry)
        return self.emit(
            f"command.{command}",
            "OK" if success else "ERROR",
            {**entry, "chain_length": len(self.command_chain)},
        )

    def record_command_chain(
        self,
        chain: List[Dict[str, Any]],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """Record a pre-computed command chain in a single span."""
        with self._lock:
            self.command_chain.extend(chain)
        return self.emit(
            "command_chain",
            "OK",
            {"chain": chain, "chain_length": len(self.command_chain), **(attributes or {})},
        )

    def append_span(self, span: Any) -> bool:
        """Append an externally-created span to this node stream.

        Used by TerradevTelemetry to mirror all command spans into active
        node streams.
        """
        if self._closed:
            return False
        if self.exporter.is_available():
            self.exporter.append_span(self.stream_key, span)
            return True
        return False

    def heartbeat(self, attributes: Optional[Dict[str, Any]] = None) -> Optional[Span]:
        """Ping the pod/sequence by writing a heartbeat span and refreshing TTL."""
        if self._closed:
            return None
        span = self._node_span("heartbeat", "OK", attributes)
        if self.exporter.is_available():
            self.exporter.append_span(self.stream_key, span)
            self.exporter.refresh_ttl(self.stream_key)
        return span

    def ping(self, attributes: Optional[Dict[str, Any]] = None) -> tuple:
        """Ping the stream and return liveness status."""
        self.heartbeat(attributes)
        return self.is_up()

    def is_up(self, max_age_seconds: int = 120) -> tuple:
        """Check whether the linked pod/sequence is alive."""
        return self.exporter.is_stream_alive(self.stream_key, max_age_seconds=max_age_seconds)

    def start_heartbeat(self, interval: int = 30) -> "NodeSpanStream":
        """Start a background daemon thread that pings the stream.

        The heartbeat keeps the stream TTL fresh and gives downstream consumers
        a real-time liveness signal. It stops automatically when the stream is
        closed or the process exits.
        """
        if self._closed or self._heartbeat_thread is not None:
            return self

        def _loop():
            while not self._heartbeat_stop.is_set() and not self._closed:
                self.heartbeat({"heartbeat_interval_seconds": interval})
                self._heartbeat_stop.wait(interval)

        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=_loop,
            name=f"terradev-heartbeat-{self.instance_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        return self

    def _stop_heartbeat(self) -> None:
        """Stop the background heartbeat, if any."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None

    def end(
        self,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None,
        tombstone_ttl: int = 60,
    ) -> "NodeSpanStream":
        """End the sidecar gracefully, leaving a short tombstone."""
        with self._lock:
            if self._closed:
                return self
            self._closed = True
            self._stop_heartbeat()
            self._detach()

            summary_attrs = {
                "command_chain": self.command_chain,
                "command_count": len(self.command_chain),
                **(attributes or {}),
            }
            span = self._node_span("provision.end", status, summary_attrs)
            telemetry = get_telemetry()
            telemetry.log_event("terradev.node_stream.end", attributes=span.to_dict())

            if self.exporter.is_available():
                self.exporter.end_stream(self.stream_key, span, tombstone_ttl=tombstone_ttl)
                logger.info(f"Ended Redis span stream {self.stream_key}")

            # Flush any queued Telinea telemetry. Fail-safe wrapper.
            if self._telinea_connector is not None:
                try:
                    self._telinea_connector.close(extra={"stream_status": "ended"})
                except Exception:  # noqa: BLE001
                    logger.debug("Telinea connector close failed", exc_info=True)

        return self

    def destroy(self, attributes: Optional[Dict[str, Any]] = None) -> "NodeSpanStream":
        """End the sidecar and delete the Redis stream."""
        if self._closed:
            self.exporter.delete_stream(self.stream_key)
            return self
        self._closed = True
        self._detach()

        span = self._node_span("destroy", "OK", attributes)
        telemetry = get_telemetry()
        telemetry.log_event("terradev.node_stream.destroy", attributes=span.to_dict())

        if self.exporter.is_available():
            self.exporter.append_span(self.stream_key, span)
            self.exporter.delete_stream(self.stream_key)
            logger.info(f"Destroyed Redis span stream {self.stream_key}")

        # Flush any queued Telinea telemetry. Fail-safe wrapper.
        if self._telinea_connector is not None:
            try:
                self._telinea_connector.close(extra={"stream_status": "destroyed"})
            except Exception:  # noqa: BLE001
                logger.debug("Telinea connector close failed", exc_info=True)

        return self

    def __enter__(self) -> "NodeSpanStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        status = "ERROR" if exc is not None else "OK"
        if exc is not None:
            telemetry = get_telemetry()
            telemetry.log_error(
                TerradevError(
                    code=ErrorCode.UNKNOWN,
                    message=str(exc),
                    category=ErrorCategory.INTERNAL,
                    severity=Severity.ERROR,
                    context={"stream_key": self.stream_key, "instance_id": self.instance_id},
                )
            )
        self.end(status=status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job": self.job,
            "version": self.version,
            "instance_id": self.instance_id,
            "provider": self.provider,
            "region": self.region,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "stream_key": self.stream_key,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "ttl": self.ttl,
            "active": self.active,
            "closed": self._closed,
            "redis_available": self.exporter.is_available(),
        }
