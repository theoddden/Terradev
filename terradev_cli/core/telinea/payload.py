"""Payload builder for Telinea telemetry ingestion.

Turns the local span stream, command chain, and execution metadata into a
structured JSON payload that the Telinea dashboard can render.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..telemetry import TerradevTelemetry
from .config import TelineaConfig


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_stream_payload(
    stream: Any,
    spans: List[Dict[str, Any]],
    config: Optional[TelineaConfig] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a Telinea payload from a NodeSpanStream and a list of span dicts.

    This deliberately copies only OTel-compatible span data and redacted
    command metadata. No raw secrets are included.
    """
    from ..node_span_stream import NodeSpanStream

    if not isinstance(stream, NodeSpanStream):
        # Defensive: if a generic object is passed, extract what we can.
        return {
            "source": "terradev-cli",
            "ingested_at": _now(),
            "spans": spans,
            "metadata": extra or {},
        }

    payload: Dict[str, Any] = {
        "source": "terradev-cli",
        "ingested_at": _now(),
        "event_type": "node_stream",
        "trace_id": stream.trace_id,
        "span_id": stream.span_id,
        "stream_key": stream.stream_key,
        "workspace_id": getattr(config, "workspace_id", None) if config else None,
        "project_id": getattr(config, "project_id", None) if config else None,
        "node": {
            "job": stream.job,
            "version": stream.version,
            "instance_id": stream.instance_id,
            "provider": stream.provider,
            "region": stream.region,
            "gpu_type": stream.gpu_type,
            "gpu_count": stream.gpu_count,
            "ttl": stream.ttl,
            "active": stream.active,
            "closed": stream._closed,
        },
        "command_chain": list(getattr(stream, "command_chain", [])),
        "spans": spans,
    }
    if extra:
        payload["metadata"] = extra
    return payload


def build_telemetry_payload(
    telemetry: TerradevTelemetry,
    config: Optional[TelineaConfig] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a payload from the in-memory TerradevTelemetry summary.

    Used as a fallback when no NodeSpanStream is active, or as a heartbeat
    when a command finishes.
    """
    summary = telemetry.get_trace_summary()
    spans = [span.to_dict() for span in telemetry._spans]
    payload: Dict[str, Any] = {
        "source": "terradev-cli",
        "ingested_at": _now(),
        "event_type": "telemetry_summary",
        "workspace_id": getattr(config, "workspace_id", None) if config else None,
        "project_id": getattr(config, "project_id", None) if config else None,
        "trace_id": summary.get("trace_id"),
        "spans_in_memory": summary.get("spans_in_memory"),
        "span_stack_depth": summary.get("span_stack_depth"),
        "pending_events": summary.get("pending_events"),
        "flush_count": summary.get("flush_count"),
        "enabled": summary.get("enabled"),
        "trace_dir": str(summary.get("trace_dir", "")),
        "spans": spans,
    }
    if extra:
        payload["metadata"] = extra
    return payload
