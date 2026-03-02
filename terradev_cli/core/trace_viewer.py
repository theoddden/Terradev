#!/usr/bin/env python3
"""
Trace Viewer — renders Phoenix span trees in terminal.
Queries Phoenix REST API and displays execution chains like `terradev status`.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _duration_ms(start: Optional[str], end: Optional[str]) -> Optional[float]:
    s, e = _parse_iso(start), _parse_iso(end)
    if s and e:
        return (e - s).total_seconds() * 1000
    return None


def build_span_tree(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a tree structure from flat span list.

    Each span has context.trace_id, context.span_id, parent_id.
    Returns list of root spans with 'children' key populated.
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    for span in spans:
        sid = span.get("context", {}).get("span_id", span.get("id", ""))
        span["_id"] = sid
        span["children"] = []
        by_id[sid] = span

    roots: List[Dict[str, Any]] = []
    for span in spans:
        pid = span.get("parent_id")
        if pid and pid in by_id:
            by_id[pid]["children"].append(span)
        else:
            roots.append(span)

    return roots


def render_span_tree(roots: List[Dict[str, Any]], indent: int = 0) -> str:
    """Render span tree as indented text for terminal output."""
    lines: List[str] = []
    for span in roots:
        name = span.get("name", "unknown")
        kind = span.get("span_kind", "")
        status = span.get("status_code", "OK")
        dur = _duration_ms(span.get("start_time"), span.get("end_time"))
        dur_str = f"{dur:.1f}ms" if dur is not None else "?"

        # Status indicator
        icon = "✅" if status in ("OK", "UNSET") else "❌"
        prefix = "  " * indent

        line = f"{prefix}{icon} {name}"
        if kind:
            line += f" [{kind}]"
        line += f" ({dur_str})"

        # Show error message if present
        if span.get("status_message"):
            line += f" — {span['status_message']}"

        lines.append(line)

        # Recurse into children
        if span.get("children"):
            lines.append(render_span_tree(span["children"], indent + 1))

    return "\n".join(lines)


def format_trace_summary(spans: List[Dict[str, Any]]) -> str:
    """Format a summary of a trace for terminal display."""
    if not spans:
        return "No spans found."

    trace_id = spans[0].get("context", {}).get("trace_id", "unknown")
    total = len(spans)
    errors = sum(1 for s in spans if s.get("status_code") == "ERROR")
    kinds = {}
    for s in spans:
        k = s.get("span_kind", "UNKNOWN")
        kinds[k] = kinds.get(k, 0) + 1

    # Total trace duration
    starts = [_parse_iso(s.get("start_time")) for s in spans]
    ends = [_parse_iso(s.get("end_time")) for s in spans]
    starts = [s for s in starts if s]
    ends = [e for e in ends if e]
    total_ms = None
    if starts and ends:
        total_ms = (max(ends) - min(starts)).total_seconds() * 1000

    lines = [
        f"🔍 Trace: {trace_id}",
        f"   Spans: {total} ({errors} errors)",
        f"   Duration: {total_ms:.1f}ms" if total_ms else "   Duration: unknown",
        f"   Span kinds: {', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))}",
        "",
    ]

    roots = build_span_tree(spans)
    lines.append(render_span_tree(roots))
    return "\n".join(lines)


async def view_trace(phoenix_service, trace_id: str, project: Optional[str] = None) -> str:
    """Fetch and render a trace from Phoenix."""
    data = await phoenix_service.get_trace(trace_id, project_identifier=project)
    spans = data.get("data", [])
    return format_trace_summary(spans)


async def view_recent_spans(
    phoenix_service, *,
    project: Optional[str] = None,
    limit: int = 20,
    filter_condition: Optional[str] = None,
) -> str:
    """Fetch and render recent spans."""
    data = await phoenix_service.list_spans(
        project, limit=limit, filter_condition=filter_condition,
    )
    spans = data.get("data", [])
    if not spans:
        return "No spans found."

    lines = [f"📊 Recent spans ({len(spans)} shown):"]
    for s in spans:
        name = s.get("name", "?")
        kind = s.get("span_kind", "?")
        status = s.get("status_code", "OK")
        dur = _duration_ms(s.get("start_time"), s.get("end_time"))
        dur_str = f"{dur:.1f}ms" if dur is not None else "?"
        icon = "✅" if status in ("OK", "UNSET") else "❌"
        tid = s.get("context", {}).get("trace_id", "?")[:12]
        lines.append(f"  {icon} {name} [{kind}] {dur_str} trace={tid}…")

    cursor = data.get("next_cursor")
    if cursor:
        lines.append(f"\n  ▶ More results available (cursor: {cursor[:16]}…)")
    return "\n".join(lines)
