"""Tests for terradev_cli.core.trace_viewer — pure data-processing functions."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from terradev_cli.core.trace_viewer import (
    _parse_iso,
    _duration_ms,
    build_span_tree,
    render_span_tree,
    format_trace_summary,
    view_trace,
    view_recent_spans,
)


# ── _parse_iso ─────────────────────────────────────────────────────────────

class TestParseIso:
    def test_valid_utc_timestamp(self):
        dt = _parse_iso("2024-01-15T10:30:00Z")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1

    def test_valid_offset_timestamp(self):
        dt = _parse_iso("2024-01-15T10:30:00+00:00")
        assert dt is not None

    def test_none_returns_none(self):
        assert _parse_iso(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_iso("") is None

    def test_invalid_format_returns_none(self):
        assert _parse_iso("not-a-date") is None


# ── _duration_ms ───────────────────────────────────────────────────────────

class TestDurationMs:
    def test_one_second(self):
        dur = _duration_ms("2024-01-15T10:00:00Z", "2024-01-15T10:00:01Z")
        assert dur == pytest.approx(1000.0)

    def test_none_start_returns_none(self):
        assert _duration_ms(None, "2024-01-15T10:00:01Z") is None

    def test_none_end_returns_none(self):
        assert _duration_ms("2024-01-15T10:00:00Z", None) is None

    def test_both_none_returns_none(self):
        assert _duration_ms(None, None) is None

    def test_half_second(self):
        dur = _duration_ms("2024-01-15T10:00:00Z", "2024-01-15T10:00:00.500Z")
        assert dur == pytest.approx(500.0, abs=1.0)


# ── build_span_tree ────────────────────────────────────────────────────────

class TestBuildSpanTree:
    def test_empty_returns_empty(self):
        assert build_span_tree([]) == []

    def test_single_root_span(self):
        spans = [{"context": {"span_id": "abc"}, "parent_id": None, "name": "root"}]
        roots = build_span_tree(spans)
        assert len(roots) == 1
        assert roots[0]["name"] == "root"
        assert roots[0]["children"] == []

    def test_parent_child_relationship(self):
        spans = [
            {"context": {"span_id": "parent"}, "parent_id": None, "name": "parent"},
            {"context": {"span_id": "child"}, "parent_id": "parent", "name": "child"},
        ]
        roots = build_span_tree(spans)
        assert len(roots) == 1
        assert len(roots[0]["children"]) == 1
        assert roots[0]["children"][0]["name"] == "child"

    def test_multiple_roots(self):
        spans = [
            {"context": {"span_id": "a"}, "parent_id": None, "name": "span_a"},
            {"context": {"span_id": "b"}, "parent_id": None, "name": "span_b"},
        ]
        roots = build_span_tree(spans)
        assert len(roots) == 2

    def test_orphan_child_becomes_root(self):
        spans = [
            {"context": {"span_id": "child"}, "parent_id": "nonexistent", "name": "orphan"},
        ]
        roots = build_span_tree(spans)
        assert len(roots) == 1
        assert roots[0]["name"] == "orphan"

    def test_deep_nesting(self):
        spans = [
            {"context": {"span_id": "l1"}, "parent_id": None, "name": "level1"},
            {"context": {"span_id": "l2"}, "parent_id": "l1", "name": "level2"},
            {"context": {"span_id": "l3"}, "parent_id": "l2", "name": "level3"},
        ]
        roots = build_span_tree(spans)
        assert len(roots) == 1
        assert len(roots[0]["children"]) == 1
        assert len(roots[0]["children"][0]["children"]) == 1

    def test_fallback_to_id_key(self):
        spans = [{"id": "abc", "parent_id": None, "name": "span"}]
        roots = build_span_tree(spans)
        assert len(roots) == 1


# ── render_span_tree ───────────────────────────────────────────────────────

class TestRenderSpanTree:
    def test_empty_returns_empty_string(self):
        assert render_span_tree([]) == ""

    def test_renders_span_name(self):
        spans = build_span_tree([
            {"context": {"span_id": "x"}, "parent_id": None, "name": "my-span",
             "status_code": "OK", "start_time": "2024-01-15T10:00:00Z",
             "end_time": "2024-01-15T10:00:01Z"},
        ])
        result = render_span_tree(spans)
        assert "my-span" in result
        assert "✅" in result
        assert "1000.0ms" in result

    def test_error_span_shows_x(self):
        spans = build_span_tree([
            {"context": {"span_id": "x"}, "parent_id": None, "name": "err-span",
             "status_code": "ERROR", "start_time": None, "end_time": None},
        ])
        result = render_span_tree(spans)
        assert "❌" in result

    def test_span_kind_included(self):
        spans = build_span_tree([
            {"context": {"span_id": "x"}, "parent_id": None, "name": "span",
             "span_kind": "SERVER", "status_code": "OK",
             "start_time": None, "end_time": None},
        ])
        result = render_span_tree(spans)
        assert "[SERVER]" in result

    def test_status_message_included(self):
        spans = build_span_tree([
            {"context": {"span_id": "x"}, "parent_id": None, "name": "span",
             "status_code": "ERROR", "status_message": "connection refused",
             "start_time": None, "end_time": None},
        ])
        result = render_span_tree(spans)
        assert "connection refused" in result

    def test_child_indented(self):
        spans = build_span_tree([
            {"context": {"span_id": "p"}, "parent_id": None, "name": "parent",
             "status_code": "OK", "start_time": None, "end_time": None},
            {"context": {"span_id": "c"}, "parent_id": "p", "name": "child",
             "status_code": "OK", "start_time": None, "end_time": None},
        ])
        result = render_span_tree(spans)
        lines = result.split("\n")
        parent_line = next(l for l in lines if "parent" in l)
        child_line = next(l for l in lines if "child" in l)
        assert child_line.startswith("  ")
        assert not parent_line.startswith("  ")

    def test_unknown_duration_shown_as_question_mark(self):
        spans = build_span_tree([
            {"context": {"span_id": "x"}, "parent_id": None, "name": "span",
             "status_code": "OK", "start_time": None, "end_time": None},
        ])
        result = render_span_tree(spans)
        assert "(?)" in result


# ── format_trace_summary ───────────────────────────────────────────────────

class TestFormatTraceSummary:
    def test_empty_spans(self):
        result = format_trace_summary([])
        assert "No spans found" in result

    def test_trace_id_shown(self):
        spans = [{
            "context": {"trace_id": "abc123", "span_id": "s1"},
            "parent_id": None,
            "name": "root",
            "status_code": "OK",
            "start_time": "2024-01-15T10:00:00Z",
            "end_time": "2024-01-15T10:00:01Z",
        }]
        result = format_trace_summary(spans)
        assert "abc123" in result

    def test_span_count_shown(self):
        spans = [
            {"context": {"trace_id": "t1", "span_id": "s1"}, "parent_id": None,
             "name": "a", "status_code": "OK", "start_time": None, "end_time": None},
            {"context": {"trace_id": "t1", "span_id": "s2"}, "parent_id": None,
             "name": "b", "status_code": "OK", "start_time": None, "end_time": None},
        ]
        result = format_trace_summary(spans)
        assert "2" in result

    def test_error_count_shown(self):
        spans = [
            {"context": {"trace_id": "t1", "span_id": "s1"}, "parent_id": None,
             "name": "ok", "status_code": "OK", "start_time": None, "end_time": None},
            {"context": {"trace_id": "t1", "span_id": "s2"}, "parent_id": None,
             "name": "err", "status_code": "ERROR", "start_time": None, "end_time": None},
        ]
        result = format_trace_summary(spans)
        assert "1 errors" in result

    def test_span_kinds_shown(self):
        spans = [{
            "context": {"trace_id": "t1", "span_id": "s1"},
            "parent_id": None,
            "name": "span",
            "status_code": "OK",
            "span_kind": "SERVER",
            "start_time": None,
            "end_time": None,
        }]
        result = format_trace_summary(spans)
        assert "SERVER" in result

    def test_total_duration_calculated(self):
        spans = [{
            "context": {"trace_id": "t1", "span_id": "s1"},
            "parent_id": None,
            "name": "span",
            "status_code": "OK",
            "start_time": "2024-01-15T10:00:00Z",
            "end_time": "2024-01-15T10:00:02Z",
        }]
        result = format_trace_summary(spans)
        assert "2000.0ms" in result


# ── view_trace / view_recent_spans (async, service mocked) ─────────────────

class TestViewTrace:
    async def test_view_trace_calls_service(self):
        svc = MagicMock()
        svc.get_trace = AsyncMock(return_value={"data": []})
        result = await view_trace(svc, "trace-123")
        svc.get_trace.assert_called_once_with("trace-123", project_identifier=None)
        assert "No spans found" in result

    async def test_view_trace_with_project(self):
        svc = MagicMock()
        svc.get_trace = AsyncMock(return_value={"data": []})
        await view_trace(svc, "trace-abc", project="my-project")
        svc.get_trace.assert_called_once_with("trace-abc", project_identifier="my-project")


class TestViewRecentSpans:
    async def test_no_spans_returns_message(self):
        svc = MagicMock()
        svc.list_spans = AsyncMock(return_value={"data": []})
        result = await view_recent_spans(svc)
        assert "No spans found" in result

    async def test_spans_listed(self):
        svc = MagicMock()
        svc.list_spans = AsyncMock(return_value={
            "data": [{
                "name": "my-span",
                "span_kind": "CLIENT",
                "status_code": "OK",
                "start_time": "2024-01-15T10:00:00Z",
                "end_time": "2024-01-15T10:00:01Z",
                "context": {"trace_id": "abc123def456"},
            }]
        })
        result = await view_recent_spans(svc)
        assert "my-span" in result
        assert "CLIENT" in result
        assert "✅" in result

    async def test_error_span_shows_x(self):
        svc = MagicMock()
        svc.list_spans = AsyncMock(return_value={
            "data": [{
                "name": "bad-span",
                "span_kind": "SERVER",
                "status_code": "ERROR",
                "start_time": None,
                "end_time": None,
                "context": {"trace_id": "abc123def456"},
            }]
        })
        result = await view_recent_spans(svc)
        assert "❌" in result

    async def test_cursor_shown_when_present(self):
        svc = MagicMock()
        svc.list_spans = AsyncMock(return_value={
            "data": [{
                "name": "s", "span_kind": "X", "status_code": "OK",
                "start_time": None, "end_time": None,
                "context": {"trace_id": "abc123def456"},
            }],
            "next_cursor": "cursor-value-here-xyz",
        })
        result = await view_recent_spans(svc)
        assert "cursor" in result.lower()

    async def test_filter_and_limit_passed(self):
        svc = MagicMock()
        svc.list_spans = AsyncMock(return_value={"data": []})
        await view_recent_spans(svc, project="proj", limit=5, filter_condition="status=ERROR")
        svc.list_spans.assert_called_once_with("proj", limit=5, filter_condition="status=ERROR")
