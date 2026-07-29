"""Tests for terradev_cli.core.trace_viewer.

Trace viewer converts flat Phoenix-style span lists into readable trees.
"""

from datetime import datetime, timedelta, timezone

from terradev_cli.core.trace_viewer import (
    build_span_tree,
    format_trace_summary,
    render_span_tree,
    _duration_ms,
    _parse_iso,
)


def test_parse_iso():
    """_parse_iso handles ISO and Zulu timestamps."""
    assert _parse_iso("2024-01-01T12:00:00Z") == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert _parse_iso("2024-01-01T12:00:00+00:00") == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert _parse_iso(None) is None
    assert _parse_iso("not a date") is None


def test_duration_ms():
    """_duration_ms returns milliseconds between two timestamps."""
    start = "2024-01-01T12:00:00.000Z"
    end = "2024-01-01T12:00:01.500Z"
    assert _duration_ms(start, end) == 1500.0
    assert _duration_ms(start, None) is None


def test_build_span_tree():
    """build_span_tree links child spans to parents."""
    spans = [
        {"context": {"span_id": "root", "trace_id": "t1"}, "name": "root"},
        {"context": {"span_id": "child1", "trace_id": "t1"}, "parent_id": "root", "name": "child1"},
        {"context": {"span_id": "child2", "trace_id": "t1"}, "parent_id": "root", "name": "child2"},
    ]
    roots = build_span_tree(spans)
    assert len(roots) == 1
    assert roots[0]["_id"] == "root"
    assert len(roots[0]["children"]) == 2


def test_build_span_tree_missing_parent():
    """Spans with missing parents become roots."""
    spans = [
        {"context": {"span_id": "a"}, "name": "a"},
        {"context": {"span_id": "b"}, "parent_id": "missing", "name": "b"},
    ]
    roots = build_span_tree(spans)
    assert len(roots) == 2


def test_render_span_tree():
    """render_span_tree produces an indented string with durations."""
    root = {
        "_id": "root",
        "name": "root",
        "span_kind": "SERVER",
        "status_code": "OK",
        "start_time": "2024-01-01T12:00:00.000Z",
        "end_time": "2024-01-01T12:00:00.100Z",
        "children": [
            {
                "_id": "child",
                "name": "child",
                "status_code": "OK",
                "start_time": "2024-01-01T12:00:00.010Z",
                "end_time": "2024-01-01T12:00:00.050Z",
                "children": [],
            }
        ],
    }
    text = render_span_tree([root])
    assert "root" in text
    assert "SERVER" in text
    assert "100.0ms" in text or "100.0" in text
    assert "child" in text


def test_render_span_tree_error():
    """Error status is rendered differently."""
    root = {
        "_id": "root",
        "name": "fail",
        "status_code": "ERROR",
        "status_message": "boom",
        "children": [],
    }
    text = render_span_tree([root])
    assert "fail" in text
    assert "boom" in text


def test_format_trace_summary():
    """format_trace_summary builds a complete trace overview."""
    spans = [
        {
            "context": {"trace_id": "trace-1"},
            "name": "root",
            "span_kind": "SERVER",
            "status_code": "OK",
            "start_time": "2024-01-01T12:00:00.000Z",
            "end_time": "2024-01-01T12:00:00.500Z",
        },
        {
            "context": {"trace_id": "trace-1"},
            "name": "child",
            "span_kind": "CLIENT",
            "status_code": "ERROR",
            "start_time": "2024-01-01T12:00:00.100Z",
            "end_time": "2024-01-01T12:00:00.400Z",
        },
    ]
    summary = format_trace_summary(spans)
    assert "trace-1" in summary
    assert "Spans: 2" in summary
    assert "(1 errors)" in summary
    assert "root" in summary
    assert "CLIENT" in summary


def test_format_trace_summary_empty():
    """An empty span list returns a no-spans message."""
    assert format_trace_summary([]) == "No spans found."
