"""Tests for terradev_cli.core.monitoring.

MetricsCollector is a lightweight in-memory metrics sink used by the
optimization modules.
"""

from terradev_cli.core.monitoring import MetricsCollector


def test_record_and_get_metrics():
    """Metrics can be recorded and retrieved."""
    collector = MetricsCollector()
    collector.record("latency_ms", 42.0, {"model": "m1"})
    collector.record("latency_ms", 55.0, {"model": "m1"})

    metrics = collector.get_metrics()
    assert len(metrics) == 2
    assert metrics[0]["name"] == "latency_ms"
    assert metrics[0]["value"] == 42.0
    assert metrics[0]["tags"] == {"model": "m1"}
    assert "ts" in metrics[0]


def test_increment():
    """increment records a value of 1.0."""
    collector = MetricsCollector()
    collector.increment("requests")
    metrics = collector.get_metrics()
    assert len(metrics) == 1
    assert metrics[0]["name"] == "requests"
    assert metrics[0]["value"] == 1.0


def test_reset():
    """reset clears all metrics."""
    collector = MetricsCollector()
    collector.record("x", 1.0)
    assert len(collector.get_metrics()) == 1

    collector.reset()
    assert collector.get_metrics() == []
