"""Tests for terradev_cli.core.inference_router.

The inference router is the latency-aware, KV-cache-aware serving layer.
These tests cover the prefix cache index and prefill/decode handoff tracker.
"""

import time

import pytest

from terradev_cli.core.inference_router import (
    EndpointHealth,
    EndpointPhase,
    KVConnectorType,
    PrefixCacheIndex,
    PrefillDecodeTracker,
)


@pytest.fixture
def prefix_index():
    return PrefixCacheIndex(max_entries=100, prefix_tokens=4)


def test_prefix_index_records_and_lookups(prefix_index):
    """The index records prefixes and finds matching endpoints."""
    prompt = "The quick brown fox jumps over the lazy dog"
    prefix_index.record(prompt, "endpoint-1")

    matches = prefix_index.lookup(prompt)
    assert len(matches) == 1
    assert matches[0][0] == "endpoint-1"
    assert 0.0 <= matches[0][1] <= 1.0


def test_prefix_index_evicts_old_prefixes(prefix_index):
    """LRU eviction removes the oldest prefix when capacity is reached."""
    for i in range(101):
        prefix_index.record(f"prompt number {i}", f"endpoint-{i}")

    assert prefix_index.size == 100


def test_prefix_index_evicts_endpoint(prefix_index):
    """When an endpoint is evicted, it is removed from all prefixes."""
    prefix_index.record("hello world", "a")
    prefix_index.record("hello again", "a")
    prefix_index.record("goodbye", "b")

    prefix_index.evict_endpoint("a")
    assert prefix_index.lookup("hello world") == []
    assert prefix_index.lookup("hello again") == []
    assert prefix_index.lookup("goodbye") != []


def test_prefix_index_freshness_decay(prefix_index):
    """Old entries are skipped when they exceed max_age_s."""
    old_prompt = "old prompt"
    prefix_index.record(old_prompt, "endpoint-1")
    time.sleep(0.1)
    matches = prefix_index.lookup(old_prompt, max_age_s=0.05)
    assert matches == []


@pytest.fixture
def tracker():
    return PrefillDecodeTracker(max_links=100)


def test_tracker_records_handoff(tracker):
    """A handoff links a prefill endpoint to a decode endpoint."""
    tracker.record_handoff("prefill-1", "decode-1", "llama-3-8b", transfer_ms=0.5)
    assert tracker.get_decode_for_prefill("prefill-1", "llama-3-8b") == "decode-1"


def test_tracker_get_prefill_for_decode(tracker):
    """The tracker can find the prefill endpoint associated with a decode endpoint."""
    tracker.record_handoff("prefill-1", "decode-1", "llama-3-8b")
    assert tracker.get_prefill_for_decode("decode-1", "llama-3-8b") == "prefill-1"


def test_tracker_lru_eviction(tracker):
    """Old handoff links are evicted when the cap is reached."""
    for i in range(101):
        tracker.record_handoff(f"prefill-{i}", f"decode-{i}", f"model-{i}")
    assert len(tracker._links) == 100


def test_tracker_get_best_decode_prefers_nixl_rdma(tracker):
    """get_best_decode_by_transport prefers NIXL with active RDMA."""
    tracker.record_handoff(
        "prefill-1", "decode-lmcache", "model", transfer_ms=5.0, kv_connector="LMCacheConnector"
    )
    tracker.record_handoff(
        "prefill-2",
        "decode-nixl",
        "model",
        transfer_ms=0.5,
        kv_connector="NixlConnector",
        rdma_active=True,
    )

    best = tracker.get_best_decode_by_transport("model")
    assert best == "decode-nixl"


def test_tracker_connector_summary(tracker):
    """The connector summary aggregates handoff statistics."""
    tracker.record_handoff(
        "prefill-1",
        "decode-1",
        "model",
        transfer_ms=0.5,
        kv_connector="NixlConnector",
        rdma_active=True,
        kv_transfer_bytes=1024,
    )
    summary = tracker.get_connector_summary()
    assert summary["total_active_links"] == 1
    assert summary["rdma_active_count"] == 1
    assert summary["total_transfer_bytes"] == 1024
    assert "NixlConnector" in summary["connector_types"]


def test_tracker_link_details(tracker):
    """get_link_details returns a serialized handoff record."""
    tracker.record_handoff("prefill-1", "decode-1", "model")
    details = tracker.get_link_details("prefill-1", "model")
    assert details is not None
    assert details["prefill_endpoint_id"] == "prefill-1"
    assert details["decode_endpoint_id"] == "decode-1"
    assert details["handoff_count"] == 1


def test_endpoint_enums_values():
    """Endpoint health and phase enums have the expected string values."""
    assert EndpointHealth.HEALTHY.value == "healthy"
    assert EndpointPhase.PREFILL.value == "prefill"
