"""Tests for terradev_cli.core.drift_detector.

Drift detection and idempotent re-provision keep client clusters in sync with
their declared manifests. These tests cover detection, fixing, and rollback.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from terradev_cli.core.drift_detector import DriftDetector, DriftReport
from terradev_cli.core.manifest_cache import Manifest, ManifestCache, ManifestNode


def _node(pod_id, provider="demo", status="running", gpus=1, gpu_type="A100", region="us-east"):
    return ManifestNode(
        provider=provider,
        pod_id=pod_id,
        instance_id=f"i-{pod_id}",
        gpus=gpus,
        gpu_type=gpu_type,
        region=region,
        status=status,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        ttl="3600",
    )


def _manifest(tmp_path, nodes):
    cache = ManifestCache(str(tmp_path))
    manifest = Manifest(
        job="test-job",
        version="v1",
        nodes=nodes,
        dataset_hash="sha256:abc",
        ttl="3600",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        metadata={},
    )
    cache.store_manifest(manifest)
    return cache


def _instances_from_nodes(nodes):
    return [
        {
            "provider": n.provider,
            "pod_id": n.pod_id,
            "instance_id": n.instance_id,
            "status": n.status,
            "gpus": n.gpus,
            "gpu_type": n.gpu_type,
            "region": n.region,
        }
        for n in nodes
    ]


def _make_detector(tmp_path, manifest_nodes):
    """Create a detector with a manifest and a mocked provider factory."""
    _manifest(tmp_path, manifest_nodes)
    detector = DriftDetector(str(tmp_path))

    provider = AsyncMock()
    provider.list_instances = AsyncMock(return_value=[])
    provider.terminate_instance = AsyncMock(return_value={"terminated": True})
    provider.provision_instance = AsyncMock(return_value={"instance_id": "i-new"})

    detector.provider_factory = MagicMock()
    detector.provider_factory.get_provider = MagicMock(return_value=provider)
    return detector


def _set_actual_state(detector, instances):
    """Override _query_provider to return a fixed list of instances."""
    detector._query_provider = AsyncMock(return_value=instances)


@pytest.mark.asyncio
async def test_no_drift_when_state_matches(tmp_path):
    """No drift reported when actual state matches the manifest."""
    nodes = [_node("pod-1")]
    detector = _make_detector(tmp_path, nodes)
    _set_actual_state(detector, _instances_from_nodes(nodes))

    report = await detector.detect_drift("test-job")

    assert isinstance(report, DriftReport)
    assert report.drifted_nodes == []
    assert report.missing_nodes == []
    assert report.extra_nodes == []


@pytest.mark.asyncio
async def test_missing_node_reported(tmp_path):
    """A missing node is reported in the drift report."""
    nodes = [_node("pod-1"), _node("pod-2")]
    actual = _instances_from_nodes([_node("pod-1")])

    detector = _make_detector(tmp_path, nodes)
    _set_actual_state(detector, actual)

    report = await detector.detect_drift("test-job")

    assert len(report.missing_nodes) == 1
    assert report.missing_nodes[0].pod_id == "pod-2"


@pytest.mark.asyncio
async def test_extra_node_reported(tmp_path):
    """Nodes in actual state but not in the manifest are reported as extra."""
    nodes = [_node("pod-1")]
    actual = _instances_from_nodes(nodes) + [
        {
            "provider": "demo",
            "pod_id": "pod-orphan",
            "instance_id": "i-orphan",
            "status": "running",
            "gpus": 1,
            "gpu_type": "A100",
            "region": "us-east",
        }
    ]

    detector = _make_detector(tmp_path, nodes)
    _set_actual_state(detector, actual)

    report = await detector.detect_drift("test-job")

    assert len(report.extra_nodes) == 1
    assert report.extra_nodes[0]["pod_id"] == "pod-orphan"


@pytest.mark.asyncio
async def test_drifted_node_detected(tmp_path):
    """A node whose status changed is marked as drifted."""
    nodes = [_node("pod-1", status="running")]
    actual = _instances_from_nodes([_node("pod-1", status="terminated")])

    detector = _make_detector(tmp_path, nodes)
    _set_actual_state(detector, actual)

    report = await detector.detect_drift("test-job")

    assert len(report.drifted_nodes) == 1
    assert report.drifted_nodes[0].pod_id == "pod-1"


@pytest.mark.asyncio
async def test_fix_drift_terminates_and_recreates(tmp_path):
    """fix_drift terminates drifted nodes and recreates missing ones."""
    nodes = [_node("pod-1", status="running"), _node("pod-2")]

    # Initial state: pod-1 has drifted to terminated, pod-2 is missing
    actual_before = [
        {
            "provider": "demo",
            "pod_id": "pod-1",
            "instance_id": "i-pod-1",
            "status": "terminated",
            "gpus": 1,
            "gpu_type": "A100",
            "region": "us-east",
        }
    ]

    # After fix, both nodes are present and running
    actual_after = _instances_from_nodes([_node("pod-1", status="running"), _node("pod-2")])

    detector = _make_detector(tmp_path, nodes)
    call_count = 0

    async def _query_provider(provider_name, manifest_nodes):
        nonlocal call_count
        call_count += 1
        return actual_after if call_count > 1 else actual_before

    detector._query_provider = _query_provider

    result = await detector.fix_drift("test-job")
    assert result["status"] == "fixed"
    assert result["terminated"] == 1
    assert result["recreated"] == 1


@pytest.mark.asyncio
async def test_detect_drift_missing_manifest_raises(tmp_path):
    """Detecting drift without a stored manifest raises ValueError."""
    detector = DriftDetector(str(tmp_path))
    with pytest.raises(ValueError, match="Manifest not found"):
        await detector.detect_drift("missing-job")
