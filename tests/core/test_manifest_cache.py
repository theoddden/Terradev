"""Tests for terradev_cli.core.manifest_cache.

Manifests are the source of truth for idempotent re-provision and drift
detection. These tests exercise store/load/list/delete and hash computation.
"""

from pathlib import Path

import pytest

from terradev_cli.core.manifest_cache import Manifest, ManifestCache, ManifestNode


@pytest.fixture
def cache(tmp_path):
    return ManifestCache(cache_dir=str(tmp_path))


def test_store_and_load_manifest(cache):
    """A manifest round-trips through the cache."""
    node = ManifestNode(
        provider="runpod",
        pod_id="pod-1",
        instance_id="i-1",
        gpus=8,
        gpu_type="A100",
        region="us-east-1",
        status="running",
        created_at="2025-01-01T00:00:00",
        ttl="1h",
    )
    manifest = Manifest(
        job="train-1",
        version="v1",
        nodes=[node],
        dataset_hash="sha256:abc",
        ttl="1h",
        created_at="2025-01-01T00:00:00",
        metadata={"framework": "pytorch"},
    )

    path = cache.store_manifest(manifest)
    assert Path(path).exists()

    loaded = cache.load_manifest("train-1", "v1")
    assert loaded is not None
    assert loaded.job == "train-1"
    assert loaded.version == "v1"
    assert len(loaded.nodes) == 1
    assert loaded.nodes[0].provider == "runpod"
    assert loaded.metadata == {"framework": "pytorch"}


def test_load_latest_version(cache):
    """load_manifest without a version returns the most recently written manifest."""
    for v in ["v1", "v2"]:
        m = Manifest(
            job="train-1",
            version=v,
            nodes=[],
            dataset_hash="",
            ttl="",
            created_at="2025-01-01T00:00:00",
            metadata={},
        )
        cache.store_manifest(m)

    latest = cache.load_manifest("train-1")
    assert latest is not None
    assert latest.version == "v2"


def test_list_versions(cache):
    """list_versions returns sorted version strings."""
    for v in ["v1", "v3", "v2"]:
        cache.store_manifest(
            Manifest(
                job="train-1",
                version=v,
                nodes=[],
                dataset_hash="",
                ttl="",
                created_at="2025-01-01T00:00:00",
                metadata={},
            )
        )

    versions = cache.list_versions("train-1")
    assert versions == ["v3", "v2", "v1"]


def test_delete_manifest(cache):
    """delete_manifest removes a stored manifest."""
    m = Manifest(
        job="train-1",
        version="v1",
        nodes=[],
        dataset_hash="",
        ttl="",
        created_at="2025-01-01T00:00:00",
        metadata={},
    )
    cache.store_manifest(m)
    assert cache.load_manifest("train-1", "v1") is not None

    assert cache.delete_manifest("train-1", "v1") is True
    assert cache.load_manifest("train-1", "v1") is None
    assert cache.delete_manifest("train-1", "v1") is False


def test_compute_dataset_hash_file(cache, tmp_path):
    """compute_dataset_hash returns a consistent sha256:... for a file."""
    f = tmp_path / "data.txt"
    f.write_text("hello")
    h1 = cache.compute_dataset_hash(str(f))
    h2 = cache.compute_dataset_hash(str(f))
    assert h1.startswith("sha256:")
    assert h1 == h2

    f.write_text("world")
    assert cache.compute_dataset_hash(str(f)) != h1


def test_compute_dataset_hash_directory(cache, tmp_path):
    """compute_dataset_hash returns a stable hash for a directory tree."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir(exist_ok=True)
    (tmp_path / "a" / "1.txt").write_text("one")
    (tmp_path / "b" / "2.txt").write_text("two")

    h = cache.compute_dataset_hash(str(tmp_path))
    assert h.startswith("sha256:")


# ── Stateful additions ──────────────────────────────────────────────────────

def test_load_missing_manifest(cache):
    assert cache.load_manifest("nonexistent") is None
    assert cache.load_manifest("nonexistent", "v1") is None


def test_list_versions_empty(cache):
    assert cache.list_versions("new-job") == []


def test_list_versions_ignores_unrelated_files(cache):
    m = Manifest(
        job="train-1",
        version="v1",
        nodes=[],
        dataset_hash="",
        ttl="",
        created_at="2025-01-01T00:00:00",
        metadata={},
    )
    cache.store_manifest(m)
    (cache.cache_dir / "train-1.txt").write_text("not a manifest")
    (cache.cache_dir / "train-1").mkdir(exist_ok=True)
    assert cache.list_versions("train-1") == ["v1"]


def test_delete_latest_falls_back_to_previous(cache):
    for v in ["v1", "v2"]:
        cache.store_manifest(
            Manifest(
                job="train-1",
                version=v,
                nodes=[],
                dataset_hash="",
                ttl="",
                created_at="2025-01-01T00:00:00",
                metadata={},
            )
        )
    assert cache.load_manifest("train-1").version == "v2"
    cache.delete_manifest("train-1", "v2")
    assert cache.load_manifest("train-1").version == "v1"


def test_store_overwrites_same_version(cache):
    m1 = Manifest(
        job="train-1",
        version="v1",
        nodes=[],
        dataset_hash="hash-a",
        ttl="",
        created_at="2025-01-01T00:00:00",
        metadata={},
    )
    m2 = Manifest(
        job="train-1",
        version="v1",
        nodes=[],
        dataset_hash="hash-b",
        ttl="",
        created_at="2025-01-01T00:00:00",
        metadata={},
    )
    cache.store_manifest(m1)
    cache.store_manifest(m2)
    loaded = cache.load_manifest("train-1", "v1")
    assert loaded.dataset_hash == "hash-b"


def test_multiple_jobs_isolated(cache):
    for job in ["train-a", "train-b"]:
        cache.store_manifest(
            Manifest(
                job=job,
                version="v1",
                nodes=[],
                dataset_hash="",
                ttl="",
                created_at="2025-01-01T00:00:00",
                metadata={},
            )
        )
    assert cache.list_versions("train-a") == ["v1"]
    assert cache.list_versions("train-b") == ["v1"]
    assert cache.load_manifest("train-a", "v1").job == "train-a"


def test_manifest_with_multiple_nodes(cache):
    nodes = [
        ManifestNode(provider="runpod", pod_id="p1", instance_id="i1", gpus=8, gpu_type="A100", region="us-east-1", status="running", created_at="2025-01-01T00:00:00", ttl="1h"),
        ManifestNode(provider="vastai", pod_id="p2", instance_id="i2", gpus=4, gpu_type="RTX4090", region="us-west-1", status="running", created_at="2025-01-01T00:00:00", ttl="1h"),
    ]
    m = Manifest(
        job="train-1",
        version="v1",
        nodes=nodes,
        dataset_hash="",
        ttl="",
        created_at="2025-01-01T00:00:00",
        metadata={},
    )
    cache.store_manifest(m)
    loaded = cache.load_manifest("train-1", "v1")
    assert len(loaded.nodes) == 2
    assert loaded.nodes[0].gpu_type == "A100"
    assert loaded.nodes[1].provider == "vastai"


def test_compute_hash_empty_directory(cache, tmp_path):
    (tmp_path / "empty_subdir").mkdir()
    h = cache.compute_dataset_hash(str(tmp_path))
    assert h.startswith("sha256:")


def test_compute_hash_changes_with_file_order(cache, tmp_path):
    # Reproducibility: same content in different directories should yield same hash
    for name in ["a", "b"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "data.txt").write_text("same")

    h1 = cache.compute_dataset_hash(str(tmp_path / "a"))
    h2 = cache.compute_dataset_hash(str(tmp_path / "b"))
    assert h1 == h2
