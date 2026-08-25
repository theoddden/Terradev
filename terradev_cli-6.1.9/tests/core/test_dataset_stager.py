"""Tests for terradev_cli.core.dataset_stager.

Dataset staging determines compression, chunking, and checksums before data
movement. These tests cover the helper functions and the stager plan builder.
"""

from pathlib import Path

import pytest

from terradev_cli.core.dataset_stager import (
    DatasetStager,
    StagingPlan,
    _detect_size,
    _human_size,
    _pick_compression,
    chunk_file,
    compress_file,
    compute_checksum,
)


def test_detect_size_file(tmp_path):
    """_detect_size returns the byte size of an existing file."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"x" * 1024)
    assert _detect_size(str(f)) == 1024


def test_detect_size_directory(tmp_path):
    """_detect_size sums the sizes of all files in a directory."""
    (tmp_path / "a.txt").write_bytes(b"a" * 100)
    (tmp_path / "b.txt").write_bytes(b"b" * 200)
    assert _detect_size(str(tmp_path)) == 300


def test_detect_size_missing():
    """_detect_size returns 0 for a missing path."""
    assert _detect_size("/nonexistent/path/that/should/not/exist") == 0


def test_human_size():
    """_human_size formats bytes into human-readable units."""
    assert _human_size(512) == "512.0 B"
    assert _human_size(1024) == "1.0 KB"
    assert _human_size(1024**3) == "1.0 GB"


def test_pick_compression():
    """_pick_compression respects auto mode and falls back to gzip."""
    assert _pick_compression(False, 1024) == "none"
    algo = _pick_compression(True, 1024)
    assert algo in ("zstd", "gzip")


def test_compress_and_chunk_file(tmp_path):
    """Files can be compressed and split into chunks."""
    src = tmp_path / "big.bin"
    src.write_bytes(b"0" * (2 * 1024 * 1024 + 1))  # 2 MB + 1 byte

    # Test none compression (copy) to guarantee a predictable large file
    dst = tmp_path / "big.bin.copy"
    original, compressed = compress_file(str(src), str(dst), "none")
    assert original == 2 * 1024 * 1024 + 1
    assert compressed == original

    chunks = chunk_file(str(dst), chunk_size=1024 * 1024)  # 1 MB chunks
    assert len(chunks) >= 2
    assert all(Path(c).exists() for c in chunks)


def test_compute_checksum_stable(tmp_path):
    """compute_checksum returns the same SHA-256 for the same file."""
    f = tmp_path / "data.txt"
    f.write_text("hello")
    h1 = compute_checksum(str(f))
    h2 = compute_checksum(str(f))
    assert len(h1) == 64
    assert h1 == h2


def test_staging_plan(tmp_path):
    """DatasetStager builds a plan with size and compression estimates."""
    f = tmp_path / "dataset.txt"
    f.write_text("a" * 10000)

    stager = DatasetStager(chunk_size=1024 * 1024)
    plan = stager.plan(str(f), ["us-east-1", "us-west-2"], compression="auto")

    assert isinstance(plan, StagingPlan)
    assert plan.dataset == str(f)
    assert plan.size_bytes == 10000
    assert plan.regions == ["us-east-1", "us-west-2"]
    assert plan.compression in ("zstd", "gzip", "none")
    assert plan.chunks >= 1
    assert plan.to_dict()["dataset"] == str(f)


def test_staging_plan_with_egress(tmp_path):
    """plan_with_egress adds egress routing info to the plan."""
    f = tmp_path / "dataset.txt"
    f.write_text("x" * 1000)

    stager = DatasetStager()
    result = stager.plan_with_egress(
        str(f),
        ["aws:us-east-1"],
        data_provider="local",
        data_region="us-east-1",
    )

    assert "egress" in result
    assert result["dataset"] == str(f)
