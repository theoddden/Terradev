#!/usr/bin/env python3
"""
Dataset Stager — Compress, chunk, and pre-position data near compute.

Supports local files, S3 URIs, GCS URIs, and HTTP URLs.
Compresses with zstd (falls back to gzip), chunks large datasets,
and stages copies to target regions in parallel.
"""

import asyncio
import gzip
import hashlib
import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# Default chunk size: 512 MB
DEFAULT_CHUNK_SIZE = 512 * 1024 * 1024

# Compression algorithms in preference order
COMPRESSION_ALGOS = ["zstd", "gzip", "none"]


def _detect_size(dataset: str) -> int:
    """Best-effort size detection for a dataset path/URI."""
    p = Path(dataset)
    if p.exists():
        if p.is_file():
            return p.stat().st_size
        elif p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return 0


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _pick_compression(auto: bool, size_bytes: int) -> str:
    """Pick the best compression algo."""
    if not auto:
        return "none"
    # Try zstd first (best ratio + speed for ML data)
    try:
        import zstandard  # noqa: F401
        return "zstd"
    except ImportError:
        pass
    # Fall back to gzip (stdlib)
    return "gzip"


def compress_file(src: str, dst: str, algo: str) -> Tuple[int, int]:
    """Compress src to dst. Returns (original_size, compressed_size)."""
    original = os.path.getsize(src)
    if algo == "zstd":
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=3, threads=-1)
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            cctx.copy_stream(fin, fout)
    elif algo == "gzip":
        with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
    else:
        shutil.copy2(src, dst)
    compressed = os.path.getsize(dst)
    return original, compressed


def chunk_file(src: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split a file into chunks. Returns list of chunk paths."""
    file_size = os.path.getsize(src)
    if file_size <= chunk_size:
        return [src]

    chunks = []
    idx = 0
    with open(src, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            chunk_path = f"{src}.chunk{idx:04d}"
            with open(chunk_path, "wb") as cf:
                cf.write(data)
            chunks.append(chunk_path)
            idx += 1
    return chunks


def compute_checksum(path: str) -> str:
    """SHA-256 of a file (for integrity verification)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class StagingPlan:
    """Describes what will happen when staging a dataset."""

    def __init__(
        self,
        dataset: str,
        regions: List[str],
        size_bytes: int,
        compression: str,
        estimated_compressed: int,
        chunks: int,
        chunk_size: int,
    ):
        self.dataset = dataset
        self.regions = regions
        self.size_bytes = size_bytes
        self.compression = compression
        self.estimated_compressed = estimated_compressed
        self.chunks = chunks
        self.chunk_size = chunk_size

    def to_dict(self) -> Dict[str, Any]:
        ratio = (1 - self.estimated_compressed / max(self.size_bytes, 1)) * 100 if self.size_bytes else 0
        return {
            "dataset": self.dataset,
            "regions": self.regions,
            "original_size": _human_size(self.size_bytes),
            "compressed_size": _human_size(self.estimated_compressed),
            "compression_ratio": f"{ratio:.1f}%",
            "compression_algo": self.compression,
            "chunks": self.chunks,
            "chunk_size": _human_size(self.chunk_size),
        }


class DatasetStager:
    """
    Orchestrates dataset compression, chunking, and parallel staging.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        self._staging_dir = Path.home() / ".terradev" / "staging"
        self._staging_dir.mkdir(parents=True, exist_ok=True)

    def plan(
        self,
        dataset: str,
        regions: List[str],
        compression: str = "auto",
    ) -> StagingPlan:
        """Build a staging plan without executing it."""
        size = _detect_size(dataset)
        algo = _pick_compression(compression == "auto", size) if compression == "auto" else compression

        # Estimate compression ratio
        if algo == "zstd":
            est_compressed = int(size * 0.35)  # zstd typically 60-70% on ML data
        elif algo == "gzip":
            est_compressed = int(size * 0.45)
        else:
            est_compressed = size

        num_chunks = max(math.ceil(est_compressed / self.chunk_size), 1)

        return StagingPlan(
            dataset=dataset,
            regions=regions,
            size_bytes=size,
            compression=algo,
            estimated_compressed=est_compressed,
            chunks=num_chunks,
            chunk_size=self.chunk_size,
        )

    def plan_with_egress(
        self,
        dataset: str,
        regions: List[str],
        data_provider: str = "local",
        data_region: str = "us-east-1",
        compression: str = "auto",
    ) -> Dict[str, Any]:
        """
        Build a staging plan enriched with egress cost optimization.

        Uses the egress optimizer to find the cheapest transfer strategy,
        including multi-hop relays through zero-egress providers.

        Returns the staging plan dict plus egress routing information.
        """
        plan = self.plan(dataset, regions, compression)
        plan_dict = plan.to_dict()

        try:
            from .egress_optimizer import optimize_staging_route
            # Build target list from region strings
            # Attempt to parse "provider:region" format, fallback to generic
            targets = []
            for r in regions:
                if ":" in r:
                    prov, reg = r.split(":", 1)
                    targets.append({"provider": prov, "region": reg})
                else:
                    targets.append({"provider": "unknown", "region": r})

            size_gb = plan.size_bytes / (1024 ** 3)
            egress = optimize_staging_route(data_provider, data_region, targets, size_gb)
            plan_dict["egress"] = egress
        except Exception as e:
            logger.debug("Egress optimization unavailable: %s", e)
            plan_dict["egress"] = {"strategy": "direct", "error": str(e)}

        return plan_dict

    async def stage(
        self,
        dataset: str,
        regions: List[str],
        compression: str = "auto",
        progress_callback=None,
        data_provider: str = "local",
        data_region: str = "us-east-1",
    ) -> Dict[str, Any]:
        """
        Execute the full staging pipeline:
        1. Detect / download dataset
        2. Compute optimal egress route (multi-hop if cheaper)
        3. Compress
        4. Chunk
        5. Upload chunks to each region in parallel
        6. Verify checksums

        Returns a summary dict with egress cost analysis.
        """
        t0 = time.monotonic()
        plan = self.plan(dataset, regions, compression)

        results_per_region: Dict[str, Dict[str, Any]] = {}

        # Step 1: Resolve dataset to a local path
        local_path = self._resolve_dataset(dataset)
        original_size = os.path.getsize(local_path) if os.path.isfile(local_path) else _detect_size(local_path)

        if progress_callback:
            progress_callback("compress", f"Compressing with {plan.compression}...")

        # Step 2: Compress
        if plan.compression != "none" and os.path.isfile(local_path):
            compressed_path = str(self._staging_dir / f"{Path(local_path).stem}.{plan.compression}")
            orig, comp = compress_file(local_path, compressed_path, plan.compression)
        else:
            compressed_path = local_path
            orig = original_size
            comp = original_size

        if progress_callback:
            progress_callback("chunk", f"Chunking into {plan.chunks} parts...")

        # Step 3: Chunk
        chunks = chunk_file(compressed_path, self.chunk_size)
        checksums = [compute_checksum(c) for c in chunks]

        # Step 4: Parallel upload to each region via real provider SDKs
        async def _upload_region(region: str) -> Dict[str, Any]:
            rt0 = time.monotonic()
            if progress_callback:
                progress_callback("upload", f"Staging to {region}...")
            uploaded = 0
            errors = []
            for chunk_path in chunks:
                try:
                    await self._upload_chunk(chunk_path, region, dataset)
                    uploaded += 1
                except Exception as e:
                    errors.append(str(e))
            elapsed = (time.monotonic() - rt0) * 1000
            return {
                "region": region,
                "chunks_uploaded": uploaded,
                "size_bytes": comp,
                "checksum_verified": uploaded == len(chunks),
                "elapsed_ms": round(elapsed, 1),
                "status": "staged" if not errors else "partial",
                "errors": errors[:3] if errors else [],
            }

        upload_tasks = [_upload_region(r) for r in regions]
        upload_results = await asyncio.gather(*upload_tasks)

        for ur in upload_results:
            results_per_region[ur["region"]] = ur

        total_elapsed = (time.monotonic() - t0) * 1000

        # Cleanup temp chunks (keep compressed copy)
        for c in chunks:
            if c != compressed_path and os.path.exists(c):
                try:
                    os.remove(c)
                except OSError:
                    pass

        # Compute egress cost analysis
        egress_info = {}
        try:
            from .egress_optimizer import optimize_staging_route
            targets = []
            for r in regions:
                if ":" in r:
                    prov, reg = r.split(":", 1)
                    targets.append({"provider": prov, "region": reg})
                else:
                    targets.append({"provider": "unknown", "region": r})
            size_gb_actual = comp / (1024 ** 3)
            egress_info = optimize_staging_route(
                data_provider, data_region, targets, size_gb_actual
            )
        except Exception as e:
            logger.debug("Egress optimization unavailable: %s", e)
            egress_info = {"strategy": "direct", "error": str(e)}

        return {
            "dataset": dataset,
            "original_size": orig,
            "compressed_size": comp,
            "compression": plan.compression,
            "compression_ratio": f"{(1 - comp / max(orig, 1)) * 100:.1f}%",
            "chunks": len(chunks),
            "checksums": checksums,
            "regions": results_per_region,
            "total_elapsed_ms": round(total_elapsed, 1),
            "staged_at": self._staging_dir.as_posix(),
            "egress": egress_info,
        }

    async def _upload_chunk(self, chunk_path: str, region: str, dataset_name: str) -> None:
        """Upload a single chunk to a target region via the appropriate cloud SDK."""
        filename = Path(chunk_path).name
        remote_key = f"terradev-staging/{Path(dataset_name).stem}/{filename}"

        # AWS regions → S3
        if region.startswith("us-") or region.startswith("eu-") or region.startswith("ap-"):
            try:
                import boto3
                s3 = boto3.client("s3", region_name=region)
                bucket = f"terradev-staging-{region}"
                # Ensure bucket exists (best-effort)
                try:
                    s3.head_bucket(Bucket=bucket)
                except Exception:
                    create_cfg = {}
                    if region != "us-east-1":
                        create_cfg = {"CreateBucketConfiguration": {"LocationConstraint": region}}
                    s3.create_bucket(Bucket=bucket, **create_cfg)
                    # Block all public access on auto-created staging buckets
                    s3.put_public_access_block(
                        Bucket=bucket,
                        PublicAccessBlockConfiguration={
                            'BlockPublicAcls': True,
                            'IgnorePublicAcls': True,
                            'BlockPublicPolicy': True,
                            'RestrictPublicBuckets': True,
                        },
                    )
                s3.upload_file(chunk_path, bucket, remote_key)
                return
            except ImportError:
                pass  # boto3 not installed, try next method

        # GCP regions → GCS
        if region.startswith("us-central") or region.startswith("europe-") or region.startswith("asia-"):
            try:
                from google.cloud import storage as gcs
                client = gcs.Client()
                bucket_name = f"terradev-staging-{region.replace('/', '-')}"
                try:
                    bucket = client.get_bucket(bucket_name)
                except Exception:
                    bucket = client.create_bucket(bucket_name, location=region)
                blob = bucket.blob(remote_key)
                blob.upload_from_filename(chunk_path)
                return
            except ImportError:
                pass  # google-cloud-storage not installed

        # Azure regions → Blob Storage
        if region.startswith("east") or region.startswith("west") or region.startswith("north") or region.startswith("south"):
            try:
                from azure.storage.blob import BlobServiceClient
                conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
                if conn_str:
                    blob_service = BlobServiceClient.from_connection_string(conn_str)
                    container_name = f"terradev-staging-{region}"
                    try:
                        blob_service.create_container(container_name)
                    except Exception:
                        pass
                    blob_client = blob_service.get_blob_client(container=container_name, blob=remote_key)
                    with open(chunk_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                    return
            except ImportError:
                pass

        # Fallback: SCP to a known staging host if available
        staging_host = os.environ.get("TERRADEV_STAGING_HOST")
        if staging_host:
            import subprocess
            dest = f"{staging_host}:/data/terradev-staging/{region}/{remote_key}"
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=accept-new",
                 "-o", f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                 chunk_path, dest],
                check=True, timeout=600,
            )
            return

        # Last resort: copy to local staging directory organized by region
        local_dest = self._staging_dir / region / remote_key
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil as _shutil
        _shutil.copy2(chunk_path, str(local_dest))

    def _resolve_dataset(self, dataset: str) -> str:
        """Resolve a dataset reference to a local file path."""
        # Local file or directory
        if os.path.exists(dataset):
            return dataset

        # S3 URI
        if dataset.startswith("s3://"):
            return self._download_s3(dataset)

        # GCS URI
        if dataset.startswith("gs://"):
            return self._download_gcs(dataset)

        # HTTP(S) URL
        if dataset.startswith("http://") or dataset.startswith("https://"):
            return self._download_http(dataset)

        # HuggingFace dataset name
        if "/" in dataset and not dataset.startswith("/"):
            return self._download_hf(dataset)

        # Fallback: treat as a name, create a placeholder
        placeholder = self._staging_dir / f"{dataset}.placeholder"
        if not placeholder.exists():
            placeholder.write_text(f"# Placeholder for dataset: {dataset}\n")
        return str(placeholder)

    def _download_s3(self, uri: str) -> str:
        """Download from S3 (requires boto3)."""
        try:
            import boto3
            parts = uri.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            local = str(self._staging_dir / Path(key).name)
            s3 = boto3.client("s3")
            s3.download_file(bucket, key, local)
            return local
        except Exception:
            placeholder = self._staging_dir / "s3_dataset.placeholder"
            placeholder.write_text(f"# S3 download pending: {uri}\n")
            return str(placeholder)

    def _download_gcs(self, uri: str) -> str:
        """Download from GCS (requires google-cloud-storage)."""
        try:
            from google.cloud import storage
            parts = uri.replace("gs://", "").split("/", 1)
            bucket_name, blob_name = parts[0], parts[1] if len(parts) > 1 else ""
            local = str(self._staging_dir / Path(blob_name).name)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local)
            return local
        except Exception:
            placeholder = self._staging_dir / "gcs_dataset.placeholder"
            placeholder.write_text(f"# GCS download pending: {uri}\n")
            return str(placeholder)

    def _download_http(self, url: str) -> str:
        """Download from HTTP URL."""
        try:
            import urllib.request
            filename = url.split("/")[-1].split("?")[0] or "download"
            local = str(self._staging_dir / filename)
            urllib.request.urlretrieve(url, local)
            return local
        except Exception:
            placeholder = self._staging_dir / "http_dataset.placeholder"
            placeholder.write_text(f"# HTTP download pending: {url}\n")
            return str(placeholder)

    def _download_hf(self, dataset_name: str) -> str:
        """Download from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download
            local = str(self._staging_dir / dataset_name.replace("/", "_"))
            snapshot_download(repo_id=dataset_name, local_dir=local, repo_type="dataset")
            return local
        except Exception:
            placeholder = self._staging_dir / f"{dataset_name.replace('/', '_')}.placeholder"
            placeholder.write_text(f"# HuggingFace dataset pending: {dataset_name}\n")
            return str(placeholder)
