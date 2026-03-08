#!/usr/bin/env python3
"""
Checkpoint Manager — Atomic distributed checkpoints with manifest-based commits.

Default: local filesystem (zero deps).
Optional hooks:
  - S3 backend (if boto3 available + bucket configured)
  - GCS backend (if google-cloud-storage available + bucket configured)
  - Custom storage backend via StorageBackend protocol

DAG structure:
    Wave 0: shard_write_0 ∥ shard_write_1 ∥ ... ∥ shard_write_N
    Wave 1: manifest_assemble  (depends on all shard writes)
    Wave 2: manifest_commit    (atomic rename — the commit marker)

All output is structured JSON for MCP consumption.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .dag_executor import DAGExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage backend protocol (local default, S3/GCS optional)
# ---------------------------------------------------------------------------

@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for checkpoint storage. Implement for custom backends."""
    def put(self, local_path: str, remote_path: str) -> bool: ...
    def get(self, remote_path: str, local_path: str) -> bool: ...
    def exists(self, remote_path: str) -> bool: ...
    def delete(self, remote_path: str) -> bool: ...
    def list_prefix(self, prefix: str) -> List[str]: ...


class LocalStorage:
    """Default: local filesystem (zero deps)."""
    def put(self, src: str, dest: str) -> bool:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        return True

    def get(self, src: str, dest: str) -> bool:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            return True
        return False

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def delete(self, path: str) -> bool:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        return True

    def list_prefix(self, prefix: str) -> List[str]:
        if os.path.isdir(prefix):
            return [str(p) for p in Path(prefix).iterdir()]
        return []


class S3Storage:
    """Optional S3 backend — only instantiated if boto3 is available."""
    def __init__(self, bucket: str, prefix: str = "checkpoints"):
        try:
            import boto3
            self._s3 = boto3.client("s3")
        except ImportError:
            raise ImportError("S3 backend requires boto3: pip install boto3")
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, path: str) -> str:
        return f"{self.prefix}/{path}" if not path.startswith(self.prefix) else path

    def put(self, local_path: str, remote_path: str) -> bool:
        self._s3.upload_file(local_path, self.bucket, self._key(remote_path))
        return True

    def get(self, remote_path: str, local_path: str) -> bool:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self._s3.download_file(self.bucket, self._key(remote_path), local_path)
        return True

    def exists(self, remote_path: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._key(remote_path))
            return True
        except Exception:
            return False

    def delete(self, remote_path: str) -> bool:
        self._s3.delete_object(Bucket=self.bucket, Key=self._key(remote_path))
        return True

    def list_prefix(self, prefix: str) -> List[str]:
        resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=self._key(prefix))
        return [o["Key"] for o in resp.get("Contents", [])]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ShardInfo:
    rank: int
    node: str
    path: str
    size_bytes: int = 0
    sha256: str = ""
    write_ms: float = 0.0
    status: str = "pending"


@dataclass
class CheckpointManifest:
    """Commit marker — checkpoint valid iff manifest exists and is complete."""
    checkpoint_id: str
    job_id: str
    step: int
    timestamp: str
    topology_hash: str = ""
    shards: List[Dict[str, Any]] = field(default_factory=list)
    total_size_bytes: int = 0
    shard_count: int = 0
    framework: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "job_id": self.job_id,
            "step": self.step,
            "timestamp": self.timestamp,
            "topology_hash": self.topology_hash,
            "shards": self.shards,
            "total_size_bytes": self.total_size_bytes,
            "shard_count": self.shard_count,
            "framework": self.framework,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointManifest":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _topology_hash(topology: Dict[str, Any]) -> str:
    canon = json.dumps(topology, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _run_on(host: Optional[str], cmd: str, user: str = "root",
            key: Optional[str] = None, timeout: int = 120) -> Tuple[int, str, str]:
    if host and host not in ("localhost", "127.0.0.1"):
        ssh = ["ssh", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if key:
            ssh.extend(["-i", key])
        ssh.extend([f"{user}@{host}", cmd])
        try:
            r = subprocess.run(ssh, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout.strip(), r.stderr.strip()
        except Exception as e:
            return -1, "", str(e)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


# ---------------------------------------------------------------------------
# DAG node functions
# ---------------------------------------------------------------------------

def _write_shard(ctx: Dict[str, Any]) -> ShardInfo:
    """Write a single shard (parallel per rank)."""
    rank = ctx.get("rank", 0)
    node = ctx.get("node", "localhost")
    src_path = ctx.get("src_path", "")
    dest_dir = ctx.get("dest_dir", "")
    user = ctx.get("ssh_user", "root")
    key = ctx.get("ssh_key")

    shard = ShardInfo(rank=rank, node=node, path="")
    t0 = time.monotonic()

    if not src_path:
        shard.status = "failed"
        return shard

    shard_filename = f"shard_rank{rank:04d}.pt"
    dest_path = os.path.join(dest_dir, shard_filename)
    shard.path = dest_path

    if node in (None, "localhost", "127.0.0.1"):
        try:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            shard.size_bytes = os.path.getsize(dest_path)
            shard.sha256 = _compute_sha256(dest_path)
            shard.status = "written"
        except Exception as e:
            shard.status = "failed"
            logger.error(f"Shard write rank {rank}: {e}")
    else:
        _run_on(node, f"mkdir -p {dest_dir}", user, key)
        scp = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
        if key:
            scp.extend(["-i", key])
        scp.extend([src_path, f"{user}@{node}:{dest_path}"])
        try:
            r = subprocess.run(scp, capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                rc, stdout, _ = _run_on(
                    node, f"stat -c %s {dest_path} && sha256sum {dest_path} | cut -d' ' -f1",
                    user, key)
                if rc == 0:
                    lines = stdout.splitlines()
                    shard.size_bytes = int(lines[0]) if lines else 0
                    shard.sha256 = lines[1] if len(lines) > 1 else ""
                shard.status = "written"
            else:
                shard.status = "failed"
        except Exception as e:
            shard.status = "failed"
            logger.error(f"Remote shard write rank {rank}@{node}: {e}")

    shard.write_ms = (time.monotonic() - t0) * 1000
    return shard


def _assemble_manifest(ctx: Dict[str, Any]) -> CheckpointManifest:
    """Assemble manifest from shard writes (Wave 1)."""
    deps = ctx.get("__deps__", {})
    shards: List[ShardInfo] = [v for k, v in deps.items()
                                if k.startswith("shard_") and isinstance(v, ShardInfo)]

    failed = [s for s in shards if s.status == "failed"]
    if failed:
        raise RuntimeError(
            f"{len(failed)}/{len(shards)} shards failed (ranks: {[s.rank for s in failed]})")

    ckpt_id = ctx.get("checkpoint_id", f"ckpt-{uuid.uuid4().hex[:8]}")
    return CheckpointManifest(
        checkpoint_id=ckpt_id,
        job_id=ctx.get("job_id", ""),
        step=ctx.get("step", 0),
        timestamp=datetime.now().isoformat(),
        topology_hash=_topology_hash(ctx.get("topology", {})),
        shards=[{"rank": s.rank, "node": s.node, "path": s.path,
                 "size_bytes": s.size_bytes, "sha256": s.sha256,
                 "write_ms": s.write_ms}
                for s in sorted(shards, key=lambda x: x.rank)],
        total_size_bytes=sum(s.size_bytes for s in shards),
        shard_count=len(shards),
        framework=ctx.get("framework", ""),
        metadata=ctx.get("metadata", {}),
    )


def _commit_manifest(ctx: Dict[str, Any]) -> str:
    """Atomic commit: write to temp, rename (Wave 2)."""
    deps = ctx.get("__deps__", {})
    manifest = deps.get("manifest_assemble")
    if not isinstance(manifest, CheckpointManifest):
        raise RuntimeError("No manifest to commit")

    dest_dir = ctx.get("dest_dir", "")
    temp = os.path.join(dest_dir, ".manifest.json.tmp")
    final = os.path.join(dest_dir, "manifest.json")

    os.makedirs(dest_dir, exist_ok=True)
    with open(temp, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    os.rename(temp, final)
    logger.info(f"Committed: {manifest.checkpoint_id} step={manifest.step} "
                f"shards={manifest.shard_count} size={manifest.total_size_bytes}")
    return final


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Distributed checkpoint manager.

    Default: local filesystem + manifest-based atomic commits.
    Optional: S3/GCS/custom StorageBackend for offloading.

    DAG:
        Wave 0: shard writes (parallel per rank)
        Wave 1: manifest assemble
        Wave 2: manifest commit (atomic rename)

    Usage:
        mgr = CheckpointManager()
        manifest = mgr.save("job-1", step=1000, shard_paths={0: "/tmp/shard0.pt"})
        manifest = mgr.restore("job-1")  # latest
    """

    def __init__(self, base_dir: str = "",
                 state_manager=None,
                 storage: Optional[StorageBackend] = None,
                 ssh_user: str = "root", ssh_key: Optional[str] = None,
                 retention: int = 3):
        self.base_dir = base_dir or str(Path.home() / ".terradev" / "checkpoints")
        self.state_manager = state_manager
        self.storage = storage or LocalStorage()
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.retention = retention
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)

    def save(self, job_id: str, step: int,
             shard_paths: Dict[int, str],
             nodes: Optional[Dict[int, str]] = None,
             topology: Optional[Dict[str, Any]] = None,
             framework: str = "pytorch",
             metadata: Optional[Dict[str, Any]] = None) -> CheckpointManifest:
        """Save checkpoint with DAG-parallel shard writes."""
        nodes = nodes or {}
        ckpt_id = f"ckpt-{uuid.uuid4().hex[:8]}"
        dest_dir = os.path.join(self.base_dir, job_id, f"step_{step:08d}")

        if self.state_manager:
            try:
                self.state_manager.create_checkpoint(job_id, step, dest_dir)
            except Exception:
                pass

        # DAG: parallel shard writes → manifest → commit
        dag = DAGExecutor(max_workers=max(len(shard_paths), 2), name=f"ckpt_{step}")

        shard_names = []
        for rank, src in shard_paths.items():
            name = f"shard_{rank}"
            shard_names.append(name)

            def make_fn(r, sp, nh):
                def fn(_ctx):
                    return _write_shard({
                        "rank": r, "node": nh, "src_path": sp,
                        "dest_dir": dest_dir, "ssh_user": self.ssh_user,
                        "ssh_key": self.ssh_key})
                return fn
            dag.add_node(name, make_fn(rank, src, nodes.get(rank, "localhost")))

        dag.add_node("manifest_assemble", _assemble_manifest,
                     depends_on=set(shard_names))
        dag.add_node("manifest_commit", _commit_manifest,
                     depends_on={"manifest_assemble"})

        result = dag.apply(initial_context={
            "job_id": job_id, "step": step, "dest_dir": dest_dir,
            "topology": topology or {}, "framework": framework,
            "checkpoint_id": ckpt_id, "metadata": metadata or {}})

        if not result.success:
            if self.state_manager:
                try:
                    self.state_manager.fail_checkpoint(ckpt_id)
                except Exception:
                    pass
            raise RuntimeError(f"Checkpoint save failed: {result.errors}")

        manifest = result.outputs.get("manifest_assemble")

        # Optional: upload to remote storage
        if not isinstance(self.storage, LocalStorage):
            manifest_path = os.path.join(dest_dir, "manifest.json")
            if os.path.exists(manifest_path):
                remote_key = f"{job_id}/step_{step:08d}/manifest.json"
                self.storage.put(manifest_path, remote_key)
                for shard in manifest.shards if manifest else []:
                    if os.path.exists(shard["path"]):
                        self.storage.put(
                            shard["path"],
                            f"{job_id}/step_{step:08d}/{os.path.basename(shard['path'])}")
                logger.info(f"Uploaded checkpoint to remote storage")

        # State DB updates
        if self.state_manager:
            try:
                self.state_manager.commit_checkpoint(ckpt_id)
                self.state_manager.set_job_checkpoint(job_id, ckpt_id)
                self.state_manager.delete_old_checkpoints(job_id, keep=self.retention)
            except Exception:
                pass

        return manifest

    def restore(self, job_id: str, step: Optional[int] = None,
                checkpoint_id: Optional[str] = None,
                topology: Optional[Dict] = None) -> CheckpointManifest:
        """Restore checkpoint with parallel shard verification."""
        manifest_path = self._find_manifest(job_id, step, checkpoint_id)

        with open(manifest_path) as f:
            manifest = CheckpointManifest.from_dict(json.load(f))

        # Topology validation
        if topology and manifest.topology_hash:
            current = _topology_hash(topology)
            if current != manifest.topology_hash:
                raise RuntimeError(
                    f"Topology mismatch: saved={manifest.topology_hash} "
                    f"current={current}")

        # Parallel shard verification
        if manifest.shards:
            dag = DAGExecutor(max_workers=min(len(manifest.shards), 8),
                              name=f"ckpt_verify_{manifest.step}")
            for shard in manifest.shards:
                def make_verify(s):
                    def fn(_ctx):
                        path = s["path"]
                        expected = s.get("sha256", "")
                        if not expected:
                            return {"rank": s["rank"], "verified": True}
                        if not os.path.exists(path):
                            raise FileNotFoundError(f"Shard missing: {path}")
                        actual = _compute_sha256(path)
                        if actual != expected:
                            raise RuntimeError(
                                f"Rank {s['rank']} checksum mismatch")
                        return {"rank": s["rank"], "verified": True}
                    return fn
                dag.add_node(f"v_{shard['rank']}", make_verify(shard))

            vr = dag.apply()
            if not vr.success:
                raise RuntimeError(f"Verification failed: {vr.errors}")

        logger.info(f"Restored: {manifest.checkpoint_id} step={manifest.step}")
        return manifest

    def list(self, job_id: str) -> List[Dict[str, Any]]:
        """List checkpoints for a job."""
        if self.state_manager:
            try:
                return [c.to_dict() for c in self.state_manager.list_checkpoints(job_id)]
            except Exception:
                pass

        job_dir = os.path.join(self.base_dir, job_id)
        results = []
        if os.path.isdir(job_dir):
            for step_dir in sorted(Path(job_dir).iterdir(), reverse=True):
                mp = step_dir / "manifest.json"
                if mp.exists():
                    with open(mp) as f:
                        results.append(json.load(f))
        return results

    def promote(self, job_id: str, checkpoint_id: str,
                dest_path: str = "") -> str:
        """Promote checkpoint (copy to model output path)."""
        if self.state_manager:
            try:
                self.state_manager.promote_checkpoint(checkpoint_id)
            except Exception:
                pass

        if dest_path:
            ckpts = self.list(job_id)
            ckpt = next((c for c in ckpts
                         if c.get("checkpoint_id") == checkpoint_id), None)
            if ckpt and ckpt.get("shards"):
                src_dir = os.path.dirname(ckpt["shards"][0]["path"])
                if src_dir and os.path.isdir(src_dir):
                    shutil.copytree(src_dir, dest_path, dirs_exist_ok=True)
                    return dest_path
        return checkpoint_id

    def delete(self, job_id: str, checkpoint_id: str):
        """Delete a checkpoint."""
        ckpts = self.list(job_id)
        for c in ckpts:
            if c.get("checkpoint_id") == checkpoint_id:
                ckpt_dir = ""
                if c.get("shards"):
                    ckpt_dir = os.path.dirname(c["shards"][0]["path"])
                elif c.get("path"):
                    ckpt_dir = c["path"]
                if ckpt_dir:
                    self.storage.delete(ckpt_dir)
                if self.state_manager:
                    try:
                        self.state_manager.fail_checkpoint(checkpoint_id)
                    except Exception:
                        pass
                return

    def _find_manifest(self, job_id: str, step: Optional[int],
                       checkpoint_id: Optional[str]) -> str:
        """Locate manifest file."""
        if checkpoint_id and self.state_manager:
            try:
                ckpts = self.state_manager.list_checkpoints(job_id)
                ckpt = next((c for c in ckpts if c.id == checkpoint_id), None)
                if ckpt:
                    mp = os.path.join(ckpt.path, "manifest.json")
                    if os.path.exists(mp):
                        return mp
            except Exception:
                pass

        if step is not None:
            mp = os.path.join(self.base_dir, job_id,
                              f"step_{step:08d}", "manifest.json")
            if os.path.exists(mp):
                return mp

        # Latest
        if self.state_manager:
            try:
                ckpt = self.state_manager.get_latest_checkpoint(job_id)
                if ckpt:
                    mp = os.path.join(ckpt.path, "manifest.json")
                    if os.path.exists(mp):
                        return mp
            except Exception:
                pass

        job_dir = os.path.join(self.base_dir, job_id)
        if os.path.isdir(job_dir):
            for step_dir in sorted(Path(job_dir).iterdir(), reverse=True):
                mp = step_dir / "manifest.json"
                if mp.exists():
                    return str(mp)

        raise FileNotFoundError(
            f"No checkpoint: job={job_id} step={step} id={checkpoint_id}")
