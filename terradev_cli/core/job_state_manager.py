#!/usr/bin/env python3
"""
Job State Manager — SQLite-backed job database.

Tracks: what's running, where, since when, cost-to-date, checkpoint state.
All queries are synchronous (SQLite is single-writer anyway); writes use
WAL mode for concurrent reads.

Schema:
    jobs: id, name, framework, status, config_json, topology_json,
          nodes_json, created_at, started_at, finished_at,
          last_checkpoint_id, current_step, total_steps,
          cost_usd, error_message
    checkpoints: id, job_id, step, path, manifest_json, size_bytes,
                 created_at, status, promoted
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    CREATED = "created"
    PREFLIGHT = "preflight"
    LAUNCHING = "launching"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


class CheckpointStatus(Enum):
    WRITING = "writing"
    COMMITTED = "committed"
    FAILED = "failed"
    PROMOTED = "promoted"
    DELETED = "deleted"


@dataclass
class JobRecord:
    """A training job record."""
    id: str
    name: str
    framework: str
    status: JobStatus
    config: Dict[str, Any] = field(default_factory=dict)
    topology: Dict[str, Any] = field(default_factory=dict)
    nodes: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    last_checkpoint_id: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    cost_usd: float = 0.0
    cost_per_gpu_hour: float = 0.0
    error_message: str = ""

    @property
    def elapsed_hours(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or datetime.now()
        return (end - self.started_at).total_seconds() / 3600

    @property
    def gpu_count(self) -> int:
        gpus_per_node = self.config.get("gpus_per_node", 8) if self.config else 8
        return max(len(self.nodes), 1) * gpus_per_node

    @property
    def gpu_hours(self) -> float:
        return self.elapsed_hours * self.gpu_count

    @property
    def eta_hours(self) -> Optional[float]:
        if not self.total_steps or not self.current_step or not self.started_at:
            return None
        if self.current_step <= 0:
            return None
        rate = self.current_step / max(self.elapsed_hours, 0.001)
        remaining = self.total_steps - self.current_step
        return remaining / rate if rate > 0 else None

    @property
    def efficiency(self) -> float:
        """Steps per GPU-hour — higher is better."""
        return self.current_step / max(self.gpu_hours, 0.001)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "framework": self.framework,
            "status": self.status.value,
            "nodes": self.nodes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_pct": round(self.current_step / self.total_steps * 100, 1) if self.total_steps else 0,
            "elapsed_hours": round(self.elapsed_hours, 2),
            "gpu_hours": round(self.gpu_hours, 2),
            "eta_hours": round(self.eta_hours, 2) if self.eta_hours is not None else None,
            "cost_usd": round(self.cost_usd, 4),
            "cost_per_gpu_hour": self.cost_per_gpu_hour,
            "efficiency_steps_per_gpuh": round(self.efficiency, 2),
            "last_checkpoint_id": self.last_checkpoint_id,
            "error_message": self.error_message,
        }


@dataclass
class CheckpointRecord:
    """A checkpoint record."""
    id: str
    job_id: str
    step: int
    path: str
    manifest: Dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0
    created_at: Optional[datetime] = None
    status: CheckpointStatus = CheckpointStatus.WRITING
    promoted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "step": self.step,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "status": self.status.value,
            "promoted": self.promoted,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    framework TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    config_json TEXT DEFAULT '{}',
    topology_json TEXT DEFAULT '{}',
    nodes_json TEXT DEFAULT '[]',
    created_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    last_checkpoint_id TEXT,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    cost_per_gpu_hour REAL DEFAULT 0.0,
    error_message TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    path TEXT NOT NULL,
    manifest_json TEXT DEFAULT '{}',
    size_bytes INTEGER DEFAULT 0,
    created_at TEXT,
    status TEXT NOT NULL DEFAULT 'writing',
    promoted INTEGER DEFAULT 0,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_checkpoints_job ON checkpoints(job_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_step ON checkpoints(job_id, step);
"""


class JobStateManager:
    """SQLite-backed job and checkpoint state manager."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(
            Path.home() / ".terradev" / "jobs.db"
        )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Job CRUD ──────────────────────────────────────────────────────────

    def create_job(self, name: str, framework: str, config: Dict[str, Any],
                   nodes: List[str], topology: Optional[Dict] = None,
                   total_steps: int = 0) -> JobRecord:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        self._conn.execute(
            "INSERT INTO jobs (id, name, framework, status, config_json, "
            "topology_json, nodes_json, created_at, total_steps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, name, framework, JobStatus.CREATED.value,
             json.dumps(config), json.dumps(topology or {}),
             json.dumps(nodes), now.isoformat(), total_steps)
        )
        self._conn.commit()
        return JobRecord(id=job_id, name=name, framework=framework,
                         status=JobStatus.CREATED, config=config,
                         topology=topology or {}, nodes=nodes,
                         created_at=now, total_steps=total_steps)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, status: Optional[str] = None,
                  limit: int = 50) -> List[JobRecord]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def update_job_status(self, job_id: str, status: JobStatus,
                          error_message: str = ""):
        updates = {"status": status.value}
        if status == JobStatus.RUNNING:
            updates["started_at"] = datetime.now().isoformat()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            updates["finished_at"] = datetime.now().isoformat()
        if error_message:
            updates["error_message"] = error_message

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        self._conn.execute(
            f"UPDATE jobs SET {set_clause} WHERE id = ?",
            (*updates.values(), job_id)
        )
        self._conn.commit()

    def update_job_step(self, job_id: str, step: int, cost_usd: float = 0.0):
        self._conn.execute(
            "UPDATE jobs SET current_step = ?, cost_usd = cost_usd + ? WHERE id = ?",
            (step, cost_usd, job_id)
        )
        self._conn.commit()

    def set_cost_rate(self, job_id: str, cost_per_gpu_hour: float):
        self._conn.execute(
            "UPDATE jobs SET cost_per_gpu_hour = ? WHERE id = ?",
            (cost_per_gpu_hour, job_id)
        )
        self._conn.commit()

    def set_job_checkpoint(self, job_id: str, checkpoint_id: str):
        self._conn.execute(
            "UPDATE jobs SET last_checkpoint_id = ? WHERE id = ?",
            (checkpoint_id, job_id)
        )
        self._conn.commit()

    # ── Checkpoint CRUD ───────────────────────────────────────────────────

    def create_checkpoint(self, job_id: str, step: int, path: str,
                          manifest: Optional[Dict] = None,
                          size_bytes: int = 0) -> CheckpointRecord:
        ckpt_id = f"ckpt-{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        self._conn.execute(
            "INSERT INTO checkpoints (id, job_id, step, path, manifest_json, "
            "size_bytes, created_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ckpt_id, job_id, step, path, json.dumps(manifest or {}),
             size_bytes, now.isoformat(), CheckpointStatus.WRITING.value)
        )
        self._conn.commit()
        return CheckpointRecord(id=ckpt_id, job_id=job_id, step=step,
                                path=path, manifest=manifest or {},
                                size_bytes=size_bytes, created_at=now)

    def commit_checkpoint(self, checkpoint_id: str):
        self._conn.execute(
            "UPDATE checkpoints SET status = ? WHERE id = ?",
            (CheckpointStatus.COMMITTED.value, checkpoint_id)
        )
        self._conn.commit()

    def fail_checkpoint(self, checkpoint_id: str):
        self._conn.execute(
            "UPDATE checkpoints SET status = ? WHERE id = ?",
            (CheckpointStatus.FAILED.value, checkpoint_id)
        )
        self._conn.commit()

    def promote_checkpoint(self, checkpoint_id: str):
        self._conn.execute(
            "UPDATE checkpoints SET status = ?, promoted = 1 WHERE id = ?",
            (CheckpointStatus.PROMOTED.value, checkpoint_id)
        )
        self._conn.commit()

    def list_checkpoints(self, job_id: str) -> List[CheckpointRecord]:
        rows = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY step DESC",
            (job_id,)
        ).fetchall()
        return [self._row_to_checkpoint(r) for r in rows]

    def get_latest_checkpoint(self, job_id: str) -> Optional[CheckpointRecord]:
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? AND status = ? "
            "ORDER BY step DESC LIMIT 1",
            (job_id, CheckpointStatus.COMMITTED.value)
        ).fetchone()
        return self._row_to_checkpoint(row) if row else None

    def delete_old_checkpoints(self, job_id: str, keep: int = 3) -> int:
        """Retain only the latest `keep` committed checkpoints. Returns count deleted."""
        rows = self._conn.execute(
            "SELECT id FROM checkpoints WHERE job_id = ? AND status = ? "
            "ORDER BY step DESC",
            (job_id, CheckpointStatus.COMMITTED.value)
        ).fetchall()
        to_delete = [r[0] for r in rows[keep:]]
        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            self._conn.execute(
                f"UPDATE checkpoints SET status = ? WHERE id IN ({placeholders})",
                (CheckpointStatus.DELETED.value, *to_delete)
            )
            self._conn.commit()
        return len(to_delete)

    # ── Aggregate queries ─────────────────────────────────────────────────

    def running_jobs_summary(self) -> List[Dict[str, Any]]:
        """What's running — structured JSON for MCP."""
        jobs = self.list_jobs(status=JobStatus.RUNNING.value)
        return [j.to_dict() for j in jobs]

    def job_metrics(self, job_id: str) -> Dict[str, Any]:
        """Full metrics for a single job — MCP queryable."""
        job = self.get_job(job_id)
        if not job:
            return {"error": f"Job not found: {job_id}"}
        d = job.to_dict()
        ckpts = self.list_checkpoints(job_id)
        d["checkpoint_count"] = len(ckpts)
        d["last_checkpoint_step"] = ckpts[0].step if ckpts else None
        return d

    def total_cost(self, job_id: Optional[str] = None) -> float:
        if job_id:
            row = self._conn.execute(
                "SELECT cost_usd FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return row[0] if row else 0.0
        row = self._conn.execute("SELECT SUM(cost_usd) FROM jobs").fetchone()
        return row[0] if row and row[0] else 0.0

    # ── Row mappers ───────────────────────────────────────────────────────

    def _row_to_job(self, row) -> JobRecord:
        return JobRecord(
            id=row[0], name=row[1], framework=row[2],
            status=JobStatus(row[3]),
            config=json.loads(row[4]) if row[4] else {},
            topology=json.loads(row[5]) if row[5] else {},
            nodes=json.loads(row[6]) if row[6] else [],
            created_at=datetime.fromisoformat(row[7]) if row[7] else None,
            started_at=datetime.fromisoformat(row[8]) if row[8] else None,
            finished_at=datetime.fromisoformat(row[9]) if row[9] else None,
            last_checkpoint_id=row[10],
            current_step=row[11] or 0,
            total_steps=row[12] or 0,
            cost_usd=row[13] or 0.0,
            cost_per_gpu_hour=row[14] or 0.0,
            error_message=row[15] or "",
        )

    def _row_to_checkpoint(self, row) -> CheckpointRecord:
        return CheckpointRecord(
            id=row[0], job_id=row[1], step=row[2], path=row[3],
            manifest=json.loads(row[4]) if row[4] else {},
            size_bytes=row[5] or 0,
            created_at=datetime.fromisoformat(row[6]) if row[6] else None,
            status=CheckpointStatus(row[7]),
            promoted=bool(row[8]),
        )
