#!/usr/bin/env python3
"""
Cost Tracker — Persistent cost tracking wired into every provision/quote call.
Uses ~/.terradev/cost_tracking.db (SQLite).
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional


DB_PATH = Path.home() / ".terradev" / "cost_tracking.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not DB_PATH.exists()
    conn = sqlite3.connect(str(DB_PATH))
    if is_new:
        import os
        os.chmod(DB_PATH, 0o600)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS quotes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL DEFAULT (datetime('now')),
            gpu_type    TEXT    NOT NULL,
            provider    TEXT    NOT NULL,
            region      TEXT,
            price_hr    REAL    NOT NULL,
            spot        INTEGER NOT NULL DEFAULT 0,
            selected    INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS provisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL DEFAULT (datetime('now')),
            instance_id     TEXT    NOT NULL,
            provider        TEXT    NOT NULL,
            gpu_type        TEXT    NOT NULL,
            region          TEXT,
            price_hr        REAL    NOT NULL,
            spot            INTEGER NOT NULL DEFAULT 0,
            status          TEXT    NOT NULL DEFAULT 'provisioning',
            parallel_group  TEXT,
            end_ts          TEXT,
            total_cost      REAL    NOT NULL DEFAULT 0.0,
            ip_address      TEXT    DEFAULT '',
            ssh_key_path    TEXT    DEFAULT ''
        );

        -- Migration: add ssh_key_path if missing (existing DBs)
        -- SQLite ignores ALTER TABLE errors for duplicate columns via pragma
    """)
    # Safe migration for existing DBs
    try:
        conn.execute("ALTER TABLE provisions ADD COLUMN ssh_key_path TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS egress (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           TEXT    NOT NULL DEFAULT (datetime('now')),
            src_provider TEXT    NOT NULL,
            src_region   TEXT    NOT NULL,
            dst_provider TEXT    NOT NULL,
            dst_region   TEXT    NOT NULL,
            bytes_moved  INTEGER NOT NULL,
            cost         REAL    NOT NULL,
            optimized    INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS staging (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL DEFAULT (datetime('now')),
            dataset     TEXT    NOT NULL,
            region      TEXT    NOT NULL,
            size_bytes  INTEGER NOT NULL DEFAULT 0,
            compressed  INTEGER NOT NULL DEFAULT 0,
            chunks      INTEGER NOT NULL DEFAULT 1,
            status      TEXT    NOT NULL DEFAULT 'staged'
        );

        CREATE TABLE IF NOT EXISTS daily_spend (
            date        TEXT    PRIMARY KEY,
            total       REAL    NOT NULL DEFAULT 0.0,
            providers   TEXT    NOT NULL DEFAULT '{}'
        );
    """)
    conn.commit()


# ── Quote tracking ────────────────────────────────────────────────────

def record_quotes(quotes: List[Dict[str, Any]], selected_idx: Optional[int] = None):
    """Record every quote returned by a quote command."""
    conn = _conn()
    for i, q in enumerate(quotes):
        conn.execute(
            "INSERT INTO quotes (gpu_type, provider, region, price_hr, spot, selected) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                q.get("gpu_type", ""),
                q.get("provider", ""),
                q.get("region", ""),
                q.get("price", q.get("price_per_hour", 0)),
                1 if q.get("spot") or q.get("availability") == "spot" else 0,
                1 if i == selected_idx else 0,
            ),
        )
    conn.commit()
    conn.close()


# ── Provision tracking ────────────────────────────────────────────────

def record_provision(
    instance_id: str,
    provider: str,
    gpu_type: str,
    region: str,
    price_hr: float,
    spot: bool = False,
    parallel_group: Optional[str] = None,
) -> int:
    """Record a new provision event. Returns the row id."""
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO provisions "
        "(instance_id, provider, gpu_type, region, price_hr, spot, parallel_group, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'active')",
        (instance_id, provider, gpu_type, region, price_hr, 1 if spot else 0, parallel_group),
    )
    conn.commit()
    row_id = cur.lastrowid
    _update_daily_spend(conn, price_hr, provider)
    conn.close()
    return row_id


def end_provision(instance_id: str):
    """Mark a provision as ended and calculate total cost."""
    conn = _conn()
    row = conn.execute(
        "SELECT id, ts, price_hr FROM provisions WHERE instance_id = ? AND end_ts IS NULL",
        (instance_id,),
    ).fetchone()
    if row:
        start = datetime.fromisoformat(row["ts"])
        hours = max((datetime.utcnow() - start).total_seconds() / 3600, 0.01)
        total = round(hours * row["price_hr"], 4)
        conn.execute(
            "UPDATE provisions SET end_ts = datetime('now'), status = 'terminated', total_cost = ? WHERE id = ?",
            (total, row["id"]),
        )
        conn.commit()
    conn.close()


# ── Egress tracking ──────────────────────────────────────────────────

def record_egress(
    src_provider: str, src_region: str,
    dst_provider: str, dst_region: str,
    bytes_moved: int, cost: float, optimized: bool = False,
):
    conn = _conn()
    conn.execute(
        "INSERT INTO egress (src_provider, src_region, dst_provider, dst_region, bytes_moved, cost, optimized) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (src_provider, src_region, dst_provider, dst_region, bytes_moved, cost, 1 if optimized else 0),
    )
    conn.commit()
    conn.close()


# ── Staging tracking ─────────────────────────────────────────────────

def record_staging(dataset: str, original_size: int, compressed_size: int,
                   compression: str, chunks: int, regions: List[str]):
    """Record a staging event for each target region."""
    conn = _conn()
    for region in regions:
        conn.execute(
            "INSERT INTO staging (dataset, region, size_bytes, compressed, chunks) VALUES (?, ?, ?, ?, ?)",
            (dataset, region, original_size, compressed_size, chunks),
        )
    conn.commit()
    conn.close()


# ── Daily spend helper ───────────────────────────────────────────────

def _update_daily_spend(conn: sqlite3.Connection, price_hr: float, provider: str):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    row = conn.execute("SELECT total, providers FROM daily_spend WHERE date = ?", (today,)).fetchone()
    if row:
        providers = json.loads(row["providers"])
        providers[provider] = providers.get(provider, 0) + price_hr
        conn.execute(
            "UPDATE daily_spend SET total = total + ?, providers = ? WHERE date = ?",
            (price_hr, json.dumps(providers), today),
        )
    else:
        conn.execute(
            "INSERT INTO daily_spend (date, total, providers) VALUES (?, ?, ?)",
            (today, price_hr, json.dumps({provider: price_hr})),
        )
    conn.commit()


# ── Analytics queries ────────────────────────────────────────────────

def get_spend_summary(days: int = 30) -> Dict[str, Any]:
    """Get spend summary for the last N days."""
    conn = _conn()
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    total = conn.execute(
        "SELECT COALESCE(SUM(total_cost), 0) as s FROM provisions WHERE ts >= ?", (cutoff,)
    ).fetchone()["s"]

    by_provider = {}
    for row in conn.execute(
        "SELECT provider, SUM(total_cost) as s, COUNT(*) as c FROM provisions WHERE ts >= ? GROUP BY provider",
        (cutoff,),
    ):
        by_provider[row["provider"]] = {"cost": row["s"], "count": row["c"]}

    quote_count = conn.execute(
        "SELECT COUNT(*) as c FROM quotes WHERE ts >= ?", (cutoff,)
    ).fetchone()["c"]

    egress_cost = conn.execute(
        "SELECT COALESCE(SUM(cost), 0) as s FROM egress WHERE ts >= ?", (cutoff,)
    ).fetchone()["s"]

    prov_count = conn.execute(
        "SELECT COUNT(*) as c FROM provisions WHERE ts >= ?", (cutoff,)
    ).fetchone()["c"]

    conn.close()
    return {
        "total_provision_cost": round(total, 2),
        "total_provisions": prov_count,
        "by_provider": by_provider,
        "quotes_fetched": quote_count,
        "egress_cost": round(egress_cost, 2),
        "days": days,
    }


def get_daily_spend(days: int = 7) -> List[Dict[str, Any]]:
    """Get daily spend for the last N days."""
    conn = _conn()
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT date, total as cost, providers FROM daily_spend WHERE date >= ? ORDER BY date",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [{"date": r["date"], "cost": r["cost"], "providers": json.loads(r["providers"])} for r in rows]


def get_parallel_group_summary(group_id: str) -> List[Dict[str, Any]]:
    """Get all provisions in a parallel group."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM provisions WHERE parallel_group = ? ORDER BY ts", (group_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_instance_ip(instance_id: str, ip_address: str):
    """Store resolved IP for a provisioned instance."""
    conn = _conn()
    conn.execute(
        "UPDATE provisions SET ip_address = ? WHERE instance_id = ?",
        (ip_address, instance_id),
    )
    conn.commit()
    conn.close()


def set_ssh_key_path(parallel_group: str, ssh_key_path: str):
    """Store the encrypted SSH key path for all instances in a provision group."""
    conn = _conn()
    conn.execute(
        "UPDATE provisions SET ssh_key_path = ? WHERE parallel_group = ?",
        (ssh_key_path, parallel_group),
    )
    conn.commit()
    conn.close()


def get_provision_ssh_key_path(parallel_group: str) -> Optional[str]:
    """Get the SSH key path stored for a provision group."""
    conn = _conn()
    row = conn.execute(
        "SELECT ssh_key_path FROM provisions WHERE parallel_group = ? AND ssh_key_path != '' LIMIT 1",
        (parallel_group,),
    ).fetchone()
    conn.close()
    return row["ssh_key_path"] if row else None


def get_active_instances(parallel_group: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get active (non-terminated) instances, optionally filtered by group."""
    conn = _conn()
    if parallel_group:
        rows = conn.execute(
            "SELECT * FROM provisions WHERE parallel_group = ? AND status = 'active' ORDER BY ts",
            (parallel_group,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM provisions WHERE status = 'active' ORDER BY ts DESC",
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_parallel_group() -> Optional[str]:
    """Get the most recent parallel group ID."""
    conn = _conn()
    row = conn.execute(
        "SELECT parallel_group FROM provisions WHERE parallel_group IS NOT NULL "
        "ORDER BY ts DESC LIMIT 1",
    ).fetchone()
    conn.close()
    return row["parallel_group"] if row else None
