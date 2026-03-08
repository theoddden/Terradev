#!/usr/bin/env python3
"""
GPU Price Intelligence Engine — delta, gamma, realized volatility, and
training/inference price segmentation.

All data lives in the same ~/.terradev/cost_tracking.db alongside the
existing cost tracker tables.  New tables:

    price_ticks     – raw tick-level price observations (one row per
                      gpu_type × provider × region × workload_type per
                      observation timestamp)
    price_stats     – pre-computed rolling statistics (σ, δ, γ) refreshed
                      on every new tick batch
    price_alerts    – user-defined price thresholds and trigger history

The module exposes:
    record_price_tick()          – called by quote/provision paths
    compute_delta()              – first derivative (rate of change)
    compute_gamma()              – second derivative (acceleration)
    compute_realized_volatility() – annualized realized volatility
    get_price_series()           – retrieve time-series for a key
    get_insights()               – full dashboard dict for a GPU type
    refresh_stats()              – batch recompute δ/γ/σ for all series
"""

import sqlite3
import math
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

DB_PATH = Path.home() / ".terradev" / "cost_tracking.db"


# ═══════════════════════════════════════════════════════════════════════
# Database helpers
# ═══════════════════════════════════════════════════════════════════════

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_price_schema(conn)
    return conn


def _ensure_price_schema(conn: sqlite3.Connection):
    conn.executescript("""
        -- Raw price observations (tick level)
        CREATE TABLE IF NOT EXISTS price_ticks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL DEFAULT (datetime('now')),
            gpu_type        TEXT    NOT NULL,
            provider        TEXT    NOT NULL,
            region          TEXT    NOT NULL DEFAULT '',
            price_hr        REAL    NOT NULL,
            spot            INTEGER NOT NULL DEFAULT 0,
            workload_type   TEXT    NOT NULL DEFAULT 'training',
            source          TEXT    NOT NULL DEFAULT 'quote'
        );

        CREATE INDEX IF NOT EXISTS idx_ticks_series
            ON price_ticks (gpu_type, provider, spot, workload_type, ts);

        CREATE INDEX IF NOT EXISTS idx_ticks_ts
            ON price_ticks (ts);

        -- Pre-computed rolling statistics per series
        CREATE TABLE IF NOT EXISTS price_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL DEFAULT (datetime('now')),
            gpu_type        TEXT    NOT NULL,
            provider        TEXT    NOT NULL,
            spot            INTEGER NOT NULL DEFAULT 0,
            workload_type   TEXT    NOT NULL DEFAULT 'training',
            -- Current price snapshot
            price_latest    REAL,
            -- Delta (rate of change)
            delta_1h        REAL,
            delta_24h       REAL,
            delta_7d        REAL,
            -- Gamma (acceleration)
            gamma_1h        REAL,
            gamma_24h       REAL,
            gamma_7d        REAL,
            -- Volatility
            volatility_24h  REAL,
            volatility_7d   REAL,
            volatility_30d  REAL,
            -- Descriptive
            price_min_24h   REAL,
            price_max_24h   REAL,
            price_mean_24h  REAL,
            price_min_7d    REAL,
            price_max_7d    REAL,
            price_mean_7d   REAL,
            tick_count       INTEGER NOT NULL DEFAULT 0,
            UNIQUE(gpu_type, provider, spot, workload_type)
        );

        -- User-defined price thresholds and trigger history
        CREATE TABLE IF NOT EXISTS price_alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL DEFAULT (datetime('now')),
            gpu_type        TEXT    NOT NULL,
            provider        TEXT    NOT NULL,
            spot            INTEGER NOT NULL DEFAULT 0,
            workload_type   TEXT    NOT NULL DEFAULT 'training',
            threshold_type  TEXT    NOT NULL,  -- 'above', 'below', 'delta_above', 'delta_below'
            threshold_value REAL    NOT NULL,
            active          INTEGER NOT NULL DEFAULT 1,
            triggered_ts    TEXT,
            triggered_value REAL,
            UNIQUE(gpu_type, provider, spot, workload_type, threshold_type, threshold_value)
        );

        -- Availability and performance logs
        CREATE TABLE IF NOT EXISTS availability_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL DEFAULT (datetime('now')),
            gpu_type        TEXT    NOT NULL,
            provider        TEXT    NOT NULL,
            region          TEXT    NOT NULL DEFAULT '',
            available       INTEGER NOT NULL DEFAULT 1,
            response_time_ms REAL,
            error_message   TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_availability_series
            ON availability_log (gpu_type, provider, region, ts);

        -- Provider events (maintenance, outages, reliability metrics)
        CREATE TABLE IF NOT EXISTS provider_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL DEFAULT (datetime('now')),
            provider        TEXT    NOT NULL,
            event_type      TEXT    NOT NULL,  -- 'maintenance', 'outage', 'price_change', 'capacity', 'quote', 'provision', 'execute', 'manage', 'terminate'
            severity        TEXT,              -- 'low', 'medium', 'high', 'critical' (infrastructure events)
            title           TEXT,              -- infrastructure events
            description     TEXT,
            affected_regions TEXT,             -- JSON array of regions (infrastructure events)
            resolved_ts     TEXT,              -- infrastructure events
            gpu_type        TEXT    NOT NULL DEFAULT '',   -- reliability metrics
            region          TEXT    NOT NULL DEFAULT '',   -- reliability metrics
            success         INTEGER,           -- reliability metrics (1=ok, 0=fail)
            latency_ms      REAL,              -- reliability metrics
            error           TEXT               -- reliability metrics
        );

        CREATE INDEX IF NOT EXISTS idx_events_provider
            ON provider_events (provider, ts);
    """)
    conn.commit()

    # Migrate existing DBs: add reliability columns if missing
    cursor = conn.execute("PRAGMA table_info(provider_events)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    _new_cols = [
        ("gpu_type",   "TEXT NOT NULL DEFAULT ''"),
        ("region",     "TEXT NOT NULL DEFAULT ''"),
        ("success",    "INTEGER"),
        ("latency_ms", "REAL"),
        ("error",      "TEXT"),
    ]
    for col_name, col_def in _new_cols:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE provider_events ADD COLUMN {col_name} {col_def}")
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════
# Tick recording
# ═══════════════════════════════════════════════════════════════════════

def record_price_tick(
    gpu_type: str,
    provider: str,
    price_hr: float,
    region: str = "",
    spot: bool = False,
    workload_type: str = "training",
    source: str = "quote",
):
    """Record a single price observation."""
    conn = _conn()
    conn.execute(
        "INSERT INTO price_ticks "
        "(gpu_type, provider, region, price_hr, spot, workload_type, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (gpu_type.upper(), provider.lower(), region, price_hr,
         1 if spot else 0, workload_type.lower(), source),
    )
    conn.commit()
    conn.close()


def record_price_ticks_batch(ticks: List[Dict[str, Any]]):
    """Record multiple price observations in one transaction."""
    conn = _conn()
    for t in ticks:
        conn.execute(
            "INSERT INTO price_ticks "
            "(gpu_type, provider, region, price_hr, spot, workload_type, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                t.get("gpu_type", "").upper(),
                t.get("provider", "").lower(),
                t.get("region", ""),
                t.get("price", t.get("price_hr", 0)),
                1 if t.get("spot") or t.get("availability") == "spot" else 0,
                t.get("workload_type", "training").lower(),
                t.get("source", "quote"),
            ),
        )
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════
# Time-series retrieval
# ═══════════════════════════════════════════════════════════════════════

def get_price_series(
    gpu_type: str,
    provider: Optional[str] = None,
    spot: Optional[bool] = None,
    workload_type: Optional[str] = None,
    hours: int = 168,
) -> List[Dict[str, Any]]:
    """Retrieve price time-series for a given GPU type.

    Returns list of {ts, price_hr, provider, spot, workload_type}.
    Default window: 168 hours (7 days).
    """
    conn = _conn()
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    sql = "SELECT ts, price_hr, provider, region, spot, workload_type FROM price_ticks WHERE gpu_type = ? AND ts >= ?"
    params: list = [gpu_type.upper(), cutoff]

    if provider:
        sql += " AND provider = ?"
        params.append(provider.lower())
    if spot is not None:
        sql += " AND spot = ?"
        params.append(1 if spot else 0)
    if workload_type:
        sql += " AND workload_type = ?"
        params.append(workload_type.lower())

    sql += " ORDER BY ts ASC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _get_prices_array(
    conn: sqlite3.Connection,
    gpu_type: str,
    provider: str,
    spot: int,
    workload_type: str,
    hours: int,
) -> List[float]:
    """Internal: get raw price array for a series within a time window."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        "SELECT price_hr FROM price_ticks "
        "WHERE gpu_type = ? AND provider = ? AND spot = ? AND workload_type = ? AND ts >= ? "
        "ORDER BY ts ASC",
        (gpu_type, provider, spot, workload_type, cutoff),
    ).fetchall()
    return [r["price_hr"] for r in rows]


# ═══════════════════════════════════════════════════════════════════════
# Delta — first derivative (rate of price change)
# ═══════════════════════════════════════════════════════════════════════

def compute_delta(prices: List[float], window: int = 1) -> Optional[float]:
    """Compute percentage price change over `window` observations.

    δ = (P_t - P_{t-window}) / P_{t-window}

    Returns None if insufficient data.
    """
    if len(prices) < window + 1:
        return None
    p_now = prices[-1]
    p_prev = prices[-(window + 1)]
    if p_prev == 0:
        return None
    return (p_now - p_prev) / p_prev


def compute_delta_absolute(prices: List[float], window: int = 1) -> Optional[float]:
    """Absolute price change over `window` observations."""
    if len(prices) < window + 1:
        return None
    return prices[-1] - prices[-(window + 1)]


# ═══════════════════════════════════════════════════════════════════════
# Gamma — second derivative (acceleration of price change)
# ═══════════════════════════════════════════════════════════════════════

def compute_gamma(prices: List[float], window: int = 1) -> Optional[float]:
    """Compute acceleration of price change (second derivative).

    γ = δ_t - δ_{t-1}

    where δ_t = (P_t - P_{t-1}) / P_{t-1}

    Positive gamma = price acceleration (increasing faster)
    Negative gamma = price deceleration (slowing down or reversing)

    Returns None if insufficient data.
    """
    if len(prices) < window + 2:
        return None

    # Current delta
    delta_now = compute_delta(prices, window)
    # Previous delta (shift back by 1)
    delta_prev = compute_delta(prices[:-1], window)

    if delta_now is None or delta_prev is None:
        return None

    return delta_now - delta_prev


# ═══════════════════════════════════════════════════════════════════════
# Realized volatility
# ═══════════════════════════════════════════════════════════════════════

def compute_realized_volatility(prices: List[float]) -> Optional[float]:
    """Compute annualized realized volatility from log returns.

    σ = std(log_returns) × √(observations_per_year)

    Assumes ~hourly observations → 8760 obs/year.
    Returns None if < 3 observations.
    """
    if len(prices) < 3:
        return None

    log_returns = []
    for i in range(1, len(prices)):
        if prices[i] > 0 and prices[i - 1] > 0:
            log_returns.append(math.log(prices[i] / prices[i - 1]))

    if len(log_returns) < 2:
        return None

    mean_r = sum(log_returns) / len(log_returns)
    variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
    std_dev = math.sqrt(variance)

    # Annualize assuming hourly observations
    annualized = std_dev * math.sqrt(8760)
    return annualized


# ═══════════════════════════════════════════════════════════════════════
# Stats refresh — recompute all rolling statistics
# ═══════════════════════════════════════════════════════════════════════

def refresh_stats():
    """Recompute δ, γ, and σ for all active series."""
    conn = _conn()

    # Get all unique series
    series = conn.execute(
        "SELECT DISTINCT gpu_type, provider, spot, workload_type FROM price_ticks"
    ).fetchall()

    for s in series:
        gpu_type = s["gpu_type"]
        provider = s["provider"]
        spot = s["spot"]
        wtype = s["workload_type"]

        # Get price arrays for different windows
        prices_1h = _get_prices_array(conn, gpu_type, provider, spot, wtype, 1)
        prices_24h = _get_prices_array(conn, gpu_type, provider, spot, wtype, 24)
        prices_7d = _get_prices_array(conn, gpu_type, provider, spot, wtype, 168)
        prices_30d = _get_prices_array(conn, gpu_type, provider, spot, wtype, 720)

        # Latest price
        price_latest = prices_7d[-1] if prices_7d else None

        # Delta (percentage change)
        delta_1h = compute_delta(prices_1h) if len(prices_1h) >= 2 else None
        delta_24h = compute_delta(prices_24h) if len(prices_24h) >= 2 else None
        delta_7d = compute_delta(prices_7d) if len(prices_7d) >= 2 else None

        # Gamma (acceleration)
        gamma_1h = compute_gamma(prices_1h) if len(prices_1h) >= 3 else None
        gamma_24h = compute_gamma(prices_24h) if len(prices_24h) >= 3 else None
        gamma_7d = compute_gamma(prices_7d) if len(prices_7d) >= 3 else None

        # Realized volatility
        vol_24h = compute_realized_volatility(prices_24h)
        vol_7d = compute_realized_volatility(prices_7d)
        vol_30d = compute_realized_volatility(prices_30d)

        # Descriptive stats
        price_min_24h = min(prices_24h) if prices_24h else None
        price_max_24h = max(prices_24h) if prices_24h else None
        price_mean_24h = sum(prices_24h) / len(prices_24h) if prices_24h else None
        price_min_7d = min(prices_7d) if prices_7d else None
        price_max_7d = max(prices_7d) if prices_7d else None
        price_mean_7d = sum(prices_7d) / len(prices_7d) if prices_7d else None

        tick_count = len(prices_30d)

        # Upsert price_stats
        conn.execute(
            """INSERT INTO price_stats
               (gpu_type, provider, spot, workload_type,
                price_latest,
                delta_1h, delta_24h, delta_7d,
                gamma_1h, gamma_24h, gamma_7d,
                volatility_24h, volatility_7d, volatility_30d,
                price_min_24h, price_max_24h, price_mean_24h,
                price_min_7d, price_max_7d, price_mean_7d,
                tick_count, ts)
               VALUES (?, ?, ?, ?,
                       ?, ?, ?, ?,
                       ?, ?, ?,
                       ?, ?, ?,
                       ?, ?, ?,
                       ?, ?, ?,
                       ?, datetime('now'))
               ON CONFLICT(gpu_type, provider, spot, workload_type)
               DO UPDATE SET
                   price_latest = excluded.price_latest,
                   delta_1h = excluded.delta_1h,
                   delta_24h = excluded.delta_24h,
                   delta_7d = excluded.delta_7d,
                   gamma_1h = excluded.gamma_1h,
                   gamma_24h = excluded.gamma_24h,
                   gamma_7d = excluded.gamma_7d,
                   volatility_24h = excluded.volatility_24h,
                   volatility_7d = excluded.volatility_7d,
                   volatility_30d = excluded.volatility_30d,
                   price_min_24h = excluded.price_min_24h,
                   price_max_24h = excluded.price_max_24h,
                   price_mean_24h = excluded.price_mean_24h,
                   price_min_7d = excluded.price_min_7d,
                   price_max_7d = excluded.price_max_7d,
                   price_mean_7d = excluded.price_mean_7d,
                   tick_count = excluded.tick_count,
                   ts = excluded.ts
            """,
            (gpu_type, provider, spot, wtype,
             price_latest,
             delta_1h, delta_24h, delta_7d,
             gamma_1h, gamma_24h, gamma_7d,
             vol_24h, vol_7d, vol_30d,
             price_min_24h, price_max_24h, price_mean_24h,
             price_min_7d, price_max_7d, price_mean_7d,
             tick_count),
        )

    conn.commit()
    conn.close()
    return len(series)


# ═══════════════════════════════════════════════════════════════════════
# Insights — full dashboard for a GPU type
# ═══════════════════════════════════════════════════════════════════════

def get_insights(
    gpu_type: str,
    workload_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Get full price intelligence dashboard for a GPU type.

    Returns dict with per-provider stats, deltas, gammas, volatility,
    and recommendations.
    """
    conn = _conn()

    sql = "SELECT * FROM price_stats WHERE gpu_type = ?"
    params: list = [gpu_type.upper()]
    if workload_type:
        sql += " AND workload_type = ?"
        params.append(workload_type.lower())

    rows = conn.execute(sql, params).fetchall()

    providers = []
    for r in rows:
        entry = dict(r)

        # Stability score: lower volatility + lower |gamma| = more stable
        vol = r["volatility_7d"] or 0
        gamma = abs(r["gamma_24h"] or 0)
        entry["stability_score"] = round(max(0, 100 - (vol * 100) - (gamma * 500)), 1)

        providers.append(entry)

    conn.close()

    # Sort by price
    providers.sort(key=lambda x: x.get("price_latest") or 999)

    # Recommendations
    cheapest = providers[0] if providers else None
    most_stable = max(providers, key=lambda x: x.get("stability_score", 0)) if providers else None

    # Find best value: cheapest among stable options (stability > 60)
    stable_options = [p for p in providers if p.get("stability_score", 0) > 60]
    best_value = min(stable_options, key=lambda x: x.get("price_latest") or 999) if stable_options else cheapest

    return {
        "gpu_type": gpu_type.upper(),
        "workload_type": workload_type or "all",
        "providers": providers,
        "recommendations": {
            "cheapest": {
                "provider": cheapest["provider"] if cheapest else None,
                "price": cheapest["price_latest"] if cheapest else None,
                "spot": bool(cheapest["spot"]) if cheapest else None,
            } if cheapest else None,
            "most_stable": {
                "provider": most_stable["provider"] if most_stable else None,
                "stability_score": most_stable["stability_score"] if most_stable else None,
                "price": most_stable["price_latest"] if most_stable else None,
            } if most_stable else None,
            "best_value": {
                "provider": best_value["provider"] if best_value else None,
                "price": best_value["price_latest"] if best_value else None,
                "stability_score": best_value["stability_score"] if best_value else None,
                "reason": "Cheapest option with stability score > 60",
            } if best_value else None,
        },
        "summary": {
            "total_providers": len(providers),
            "price_range": (
                f"${min(p['price_latest'] for p in providers if p.get('price_latest')):.2f}"
                f" — ${max(p['price_latest'] for p in providers if p.get('price_latest')):.2f}"
            ) if providers and any(p.get("price_latest") for p in providers) else "N/A",
        },
    }


def get_all_tracked_gpus() -> List[str]:
    """Return list of all GPU types with price data."""
    conn = _conn()
    rows = conn.execute("SELECT DISTINCT gpu_type FROM price_ticks ORDER BY gpu_type").fetchall()
    conn.close()
    return [r["gpu_type"] for r in rows]


def get_training_vs_inference(gpu_type: str) -> Dict[str, Any]:
    """Compare training vs inference pricing for a GPU type."""
    conn = _conn()

    result = {}
    for wtype in ["training", "inference"]:
        row = conn.execute(
            "SELECT * FROM price_stats WHERE gpu_type = ? AND workload_type = ? ORDER BY price_latest ASC",
            (gpu_type.upper(), wtype),
        ).fetchall()
        result[wtype] = {
            "providers": len(row),
            "cheapest": dict(row[0]) if row else None,
            "all": [dict(r) for r in row],
        }

    conn.close()

    # Compute premium/discount
    train_price = result["training"]["cheapest"]["price_latest"] if result["training"]["cheapest"] else None
    infer_price = result["inference"]["cheapest"]["price_latest"] if result["inference"]["cheapest"] else None

    if train_price and infer_price and train_price > 0:
        result["inference_premium"] = round((infer_price - train_price) / train_price * 100, 1)
    else:
        result["inference_premium"] = None

    return result


# ═══════════════════════════════════════════════════════════════════════
# FEATURE: Historical Price Percentiles
# ═══════════════════════════════════════════════════════════════════════

def _percentile(sorted_values: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


def compute_percentiles(
    gpu_type: str,
    provider: Optional[str] = None,
    spot: Optional[bool] = None,
    hours: int = 720,
) -> Dict[str, Any]:
    """Compute price percentiles (p10, p25, p50, p75, p90, p99) per provider.

    Default window: 720 hours (30 days).
    """
    conn = _conn()
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    sql = ("SELECT provider, price_hr FROM price_ticks "
           "WHERE gpu_type = ? AND ts >= ?")
    params: list = [gpu_type.upper(), cutoff]
    if provider:
        sql += " AND provider = ?"
        params.append(provider.lower())
    if spot is not None:
        sql += " AND spot = ?"
        params.append(1 if spot else 0)
    sql += " ORDER BY provider, price_hr ASC"

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    from collections import defaultdict
    by_provider: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_provider[r["provider"]].append(r["price_hr"])

    providers: Dict[str, Any] = {}
    for prov, prices in by_provider.items():
        prices.sort()
        providers[prov] = {
            "p10": round(_percentile(prices, 10), 4),
            "p25": round(_percentile(prices, 25), 4),
            "p50": round(_percentile(prices, 50), 4),
            "p75": round(_percentile(prices, 75), 4),
            "p90": round(_percentile(prices, 90), 4),
            "p99": round(_percentile(prices, 99), 4),
            "min": round(prices[0], 4),
            "max": round(prices[-1], 4),
            "count": len(prices),
        }

    return {
        "gpu_type": gpu_type.upper(),
        "window_hours": hours,
        "providers": providers,
    }


# ═══════════════════════════════════════════════════════════════════════
# FEATURE: Availability / Stock Tracking
# ═══════════════════════════════════════════════════════════════════════

def record_availability(
    gpu_type: str,
    provider: str,
    available: bool,
    region: str = "",
    response_ms: Optional[float] = None,
    error: Optional[str] = None,
):
    """Record a single availability check (called during quote/provision)."""
    conn = _conn()
    conn.execute(
        "INSERT INTO availability_log "
        "(gpu_type, provider, region, available, response_ms, error) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (gpu_type.upper(), provider.lower(), region,
         1 if available else 0, response_ms, error),
    )
    conn.commit()
    conn.close()


def record_availability_batch(checks: List[Dict[str, Any]]):
    """Record multiple availability checks in one transaction."""
    conn = _conn()
    for c in checks:
        conn.execute(
            "INSERT INTO availability_log "
            "(gpu_type, provider, region, available, response_ms, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                c.get("gpu_type", "").upper(),
                c.get("provider", "").lower(),
                c.get("region", ""),
                1 if c.get("available", True) else 0,
                c.get("response_ms"),
                c.get("error"),
            ),
        )
    conn.commit()
    conn.close()


def get_availability(
    gpu_type: str,
    hours: int = 24,
) -> Dict[str, Any]:
    """Get availability status for a GPU type across all providers.

    Returns per-provider: available (bool), availability_rate (0-1),
    avg_response_ms, total_checks, last_seen, last_error.
    """
    conn = _conn()
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    rows = conn.execute(
        "SELECT provider, region, available, response_ms, error, ts "
        "FROM availability_log "
        "WHERE gpu_type = ? AND ts >= ? "
        "ORDER BY provider, ts DESC",
        (gpu_type.upper(), cutoff),
    ).fetchall()
    conn.close()

    from collections import defaultdict
    by_provider: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_provider[r["provider"]].append(dict(r))

    result = {}
    for prov, checks in by_provider.items():
        total = len(checks)
        available_count = sum(1 for c in checks if c["available"])
        ms_values = [c["response_ms"] for c in checks if c["response_ms"] is not None]
        avg_ms = sum(ms_values) / len(ms_values) if ms_values else 0
        last = checks[0]
        last_error = next((c["error"] for c in checks if c["error"]), None)

        result[prov] = {
            "available": bool(last["available"]),
            "last_seen": last["ts"],
            "region": last["region"],
            "availability_rate": round(available_count / total, 4) if total else 0,
            "avg_response_ms": round(avg_ms, 1),
            "total_checks": total,
            "available_checks": available_count,
            "last_error": last_error,
        }

    return {
        "gpu_type": gpu_type.upper(),
        "window_hours": hours,
        "providers": result,
    }


def get_availability_summary() -> Dict[str, Dict[str, bool]]:
    """Quick summary: {gpu_type: {provider: True/False}} based on last check."""
    conn = _conn()
    rows = conn.execute(
        "SELECT gpu_type, provider, available, MAX(ts) as last_ts "
        "FROM availability_log "
        "GROUP BY gpu_type, provider "
        "ORDER BY gpu_type, provider"
    ).fetchall()
    conn.close()

    from collections import defaultdict
    summary: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for r in rows:
        summary[r["gpu_type"]][r["provider"]] = bool(r["available"])
    return dict(summary)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE: Provider Reliability Scoring
# ═══════════════════════════════════════════════════════════════════════

def record_provider_event(
    provider: str,
    event_type: str,
    success: bool,
    gpu_type: str = "",
    region: str = "",
    latency_ms: Optional[float] = None,
    error: Optional[str] = None,
):
    """Record a provider interaction event.

    event_type: 'quote', 'provision', 'execute', 'manage', 'terminate'
    """
    conn = _conn()
    conn.execute(
        "INSERT INTO provider_events "
        "(provider, event_type, gpu_type, region, success, latency_ms, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (provider.lower(), event_type.lower(), gpu_type.upper(), region,
         1 if success else 0, latency_ms, error),
    )
    conn.commit()
    conn.close()


def get_provider_reliability(
    provider: Optional[str] = None,
    hours: int = 720,
) -> Dict[str, Any]:
    """Compute reliability scores for providers.

    Returns per-provider:
        overall_score (0-100), quote_success_rate, provision_success_rate,
        avg_quote_latency_ms, avg_provision_latency_ms, total_events,
        error_breakdown {error_msg: count}.
    """
    conn = _conn()
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    sql = ("SELECT provider, event_type, success, latency_ms, error "
           "FROM provider_events WHERE ts >= ?")
    params: list = [cutoff]
    if provider:
        sql += " AND provider = ?"
        params.append(provider.lower())
    sql += " ORDER BY provider"

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    from collections import defaultdict
    by_provider: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_provider[r["provider"]].append(dict(r))

    result = {}
    for prov, events in by_provider.items():
        total = len(events)

        # Per event-type breakdown
        quotes = [e for e in events if e["event_type"] == "quote"]
        provisions = [e for e in events if e["event_type"] == "provision"]
        other = [e for e in events if e["event_type"] not in ("quote", "provision")]

        quote_success = sum(1 for e in quotes if e["success"]) / len(quotes) if quotes else 1.0
        provision_success = sum(1 for e in provisions if e["success"]) / len(provisions) if provisions else 1.0
        other_success = sum(1 for e in other if e["success"]) / len(other) if other else 1.0

        # Latency averages
        quote_latencies = [e["latency_ms"] for e in quotes if e["latency_ms"] is not None]
        provision_latencies = [e["latency_ms"] for e in provisions if e["latency_ms"] is not None]
        avg_quote_ms = sum(quote_latencies) / len(quote_latencies) if quote_latencies else 0
        avg_provision_ms = sum(provision_latencies) / len(provision_latencies) if provision_latencies else 0

        # Error breakdown
        errors: Dict[str, int] = defaultdict(int)
        for e in events:
            if e["error"]:
                errors[e["error"][:80]] += 1

        # Overall score: weighted average
        # Provision success matters most (50%), quote success (30%), other (20%)
        overall = (
            provision_success * 50 +
            quote_success * 30 +
            other_success * 20
        )
        # Latency penalty: subtract up to 10 points for slow providers
        if avg_quote_ms > 5000:
            overall -= min(10, (avg_quote_ms - 5000) / 1000)

        result[prov] = {
            "overall_score": round(max(0, min(100, overall)), 1),
            "quote_success_rate": round(quote_success, 4),
            "provision_success_rate": round(provision_success, 4),
            "avg_quote_latency_ms": round(avg_quote_ms, 1),
            "avg_provision_latency_ms": round(avg_provision_ms, 1),
            "total_events": total,
            "quotes": len(quotes),
            "provisions": len(provisions),
            "errors": dict(errors),
        }

    return {
        "window_hours": hours,
        "providers": result,
    }


def get_provider_ranking() -> List[Dict[str, Any]]:
    """Rank all providers by reliability score (descending)."""
    data = get_provider_reliability()
    ranked = []
    for prov, stats in data["providers"].items():
        ranked.append({"provider": prov, **stats})
    ranked.sort(key=lambda x: x["overall_score"], reverse=True)
    return ranked
