#!/usr/bin/env python3
"""
Egress Optimizer — Intelligent data routing to minimize transfer costs between clouds.

Features:
  1. Live pricing cache — refreshes rates from provider APIs and caches in
     ~/.terradev/cost_tracking.db, falling back to built-in static rates.
  2. Multi-hop route optimization — Dijkstra over the provider graph to find
     the cheapest A→…→Z path (e.g. AWS→RunPod is free because RunPod egress
     is $0, so staging through RunPod saves money).
  3. Dataset stager integration — optimize_staging_route() returns the cheapest
     route for DatasetStager to use automatically.

All 19 Terradev-supported providers are covered.
"""

import heapq
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path.home() / ".terradev" / "cost_tracking.db"

# How often to re-fetch live pricing (seconds).  Default: 6 hours.
CACHE_TTL_S = 6 * 3600


# ═══════════════════════════════════════════════════════════════════════
# Static egress pricing per GB (USD) — all 19 providers
# ═══════════════════════════════════════════════════════════════════════
# Source: public pricing pages as of 2025-Q4.  Intra-region is free on
# most clouds; same-continent cross-region is cheaper than intercontinental.

_STATIC_EGRESS_RATES: Dict[str, Dict[str, float]] = {
    # ── Hyperscalers ─────────────────────────────────────────────────
    "aws": {
        "same_region": 0.00,
        "same_continent": 0.01,
        "cross_continent": 0.09,
        "internet": 0.09,
    },
    "gcp": {
        "same_region": 0.00,
        "same_continent": 0.01,
        "cross_continent": 0.08,
        "internet": 0.12,
    },
    "azure": {
        "same_region": 0.00,
        "same_continent": 0.01,
        "cross_continent": 0.08,
        "internet": 0.087,
    },
    "oracle": {
        "same_region": 0.00,
        "same_continent": 0.0085,
        "cross_continent": 0.0085,
        "internet": 0.0085,
    },
    "alibaba": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.074,
        "internet": 0.074,
    },
    # ── GPU-native clouds (zero or near-zero egress) ─────────────────
    "runpod": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.00,
        "internet": 0.00,
    },
    "vastai": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.00,
        "internet": 0.00,
    },
    "lambda": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.00,
        "internet": 0.00,
    },
    "tensordock": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.00,
        "internet": 0.00,
    },
    "fluidstack": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.00,
        "internet": 0.00,
    },
    "siliconflow": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.00,
        "internet": 0.00,
    },
    # ── Mid-tier clouds ──────────────────────────────────────────────
    "coreweave": {
        "same_region": 0.00,
        "same_continent": 0.01,
        "cross_continent": 0.05,
        "internet": 0.05,
    },
    "crusoe": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.02,
        "internet": 0.02,
    },
    "hyperstack": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.01,
        "internet": 0.01,
    },
    "digitalocean": {
        "same_region": 0.00,
        "same_continent": 0.01,
        "cross_continent": 0.01,
        "internet": 0.01,
    },
    "ovhcloud": {
        "same_region": 0.00,
        "same_continent": 0.01,
        "cross_continent": 0.011,
        "internet": 0.011,
    },
    "hetzner": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.00,
        "internet": 0.00,    # included bandwidth is generous
    },
    # ── Inference platforms ───────────────────────────────────────────
    "baseten": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.00,
        "internet": 0.00,
    },
    "huggingface": {
        "same_region": 0.00,
        "same_continent": 0.00,
        "cross_continent": 0.00,
        "internet": 0.00,
    },
}

# Alias for backward-compat (old code references EGRESS_RATES directly)
EGRESS_RATES = _STATIC_EGRESS_RATES

# ── Zero-egress providers (used for multi-hop relay selection) ───────
_ZERO_EGRESS_PROVIDERS = frozenset(
    p for p, rates in _STATIC_EGRESS_RATES.items()
    if all(v == 0.0 for v in rates.values())
)

# ═══════════════════════════════════════════════════════════════════════
# Region → continent mapping (expanded for all 19 providers)
# ═══════════════════════════════════════════════════════════════════════

REGION_CONTINENT: Dict[str, str] = {
    # AWS
    "us-east-1": "na", "us-east-2": "na", "us-west-1": "na", "us-west-2": "na",
    "ca-central-1": "na",
    "eu-west-1": "eu", "eu-west-2": "eu", "eu-west-3": "eu",
    "eu-central-1": "eu", "eu-north-1": "eu", "eu-south-1": "eu",
    "ap-southeast-1": "ap", "ap-southeast-2": "ap",
    "ap-northeast-1": "ap", "ap-northeast-2": "ap", "ap-south-1": "ap",
    "sa-east-1": "sa", "me-south-1": "me", "af-south-1": "af",
    # GCP
    "us-central1": "na", "us-east4": "na", "us-west1": "na", "us-west4": "na",
    "europe-west1": "eu", "europe-west2": "eu", "europe-west3": "eu",
    "europe-west4": "eu", "europe-north1": "eu",
    "asia-east1": "ap", "asia-east2": "ap", "asia-southeast1": "ap",
    "asia-northeast1": "ap", "asia-south1": "ap",
    # Azure
    "eastus": "na", "eastus2": "na", "westus": "na", "westus2": "na",
    "westus3": "na", "centralus": "na", "southcentralus": "na",
    "northeurope": "eu", "westeurope": "eu", "uksouth": "eu", "ukwest": "eu",
    "francecentral": "eu", "germanywestcentral": "eu", "swedencentral": "eu",
    "southeastasia": "ap", "eastasia": "ap", "japaneast": "ap",
    "australiaeast": "ap", "centralindia": "ap",
    # Oracle
    "us-ashburn-1": "na", "us-phoenix-1": "na", "us-chicago-1": "na",
    "uk-london-1": "eu", "eu-frankfurt-1": "eu", "eu-amsterdam-1": "eu",
    "ap-tokyo-1": "ap", "ap-mumbai-1": "ap", "ap-sydney-1": "ap",
    # Alibaba
    "cn-hangzhou": "ap", "cn-shanghai": "ap", "cn-beijing": "ap",
    "cn-shenzhen": "ap", "cn-hongkong": "ap",
    "us-west-1-alibaba": "na", "eu-central-1-alibaba": "eu",
    # OVHcloud
    "gra": "eu", "sbg": "eu", "bhs": "na", "de1": "eu", "uk1": "eu",
    "waw": "eu", "sgp": "ap",
    # Hetzner
    "fsn1": "eu", "nbg1": "eu", "hel1": "eu", "ash": "na", "hil": "na",
    # GPU cloud regions (generic, from smaller providers)
    "us-east": "na", "us-west": "na", "us-central": "na",
    "eu-west": "eu", "eu-central": "eu",
    "ap-east": "ap", "ap-south": "ap",
}


# ═══════════════════════════════════════════════════════════════════════
# Live pricing cache (SQLite)
# ═══════════════════════════════════════════════════════════════════════

def _db_conn() -> sqlite3.Connection:
    """Open/create the cost_tracking DB with the egress_cache table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS egress_rate_cache (
            provider        TEXT    NOT NULL,
            dest_class      TEXT    NOT NULL,
            rate_per_gb     REAL    NOT NULL,
            source          TEXT    NOT NULL DEFAULT 'static',
            updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (provider, dest_class)
        );
    """)
    return conn


def refresh_egress_cache(force: bool = False) -> Dict[str, int]:
    """
    Refresh the egress_rate_cache table.

    Attempts to pull live rates from provider pricing APIs.  Falls back to
    the built-in static table for any provider whose API is unavailable.

    Returns {"updated": N, "cached": M, "errors": E}.
    """
    conn = _db_conn()
    stats = {"updated": 0, "cached": 0, "errors": 0}

    # Check cache freshness
    if not force:
        row = conn.execute(
            "SELECT MIN(updated_at) AS oldest FROM egress_rate_cache"
        ).fetchone()
        if row and row["oldest"]:
            try:
                oldest = datetime.fromisoformat(row["oldest"])
                if (datetime.utcnow() - oldest).total_seconds() < CACHE_TTL_S:
                    stats["cached"] = conn.execute(
                        "SELECT COUNT(*) AS n FROM egress_rate_cache"
                    ).fetchone()["n"]
                    conn.close()
                    return stats
            except (ValueError, TypeError):
                pass  # corrupt timestamp, force refresh

    # Try live pricing for each provider, fall back to static
    for provider, static_rates in _STATIC_EGRESS_RATES.items():
        live_rates = _fetch_live_rates(provider)
        rates = live_rates if live_rates else static_rates
        source = "live" if live_rates else "static"
        if not live_rates:
            stats["errors"] += 1

        for dest_class, rate in rates.items():
            conn.execute("""
                INSERT INTO egress_rate_cache (provider, dest_class, rate_per_gb, source, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(provider, dest_class) DO UPDATE SET
                    rate_per_gb = excluded.rate_per_gb,
                    source      = excluded.source,
                    updated_at  = excluded.updated_at
            """, (provider, dest_class, rate, source))
        stats["updated"] += 1

    conn.commit()
    conn.close()
    logger.info("Egress cache refreshed: %s", stats)
    return stats


def _fetch_live_rates(provider: str) -> Optional[Dict[str, float]]:
    """
    Attempt to fetch live egress rates from a provider's pricing API.

    Returns None if the API is unavailable or unsupported.
    Providers with known public pricing endpoints are queried here;
    GPU-native clouds with zero egress always return the static dict.
    """
    # Zero-egress providers never change — skip the API call
    if provider in _ZERO_EGRESS_PROVIDERS:
        return _STATIC_EGRESS_RATES[provider]

    # AWS — use the Price List bulk JSON
    if provider == "aws":
        try:
            import urllib.request
            url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSDataTransfer/current/index.json"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            # Extract per-GB internet egress from the US Standard tier
            for sku, product in data.get("products", {}).items():
                attrs = product.get("attributes", {})
                if (attrs.get("transferType") == "AWS Outbound"
                        and attrs.get("fromLocation") == "US East (N. Virginia)"):
                    terms = data.get("terms", {}).get("OnDemand", {}).get(sku, {})
                    for _, dim in terms.items():
                        for _, price_dim in dim.get("priceDimensions", {}).items():
                            usd = float(price_dim.get("pricePerUnit", {}).get("USD", "0"))
                            if usd > 0:
                                return {
                                    "same_region": 0.00,
                                    "same_continent": 0.01,
                                    "cross_continent": usd,
                                    "internet": usd,
                                }
        except Exception as e:
            logger.debug("AWS live pricing unavailable: %s", e)
        return None

    # GCP — use the Cloud Billing catalog (simplified)
    if provider == "gcp":
        try:
            import urllib.request
            # GCP doesn't have a simple unauthenticated egress endpoint,
            # but their pricing page JSON is publicly accessible
            url = "https://cloudpricingcalculator.appspot.com/static/data/pricelist.json"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            egress_key = "CP-COMPUTEENGINE-INTERNET-EGRESS-NA-NA"
            if egress_key in data.get("gcp_price_list", {}):
                usd = data["gcp_price_list"][egress_key].get("us", 0)
                if usd > 0:
                    return {
                        "same_region": 0.00,
                        "same_continent": 0.01,
                        "cross_continent": usd,
                        "internet": usd,
                    }
        except Exception as e:
            logger.debug("GCP live pricing unavailable: %s", e)
        return None

    # All other providers — no public pricing API, use static
    return None


def get_cached_rate(provider: str, dest_class: str) -> float:
    """
    Get an egress rate, preferring cached live data, falling back to static.
    """
    try:
        conn = _db_conn()
        row = conn.execute(
            "SELECT rate_per_gb FROM egress_rate_cache WHERE provider = ? AND dest_class = ?",
            (provider.lower(), dest_class),
        ).fetchone()
        conn.close()
        if row:
            return row["rate_per_gb"]
    except Exception:
        pass
    # Fallback to static
    rates = _STATIC_EGRESS_RATES.get(provider.lower(), _STATIC_EGRESS_RATES.get("aws", {}))
    return rates.get(dest_class, rates.get("internet", 0.09))


# ═══════════════════════════════════════════════════════════════════════
# Core routing logic
# ═══════════════════════════════════════════════════════════════════════

def _continent(region: str) -> str:
    """Best-effort continent lookup."""
    r = region.lower().strip()
    if r in REGION_CONTINENT:
        return REGION_CONTINENT[r]
    if r.startswith("us") or r.startswith("na") or r.startswith("ca-"):
        return "na"
    if r.startswith("eu") or r.startswith("north") or r.startswith("west") or r.startswith("uk"):
        return "eu"
    if r.startswith("ap") or r.startswith("asia") or r.startswith("south") or r.startswith("cn-"):
        return "ap"
    if r.startswith("sa") or r.startswith("brazil"):
        return "sa"
    if r.startswith("me") or r.startswith("af"):
        return "me"
    return "unknown"


def _dest_class(src_provider: str, src_region: str, dst_provider: str, dst_region: str) -> str:
    """Classify the transfer destination for pricing lookup."""
    if src_provider == dst_provider and src_region == dst_region:
        return "same_region"
    if src_provider == dst_provider and _continent(src_region) == _continent(dst_region):
        return "same_continent"
    if _continent(src_region) == _continent(dst_region):
        return "same_continent"
    if src_provider != dst_provider:
        return "internet"
    return "cross_continent"


def estimate_egress_cost(
    src_provider: str,
    src_region: str,
    dst_provider: str,
    dst_region: str,
    size_gb: float,
) -> float:
    """Estimate egress cost in USD for a direct transfer."""
    dest = _dest_class(src_provider.lower(), src_region, dst_provider.lower(), dst_region)
    rate = get_cached_rate(src_provider.lower(), dest)
    return round(rate * size_gb, 4)


def find_cheapest_route(
    src_provider: str,
    src_region: str,
    dst_candidates: List[Dict[str, str]],
    size_gb: float,
) -> List[Dict[str, Any]]:
    """
    Given a source and a list of destination candidates, rank them by egress cost.

    dst_candidates: [{"provider": "aws", "region": "us-east-1"}, ...]
    Returns sorted list with cost attached.
    """
    results = []
    for dst in dst_candidates:
        cost = estimate_egress_cost(
            src_provider, src_region, dst["provider"], dst["region"], size_gb
        )
        results.append({
            "provider": dst["provider"],
            "region": dst["region"],
            "egress_cost": cost,
            "size_gb": size_gb,
            "rate_per_gb": round(cost / max(size_gb, 0.001), 4),
        })
    results.sort(key=lambda x: x["egress_cost"])
    return results


# ═══════════════════════════════════════════════════════════════════════
# Multi-hop Dijkstra route optimizer
# ═══════════════════════════════════════════════════════════════════════

def _build_provider_graph(size_gb: float) -> Dict[str, Dict[str, float]]:
    """
    Build a weighted directed graph over providers.

    Edge weight = egress cost (USD) to move `size_gb` from provider A → B
    via the 'internet' rate class (cross-provider transfers).

    This graph enables Dijkstra to find cheapest multi-hop paths like:
        AWS ($0.09/GB) → RunPod ($0.00/GB) → destination
    which is cheaper than AWS → destination directly.
    """
    providers = list(_STATIC_EGRESS_RATES.keys())
    graph: Dict[str, Dict[str, float]] = {p: {} for p in providers}

    for src in providers:
        for dst in providers:
            if src == dst:
                continue
            rate = get_cached_rate(src, "internet")
            graph[src][dst] = round(rate * size_gb, 4)

    return graph


def find_cheapest_multihop(
    src_provider: str,
    dst_provider: str,
    size_gb: float,
    max_hops: int = 3,
) -> Dict[str, Any]:
    """
    Find the cheapest path from src_provider to dst_provider, potentially
    routing through intermediate zero-egress providers.

    Uses Dijkstra's algorithm over the provider egress graph.

    Returns:
        {
            "path": ["aws", "runpod", "coreweave"],
            "hops": 2,
            "total_cost": 0.00,
            "direct_cost": 4.50,
            "savings": 4.50,
            "legs": [
                {"from": "aws", "to": "runpod", "cost": 0.00},
                {"from": "runpod", "to": "coreweave", "cost": 0.00},
            ]
        }
    """
    src = src_provider.lower()
    dst = dst_provider.lower()

    if src == dst:
        return {
            "path": [src],
            "hops": 0,
            "total_cost": 0.0,
            "direct_cost": 0.0,
            "savings": 0.0,
            "legs": [],
        }

    graph = _build_provider_graph(size_gb)
    direct_cost = graph.get(src, {}).get(dst, size_gb * 0.09)

    # Dijkstra
    # Priority queue: (cost, current_node, path)
    pq: List[Tuple[float, str, List[str]]] = [(0.0, src, [src])]
    visited: Dict[str, float] = {}

    best_path = [src, dst]
    best_cost = direct_cost

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node == dst:
            if cost < best_cost:
                best_cost = cost
                best_path = path
            continue

        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost

        # Enforce max hops
        if len(path) - 1 >= max_hops:
            continue

        for neighbor, edge_cost in graph.get(node, {}).items():
            new_cost = cost + edge_cost
            if new_cost < best_cost and (neighbor not in visited or visited[neighbor] > new_cost):
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

    # Build leg details
    legs = []
    for i in range(len(best_path) - 1):
        leg_from = best_path[i]
        leg_to = best_path[i + 1]
        leg_cost = graph.get(leg_from, {}).get(leg_to, 0.0)
        legs.append({"from": leg_from, "to": leg_to, "cost": leg_cost})

    return {
        "path": best_path,
        "hops": len(best_path) - 1,
        "total_cost": round(best_cost, 4),
        "direct_cost": round(direct_cost, 4),
        "savings": round(max(direct_cost - best_cost, 0), 4),
        "legs": legs,
    }


# ═══════════════════════════════════════════════════════════════════════
# Transfer plan (upgraded with multi-hop)
# ═══════════════════════════════════════════════════════════════════════

def optimize_transfer_plan(
    data_location: Dict[str, str],
    compute_targets: List[Dict[str, str]],
    size_gb: float,
) -> Dict[str, Any]:
    """
    Build an optimized transfer plan with multi-hop routing.

    data_location: {"provider": "aws", "region": "us-east-1"}
    compute_targets: [{"provider": "runpod", "region": "us-east"}, ...]

    Returns a plan with per-target costs, total cost, multi-hop suggestions,
    and actionable recommendations.
    """
    src_prov = data_location["provider"].lower()
    src_reg = data_location["region"]

    routes = []
    multihop_savings = 0.0

    for target in compute_targets:
        dst_prov = target["provider"].lower()
        dst_reg = target["region"]

        direct_cost = estimate_egress_cost(src_prov, src_reg, dst_prov, dst_reg, size_gb)
        multihop = find_cheapest_multihop(src_prov, dst_prov, size_gb)

        route_entry = {
            "provider": dst_prov,
            "region": dst_reg,
            "direct_cost": direct_cost,
            "egress_cost": multihop["total_cost"],
            "size_gb": size_gb,
            "rate_per_gb": round(multihop["total_cost"] / max(size_gb, 0.001), 4),
        }

        if multihop["hops"] > 1 and multihop["savings"] > 0:
            route_entry["multihop"] = {
                "path": multihop["path"],
                "hops": multihop["hops"],
                "savings": multihop["savings"],
                "legs": multihop["legs"],
            }
            multihop_savings += multihop["savings"]

        routes.append(route_entry)

    routes.sort(key=lambda x: x["egress_cost"])

    total_cost = sum(r["egress_cost"] for r in routes)
    total_direct = sum(r["direct_cost"] for r in routes)
    free_routes = [r for r in routes if r["egress_cost"] == 0]
    paid_routes = [r for r in routes if r["egress_cost"] > 0]

    recommendations = []

    # Recommend multi-hop relays
    if multihop_savings > 0:
        relay_providers = sorted(_ZERO_EGRESS_PROVIDERS)
        recommendations.append(
            f"Multi-hop routing saves ${multihop_savings:.2f}. "
            f"Zero-egress relay providers: {', '.join(relay_providers)}"
        )

    # Recommend moving compute to free-egress providers
    if paid_routes:
        cheapest_free = free_routes[0] if free_routes else None
        for pr in paid_routes:
            if cheapest_free:
                recommendations.append(
                    f"Move compute from {pr['provider']}/{pr['region']} to "
                    f"{cheapest_free['provider']}/{cheapest_free['region']} to save "
                    f"${pr['direct_cost']:.2f}"
                )
            elif pr.get("multihop"):
                hop_path = " → ".join(pr["multihop"]["path"])
                recommendations.append(
                    f"Route data via {hop_path} to save "
                    f"${pr['multihop']['savings']:.2f} vs direct transfer"
                )
            else:
                recommendations.append(
                    f"Stage data closer to {pr['provider']}/{pr['region']} to reduce "
                    f"${pr['egress_cost']:.2f} egress"
                )

    return {
        "routes": routes,
        "total_egress_cost": round(total_cost, 2),
        "total_direct_cost": round(total_direct, 2),
        "multihop_savings": round(multihop_savings, 2),
        "free_routes": len(free_routes),
        "paid_routes": len(paid_routes),
        "recommendations": recommendations,
        "data_location": data_location,
        "size_gb": size_gb,
        "zero_egress_providers": sorted(_ZERO_EGRESS_PROVIDERS),
    }


# ═══════════════════════════════════════════════════════════════════════
# Dataset stager integration
# ═══════════════════════════════════════════════════════════════════════

def optimize_staging_route(
    data_provider: str,
    data_region: str,
    target_regions: List[Dict[str, str]],
    size_gb: float,
) -> Dict[str, Any]:
    """
    Integration point for DatasetStager.

    Given where data lives and where it needs to go, returns the cheapest
    transfer strategy — including whether to relay through a zero-egress
    provider first.

    Returns:
        {
            "strategy": "direct" | "relay",
            "relay_provider": "runpod" | None,
            "routes": [...],
            "total_cost": float,
            "recommendation": str,
        }
    """
    plan = optimize_transfer_plan(
        {"provider": data_provider, "region": data_region},
        target_regions,
        size_gb,
    )

    # Check if any route benefits from multi-hop
    has_relay = any(r.get("multihop") for r in plan["routes"])

    # Find the best relay provider (most-used in multi-hop paths)
    relay_counts: Dict[str, int] = {}
    for route in plan["routes"]:
        mh = route.get("multihop")
        if mh and len(mh["path"]) > 2:
            for hop in mh["path"][1:-1]:  # intermediate nodes only
                relay_counts[hop] = relay_counts.get(hop, 0) + 1
    best_relay = max(relay_counts, key=relay_counts.get) if relay_counts else None

    if has_relay and best_relay and plan["multihop_savings"] > 0:
        strategy = "relay"
        recommendation = (
            f"Stage data to {best_relay} first (zero egress), then distribute "
            f"to {len(target_regions)} targets. Saves ${plan['multihop_savings']:.2f} "
            f"vs direct transfers."
        )
    else:
        strategy = "direct"
        recommendation = (
            f"Direct transfer is optimal. Total egress: ${plan['total_egress_cost']:.2f} "
            f"for {size_gb:.1f} GB to {len(target_regions)} targets."
        )

    return {
        "strategy": strategy,
        "relay_provider": best_relay,
        "routes": plan["routes"],
        "total_cost": plan["total_egress_cost"],
        "total_direct_cost": plan["total_direct_cost"],
        "savings": plan["multihop_savings"],
        "recommendation": recommendation,
        "zero_egress_providers": plan["zero_egress_providers"],
    }
