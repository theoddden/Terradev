#!/usr/bin/env python3
"""
Inference Router — Auto-failover + latency-aware routing for inference endpoints.

Features:
  1. Health checks on active inference endpoints (HTTP probe)
  2. Auto-failover: if primary provider goes down, traffic shifts to backup
  3. Latency-aware routing: pick the lowest-latency healthy provider
  4. Integrates with WebPageTest TTFB probes and simple ping latency
"""

import asyncio
import aiohttp
import hashlib
import json
import os
import time
import logging
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# ── KV Prefix Cache Index ────────────────────────────────────────────────────


class PrefixCacheIndex:
    """
    Tracks which endpoints hold warm KV caches for specific prompt prefixes.

    Inspired by llm-d's KV cache-aware routing (Red Hat/IBM/Google, Oct 2025).
    Routes requests to the pod most likely to have the prefix cached in GPU
    memory, avoiding redundant prefill computation.

    Performance:
      - O(1) hash lookup per route decision
      - LRU eviction keeps memory bounded
      - Prefix hashing uses first N tokens for fast comparison
    """

    def __init__(self, max_entries: int = 10_000, prefix_tokens: int = 64):
        self._max_entries = max_entries
        self._prefix_tokens = prefix_tokens
        # prefix_hash -> {endpoint_id: last_seen_timestamp}
        self._index: OrderedDict[str, Dict[str, float]] = OrderedDict()

    def _hash_prefix(self, text: str) -> str:
        """Hash the first N whitespace-delimited tokens of a prompt."""
        tokens = text.split(None, self._prefix_tokens)[:self._prefix_tokens]
        prefix = " ".join(tokens)
        return hashlib.blake2b(prefix.encode(), digest_size=16).hexdigest()

    def record(self, text: str, endpoint_id: str):
        """Record that an endpoint processed (and cached) this prefix."""
        h = self._hash_prefix(text)
        if h in self._index:
            self._index.move_to_end(h)
            self._index[h][endpoint_id] = time.monotonic()
        else:
            self._index[h] = {endpoint_id: time.monotonic()}
        # LRU eviction
        while len(self._index) > self._max_entries:
            self._index.popitem(last=False)

    def lookup(self, text: str, max_age_s: float = 300.0) -> List[Tuple[str, float]]:
        """
        Find endpoints that likely have this prefix cached.

        Returns list of (endpoint_id, freshness_score) sorted by freshness.
        freshness_score: 1.0 = just seen, 0.0 = about to expire.
        """
        h = self._hash_prefix(text)
        entry = self._index.get(h)
        if not entry:
            return []

        now = time.monotonic()
        results = []
        for eid, ts in entry.items():
            age = now - ts
            if age <= max_age_s:
                freshness = 1.0 - (age / max_age_s)
                results.append((eid, freshness))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def evict_endpoint(self, endpoint_id: str):
        """Remove an endpoint from all cache entries (e.g., on endpoint failure)."""
        for h in list(self._index):
            self._index[h].pop(endpoint_id, None)
            if not self._index[h]:
                del self._index[h]

    @property
    def size(self) -> int:
        return len(self._index)


class EndpointHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class EndpointPhase(Enum):
    """
    Disaggregated serving phase (DistServe, UCSD Hao AI Lab, 2025).

    Modern LLM serving splits inference into two phases with different
    hardware requirements:
      - PREFILL: compute-bound (high FLOPS) — processes the input prompt,
        generates the full KV cache. Benefits from high-FLOPS GPUs (H100 SXM).
      - DECODE: memory-bound (high bandwidth) — autoregressive token generation,
        reads KV cache every step. Benefits from high-bandwidth GPUs (H200, MI300X).
      - MIXED: legacy unified endpoint that handles both phases.

    Adopted by vLLM, SGLang, NVIDIA Dynamo, MoonCake in 2025.
    """
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


# ── Prefill→Decode Handoff Tracker ────────────────────────────────────────────


class KVConnectorType(Enum):
    """KV cache transfer connector types for disaggregated serving."""
    NIXL = "NixlConnector"           # NVIDIA NIXL: zero-copy GPU-GPU via RDMA/NVLink
    LMCACHE = "LMCacheConnector"     # LMCache: CPU-mediated transfer (wider compat)
    MOONCAKE = "MooncakeConnector"   # MoonCake: distributed KV store
    UNKNOWN = "unknown"


@dataclass
class KVConnectorConfig:
    """Configuration for KV cache transfer between prefill and decode endpoints.

    NIXL (NVIDIA Inference eXtension Library) enables zero-copy GPU-to-GPU
    KV cache transfer over RDMA or NVLink, avoiding CPU bounce buffers.
    This is critical for disaggregated MoE serving where KV caches can be
    several GB per request at long context lengths.
    """
    connector_type: KVConnectorType = KVConnectorType.NIXL
    buffer_size_bytes: int = 5_368_709_120   # 5GB default NIXL buffer
    rdma_enabled: bool = True
    rdma_device: str = ""                    # e.g. "mlx5_0" for ConnectX-7
    nvlink_direct: bool = False              # True if prefill/decode on same node
    max_inflight_transfers: int = 16         # Concurrent KV transfer limit
    compression_enabled: bool = False        # KV cache compression (lossy)
    prefetch_enabled: bool = True            # Speculative KV prefetch


@dataclass
class PrefillDecodeLink:
    """Tracks a KV cache handoff from a prefill endpoint to a decode endpoint."""
    prefill_endpoint_id: str
    decode_endpoint_id: str
    model: str
    last_handoff: float = 0.0       # monotonic timestamp
    handoff_count: int = 0
    avg_transfer_ms: float = 0.0    # KV cache transfer latency
    # NIXL KV connector tracking
    kv_connector: KVConnectorType = KVConnectorType.UNKNOWN
    kv_transfer_bytes: int = 0      # Avg KV cache size transferred
    rdma_active: bool = False       # Whether RDMA was used for last transfer


class PrefillDecodeTracker:
    """
    Tracks which prefill endpoints hand off KV caches to which decode endpoints.

    Enables sticky routing: once a prefill endpoint produces a KV cache,
    route the decode phase to the same decode endpoint that already received
    the KV transfer, avoiding redundant transfers.

    Design follows DistServe's disaggregated prefill/decode architecture
    and MoonCake's KV cache transfer protocol.

    KV Connector Integration:
      When NIXL is configured, the tracker records which connector type was
      used for each handoff. This enables transport-aware routing:
      - NIXL (RDMA): ~0.5ms transfer for 1GB KV cache (preferred)
      - NIXL (NVLink): ~0.2ms transfer for same-node pairs
      - LMCache (CPU): ~5ms transfer (fallback)
      The router uses this to prefer pairs with established RDMA connections.
    """

    def __init__(
        self,
        max_links: int = 1000,
        default_connector_config: Optional[KVConnectorConfig] = None,
    ):
        self._max_links = max_links
        # (prefill_id, model) -> PrefillDecodeLink
        self._links: OrderedDict[Tuple[str, str], PrefillDecodeLink] = OrderedDict()
        self.connector_config = default_connector_config or KVConnectorConfig()

    def record_handoff(
        self,
        prefill_id: str,
        decode_id: str,
        model: str,
        transfer_ms: float = 0.0,
        kv_connector: Optional[str] = None,
        kv_transfer_bytes: int = 0,
        rdma_active: bool = False,
    ):
        """Record a KV cache handoff from prefill to decode endpoint.

        Args:
            kv_connector: Connector type used ('NixlConnector', 'LMCacheConnector', etc.)
            kv_transfer_bytes: Size of KV cache transferred in bytes.
            rdma_active: Whether RDMA was used for this transfer.
        """
        key = (prefill_id, model)

        # Resolve connector type
        try:
            conn_type = KVConnectorType(kv_connector) if kv_connector else self.connector_config.connector_type
        except ValueError:
            conn_type = KVConnectorType.UNKNOWN

        if key in self._links:
            link = self._links[key]
            link.decode_endpoint_id = decode_id
            link.last_handoff = time.monotonic()
            link.handoff_count += 1
            if transfer_ms > 0:
                link.avg_transfer_ms = 0.8 * link.avg_transfer_ms + 0.2 * transfer_ms
            link.kv_connector = conn_type
            if kv_transfer_bytes > 0:
                link.kv_transfer_bytes = kv_transfer_bytes
            link.rdma_active = rdma_active
            self._links.move_to_end(key)
        else:
            self._links[key] = PrefillDecodeLink(
                prefill_endpoint_id=prefill_id,
                decode_endpoint_id=decode_id,
                model=model,
                last_handoff=time.monotonic(),
                handoff_count=1,
                avg_transfer_ms=transfer_ms,
                kv_connector=conn_type,
                kv_transfer_bytes=kv_transfer_bytes,
                rdma_active=rdma_active,
            )
        while len(self._links) > self._max_links:
            self._links.popitem(last=False)

    def get_decode_for_prefill(
        self, prefill_id: str, model: str, max_age_s: float = 120.0
    ) -> Optional[str]:
        """Get the preferred decode endpoint for a prefill endpoint + model."""
        key = (prefill_id, model)
        link = self._links.get(key)
        if link and (time.monotonic() - link.last_handoff) <= max_age_s:
            return link.decode_endpoint_id
        return None

    def get_prefill_for_decode(
        self, decode_id: str, model: str, max_age_s: float = 120.0
    ) -> Optional[str]:
        """Get a prefill endpoint that recently handed off to this decode endpoint."""
        now = time.monotonic()
        for (_pid, m), link in reversed(self._links.items()):
            if m == model and link.decode_endpoint_id == decode_id:
                if (now - link.last_handoff) <= max_age_s:
                    return _pid
        return None

    def get_link_details(
        self, prefill_id: str, model: str,
    ) -> Optional[Dict[str, Any]]:
        """Get detailed link info including KV connector state."""
        key = (prefill_id, model)
        link = self._links.get(key)
        if not link:
            return None
        return {
            "prefill_endpoint_id": link.prefill_endpoint_id,
            "decode_endpoint_id": link.decode_endpoint_id,
            "model": link.model,
            "handoff_count": link.handoff_count,
            "avg_transfer_ms": link.avg_transfer_ms,
            "kv_connector": link.kv_connector.value,
            "kv_transfer_bytes": link.kv_transfer_bytes,
            "rdma_active": link.rdma_active,
            "age_s": time.monotonic() - link.last_handoff,
        }

    def get_best_decode_by_transport(
        self, model: str, max_age_s: float = 120.0,
    ) -> Optional[str]:
        """Find the decode endpoint with the best KV transport (lowest transfer latency).

        Prefers NIXL with RDMA > NIXL without RDMA > LMCache.
        """
        now = time.monotonic()
        best_decode = None
        best_score = -1.0

        for (_pid, m), link in self._links.items():
            if m != model or (now - link.last_handoff) > max_age_s:
                continue

            # Score: NIXL+RDMA=3, NIXL=2, LMCache=1, Unknown=0
            score = 0.0
            if link.kv_connector == KVConnectorType.NIXL:
                score = 3.0 if link.rdma_active else 2.0
            elif link.kv_connector == KVConnectorType.LMCACHE:
                score = 1.0
            elif link.kv_connector == KVConnectorType.MOONCAKE:
                score = 1.5

            # Bonus for low transfer latency
            if link.avg_transfer_ms > 0:
                latency_bonus = max(0, 1.0 - (link.avg_transfer_ms / 10.0))  # <10ms is good
                score += latency_bonus

            if score > best_score:
                best_score = score
                best_decode = link.decode_endpoint_id

        return best_decode

    def get_connector_summary(self) -> Dict[str, Any]:
        """Get summary of KV connector usage across all active links."""
        connector_counts: Dict[str, int] = {}
        rdma_count = 0
        total_transfer_bytes = 0
        total_links = 0
        avg_latencies: List[float] = []

        now = time.monotonic()
        for link in self._links.values():
            if (now - link.last_handoff) > 300:  # Skip stale links
                continue
            total_links += 1
            ct = link.kv_connector.value
            connector_counts[ct] = connector_counts.get(ct, 0) + 1
            if link.rdma_active:
                rdma_count += 1
            total_transfer_bytes += link.kv_transfer_bytes
            if link.avg_transfer_ms > 0:
                avg_latencies.append(link.avg_transfer_ms)

        return {
            "total_active_links": total_links,
            "connector_types": connector_counts,
            "rdma_active_count": rdma_count,
            "total_transfer_bytes": total_transfer_bytes,
            "avg_transfer_ms": (
                sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0.0
            ),
            "default_connector": self.connector_config.connector_type.value,
            "default_buffer_size": self.connector_config.buffer_size_bytes,
            "rdma_enabled": self.connector_config.rdma_enabled,
        }

    @property
    def size(self) -> int:
        return len(self._links)


@dataclass
class HealthProbe:
    """Result of a single health probe"""
    endpoint_id: str
    provider: str
    timestamp: datetime
    latency_ms: float
    status_code: int
    healthy: bool
    error: Optional[str] = None


@dataclass
class InferenceEndpoint:
    """Tracked inference endpoint with health state"""
    endpoint_id: str
    provider: str
    url: str
    model: str
    gpu_type: str
    region: str
    price_per_hour: float
    created_at: datetime
    # Health tracking
    health: EndpointHealth = EndpointHealth.UNKNOWN
    last_probe: Optional[HealthProbe] = None
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    latency_history: List[float] = field(default_factory=list)
    # Failover
    is_primary: bool = True
    backup_endpoint_id: Optional[str] = None
    # Disaggregated serving (DistServe)
    phase: EndpointPhase = EndpointPhase.MIXED
    flops_tflops: float = 0.0           # peak TFLOPS (prefill scoring)
    memory_bandwidth_tbps: float = 0.0  # peak TB/s (decode scoring)
    kv_transfer_endpoint: Optional[str] = None  # paired endpoint for KV handoff
    # MoE Expert Parallelism topology
    ep_group_id: Optional[str] = None        # EP group this endpoint belongs to
    ep_rank: int = 0                          # Rank within the EP group
    expert_range: Tuple[int, int] = (0, 0)   # (start, end) expert indices on this rank
    nvlink_domain: Optional[str] = None       # NVLink domain ID for intra-group comms
    dp_size: int = 1                          # Total DP/EP ranks in the group
    tp_size: int = 1                          # TP degree per EP rank


class InferenceRouter:
    """
    Routes inference traffic to the healthiest, lowest-latency provider.
    Handles auto-failover when a provider goes down.

    When a topology report is available, transparently integrates
    NUMA-aware semantic routing: signal extraction → policy decision →
    NUMA-optimal endpoint selection. This adds <3ms to the routing path.
    """

    # Thresholds
    HEALTH_CHECK_INTERVAL_S = 30
    UNHEALTHY_AFTER_FAILURES = 3
    DEGRADED_AFTER_FAILURES = 1
    LATENCY_HISTORY_SIZE = 20
    FAILOVER_COOLDOWN_S = 60

    def __init__(self, config_dir: Optional[Path] = None,
                 topology_report: Optional[Dict] = None,
                 routing_policy_path: Optional[str] = None):
        self.config_dir = config_dir or Path.home() / '.terradev'
        self.endpoints_file = self.config_dir / 'inference_endpoints.json'
        self.endpoints: Dict[str, InferenceEndpoint] = {}
        self._load_endpoints()
        self._last_failover: Dict[str, float] = {}

        # KV prefix cache index — routes to pods with warm caches
        self._prefix_cache = PrefixCacheIndex()

        # Disaggregated prefill/decode handoff tracker
        self._pd_tracker = PrefillDecodeTracker()

        # Shared aiohttp session for health probes (connection pooling)
        self._probe_session: Optional[aiohttp.ClientSession] = None

        # Lazy-init semantic router (adds <3ms per request)
        self._semantic_router = None
        self._topology_report = topology_report
        self._routing_policy_path = routing_policy_path
        if topology_report or routing_policy_path:
            self._init_semantic_router()

    def _init_semantic_router(self):
        """Initialize the semantic router backend (called lazily)"""
        try:
            from .semantic_router import SemanticRouter
            self._semantic_router = SemanticRouter(
                policy_path=self._routing_policy_path,
                topology_report=self._topology_report,
            )
            logger.info("Semantic router initialized (NUMA-aware routing active)")
        except Exception as e:
            logger.debug(f"Semantic router init skipped: {e}")
            self._semantic_router = None

    # ── Persistence ──

    def _load_endpoints(self):
        """Load tracked endpoints from disk"""
        if self.endpoints_file.exists():
            try:
                with open(self.endpoints_file, 'r') as f:
                    data = json.load(f)
                for ep_data in data:
                    ep = InferenceEndpoint(
                        endpoint_id=ep_data['endpoint_id'],
                        provider=ep_data['provider'],
                        url=ep_data.get('url', ''),
                        model=ep_data.get('model', ''),
                        gpu_type=ep_data.get('gpu_type', ''),
                        region=ep_data.get('region', ''),
                        price_per_hour=ep_data.get('price_per_hour', 0.0),
                        created_at=datetime.fromisoformat(ep_data.get('created_at', datetime.now().isoformat())),
                        health=EndpointHealth(ep_data.get('health', 'unknown')),
                        is_primary=ep_data.get('is_primary', True),
                        backup_endpoint_id=ep_data.get('backup_endpoint_id'),
                        avg_latency_ms=ep_data.get('avg_latency_ms', 0.0),
                        phase=EndpointPhase(ep_data.get('phase', 'mixed')),
                        flops_tflops=ep_data.get('flops_tflops', 0.0),
                        memory_bandwidth_tbps=ep_data.get('memory_bandwidth_tbps', 0.0),
                        kv_transfer_endpoint=ep_data.get('kv_transfer_endpoint'),
                    )
                    self.endpoints[ep.endpoint_id] = ep
            except Exception:
                pass

    def _save_endpoints(self):
        """Persist endpoint state to disk"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = []
        for ep in self.endpoints.values():
            data.append({
                'endpoint_id': ep.endpoint_id,
                'provider': ep.provider,
                'url': ep.url,
                'model': ep.model,
                'gpu_type': ep.gpu_type,
                'region': ep.region,
                'price_per_hour': ep.price_per_hour,
                'created_at': ep.created_at.isoformat(),
                'health': ep.health.value,
                'is_primary': ep.is_primary,
                'backup_endpoint_id': ep.backup_endpoint_id,
                'avg_latency_ms': ep.avg_latency_ms,
                'phase': ep.phase.value,
                'flops_tflops': ep.flops_tflops,
                'memory_bandwidth_tbps': ep.memory_bandwidth_tbps,
                'kv_transfer_endpoint': ep.kv_transfer_endpoint,
            })
        with open(self.endpoints_file, 'w') as f:
            json.dump(data, f, indent=2)
        os.chmod(self.endpoints_file, 0o600)

    # ── Endpoint Management ──

    def register_endpoint(self, endpoint_id: str, provider: str, url: str,
                          model: str, gpu_type: str, region: str,
                          price_per_hour: float, is_primary: bool = True,
                          backup_endpoint_id: Optional[str] = None,
                          phase: str = "mixed",
                          flops_tflops: float = 0.0,
                          memory_bandwidth_tbps: float = 0.0,
                          kv_transfer_endpoint: Optional[str] = None,
                          ep_group_id: Optional[str] = None,
                          ep_rank: int = 0,
                          expert_range: Tuple[int, int] = (0, 0),
                          nvlink_domain: Optional[str] = None,
                          dp_size: int = 1,
                          tp_size: int = 1,
                          ) -> InferenceEndpoint:
        """Register a new inference endpoint for health tracking and routing.

        Args:
            phase: 'prefill', 'decode', or 'mixed' (default).
                   Prefill endpoints are compute-bound (high FLOPS).
                   Decode endpoints are memory-bound (high bandwidth).
            flops_tflops: Peak TFLOPS — used to score prefill endpoints.
            memory_bandwidth_tbps: Peak TB/s — used to score decode endpoints.
            kv_transfer_endpoint: Paired endpoint for KV cache handoff.
            ep_group_id: EP group identifier (endpoints in the same group
                         share expert routing via all-to-all communication).
            ep_rank: This endpoint's rank within its EP group.
            expert_range: (start, end) expert indices hosted on this rank.
            nvlink_domain: NVLink domain ID for intra-group communication.
            dp_size: Total number of DP/EP ranks in the group.
            tp_size: Tensor parallelism degree per EP rank.
        """
        ep = InferenceEndpoint(
            endpoint_id=endpoint_id,
            provider=provider,
            url=url,
            model=model,
            gpu_type=gpu_type,
            region=region,
            price_per_hour=price_per_hour,
            created_at=datetime.now(),
            is_primary=is_primary,
            backup_endpoint_id=backup_endpoint_id,
            phase=EndpointPhase(phase),
            flops_tflops=flops_tflops,
            memory_bandwidth_tbps=memory_bandwidth_tbps,
            kv_transfer_endpoint=kv_transfer_endpoint,
            ep_group_id=ep_group_id,
            ep_rank=ep_rank,
            expert_range=expert_range,
            nvlink_domain=nvlink_domain,
            dp_size=dp_size,
            tp_size=tp_size,
        )
        self.endpoints[endpoint_id] = ep
        self._save_endpoints()
        return ep

    def remove_endpoint(self, endpoint_id: str):
        """Remove an endpoint from tracking"""
        self.endpoints.pop(endpoint_id, None)
        self._save_endpoints()

    def set_backup(self, primary_id: str, backup_id: str):
        """Link a backup endpoint to a primary"""
        if primary_id in self.endpoints and backup_id in self.endpoints:
            self.endpoints[primary_id].backup_endpoint_id = backup_id
            self.endpoints[backup_id].is_primary = False
            self._save_endpoints()

    # ── Health Checks ──

    async def _get_probe_session(self) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session (connection pooling)."""
        if self._probe_session is None or self._probe_session.closed:
            connector = aiohttp.TCPConnector(
                limit=50,  # max concurrent connections
                ttl_dns_cache=60,  # DNS cache TTL
                keepalive_timeout=30,
            )
            self._probe_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._probe_session

    async def close(self):
        """Close the shared session. Call on shutdown."""
        if self._probe_session and not self._probe_session.closed:
            await self._probe_session.close()

    async def probe_endpoint(self, endpoint_id: str) -> HealthProbe:
        """HTTP health probe using shared connection pool."""
        ep = self.endpoints.get(endpoint_id)
        if not ep or not ep.url:
            return HealthProbe(
                endpoint_id=endpoint_id,
                provider=ep.provider if ep else 'unknown',
                timestamp=datetime.now(),
                latency_ms=0,
                status_code=0,
                healthy=False,
                error='No URL configured',
            )

        probe_url = ep.url.rstrip('/') + '/health'
        start = time.monotonic()
        try:
            session = await self._get_probe_session()
            async with session.get(probe_url) as resp:
                latency_ms = (time.monotonic() - start) * 1000
                healthy = resp.status < 500
                return HealthProbe(
                    endpoint_id=endpoint_id,
                    provider=ep.provider,
                    timestamp=datetime.now(),
                    latency_ms=latency_ms,
                    status_code=resp.status,
                    healthy=healthy,
                )
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthProbe(
                endpoint_id=endpoint_id,
                provider=ep.provider,
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                status_code=0,
                healthy=False,
                error=str(e),
            )

    async def check_all_endpoints(self) -> Dict[str, HealthProbe]:
        """Probe all registered endpoints in parallel"""
        tasks = {eid: self.probe_endpoint(eid) for eid in self.endpoints}
        results = {}
        if tasks:
            done = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for eid, result in zip(tasks.keys(), done):
                if isinstance(result, Exception):
                    results[eid] = HealthProbe(
                        endpoint_id=eid,
                        provider=self.endpoints[eid].provider,
                        timestamp=datetime.now(),
                        latency_ms=0,
                        status_code=0,
                        healthy=False,
                        error=str(result),
                    )
                else:
                    results[eid] = result
                self._update_health(eid, results[eid])
        self._save_endpoints()
        return results

    def _update_health(self, endpoint_id: str, probe: HealthProbe):
        """Update endpoint health state from probe result"""
        ep = self.endpoints.get(endpoint_id)
        if not ep:
            return

        ep.last_probe = probe

        if probe.healthy:
            ep.consecutive_failures = 0
            ep.health = EndpointHealth.HEALTHY
            # Track latency
            ep.latency_history.append(probe.latency_ms)
            if len(ep.latency_history) > self.LATENCY_HISTORY_SIZE:
                ep.latency_history = ep.latency_history[-self.LATENCY_HISTORY_SIZE:]
            ep.avg_latency_ms = sum(ep.latency_history) / len(ep.latency_history)
        else:
            ep.consecutive_failures += 1
            if ep.consecutive_failures >= self.UNHEALTHY_AFTER_FAILURES:
                ep.health = EndpointHealth.UNHEALTHY
            elif ep.consecutive_failures >= self.DEGRADED_AFTER_FAILURES:
                ep.health = EndpointHealth.DEGRADED

    # ── Auto-Failover ──

    async def check_and_failover(self) -> List[Dict]:
        """Run health checks and trigger failover for unhealthy primaries.
        Returns list of failover events."""
        probes = await self.check_all_endpoints()
        failover_events = []

        for eid, probe in probes.items():
            ep = self.endpoints.get(eid)
            if not ep or not ep.is_primary:
                continue

            if ep.health == EndpointHealth.UNHEALTHY and ep.backup_endpoint_id:
                # Cooldown check
                last = self._last_failover.get(eid, 0)
                if time.time() - last < self.FAILOVER_COOLDOWN_S:
                    continue

                backup = self.endpoints.get(ep.backup_endpoint_id)
                if backup and backup.health in (EndpointHealth.HEALTHY, EndpointHealth.UNKNOWN):
                    # Promote backup to primary
                    backup.is_primary = True
                    ep.is_primary = False
                    ep.backup_endpoint_id = None
                    backup.backup_endpoint_id = eid  # old primary becomes backup
                    self._last_failover[eid] = time.time()

                    event = {
                        'type': 'failover',
                        'timestamp': datetime.now().isoformat(),
                        'failed_endpoint': eid,
                        'failed_provider': ep.provider,
                        'new_primary': backup.endpoint_id,
                        'new_provider': backup.provider,
                        'reason': f'{ep.consecutive_failures} consecutive health check failures',
                    }
                    failover_events.append(event)
                    logger.warning(
                        f"FAILOVER: {ep.provider}/{eid} → {backup.provider}/{backup.endpoint_id}"
                    )

        if failover_events:
            self._save_endpoints()
            self._save_failover_log(failover_events)

        return failover_events

    def _save_failover_log(self, events: List[Dict]):
        """Append failover events to audit log"""
        log_file = self.config_dir / 'failover_log.json'
        existing = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.extend(events)
        with open(log_file, 'w') as f:
            json.dump(existing, f, indent=2)
        os.chmod(log_file, 0o600)

    # ── Latency-Aware Routing ──

    async def measure_latency_ping(self, target: str, count: int = 3) -> Optional[float]:
        """Measure latency to a target via async ping (ms)"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'ping', '-c', str(count), '-W', '2', target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = stdout.decode()
            latencies = []
            for line in output.split('\n'):
                if 'time=' in line:
                    t = float(line.split('time=')[1].split()[0])
                    latencies.append(t)
            return sum(latencies) / len(latencies) if latencies else None
        except Exception:
            return None

    async def measure_latency_wpt(self, url: str, wpt_api_key: Optional[str] = None) -> Optional[float]:
        """Measure TTFB via WebPageTest API (ms).
        Falls back to direct HTTP probe if no WPT key."""
        if wpt_api_key:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    # Submit test
                    params = {
                        'url': url,
                        'f': 'json',
                        'k': wpt_api_key,
                        'runs': '1',
                        'fvonly': '1',
                    }
                    async with session.get('https://www.webpagetest.org/runtest.php', params=params) as resp:
                        result = await resp.json()
                        test_id = result.get('data', {}).get('testId')
                        if not test_id:
                            return None

                    # Poll for result (up to 60s)
                    for _ in range(12):
                        await asyncio.sleep(5)
                        async with session.get(
                            f'https://www.webpagetest.org/jsonResult.php?test={test_id}'
                        ) as resp:
                            result = await resp.json()
                            status = result.get('statusCode', 0)
                            if status == 200:
                                ttfb = (result.get('data', {})
                                        .get('runs', {}).get('1', {})
                                        .get('firstView', {}).get('TTFB', None))
                                return float(ttfb) if ttfb is not None else None
                    return None
            except Exception:
                return None

        # Fallback: direct HTTP TTFB probe
        try:
            start = time.monotonic()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as resp:
                    ttfb_ms = (time.monotonic() - start) * 1000
                    return ttfb_ms
        except Exception:
            return None

    def get_best_endpoint(self, model: Optional[str] = None,
                          strategy: str = 'latency',
                          query: Optional[Dict] = None) -> Optional[InferenceEndpoint]:
        """Select the best healthy endpoint for routing.

        Strategies:
          - 'latency': lowest average latency (default)
          - 'cost': lowest price per hour
          - 'score': combined latency + cost score
          - 'semantic': signal-driven NUMA-aware routing (auto when query is provided)

        When `query` is provided and the semantic router is active, the
        signal extraction pipeline runs transparently (<3ms overhead),
        selects the target model via policy rules, then applies NUMA-aware
        endpoint scoring to pick the GPU with optimal PCIe locality.

        KV cache-aware routing: if query text matches a cached prefix,
        the endpoint that already holds the warm cache gets a priority boost.
        """
        # ── Semantic routing layer (transparent backend) ──
        if query and self._semantic_router:
            result = self._route_with_signals(query)
            # Record this prefix for future cache-hit routing
            if result:
                text = query.get("content") or query.get("prompt") or ""
                if text:
                    self._prefix_cache.record(text, result.endpoint_id)
            return result

        # ── KV cache-aware boost (classic path) ──
        query_text = ""
        if query:
            query_text = query.get("content") or query.get("prompt") or ""

        # ── Classic routing (latency / cost / score) ──
        candidates = [
            ep for ep in self.endpoints.values()
            if ep.health in (EndpointHealth.HEALTHY, EndpointHealth.UNKNOWN)
            and (model is None or ep.model == model)
        ]

        if not candidates:
            return None

        # Check prefix cache for a warm-cache boost
        cache_boost: Dict[str, float] = {}
        if query_text:
            hits = self._prefix_cache.lookup(query_text)
            for eid, freshness in hits:
                cache_boost[eid] = freshness

        if strategy == 'latency':
            candidates.sort(key=lambda e: (
                (e.avg_latency_ms or 9999) * (0.3 if e.endpoint_id in cache_boost else 1.0),
                e.price_per_hour,
            ))
        elif strategy == 'cost':
            candidates.sort(key=lambda e: (
                e.price_per_hour * (0.5 if e.endpoint_id in cache_boost else 1.0),
                e.avg_latency_ms or 9999,
            ))
        elif strategy == 'score':
            max_lat = max(e.avg_latency_ms for e in candidates) or 1
            max_price = max(e.price_per_hour for e in candidates) or 1
            candidates.sort(key=lambda e: (
                0.5 * ((e.avg_latency_ms or 9999) / max_lat) +
                0.3 * (e.price_per_hour / max_price) +
                (-0.2 * cache_boost.get(e.endpoint_id, 0.0))  # cache hit = lower score = better
            ))

        best = candidates[0]
        # Record prefix for future lookups
        if query_text:
            self._prefix_cache.record(query_text, best.endpoint_id)
        return best

    def _route_with_signals(self, query: Dict) -> Optional[InferenceEndpoint]:
        """Internal: run the full signal → decision → NUMA pipeline.

        Latency budget: <3ms for signal extraction + policy eval,
        plus O(n) endpoint scoring where n = candidate count.
        """
        decision = self._semantic_router.route(query)

        # Safety: blocked requests return None
        if decision.route_to == "__blocked__":
            logger.warning(
                f"Request blocked by semantic safety policy: "
                f"rule={decision.matched_rule}"
            )
            return None

        # Determine candidate pool
        target_model = decision.route_to
        candidates = [
            ep for ep in self.endpoints.values()
            if ep.health in (EndpointHealth.HEALTHY, EndpointHealth.UNKNOWN)
            and (target_model is None or ep.model == target_model
                 or target_model in ("__local_only__",))
        ]

        # __local_only__ filter: prefer endpoints on local providers
        if decision.route_to == "__local_only__":
            local = [ep for ep in candidates if ep.provider in ("local", "vllm", "self-hosted")]
            if local:
                candidates = local

        if not candidates:
            # Fallback: try classic routing with the strategy from the decision
            fallback_strategy = decision.strategy or "cost"
            return self.get_best_endpoint(strategy=fallback_strategy)

        # NUMA-aware endpoint selection
        if len(candidates) > 1 and self._semantic_router._topology_report:
            candidate_dicts = [
                {
                    "endpoint_id": ep.endpoint_id,
                    "gpu_index": getattr(ep, "gpu_index", None),
                    "numa_node": getattr(ep, "numa_node", None),
                    "avg_latency_ms": ep.avg_latency_ms or 9999,
                    "price_per_hour": ep.price_per_hour,
                }
                for ep in candidates
            ]
            best = self._semantic_router.select_numa_optimal_endpoint(
                decision, candidate_dicts
            )
            if best:
                eid = best["endpoint_id"]
                return self.endpoints.get(eid, candidates[0])

        # Single candidate or no topology — apply strategy sort
        strategy = decision.strategy or "score"
        if strategy == "latency":
            candidates.sort(key=lambda e: (e.avg_latency_ms or 9999, e.price_per_hour))
        elif strategy == "cost":
            candidates.sort(key=lambda e: (e.price_per_hour, e.avg_latency_ms or 9999))
        else:
            max_lat = max((e.avg_latency_ms or 0) for e in candidates) or 1
            max_price = max(e.price_per_hour for e in candidates) or 1
            candidates.sort(key=lambda e: (
                0.6 * ((e.avg_latency_ms or 9999) / max_lat) +
                0.4 * (e.price_per_hour / max_price)
            ))

        return candidates[0]

    # ── Disaggregated Prefill/Decode Routing (DistServe) ─────────────────

    def _score_prefill_endpoint(self, ep: InferenceEndpoint) -> float:
        """Score an endpoint for prefill phase (lower = better).
        Prefill is compute-bound: rank by FLOPS, then latency."""
        flops_score = 1.0 / max(ep.flops_tflops, 0.1)  # higher FLOPS = lower score
        latency_score = (ep.avg_latency_ms or 9999) / 10000
        return 0.6 * flops_score + 0.3 * latency_score + 0.1 * (ep.price_per_hour / 100)

    def _score_decode_endpoint(self, ep: InferenceEndpoint) -> float:
        """Score an endpoint for decode phase (lower = better).
        Decode is memory-bound: rank by bandwidth, then latency."""
        bw_score = 1.0 / max(ep.memory_bandwidth_tbps, 0.01)  # higher BW = lower score
        latency_score = (ep.avg_latency_ms or 9999) / 10000
        return 0.6 * bw_score + 0.3 * latency_score + 0.1 * (ep.price_per_hour / 100)

    def get_best_prefill_endpoint(
        self, model: Optional[str] = None, query: Optional[Dict] = None,
    ) -> Optional[InferenceEndpoint]:
        """Select the best endpoint for the prefill phase (compute-bound).
        Filters to PREFILL and MIXED endpoints, scores by FLOPS."""
        candidates = [
            ep for ep in self.endpoints.values()
            if ep.health in (EndpointHealth.HEALTHY, EndpointHealth.UNKNOWN)
            and ep.phase in (EndpointPhase.PREFILL, EndpointPhase.MIXED)
            and (model is None or ep.model == model)
        ]
        if not candidates:
            return self.get_best_endpoint(model=model, query=query)

        # KV prefix cache boost: prefer endpoint that already cached this prefix
        cache_boost: Dict[str, float] = {}
        if query:
            text = query.get("content") or query.get("prompt") or ""
            if text:
                for eid, freshness in self._prefix_cache.lookup(text):
                    cache_boost[eid] = freshness

        candidates.sort(key=lambda e: (
            self._score_prefill_endpoint(e)
            - 0.3 * cache_boost.get(e.endpoint_id, 0.0)
        ))
        return candidates[0]

    def get_best_decode_endpoint(
        self,
        model: Optional[str] = None,
        prefill_endpoint_id: Optional[str] = None,
    ) -> Optional[InferenceEndpoint]:
        """Select the best endpoint for the decode phase (memory-bound).
        Sticky routing: prefers the decode endpoint that already received
        the KV cache from the given prefill endpoint."""
        candidates = [
            ep for ep in self.endpoints.values()
            if ep.health in (EndpointHealth.HEALTHY, EndpointHealth.UNKNOWN)
            and ep.phase in (EndpointPhase.DECODE, EndpointPhase.MIXED)
            and (model is None or ep.model == model)
        ]
        if not candidates:
            return self.get_best_endpoint(model=model)

        # Sticky routing: check if prefill endpoint has a known decode partner
        if prefill_endpoint_id and model:
            sticky_id = self._pd_tracker.get_decode_for_prefill(
                prefill_endpoint_id, model
            )
            if sticky_id:
                sticky_ep = self.endpoints.get(sticky_id)
                if sticky_ep and sticky_ep in candidates:
                    return sticky_ep

        # Also check static KV transfer pairing
        if prefill_endpoint_id:
            prefill_ep = self.endpoints.get(prefill_endpoint_id)
            if prefill_ep and prefill_ep.kv_transfer_endpoint:
                paired = self.endpoints.get(prefill_ep.kv_transfer_endpoint)
                if paired and paired in candidates:
                    return paired

        candidates.sort(key=lambda e: self._score_decode_endpoint(e))
        return candidates[0]

    def get_disaggregated_pair(
        self,
        model: Optional[str] = None,
        query: Optional[Dict] = None,
    ) -> Tuple[Optional[InferenceEndpoint], Optional[InferenceEndpoint]]:
        """
        Get a (prefill_endpoint, decode_endpoint) pair for disaggregated serving.

        Uses DAGExecutor to score prefill and decode candidates IN PARALLEL
        (Terraform-style). The two scoring paths are independent DAG nodes
        that execute concurrently on the warm thread pool.

        Returns:
            (prefill_ep, decode_ep) — either may be None if no candidates.
            If all endpoints are MIXED, both will be the same endpoint.
        """
        # Check if we have any disaggregated endpoints at all
        has_prefill = any(
            ep.phase == EndpointPhase.PREFILL for ep in self.endpoints.values()
        )
        has_decode = any(
            ep.phase == EndpointPhase.DECODE for ep in self.endpoints.values()
        )

        if not has_prefill and not has_decode:
            # All MIXED — return same endpoint for both phases
            ep = self.get_best_endpoint(model=model, query=query)
            return (ep, ep)

        # ── Parallel prefill + decode selection via DAGExecutor ──
        try:
            from .dag_executor import DAGExecutor

            dag = DAGExecutor(max_workers=2, name="disaggregated_route", reuse_pool=False)

            def select_prefill(ctx):
                return self.get_best_prefill_endpoint(
                    model=ctx.get("model"), query=ctx.get("query")
                )

            def select_decode(ctx):
                # First pass: no sticky routing (prefill not known yet)
                return self.get_best_decode_endpoint(model=ctx.get("model"))

            dag.add_node("prefill", select_prefill)
            dag.add_node("decode", select_decode)
            # No edges — fully parallel

            result = dag.apply(initial_context={"model": model, "query": query})
            prefill_ep = result.outputs.get("prefill")
            decode_ep = result.outputs.get("decode")

        except ImportError:
            # Fallback: sequential
            prefill_ep = self.get_best_prefill_endpoint(model=model, query=query)
            decode_ep = self.get_best_decode_endpoint(model=model)

        # Sticky routing refinement: now that we know the prefill endpoint,
        # check if there's a better decode partner
        if prefill_ep and decode_ep and model:
            better_decode = self.get_best_decode_endpoint(
                model=model, prefill_endpoint_id=prefill_ep.endpoint_id
            )
            if better_decode:
                decode_ep = better_decode

        # Record the handoff for future sticky routing
        if prefill_ep and decode_ep and model:
            self._pd_tracker.record_handoff(
                prefill_ep.endpoint_id, decode_ep.endpoint_id, model
            )

        # Record prefix cache
        if prefill_ep and query:
            text = query.get("content") or query.get("prompt") or ""
            if text:
                self._prefix_cache.record(text, prefill_ep.endpoint_id)

        return (prefill_ep, decode_ep)

    def get_phase_summary(self) -> Dict[str, int]:
        """Count endpoints by phase."""
        counts = {"prefill": 0, "decode": 0, "mixed": 0}
        for ep in self.endpoints.values():
            counts[ep.phase.value] = counts.get(ep.phase.value, 0) + 1
        return counts

    # ── MoE EP Group Routing ──

    def get_ep_group_endpoints(self, ep_group_id: str) -> List[InferenceEndpoint]:
        """Get all endpoints in an EP group, sorted by rank."""
        eps = [
            ep for ep in self.endpoints.values()
            if ep.ep_group_id == ep_group_id
        ]
        eps.sort(key=lambda e: e.ep_rank)
        return eps

    def get_ep_group_for_model(self, model: str) -> Optional[str]:
        """Find the EP group serving a given model."""
        for ep in self.endpoints.values():
            if ep.model == model and ep.ep_group_id:
                return ep.ep_group_id
        return None

    def get_ep_group_health(self, ep_group_id: str) -> Dict[str, Any]:
        """Get health summary for an EP group.

        An EP group is only fully healthy if ALL ranks are healthy,
        because expert routing requires all-to-all communication
        across all ranks.
        """
        eps = self.get_ep_group_endpoints(ep_group_id)
        if not eps:
            return {"status": "unknown", "reason": "No endpoints in group"}

        healthy = [e for e in eps if e.health == EndpointHealth.HEALTHY]
        unhealthy = [e for e in eps if e.health == EndpointHealth.UNHEALTHY]

        if len(healthy) == len(eps):
            status = "healthy"
        elif unhealthy:
            status = "degraded" if len(unhealthy) < len(eps) else "unhealthy"
        else:
            status = "unknown"

        # Collect expert coverage
        covered_experts = set()
        for ep in healthy:
            if ep.expert_range != (0, 0):
                covered_experts.update(range(ep.expert_range[0], ep.expert_range[1]))

        total_experts = 0
        for ep in eps:
            if ep.expert_range != (0, 0):
                total_experts = max(total_experts, ep.expert_range[1])

        return {
            "ep_group_id": ep_group_id,
            "status": status,
            "total_ranks": len(eps),
            "healthy_ranks": len(healthy),
            "unhealthy_ranks": len(unhealthy),
            "total_experts": total_experts,
            "covered_experts": len(covered_experts),
            "expert_coverage_pct": (
                (len(covered_experts) / total_experts * 100)
                if total_experts > 0 else 0.0
            ),
            "nvlink_domain": eps[0].nvlink_domain if eps else None,
            "dp_size": eps[0].dp_size if eps else 0,
            "tp_size": eps[0].tp_size if eps else 0,
        }

    def route_to_ep_rank_for_experts(
        self, model: str, expert_ids: Optional[List[int]] = None,
    ) -> Optional[InferenceEndpoint]:
        """Route to the EP rank most likely to hold the needed experts.

        When the router knows which experts a request will activate
        (e.g., from a routing predictor or historical pattern), it can
        send the request directly to the rank hosting those experts.
        This minimizes cross-rank all-to-all traffic.

        Falls back to the standard routing if expert_ids are unknown.
        """
        group_id = self.get_ep_group_for_model(model)
        if not group_id or not expert_ids:
            return self.get_best_endpoint(model=model)

        eps = self.get_ep_group_endpoints(group_id)
        healthy_eps = [
            ep for ep in eps
            if ep.health in (EndpointHealth.HEALTHY, EndpointHealth.UNKNOWN)
            and ep.expert_range != (0, 0)
        ]
        if not healthy_eps:
            return self.get_best_endpoint(model=model)

        # Score each rank by how many of the target experts it hosts
        best_ep = None
        best_overlap = -1
        for ep in healthy_eps:
            start, end = ep.expert_range
            hosted = set(range(start, end))
            overlap = len(hosted.intersection(expert_ids))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ep = ep

        return best_ep or healthy_eps[0]

    def get_ep_topology_summary(self) -> Dict[str, Any]:
        """Get a summary of all EP groups and their topology."""
        groups: Dict[str, List[InferenceEndpoint]] = {}
        for ep in self.endpoints.values():
            if ep.ep_group_id:
                groups.setdefault(ep.ep_group_id, []).append(ep)

        summary = []
        for gid, eps in groups.items():
            health = self.get_ep_group_health(gid)
            summary.append(health)

        return {
            "total_ep_groups": len(groups),
            "groups": summary,
        }

    # ── Status Report ──

    def get_status(self) -> Dict:
        """Get full inference routing status"""
        endpoints = []
        for ep in self.endpoints.values():
            endpoints.append({
                'endpoint_id': ep.endpoint_id,
                'provider': ep.provider,
                'model': ep.model,
                'health': ep.health.value,
                'avg_latency_ms': round(ep.avg_latency_ms, 1),
                'price_per_hour': ep.price_per_hour,
                'is_primary': ep.is_primary,
                'backup': ep.backup_endpoint_id,
                'consecutive_failures': ep.consecutive_failures,
                'region': ep.region,
                'phase': ep.phase.value,
                'flops_tflops': ep.flops_tflops,
                'memory_bandwidth_tbps': ep.memory_bandwidth_tbps,
            })

        healthy = sum(1 for e in self.endpoints.values() if e.health == EndpointHealth.HEALTHY)
        total = len(self.endpoints)

        return {
            'total_endpoints': total,
            'healthy': healthy,
            'unhealthy': total - healthy,
            'endpoints': endpoints,
        }
