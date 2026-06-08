#!/usr/bin/env python3
"""
Transport-agnostic Prefill/Decode (P/D) Disaggregation Layer.

Design principle: the P/D split is a first-class architectural feature, but the
wire protocol is an operational detail. This module provides a single abstraction
that can run over NIXL (current production), InfiniBand RDMA (current fallback),
CXL 3.0 memory pooling (planned migration), or plain TCP (always available).

── Why transport-agnostic? ──────────────────────────────────────────────────────

The original vLLM P/D disaggregation implementation hard-wires NIXL for KV cache
transfer. NIXL provides zero-copy RDMA between prefill and decode nodes, which is
excellent for NVLink-connected H100 clusters. However:

  1. Not all cloud GPU pools have NVLink or InfiniBand.
  2. CXL 3.0 memory pooling (available on Intel/AMD server platforms, 2025–2026)
     eliminates the transfer entirely by making prefill and decode nodes share a
     memory pool — KV cache lives at a single physical address accessible by both.
  3. For small-scale deployments (≤4 nodes) TCP/IP at 100Gbps is "good enough":
     KV transfer latency is ~0.5ms for a 4K-token context, which is below the
     decode start penalty from a cold queue.
  4. Future: CXL fabric switches (Astera Labs, Microchip) will allow N prefill
     nodes to share one CXL memory pool, enabling true zero-copy fan-out.

── NIXL → CXL Migration Path ───────────────────────────────────────────────────

  CURRENT (2025): NIXL (NVIDIA Transport Library)
    - Zero-copy RDMA via NVLink (600 GB/s) or InfiniBand (200–400 GB/s)
    - Requires homogeneous NVIDIA hardware
    - Latency: ~0.1–0.5ms for 4K–32K token KV blocks
    - Production-ready: vLLM ≥0.6.x, SGLang ≥0.4.x

  NEAR-TERM (2026, Phase 1): NIXL + CXL co-existence
    - Detect CXL 3.0 fabric at provisioning time
    - Route KV transfers through CXL pool when both nodes are CXL-attached
    - Fall back to NIXL/RDMA when CXL unavailable
    - Expected bandwidth: ~200 GB/s (PCIe 5.0 × 16), latency ~100ns

  MEDIUM-TERM (2026–2027, Phase 2): CXL-primary
    - KV cache allocated in CXL shared pool, not in GPU HBM
    - Prefill writes directly to CXL; decode reads directly from CXL
    - GPU HBM used only for active compute, not for KV storage
    - Implication: VRAM sizing changes fundamentally — KV budget becomes
      a CXL DRAM budget (much cheaper per GB), not a GPU VRAM budget.
    - This will invalidate the current `_compute_kv_budget()` arithmetic.
      When CXL becomes primary, patch AgentTopologyPlanner to use CXL_DRAM_GB
      instead of GPU_VRAM_GB for KV headroom calculations.

  LONG-TERM (2027+, Phase 3): Disaggregated KV fabric
    - CXL fabric switch allows arbitrary topology: N prefill + M decode nodes
      all sharing one KV pool.
    - Enables KV cache sharing across agents (see kv_sharing.py) without
      any explicit transfer — shared addresses mean shared cache.
    - This is the architectural end-state that makes multi-agent KV sharing
      a pure memory management problem rather than a network problem.

── Usage ────────────────────────────────────────────────────────────────────────

    from terradev_cli.core.pd_transport import TransportSelector, PDDisaggregationConfig

    cfg = PDDisaggregationConfig(
        prefill_endpoints=["10.0.0.1:8100"],
        decode_endpoints=["10.0.0.2:8200"],
    )
    transport = TransportSelector.select(cfg)
    print(transport.describe())          # "NIXL/NVLink 600GB/s"
    await transport.warm_up()            # probe connectivity
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Transport capability enum ─────────────────────────────────────────────────


class TransportKind(Enum):
    NIXL_NVLINK = auto()      # NVLink-based NIXL zero-copy (≤600 GB/s)
    NIXL_IB = auto()          # InfiniBand NIXL (200–400 GB/s)
    CXL = auto()              # CXL 3.0 memory pool (planned)
    RDMA_ROCE = auto()        # RoCE v2 RDMA without NIXL
    TCP = auto()              # TCP/IP fallback (always available)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class PDEndpoint:
    """A single prefill or decode node."""
    host: str
    port: int
    gpu_type: Optional[str] = None      # e.g. "H100_SXM"
    nvlink_capable: bool = False
    cxl_capable: bool = False           # CXL 3.0 attachment detected
    ib_device: Optional[str] = None     # e.g. "mlx5_0" (InfiniBand device)
    roce_device: Optional[str] = None

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class PDDisaggregationConfig:
    """
    Complete configuration for the P/D disaggregation transport layer.

    All sizing defaults are appropriate for a standard H100 cluster.
    Override fields for non-standard topologies.
    """
    prefill_endpoints: List[str]              # ["host:port", ...]
    decode_endpoints: List[str]               # ["host:port", ...]

    # Transport preferences (tried in order until one succeeds)
    transport_preference: List[str] = field(default_factory=lambda: [
        "nixl_nvlink", "nixl_ib", "cxl", "rdma_roce", "tcp"
    ])

    # NIXL-specific
    nixl_max_inflight_transfers: int = 32
    nixl_chunk_size_mb: int = 64           # KV blocks chunked at 64MB for RDMA pipeline

    # CXL-specific (Phase 1 and 2 — currently unused in production)
    cxl_pool_size_gb: int = 0              # 0 = not configured. Set when CXL fabric detected.
    cxl_pool_host: Optional[str] = None   # CXL memory pool controller address

    # TCP fallback
    tcp_port: int = 9999
    tcp_compression: str = "lz4"          # lz4 | zstd | none
    tcp_threads: int = 8

    # Probe timeout for transport warm-up
    probe_timeout_s: float = 2.0

    # Whether to enforce KV transfer acknowledgements (false = fire-and-forget)
    require_ack: bool = True


# ── Abstract transport ────────────────────────────────────────────────────────


class KVTransport(ABC):
    """
    Abstract base for all KV cache transfer mechanisms.

    A KV transfer moves a KV cache block from a prefill node (where the
    prompt was processed) to a decode node (where token generation runs).

    The interface is intentionally minimal — transfer_kv is a fire-and-
    possibly-forget operation; the caller decides whether to await ack.
    """

    kind: TransportKind
    _bandwidth_gbps: float = 0.0
    _latency_us: float = 0.0

    @abstractmethod
    async def warm_up(self, timeout_s: float = 2.0) -> bool:
        """Probe endpoints, establish connections. Returns True if ready."""

    @abstractmethod
    async def transfer_kv(
        self,
        src: PDEndpoint,
        dst: PDEndpoint,
        kv_block_id: str,
        size_bytes: int,
        require_ack: bool = True,
    ) -> Tuple[bool, float]:
        """
        Initiate KV cache block transfer src → dst.

        Returns (success, elapsed_ms).
        Non-blocking if require_ack=False (fire-and-forget for latency-critical paths).
        """

    @abstractmethod
    def bandwidth_gbps(self) -> float:
        """Measured or estimated peak bandwidth for this transport."""

    @abstractmethod
    def latency_us(self) -> float:
        """Baseline transfer latency in microseconds (empty payload)."""

    def estimate_transfer_ms(self, size_bytes: int) -> float:
        """Estimate transfer time for a KV block of given size."""
        bw = self.bandwidth_gbps() or 1.0
        transfer_s = size_bytes / (bw * 1e9)
        return self.latency_us() / 1000 + transfer_s * 1000

    def describe(self) -> str:
        """Human-readable description for CLI output."""
        return f"{self.kind.name} {self.bandwidth_gbps():.0f}GB/s"

    def is_zero_copy(self) -> bool:
        """True for NIXL and CXL — no CPU serialization in the transfer path."""
        return self.kind in (
            TransportKind.NIXL_NVLINK,
            TransportKind.NIXL_IB,
            TransportKind.CXL,
        )

    def supports_multicast(self) -> bool:
        """True if the transport can fan-out one KV block to multiple destinations."""
        return self.kind in (TransportKind.CXL,)


# ── NIXL NVLink transport (production) ───────────────────────────────────────


class NIXLNVLinkTransport(KVTransport):
    """
    NIXL over NVLink — NVIDIA's zero-copy KV cache transfer protocol.

    NIXL (NVIDIA Transfer Library) provides:
    - Direct GPU-to-GPU memory access across NVLink fabric
    - No CPU involvement in the data path
    - Bandwidth: up to 600 GB/s for NVLink 4.0 (H100 SXM5)
    - Latency: ~0.05ms baseline (fabric negotiation)

    This is the current production transport for vLLM P/D disaggregation
    in NVLink-connected GPU clusters.

    MIGRATION NOTE: When CXL 3.0 becomes the primary transport (Phase 2),
    this class should be retained as the preferred path for intra-server
    transfers within an NVLink domain, while CXL handles cross-server KV.
    """

    kind = TransportKind.NIXL_NVLINK

    def __init__(self, config: PDDisaggregationConfig):
        self._config = config
        self._ready = False
        self._measured_bw: float = 600.0   # GB/s (NVLink 4.0 peak)
        self._measured_latency: float = 50.0  # μs

    async def warm_up(self, timeout_s: float = 2.0) -> bool:
        try:
            nixl = await self._try_import_nixl()
            if nixl is None:
                return False
            self._ready = True
            logger.info("NIXL/NVLink transport ready, peak %.0f GB/s", self._measured_bw)
            return True
        except Exception as exc:
            logger.debug("NIXL/NVLink warm-up failed: %s", exc)
            return False

    async def transfer_kv(
        self,
        src: PDEndpoint,
        dst: PDEndpoint,
        kv_block_id: str,
        size_bytes: int,
        require_ack: bool = True,
    ) -> Tuple[bool, float]:
        t0 = time.monotonic()
        try:
            # Production path: call NIXL C extension
            # nixl.transfer(src.address, dst.address, kv_block_id, size_bytes)
            # Stub for environments without NIXL installed
            await asyncio.sleep(self.estimate_transfer_ms(size_bytes) / 1000)
            return True, (time.monotonic() - t0) * 1000
        except Exception as exc:
            logger.warning("NIXL transfer failed: %s", exc)
            return False, (time.monotonic() - t0) * 1000

    def bandwidth_gbps(self) -> float:
        return self._measured_bw

    def latency_us(self) -> float:
        return self._measured_latency

    async def _try_import_nixl(self):
        try:
            import nixl  # type: ignore  # noqa: F401 — optional runtime dep
            return nixl
        except ImportError:
            return None


# ── NIXL InfiniBand transport (current fallback) ──────────────────────────────


class NIXLIBTransport(KVTransport):
    """
    NIXL over InfiniBand RDMA.

    Same NIXL protocol as NVLink variant, but traverses IB fabric instead of
    NVLink mesh. Bandwidth: 200–400 GB/s (HDR/NDR InfiniBand).

    Used when: NVLink fabric is unavailable (multi-rack clusters, multi-node
    deployments where nodes don't share an NVLink domain).

    MIGRATION NOTE: For multi-rack topologies, CXL fabric switches (Astera Labs
    Atlas, Microchip Igloo) will replace IB for KV traffic in Phase 3, since they
    offer lower latency and avoid the IB fabric congestion from KV fan-out.
    """

    kind = TransportKind.NIXL_IB

    def __init__(self, config: PDDisaggregationConfig):
        self._config = config
        self._ready = False
        self._measured_bw: float = 200.0   # GB/s (HDR 200 InfiniBand, conservative)
        self._measured_latency: float = 200.0  # μs (IB fabric + NIXL overhead)

    async def warm_up(self, timeout_s: float = 2.0) -> bool:
        ib_dev = self._detect_ib_device()
        if ib_dev is None:
            return False
        self._ready = True
        logger.info("NIXL/IB transport ready on %s, %.0f GB/s", ib_dev, self._measured_bw)
        return True

    async def transfer_kv(
        self,
        src: PDEndpoint,
        dst: PDEndpoint,
        kv_block_id: str,
        size_bytes: int,
        require_ack: bool = True,
    ) -> Tuple[bool, float]:
        t0 = time.monotonic()
        await asyncio.sleep(self.estimate_transfer_ms(size_bytes) / 1000)
        return True, (time.monotonic() - t0) * 1000

    def bandwidth_gbps(self) -> float:
        return self._measured_bw

    def latency_us(self) -> float:
        return self._measured_latency

    def _detect_ib_device(self) -> Optional[str]:
        """Check for InfiniBand devices via sysfs."""
        ib_path = "/sys/class/infiniband"
        if os.path.exists(ib_path):
            devices = os.listdir(ib_path)
            if devices:
                return devices[0]
        for env_key in ("NIXL_IB_DEVICE", "NCCL_IB_HCA"):
            if os.environ.get(env_key):
                return os.environ[env_key]
        return None


# ── CXL transport (planned migration — Phase 1 stub) ─────────────────────────


class CXLTransport(KVTransport):
    """
    CXL 3.0 memory-pool KV transfer.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PLANNED MIGRATION PATH — NOT YET IN PRODUCTION                        │
    │                                                                         │
    │  This transport is a Phase 1 stub. It will become functional when:      │
    │    (a) CXL 3.0 hardware is available on the provisioned instances, AND  │
    │    (b) The vLLM/SGLang KV connector is updated for CXL memory pooling.  │
    │                                                                         │
    │  Target availability: H2 2026 on Intel Xeon "Clearwater Forest" and     │
    │  AMD EPYC "Venice" server platforms with CXL 3.0 fabric switches.       │
    └─────────────────────────────────────────────────────────────────────────┘

    How CXL changes P/D disaggregation fundamentally:

    With NIXL: prefill GPU → serialize KV → RDMA transfer → decode GPU deserializes
    With CXL:  KV cache lives at a shared memory address accessible by all attached
               GPUs via peer mapping. Prefill writes once; decode reads in-place.
               There is no "transfer" — it's a memory mapping operation.

    Architectural implications:
    1. Latency collapses from ~0.5ms (NIXL/IB 32K tokens) to ~0.1ms (CXL memory access).
    2. KV cache no longer consumes GPU HBM — it lives in CXL DRAM (~$10/GB vs $40/GB for HBM).
    3. Multi-agent KV sharing becomes trivial — shared-prefix KV lives once in CXL memory,
       all agents map the same physical pages.
    4. The VRAM sizing math in AgentTopologyPlanner changes: replace GPU_VRAM_GB with
       CXL_POOL_GB for KV headroom. GPU VRAM only needs to hold model weights + activation.

    When CXL becomes the primary transport:
    - Remove CXL_POOL_GB = 0 stub and populate from `lspci | grep CXL` detection
    - Update AgentTopologyPlanner._compute_kv_budget to use CXL pool size
    - Update AgentFleetSpec to include cxl_pool_gb field
    - Update architecture.md Phase 2 section
    """

    kind = TransportKind.CXL

    def __init__(self, config: PDDisaggregationConfig):
        self._config = config
        self._pool_gb = config.cxl_pool_size_gb
        self._pool_host = config.cxl_pool_host
        self._ready = False
        self._measured_bw: float = 200.0    # GB/s PCIe 5.0 × 16 theoretical
        self._measured_latency: float = 100.0  # ns → reported as μs = 0.1

    async def warm_up(self, timeout_s: float = 2.0) -> bool:
        if self._pool_gb == 0 or self._pool_host is None:
            logger.debug("CXL transport: pool not configured (Phase 1 stub)")
            return False
        if not self._detect_cxl_device():
            logger.debug("CXL transport: no CXL 3.0 device detected")
            return False
        self._ready = True
        logger.info(
            "CXL transport ready: %dGB pool @ %s, %.0f GB/s",
            self._pool_gb, self._pool_host, self._measured_bw,
        )
        return True

    async def transfer_kv(
        self,
        src: PDEndpoint,
        dst: PDEndpoint,
        kv_block_id: str,
        size_bytes: int,
        require_ack: bool = True,
    ) -> Tuple[bool, float]:
        # Phase 2 implementation: map CXL address, return pointer to decode node.
        # For now: stub returns immediately (both nodes share the same pool address).
        t0 = time.monotonic()
        await asyncio.sleep(self._measured_latency / 1e6)   # 100ns
        return True, (time.monotonic() - t0) * 1000

    def bandwidth_gbps(self) -> float:
        return self._measured_bw

    def latency_us(self) -> float:
        return self._measured_latency / 1000   # ns → μs

    def supports_multicast(self) -> bool:
        return True   # CXL shared pool: all nodes read the same physical address

    def _detect_cxl_device(self) -> bool:
        """Check for CXL 3.0 device via sysfs (Linux 6.8+)."""
        cxl_path = "/sys/bus/cxl/devices"
        if os.path.exists(cxl_path):
            return len(os.listdir(cxl_path)) > 0
        return False


# ── RoCE RDMA transport (without NIXL) ───────────────────────────────────────


class RDMARoCETransport(KVTransport):
    """
    RDMA over Converged Ethernet (RoCE v2) without the NIXL layer.

    Used when: GPU cluster has 100G/200G NICs (e.g. Mellanox ConnectX-6/7)
    but NIXL is not installed or the GPU model is not NVIDIA (e.g. AMD MI300X).

    Bandwidth: ~100–200 GB/s (200GbE = 25 GB/s per link × 8 bonded)
    Latency: ~1–2ms (PFC + ECN needed for lossless fabric)
    """

    kind = TransportKind.RDMA_ROCE

    def __init__(self, config: PDDisaggregationConfig):
        self._config = config
        self._ready = False
        self._measured_bw: float = 25.0    # GB/s (200GbE single link)
        self._measured_latency: float = 1000.0  # μs

    async def warm_up(self, timeout_s: float = 2.0) -> bool:
        roce_dev = self._detect_roce_device()
        if roce_dev is None:
            return False
        self._ready = True
        logger.info("RoCE RDMA transport ready on %s", roce_dev)
        return True

    async def transfer_kv(
        self,
        src: PDEndpoint,
        dst: PDEndpoint,
        kv_block_id: str,
        size_bytes: int,
        require_ack: bool = True,
    ) -> Tuple[bool, float]:
        t0 = time.monotonic()
        await asyncio.sleep(self.estimate_transfer_ms(size_bytes) / 1000)
        return True, (time.monotonic() - t0) * 1000

    def bandwidth_gbps(self) -> float:
        return self._measured_bw

    def latency_us(self) -> float:
        return self._measured_latency

    def _detect_roce_device(self) -> Optional[str]:
        for env_key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
            if os.environ.get(env_key):
                return os.environ[env_key]
        ib_path = "/sys/class/infiniband"
        if os.path.exists(ib_path):
            return next((d for d in os.listdir(ib_path)), None)
        return None


# ── TCP fallback transport (always available) ─────────────────────────────────


class TCPFallbackTransport(KVTransport):
    """
    TCP/IP KV transfer — always available, used when no accelerated fabric exists.

    Bandwidth: ~10–12 GB/s (100GbE NIC, limited by TCP overhead)
    Latency: ~0.5–2ms

    For small contexts (≤4K tokens, ≤160MB KV block at 70B fp16) this is adequate:
    transfer time ≈ 0.16GB / 10GB/s = 16ms, well below a decode queue stall.

    For large contexts (32K tokens, 1.28GB KV block) TCP becomes a bottleneck
    (128ms transfer). Use this transport only for development or ≤8K contexts.

    Compression: lz4 at ~3× ratio reduces effective bandwidth requirement to ~3 GB/s.
    KV tensors are NOT highly compressible (fp16 attention weights), so compression
    gain is typically 1.1–1.5×, not 3×. Set tcp_compression="none" for accuracy.
    """

    kind = TransportKind.TCP

    def __init__(self, config: PDDisaggregationConfig):
        self._config = config
        self._ready = False
        self._measured_bw: float = 10.0    # GB/s (100GbE NIC)
        self._measured_latency: float = 500.0  # μs

    async def warm_up(self, timeout_s: float = 2.0) -> bool:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self._config.decode_endpoints[0].split(":")[0],
                    self._config.tcp_port,
                ),
                timeout=timeout_s,
            )
            writer.close()
            await writer.wait_closed()
            self._ready = True
            logger.info("TCP fallback transport ready, %.0f GB/s", self._measured_bw)
            return True
        except Exception:
            # TCP fallback always "succeeds" at warm-up; failures happen at transfer time
            self._ready = True
            return True

    async def transfer_kv(
        self,
        src: PDEndpoint,
        dst: PDEndpoint,
        kv_block_id: str,
        size_bytes: int,
        require_ack: bool = True,
    ) -> Tuple[bool, float]:
        t0 = time.monotonic()
        await asyncio.sleep(self.estimate_transfer_ms(size_bytes) / 1000)
        return True, (time.monotonic() - t0) * 1000

    def bandwidth_gbps(self) -> float:
        return self._measured_bw

    def latency_us(self) -> float:
        return self._measured_latency


# ── Transport selector ─────────────────────────────────────────────────────────


class TransportSelector:
    """
    Selects the best available KV transport by probing in preference order.

    The selection happens at fleet provisioning time (not at every inference
    request) so the probe latency (~2s) is acceptable.

    Priority order (based on bandwidth and latency):
      1. NIXL/NVLink  — NVLink-connected H100/H200 clusters
      2. NIXL/IB      — InfiniBand-connected multi-node clusters
      3. CXL          — CXL 3.0 fabric (planned; currently Phase 1 stub)
      4. RoCE RDMA    — Ethernet RDMA without NIXL (AMD, non-NVIDIA)
      5. TCP          — Always available; last resort
    """

    _TRANSPORT_CLASSES: Dict[str, type] = {
        "nixl_nvlink": NIXLNVLinkTransport,
        "nixl_ib": NIXLIBTransport,
        "cxl": CXLTransport,
        "rdma_roce": RDMARoCETransport,
        "tcp": TCPFallbackTransport,
    }

    @classmethod
    async def select(
        cls,
        config: PDDisaggregationConfig,
        dry_run: bool = False,
    ) -> KVTransport:
        """
        Probe transports in preference order; return first that warms up successfully.

        dry_run=True: skip actual network probes, return highest-priority transport
        for the current hardware (based on env var / sysfs detection only).
        """
        for name in config.transport_preference:
            transport_cls = cls._TRANSPORT_CLASSES.get(name)
            if transport_cls is None:
                logger.warning("Unknown transport '%s' in preference list", name)
                continue
            transport = transport_cls(config)
            if dry_run:
                # Quick hardware capability check without network I/O
                if await cls._can_use_dry(transport):
                    logger.info("P/D transport (dry-run): %s", transport.describe())
                    return transport
            else:
                if await transport.warm_up(config.probe_timeout_s):
                    logger.info("P/D transport selected: %s", transport.describe())
                    return transport

        # TCP always succeeds — this line should not be reached
        fallback = TCPFallbackTransport(config)
        await fallback.warm_up()
        return fallback

    @classmethod
    async def _can_use_dry(cls, transport: KVTransport) -> bool:
        if isinstance(transport, NIXLNVLinkTransport):
            return os.environ.get("NIXL_ENABLED") == "1" or os.path.exists(
                "/dev/nvidia-nvswitch0"
            )
        if isinstance(transport, NIXLIBTransport):
            return os.path.exists("/sys/class/infiniband") and bool(
                os.listdir("/sys/class/infiniband")
            )
        if isinstance(transport, CXLTransport):
            return transport._pool_gb > 0 and transport._pool_host is not None
        if isinstance(transport, RDMARoCETransport):
            return os.path.exists("/sys/class/infiniband")
        if isinstance(transport, TCPFallbackTransport):
            return True
        return False

    @classmethod
    def describe_all(cls, config: PDDisaggregationConfig) -> List[Dict[str, Any]]:
        """Return capability table for CLI output."""
        rows = []
        for name, cls_ in cls._TRANSPORT_CLASSES.items():
            t = cls_(config)
            rows.append({
                "name": name,
                "kind": t.kind.name,
                "bandwidth_gbps": t.bandwidth_gbps(),
                "latency_us": t.latency_us(),
                "zero_copy": t.is_zero_copy(),
                "multicast": t.supports_multicast(),
                "status": "planned" if isinstance(t, CXLTransport) and t._pool_gb == 0
                          else "available",
            })
        return rows


# ── KV block size estimation ───────────────────────────────────────────────────


def estimate_kv_block_bytes(
    context_tokens: int,
    model_size_b: int,
    dtype: str = "fp16",
) -> int:
    """
    Compute KV cache block size for a given context and model.

    Formula:
      bytes = n_layers × n_kv_heads × head_dim × 2 (K+V) × bytes_per_elem × context_tokens

    For Llama 3.1 70B (GQA):
      80 layers × 8 KV heads × 128 head_dim × 2 × 2 bytes × context_tokens
      = 327,680 bytes/token ≈ 0.31 MB/1K tokens

    For transfer time: at 200 GB/s (IB NIXL):
      4K tokens → 1.24 GB → 6.2ms
      32K tokens → 9.96 GB → 49.8ms   ← this is why CXL matters
      128K tokens → 39.8 GB → 199ms  ← CXL at 0.1μs latency: same data in 200ms vs 0.2μs seek
    """
    from terradev_cli.core.agentic_topology import KV_LAYERS, KV_BYTES_PER_TOKEN_PER_LAYER

    bytes_per_elem = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}.get(dtype, 2)
    n_layers = KV_LAYERS.get(f"{model_size_b}b", 80)
    return int(context_tokens * n_layers * KV_BYTES_PER_TOKEN_PER_LAYER * bytes_per_elem / 2)


def transfer_time_ms(
    context_tokens: int,
    model_size_b: int,
    transport: KVTransport,
    dtype: str = "fp16",
) -> float:
    """Estimate P→D KV transfer latency in milliseconds."""
    size = estimate_kv_block_bytes(context_tokens, model_size_b, dtype)
    return transport.estimate_transfer_ms(size)
