#!/usr/bin/env python3
"""
Typed domain contracts for cloud provider SDK.

Replaces Dict[str, Any] with structured dataclasses for type safety,
IDE autocomplete, and provider-agnostic interfaces.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Awaitable


class InstanceStatus(str, Enum):
    """Instance lifecycle states"""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    FAILED = "failed"
    PREEMPTED = "preempted"  # spot-specific
    UNKNOWN = "unknown"


class GPUVendor(str, Enum):
    """GPU manufacturer"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"


@dataclass
class GPUDescriptor:
    """Canonical GPU specification"""
    name: str  # canonical: "H100-80GB", "A100-40GB", "RTX-4090"
    vendor: GPUVendor
    vram_gb: int
    count: int = 1
    tflops_bf16: Optional[float] = None
    tflops_fp16: Optional[float] = None
    tflops_fp32: Optional[float] = None
    bandwidth_gb_s: Optional[float] = None
    nvlink: bool = False  # NVLink interconnect between GPUs on node
    compute_capability: Optional[str] = None  # e.g., "8.0", "9.0"


@dataclass
class QuoteRequest:
    """Request for pricing/availability from a provider"""
    gpu: GPUDescriptor
    region: Optional[str] = None
    spot: Optional[bool] = None  # None = no preference
    max_price_hr: Optional[float] = None
    min_disk_gb: int = 0
    min_vcpus: int = 0


@dataclass
class Quote:
    """Pricing/availability response from a provider"""
    provider: str
    provider_instance_type: str  # raw provider-side slug/type
    region: str
    gpu: GPUDescriptor
    price_hr: float
    spot: bool
    availability: str  # "available", "limited", "unavailable"
    latency_ms: float = 0.0
    disk_gb: int = 0
    vcpus: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvisionRequest:
    """Request to provision an instance"""
    gpu: GPUDescriptor
    region: str
    spot: bool = False
    ssh_pubkey: str = ""
    disk_gb: int = 50
    image: str = "ubuntu-22.04"  # logical name; providers map to own images
    tags: Dict[str, str] = field(default_factory=dict)
    startup_script: str = ""
    max_price_hr: Optional[float] = None  # circuit-breaker at provision time
    min_vcpus: int = 0


@dataclass
class ProvisionResult:
    """Result of a provision operation"""
    instance_id: str
    provider: str
    region: str
    gpu: GPUDescriptor
    price_hr: float
    spot: bool
    status: InstanceStatus
    ip: Optional[str] = None
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstanceInfo:
    """Current state of an instance"""
    instance_id: str
    provider: str
    status: InstanceStatus
    gpu: Optional[GPUDescriptor] = None
    ip: Optional[str] = None
    price_hr: float = 0.0
    spot: bool = False
    uptime_s: int = 0
    region: str = ""
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderEvent:
    """
    Provider-emitted event (HarmonAIze macro-level abstraction).
    Used for spot preemption detection, capacity changes, health alerts.
    """
    provider: str
    instance_id: str
    event_type: str  # "preempted", "capacity_available", "health_degraded", "recovered"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class CredentialField:
    """Credential field declaration for provider configuration"""
    name: str  # e.g. "api_key"
    required: bool
    description: str
    env_var: str  # e.g. "RUNPOD_API_KEY"
    secret: bool = True
    default: Optional[str] = None


@dataclass
class ProviderCapabilities:
    """
    Static declaration of provider capabilities.
    Used for pre-filtering and UI hints.
    """
    name: str
    supported_gpus: List[str]  # canonical GPU names from gpu_catalog
    regions: List[str]
    supports_spot: bool
    supports_ssh_keys: bool
    supports_volumes: bool
    supports_execute: bool  # can run arbitrary commands
    min_provision_time_s: int  # SLA: fastest expected boot
    max_parallel_provisions: int  # rate limit hint for semaphore sizing
    pricing_model: str  # "on_demand", "spot_only", "reserved", "serverless"
    supports_webhooks: bool = False  # native event streaming vs polling


@dataclass
class HealthStatus:
    """Health check result"""
    healthy: bool
    reason: str = ""
    latency_ms: float = 0.0
    timestamp: float = 0.0


@dataclass
class ProviderHealth:
    """
    Per-provider health metrics tracked by ProviderRegistry.
    Used for circuit breaker decisions and provider ranking.
    """
    provider: str
    consecutive_failures: int = 0
    last_failure_ts: float = 0.0
    last_success_ts: float = 0.0
    avg_latency_ms: float = 0.0
    spot_preemption_rate: float = 0.0  # from SpotHedge-style tracking
    total_provisions: int = 0
    total_failures: int = 0


@dataclass
class ProviderProfile:
    """
    Encodes provider-specific quirks and behaviors for intelligent routing.

    Used by ProviderRegistry to make intelligent decisions about:
    - Which providers to query (pre-filtering)
    - How to handle failures (fallback routing)
    - How to optimize for specific workloads (egress costs, spot reliability)
    """
    name: str
    api_style: str  # "rest", "graphql", "jsonapi", "k8s"
    auth_type: str  # "bearer", "basic", "hmac_sha256", "x_api_key", "service_account"

    # API quirks
    requires_instance_type_mapping: bool = False
    quote_method: str = "get"  # "get", "post_filter", "graphql"
    provision_requires_location_id: bool = False
    provision_requires_namespace: bool = False

    # Capacity and availability
    has_capacity_check: bool = False
    has_fallback_routing: bool = False
    fallback_providers: List[str] = field(default_factory=list)

    # Storage and persistence
    volume_required_for_persistence: bool = False
    volume_cost_separate: bool = False
    data_loss_on_restart: bool = False

    # Networking
    egress_cost: float = 0.0  # $/GB
    ssh_port_fixed: bool = True
    ssh_default_port: int = 22

    # Spot and preemption
    supports_spot: bool = True
    spot_interruption_notice_minutes: int = 0
    spot_preemption_webhook: bool = False

    # Rate limiting
    rate_limit_per_minute: int = 0  # 0 = no limit

    # Compute model
    compute_model: str = "vm"  # "vm", "pod", "k8s"
    isolation_level: str = "vm"  # "vm", "container", "k8s_pod"

    # Dependencies
    requires_boto3: bool = False
    requires_gcp_sdk: bool = False

    # Special features
    has_multi_tier_cloud: bool = False  # e.g., RunPod Community vs Secure
    has_capacity_reservations: bool = False
    has_legacy_billing: bool = False

    # Container quirks
    container_conflict_risk: bool = False
    container_image_pinning_required: bool = False

    # Lifecycle quirks
    supports_stop_start: bool = True
    supports_terminate_only: bool = False

    # Region quirks
    region_specific_availability: bool = False
    requires_zone_probing: bool = False
