# Terradev Architecture

---

## System Layers

```
┌──────────────────────────────────────────────────────────────────┐
│  User / AI Agent (Claude, Cursor, Windsurf)                      │
│  terradev <command>  ──or──  MCP tool_call("...", {...})          │
└──────────────────────────┬───────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
┌─────────────────────┐   ┌────────────────────────────────────┐
│  Python CLI         │   │  MCP Orchestrator                  │
│  (terradev_cli/     │   │  (terradev-mcp/terradev_mcp.py)    │
│   cli.py)           │   │                                    │
│  Click command tree │   │  217 tools · JSON-RPC 2.0 stdio   │
│  ~11,600 lines      │   │  DAG sequencing · idempotency      │
│                     │◄──│  Sub-ms overhead per tool call     │
└────────┬────────────┘   └────────────────────────────────────┘
         │
         ├────────────────────────────────────────────┐
         │                                            │
         ▼                                            ▼
┌─────────────────────┐                  ┌────────────────────┐
│  Provider Layer     │                  │  Core Orchestration │
│  21+ async adapters │                  │                    │
│                     │                  │  training_         │
│  RunPod  Vast.ai    │                  │  orchestrator.py   │
│  Lambda  AWS        │                  │                    │
│  GCP     Azure      │                  │  provision_        │
│  CoreWeave OCI      │                  │  orchestrator.py   │
│  Alibaba OVH        │                  │                    │
│  Hetzner FluidStack │                  │  event_system.py   │
│  ...                │                  │  (triggers/lineage)│
└─────────┬───────────┘                  └────────┬───────────┘
          │                                        │
          ▼                                        ▼
┌─────────────────────────────────────────────────────────────┐
│  NUMA Topology Engine (auto-applied on every provision)     │
│  NUMA alignment · GPUDirect RDMA · CPU pinning              │
│  SR-IOV virtual functions · NCCL tuning                     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Training    │ │  Inference   │ │  Kubernetes  │
│  Stack       │ │  Stack       │ │  Cluster     │
│              │ │              │ │              │
│  torchrun    │ │  vLLM        │ │  Karpenter   │
│  DeepSpeed   │ │  SGLang      │ │  NodePools   │
│  Accelerate  │ │  LoRA router │ │  GPU Operator│
│  Megatron    │ │  InferX      │ │  Monitoring  │
│  FlashOptim  │ │  MoE tmpl    │ │  Stack       │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## Component Detail

### MCP Orchestrator

The MCP server is the interface for AI agents. It runs as a subprocess that Claude/Cursor/Windsurf connect to via the MCP protocol (JSON-RPC 2.0 over stdio).

- **217 tools** — every Terradev capability exposed as a structured tool with JSON Schema parameter validation
- **DAG execution** — the orchestrator builds a dependency graph for multi-step workflows and enforces correct sequencing. An agent can issue `provision_gpu` and `launch_training` simultaneously; the orchestrator ensures training doesn't start until provisioning completes
- **Idempotency** — duplicate tool calls for the same resource are detected and short-circuited
- **Performance** — deserialization/routing adds <1ms overhead per call vs ~50ms for Python MCP servers. Across a 50-step agent workflow this is the difference between 50ms and 2,500ms of pure overhead

### Provider Layer

All providers implement an async base interface:

```python
class BaseProvider:
    async def list_instances() -> list[Instance]
    async def provision(config: ProvisionConfig) -> Instance
    async def terminate(instance_id: str) -> bool
    async def get_status(instance_id: str) -> InstanceStatus
    async def get_pricing(gpu_type: str) -> list[Quote]
```

**Auth patterns:**
- Standard providers: `Authorization: Bearer <key>` via `_make_request()`
- FluidStack: `api-key: <key>` header (non-standard)
- Alibaba Cloud: HMAC-SHA1 signed URL (bypasses `_make_request()` to prevent header clobbering)
- OVHcloud: `X-Ovh-Signature` HMAC (bypasses `_make_request()` for same reason)

`terradev quote` queries all configured providers in parallel via `asyncio.gather()`.

### NUMA Topology Engine

Applied at provision time on every instance, regardless of provider:

```
GPU provisioned
       │
       ├── Detect PCIe topology (lspci -tv)
       ├── Identify NUMA nodes for GPU + NIC
       ├── Apply CPU pinning (cpuset cgroups)
       ├── Load nvidia_peermem (GPUDirect RDMA)
       ├── Configure SR-IOV virtual functions
       └── Set NCCL env vars:
               NCCL_IB_DISABLE=0
               NCCL_IB_GDR_LEVEL=PIX
               NCCL_IB_GDR_READ=1
```

### Training Orchestrator

```
terradev train --script train.py --nodes 4 --gpus-per-node 8
       │
       ├── preflight()
       │     ├── GPU driver check (nvidia-smi)
       │     ├── NCCL all-reduce test
       │     ├── RDMA bandwidth test (if InfiniBand)
       │     └── FlashOptim compatibility check
       │
       ├── stage_dataset()
       │     ├── Compress (zstd, parallel)
       │     └── Pre-place to target region S3
       │
       ├── _flashoptim_auto_config()
       │     ├── Check: bf16/fp16 in script_args?
       │     ├── Check: total VRAM >= 40GB?
       │     ├── Check: not Megatron?
       │     └── Inject FLASHOPTIM_* env vars
       │
       ├── _launch_native()
       │     ├── torchrun / deepspeed / accelerate
       │     ├── SSH mesh across all nodes
       │     └── pip install flashoptim (fail-safe)
       │
       └── checkpoint_loop()
             ├── Save every N steps
             ├── Verify integrity (checksum)
             └── Auto-recover on failure
```

### Inference Stack

```
terradev infer-deploy --model ./llama-2-7b
       │
       ├── Select GPU (cheapest from pool)
       ├── Provision + NUMA topology
       │
       ├── Auto-apply (no flags needed):
       │     ├── --kv-connector=offloading
       │     ├── --speculative-config.method=mtp
       │     └── --enable-sleep-mode + VLLM_SERVER_DEV_MODE=1
       │
       └── (opt-in via lora/agentic-serving):
             ├── --enable-lora (multi-tenant adapters)
             ├── LMCache (Redis KV sharing)
             └── vLLM Router (P/D disaggregation)
```

**Disaggregated Prefill/Decode — Transport-Agnostic Layer:**

```
core/pd_transport.py — KVTransport abstraction

Client request
       │
       ▼
vLLM Router (sticky routing by prefix hash)
       │
       ├──▶ Prefill pool (H100, compute-bound)
       │    Processes input prompt
       │    KV cache block ──► TransportSelector.select(config)
       │                             │
       │              ┌──────────────┼──────────────┐
       │              ▼              ▼              ▼
       │         NIXLNVLink      NIXL/IB        TCP fallback
       │         600 GB/s       200 GB/s        10 GB/s
       │         (NVLink 4.0)   (HDR IB)        (always avail)
       │              │              │              │
       │              └──────────────┴──────────────┘
       │                             │
       └──▶ Decode pool (H100, memory-bound)
            Receives KV block, generates tokens
            Subsequent requests with same prefix
            route here to reuse KV cache
```

**Transport selection priority** (probed at provision time, ~2s):

| Priority | Transport | Bandwidth | Latency | Requirement |
|----------|-----------|-----------|---------|-------------|
| 1 | NIXL/NVLink | 600 GB/s | 0.05ms | NVLink 4.0 fabric |
| 2 | NIXL/InfiniBand | 200–400 GB/s | 0.2ms | HDR/NDR IB |
| 3 | CXL 3.0 | 200 GB/s | 0.0001ms | **PLANNED** |
| 4 | RoCE RDMA | 25 GB/s | 1ms | 200GbE NIC |
| 5 | TCP fallback | 10 GB/s | 0.5ms | Always available |

**NIXL → CXL Migration Path (planned):**

```
CURRENT (2025):  NIXL protocol — zero-copy RDMA via NVLink or InfiniBand.
                 Prefill GPU serializes KV → RDMA transfer → Decode GPU.
                 Production-ready in vLLM ≥0.6.x, SGLang ≥0.4.x.

PHASE 1 (2026):  NIXL + CXL co-existence.
                 When CXL 3.0 fabric detected (Intel Clearwater Forest /
                 AMD Venice), route KV through CXL pool. Fall back to NIXL.
                 Expected bandwidth: ~200 GB/s. Latency: ~100ns.

PHASE 2 (2026–2027): CXL-primary.
                 KV cache allocated in CXL shared DRAM pool, not GPU HBM.
                 Prefill writes directly to CXL; decode reads in-place.
                 No serialization, no transfer — pointer handoff only.
                 GPU HBM used only for model weights + active compute.
                 VRAM sizing changes: replace GPU_VRAM_GB with CXL_POOL_GB
                 in AgentTopologyPlanner._compute_kv_budget().

PHASE 3 (2027+): CXL fabric switch (Astera Labs Atlas, Microchip Igloo).
                 N prefill + M decode nodes share one CXL memory pool.
                 Multi-agent KV sharing becomes a memory management problem:
                 shared prefix maps to one physical address, accessed by all.
```

---

### Multi-Agent KV Cache Sharing (v5.3.0)

**The problem:** Each agent independently stores its KV cache. For N agents
with 70% shared context (system prompt + task), the shared portion is stored
N times — pure waste that scales linearly with fleet size.

**The math** (from `core/kv_sharing.py`):

```
20 agents × Llama-70B × 32K context (fp16):
  KV per agent:      32K × 80 layers × 512 bytes/tok/layer ÷ 2 = 0.655 GB
  Naive (N copies):  20 × 0.655 = 13.1 GB per GPU slot

  With broadcast sharing (70% shared = 22.4K shared, 9.6K unique):
    Shared KV (stored once):  22.4K × 0.0205 GB/K = 0.46 GB
    Per-agent unique KV:       9.6K × 0.0205 GB/K = 0.197 GB each
    Total fleet KV:           0.46 + 20 × 0.197   = 4.4 GB
    Saving: 13.1 - 4.4 = 8.7 GB (66% reduction)
    → 3× more agents per GPU
    → 3× fewer GPUs needed
    → ~$14/hr savings on a 20-agent H100 fleet

Eviction cost without sharing (H100 SXM, 32K context):
  Re-prefill time: 32,768 / 30,000 tokens/sec ≈ 1.1s per eviction
  With 20 agents on 6 GPUs (6 fit per GPU = 36 slots): no evictions
  With 20 agents on 2 GPUs (6 fit per GPU = 12 slots): ~8 overflow agents
  → 8 × 3,600 / 30s turns × 1.1s = 1,056s/hr wasted ≈ 29% throughput lost
```

**CLI:**
```
terradev provision -g H100 --agents 20 --context 32k --model-name llama-70b
terradev provision -g H100 --agents 20 --context 32k --sharing-topology broadcast --dry-run
```

**Output** (what no other CLI computes today):
```
KV Sharing Plan — 20 agents × llama-70b @ 32K ctx
Topology: broadcast  |  Shared prefix: 22K tokens

  VRAM without sharing: 13.1 GB (6 agents/GPU → 4 GPUs needed)
  VRAM with sharing:     4.4 GB (18 agents/GPU → 2 GPUs needed)
  Saving: 8.7 GB (66% reduction)

  Cost/hr without sharing: $9.96
  Cost/hr with sharing:    $4.98  (saves $4.98/hr = $119/day)
  ✓ With sharing: re-prefill overhead negligible (<1%)

  TIER             INSTANCES         GPU   TP   CONC  CONTEXT   $/HR
  ────────────────────────────────────────────────────────────────────
  reasoning                2     H100_SXM    1      4     32K  $ 4.98
  decode                   2  A100_SXM_80    2      2     32K  $ 5.96
  cpu_tools                3          CPU    1     10     n/a  $ 1.80
  ────────────────────────────────────────────────────────────────────
  TOTAL                                                        $12.74/hr
```

**New files:**
- `core/pd_transport.py` — `KVTransport` ABC, `NIXLNVLinkTransport`, `NIXLIBTransport`,
  `CXLTransport` (stub, Phase 1), `RDMARoCETransport`, `TCPFallbackTransport`,
  `TransportSelector`, `transfer_time_ms()`
- `core/kv_sharing.py` — `MultiAgentVRAMPlanner`, `KVSharingPlan`, `AgentKVBudget`,
  `EvictionCostModel`, `SharingTopology` enum

### Kubernetes Layer

```
terradev k8s create my-cluster
       │
       ├── EKS/GKE/AKS cluster provisioned
       ├── Karpenter installed
       │     ├── NodePool: karpenter.sh/v1
       │     └── EC2NodeClass: karpenter.k8s.aws/v1
       │
       ├── NodePool config (auto-applied):
       │     ├── kubelet.topologyManagerPolicy: single-numa-node
       │     ├── kubelet.cpuManagerPolicy: static
       │     └── tolerations: karpenter.sh/nodepool
       │
       ├── GPU Operator:
       │     ├── NVIDIA device plugin
       │     ├── MIG configuration
       │     └── Time-slicing
       │
       └── Monitoring stack:
             └── DCGM exporter (GPU metrics)
```

### RAG Stack

```
clusters/rag-template/
       │
       ├── Qdrant (vector DB)
       │     ├── REST API: port 6333
       │     ├── gRPC API: port 6334
       │     ├── Auth: api-key header (cloud) / none (self-hosted)
       │     └── HNSW index (M, ef_construction configurable)
       │
       ├── Embedding model (sidecar)
       │     └── sentence-transformers served via FastAPI
       │
       ├── vLLM (inference)
       │     └── All auto-optimizations applied
       │
       ├── Arize Phoenix (observability)
       │     ├── OTLP trace ingestion: port 6006
       │     ├── REST spans API: /v1/projects, /v1/spans
       │     └── SpanQuery DSL filtering
       │
       ├── NeMo Guardrails (safety)
       │     ├── POST /v1/chat/completions (with config_id)
       │     ├── Colang 2.x configs (topical, jailbreak, PII, factcheck)
       │     └── Memory: MemoryStore (dev) / RedisStore (prod)
       │
       └── Redis (KV cache + guardrails memory)
```

---

## Data Flow: Credentials

```
User: terradev configure --provider runpod
       │
       ├── Prompt for RUNPOD_API_KEY
       ├── Validate key against RunPod API
       └── Write to ~/.terradev/credentials.json (local only)
                   │
                   └── Never transmitted to Terradev servers
                       Never logged
                       Never included in telemetry
```

Per-provision SSH keypairs are auto-generated, used for the session, and discarded.

---

## Data Flow: Quote → Provision

```
terradev quote --gpu a100
       │
       ├── asyncio.gather() → all configured providers in parallel
       │     ├── RunPod: GET /graphql (GPU listings)
       │     ├── Vast.ai: GET /api/v0/bundles
       │     ├── Lambda: GET /api/v1/instance-types
       │     └── ...
       │
       ├── Normalize response → unified Quote schema
       ├── Sort by $/hr
       └── Display table (spot flagged, region shown)

terradev provision --gpu a100 --count 1
       │
       ├── Select cheapest quote (or specified provider)
       ├── BaseProvider.provision(config)
       ├── Wait for RUNNING state
       ├── NUMA topology engine applies
       └── Return instance_id, IP, SSH config
```

---

## Local GPU Discovery

```
terradev local scan
       │
       ├── NVML bindings (primary, 5-10x faster than nvidia-smi)
       │     ├── GPU model, VRAM, PCIe bus ID, NUMA affinity
       │     ├── Utilization, temperature, power draw
       │     ├── Driver version, CUDA version, compute capability
       │     └── NVLink topology
       │
       └── Falls back to nvidia-smi parsing if NVML unavailable

       ├── Remote scan (SSH):
       │     terradev local scan --host 192.168.1.50 --user ubuntu --key ~/.ssh/id_rsa
       │     └── Same NVML/nvidia-smi discovery over SSH tunnel
       │
       └── Registration:
             terradev local register --name workstation-4090
             └── Writes to pool at ~/.terradev/local_pool.json
                 Priced at $0.00/hr
                 Included in quote output when --include-local
```

---

## Event System (Triggers / Lineage / Environments)

```
core/event_system.py

EventBus
  └── publish(Event) → trigger_manager.evaluate()
                            └── matching triggers fire target_pipeline

trigger_manager
  └── triggers: dict[name → Trigger]
      Trigger types: EVENT_BASED | SCHEDULE | CONDITION
      Target environments: dev | staging | prod

lineage_service
  └── artifacts: dict[id → Artifact]
      executions: dict[id → Execution]
      Tracks: dataset → model → checkpoint → metrics

environment_manager
  └── promotion_requests: dict[id → PromotionRequest]
      Flow: dev → staging → prod
      Optional approval gate before promotion
```

---

## Agentic Fleet Provisioning (v5.3.0)

Purpose-built heterogeneous GPU fleet management for multi-agent LLM workloads.
Research basis: arXiv:2605.26297 "Agentic AI Workload Characteristics" (2026).

**Key empirical findings driving the design:**
- Decode dominates: 91–98% of LLM execution time is decode, not prefill
- KV cache hit rates: 84.6–99.5% when context stays resident — eviction = recompute
- Context footprint: avg 37K–80K tokens, P95 tail up to 166K tokens
- Tool execution: 2–29% of total runtime (retrieval workloads: 25–29%)

```
terradev agent deploy --agents 16 --model meta-llama/Llama-3.1-70B-Instruct
       │
       ├── AgentTopologyPlanner.infer_from_agent_count(n=16, model)
       │     ├── Parse model size (70B)
       │     ├── Select reasoning GPU: H100 SXM (KV cache preservation)
       │     ├── Select decode GPU: A100 SXM 80GB × 2 (TP=2, bandwidth-optimised)
       │     ├── CPU tools: 48-vCPU instances (Bash/WebFetch/file ops)
       │     └── AgentFleetSpec  [fleet_id, tiers, networking, autoscaling, cost]
       │
       ├── AgenticProvisioner.provision_fleet(spec)
       │     ├── Wave 0: quote_reasoning ‖ quote_decode ‖ quote_cpu
       │     ├── Wave 1: provision_reasoning ‖ provision_decode ‖ provision_cpu
       │     │           (via ParallelProvisioner across configured clouds)
       │     ├── Wave 2: configure_networking (placement group, VPC peering)
       │     ├── Wave 3: deploy_reasoning_inference ‖ deploy_decode_inference
       │     │           (vLLM: --enable-prefix-caching, --kv-connector=offloading)
       │     └── Wave 4: register_fleet → ~/.terradev/fleets/<fleet_id>.json
       │
       └── InferenceRouter (KV prefix-aware routing)
             ├── PrefixCacheIndex: route by prompt prefix hash
             ├── Sticky routing: X-Agent-Id header → same pod
             └── Preserves KV residency across agent turns
```

**Three-tier hardware model:**

| Tier | Hardware | Optimised For | Autoscale Signal |
|------|----------|---------------|------------------|
| reasoning | H100 SXM 80GB | KV cache preservation, long context | TTFT p95 > 2000ms |
| decode | A100 SXM 80GB × 2 (TP=2) | Memory bandwidth, token throughput | Decode queue depth > 6 |
| cpu_tools | 48-vCPU (no GPU) | Bash, WebFetch, Read/Edit/Grep | Tool latency p95 > 400ms |

**Why NOT GPU utilization for autoscaling:**
Agentic workloads are inherently bursty — GPU utilization oscillates between 0% (tool
execution) and 100% (decode burst) within a single agent turn. TTFT and queue depth
are stable signals that reflect actual user-visible latency.

**CLI commands:**
```
terradev agent plan     --agents 16 --model ...     # size fleet, show cost
terradev agent deploy   --agents 16 --model ...     # provision all tiers
terradev agent status   --fleet-id ag_xxx           # KV hit rate, TTFT, queue
terradev agent scale    --fleet-id ag_xxx --tier decode --count 8
terradev agent cost     --fleet-id ag_xxx           # per-tier spend breakdown
terradev agent list                                 # all known fleets
terradev agent teardown --fleet-id ag_xxx           # destroy + remove state
```

**MCP tools (6 new in v6.0.0):**
`agent_topology_plan`, `agent_fleet_provision`, `agent_fleet_status`,
`agent_fleet_scale`, `agent_fleet_cost`, `agent_fleet_teardown`

---

## File Map

```
terradev_cli/
  cli.py                        Main CLI (~14,400 lines, Click)
  cli_karpenter.py              Karpenter subcommand group
  cli_hf_spaces.py              HuggingFace Spaces subcommand group
  core/
    agentic_topology.py         AgentFleetSpec, AgentTopologyPlanner (v5.3.0)
    agentic_provisioner.py      AgenticProvisioner, fleet state mgmt (v5.3.0)
    pd_transport.py             Transport-agnostic P/D KV layer (v5.3.0)
    kv_sharing.py               Multi-agent KV sharing VRAM planner (v5.3.0)
    training_orchestrator.py    DeepSpeed/torchrun/FlashOptim
    provision_orchestrator.py   Parallel provision + NUMA
    evaluation_orchestrator.py  Model/endpoint evaluation
    event_system.py             Triggers, environments, lineage
    cost_tracker.py             Spend tracking
    price_intelligence.py       Spot market analytics + ML
    trace_viewer.py             Phoenix span tree renderer
    dag_executor.py             Wave-parallel DAG execution (Python)
    parallel_provisioner.py     Multi-cloud parallel provisioning
    warm_pool_manager.py        Pre-warming strategies for bursty workloads
    inference_router.py         KV prefix-aware routing + auto-failover
  ml_services/
    __init__.py
    vllm_service.py             vLLM config + LoRA + sleep
    phoenix_service.py          Arize Phoenix OTLP + REST
    guardrails_service.py       NeMo Guardrails Colang
    qdrant_service.py           Qdrant collections + search
  providers/
    base_provider.py
    runpod.py  vastai.py  lambda.py  aws.py  gcp.py  azure.py
    alibaba.py  ovh.py  fluidstack.py  hetzner.py  ...
  kubernetes/
    kubernetes_service.py
    kubernetes_enhanced.py      GPU operator, MIG, monitoring
clusters/
  agentic-template/             Heterogeneous agent fleet (v6.0.0)
    helm/values-agentic.yaml    Three-tier Helm values
    k8s/fleet-manifests.yaml    Deployments, HPAs, NetworkPolicies, Karpenter
  moe-template/                 MoE K8s + Helm (all opts auto-applied)
  rag-template/                 Qdrant + Phoenix + Guardrails
  glm-5/                        GLM-5 production templates
rust/                           NVML bindings and MCP codec
terradev-mcp/
  terradev_mcp.py               304-tool MCP server (~8,700 lines)
```
