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
│  Python CLI         │   │  Rust MCP Orchestrator             │
│  (terradev_cli/     │   │  (terradev-mcp/terradev_mcp.py)    │
│   cli.py)           │   │                                    │
│  Click command tree │   │  218 tools · JSON-RPC 2.0 stdio   │
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

### Rust MCP Orchestrator

The MCP server is the interface for AI agents. It runs as a subprocess that Claude/Cursor/Windsurf connect to via the MCP protocol (JSON-RPC 2.0 over stdio).

- **218 tools** — every Terradev capability exposed as a structured tool with JSON Schema parameter validation
- **DAG execution** — the orchestrator builds a dependency graph for multi-step workflows and enforces correct sequencing. An agent can issue `provision_gpu` and `launch_training` simultaneously; the orchestrator ensures training doesn't start until provisioning completes
- **Idempotency** — duplicate tool calls for the same resource are detected and short-circuited
- **Performance** — Rust deserialization/routing adds <1ms overhead per call vs ~50ms for Python MCP servers. Across a 50-step agent workflow this is the difference between 50ms and 2,500ms of pure overhead

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

**Disaggregated Prefill/Decode (advanced):**

```
Client request
       │
       ▼
vLLM Router (sticky routing by prefix hash)
       │
       ├──▶ Prefill pool (H100, compute-bound)
       │    Processes input prompt
       │    KV cache → NIXL zero-copy RDMA transfer
       │
       └──▶ Decode pool (H100, memory-bound)
            Generates tokens
            Subsequent requests with same prefix
            route here to reuse KV cache
```

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
             ├── Prometheus
             ├── Grafana (grafana/grafana Helm chart)
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
       ├── Rust NVML bindings (primary, 5-10x faster than nvidia-smi)
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

## Agentic Fleet Provisioning (v6.0.0)

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
    agentic_topology.py         AgentFleetSpec, AgentTopologyPlanner (v6.0.0)
    agentic_provisioner.py      AgenticProvisioner, fleet state mgmt (v6.0.0)
    training_orchestrator.py    DeepSpeed/torchrun/FlashOptim
    provision_orchestrator.py   Parallel provision + NUMA
    evaluation_orchestrator.py  Model/endpoint evaluation
    event_system.py             Triggers, environments, lineage
    cost_tracker.py             Spend tracking
    price_intelligence.py       Spot market analytics + ML
    trace_viewer.py             Phoenix span tree renderer
    dag_executor.py             Wave-parallel DAG execution (Rust-backed)
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
rust/                           NVML bindings, DAG orchestrator, MCP codec
terradev-mcp/
  terradev_mcp.py               304-tool MCP server (~8,700 lines)
```
