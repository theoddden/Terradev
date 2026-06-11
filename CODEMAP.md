# Terradev Codebase Architecture Map

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TERRADEV CLI v5.3.3                              │
│                        Cross-Cloud GPU Orchestration Platform                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            ┌───────▼────────┐            ┌────────▼────────┐
            │  Python CLI    │            │  Rust Accelerators│
            │  (terradev_cli)│            │  (rust/)         │
            │  211 files     │            │  30+ crates      │
            └────────────────┘            └─────────────────┘
                    │                               │
        ┌───────────┼───────────┐                   │
        │           │           │                   │
┌───────▼────┐ ┌───▼────┐ ┌────▼────┐       ┌───────▼────────┐
│   Core     │ │Providers│ │ML Services│     │  Performance   │
│  (80 files)│ │(33 files)│ │(23 files)│     │  Critical      │
└────────────┘ └─────────┘ └──────────┘     │  Components    │
                                             └────────────────┘
```

## Module Breakdown

### 1. Core Orchestration (`terradev_cli/core/`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CORE ORCHESTRATION                             │
│                              (80 modules)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌────────▼────────┐
│  Provisioning   │      │  Training & Inference│      │  Optimization   │
│                 │      │                     │      │                 │
│ • agentic_      │      │ • training_          │      │ • cost_         │
│   provisioner   │      │   orchestrator       │      │   optimizer     │
│ • parallel_     │      │ • inference_         │      │ • price_        │
│   provisioner   │      │   router             │      │   intelligence  │
│ • gpu_          │      │ • inference_         │      │ • egress_       │
│   topology      │      │   spot_manager       │      │   optimizer     │
│ • mig_manager   │      │ • warm_pool_         │      │ • cost_         │
│ • deployment_   │      │   manager            │      │   scaler        │
│   router        │      │ • model_             │      │ • quota_        │
│ • job_state_    │      │   orchestrator       │      │   manager       │
│   manager       │      │ • checkpoint_         │      │                 │
└─────────────────┘      │   manager            │      └─────────────────┘
                         │ • kv_cache_          │
                         │   checkpoint_        │
                         │   manager            │
                         │ • kv_sharing         │
                         │ • pd_transport       │
                         │   (P/D disaggregation)│
                         └──────────────────────┘
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌────────▼────────┐
│  Data & Storage│      │  Networking & I/O    │      │  Security & Auth│
│                 │      │                     │      │                 │
│ • dataset_     │      │ • pd_transport       │      │ • auth          │
│   stager       │      │   (NIXL/NVLink/CXL)  │      │ • credential_   │
│ • cache_       │      │ • egress_cost_       │      │   vault         │
│   manager      │      │   monitor            │      │ • oidc_         │
│ • manifest_    │      │ • public_ip_         │      │   provider      │
│   cache        │      │   billing_tracker    │      │ • distributed_  │
│ • weight_      │      │ • semantic_          │      │   lock          │
│   streaming_   │      │   router             │      │ • rate_         │
│   manager      │      │ • ssh_key_           │      │   limiter       │
└─────────────────┘      │   manager            │      └─────────────────┘
                         └──────────────────────┘
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌────────▼────────┐
│  Observability │      │  Governance & Compliance│      │  Automation     │
│                 │      │                        │      │                 │
│ • telemetry     │      │ • data_governance      │      │ • dag_executor  │
│ • training_    │      │ • drift_detector       │      │ • event_system  │
│   monitor       │      │ • auto_lineage         │      │ • pipeline_     │
│ • evaluation_   │      │ • migration_           │      │   schema        │
│   orchestrator  │      │   orchestrator         │      │ • gitops_       │
│ • rust_         │      │                        │      │   manager       │
│   telemetry     │      │                        │      │ • quick_start   │
└─────────────────┘      └────────────────────────┘      └─────────────────┘
```

### 2. Provider Layer (`terradev_cli/providers/`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PROVIDER LAYER                                   │
│                         (23+ Cloud Providers)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌────────▼────────┐
│  Hyperscalers   │      │  GPU Clouds         │      │  Emerging       │
│                 │      │                     │      │  Providers      │
│ • aws_provider  │      │ • runpod_provider   │      │ • yottalabs_    │
│ • gcp_provider  │      │ • vastai_provider   │      │   provider      │
│ • azure_provider│      │ • lambda_labs_      │      │ • e2e_networks_ │
│ • oracle_       │      │   provider          │      │   provider      │
│   provider      │      │ • coreweave_        │      │ • fluidstack_   │
└─────────────────┘      │   provider          │      │   provider      │
                         │ • crusoe_provider    │      │ • siliconflow_  │
                         │ • tensordock_        │      │   provider      │
                         │   provider          │      │ • hetzner_      │
                         └──────────────────────┘      │   provider      │
        ┌───────────────────────────┼───────────────────────────┐        │
        │                           │                           │        │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌────────▼────────▼────────┐
│  ML Platforms   │      │  Infrastructure     │      │  Provider System       │
│                 │      │  as Code            │      │                        │
│ • huggingface_  │      │ • digitalocean_     │      │ • base_provider         │
│   provider      │      │   provider          │      │ • provider_factory      │
│ • baseten_      │      │ • hyperstack_       │      │ • provider_profiles     │
│   provider      │      │   provider          │      │ • registry              │
│ • inferx_       │      │ • alibaba_           │      │ • gpu_catalog           │
│   provider      │      │   provider          │      │ • types                 │
│ • latitude_     │      │ • ovhcloud_         │      │ • real_pricing          │
│   provider      │      │   provider          │      └────────────────────────┘
└─────────────────┘      └──────────────────────┘
```

### 3. ML Services Layer (`terradev_cli/ml_services/`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ML SERVICES LAYER                                 │
│                         (23 Integration Services)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌────────▼────────┐
│  Inference      │      │  Training & MLOps    │      │  Orchestration   │
│  Engines        │      │                     │      │                 │
│ • vllm_service  │      │ • ray_service       │      │ • ray_enhanced   │
│ • sglang_service │      │ • ray_enhanced      │      │ • kubernetes_    │
│ • ollama_service│      │ • mlflow_service    │      │   service        │
│ • kserve_service│      │ • dvc_service       │      │ • kubernetes_    │
│ • lmcache_      │      │ • wandb_enhanced    │      │   enhanced      │
│   service       │      │ • drift_retrain_    │      │ • langchain_     │
│ • agentic_      │      │   service           │      │   service        │
│   serving       │      │                     │      │ • langgraph_     │
└─────────────────┘      └──────────────────────┘      │   service        │
        │                           │                   └─────────────────┘
        │                           │
┌───────▼────────┐      ┌──────────▼──────────┐
│  Observability  │      │  Vector DB & Safety │
│                 │      │                     │
│ • langfuse_     │      │ • qdrant_service    │
│   service       │      │ • guardrails_       │
│ • phoenix_      │      │   service           │
│   service       │      │                     │
└─────────────────┘      └──────────────────────┘
```

### 4. Rust Accelerators (`rust/`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RUST ACCELERATORS                                   │
│                    (30+ Performance-Critical Crates)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌────────▼────────┐
│  Orchestration  │      │  Resource Management │      │  Networking     │
│                 │      │                     │      │                 │
│ • terradev-dag- │      │ • terradev-resource-│      │ • terradev-     │
│   executor      │      │   pool              │      │   connection-  │
│ • terradev-     │      │ • terradev-quota-   │      │   pool         │
│   state-machine │      │   manager           │      │ • terradev-     │
│ • terradev-     │      │ • terradev-         │      │   egress-      │
│   tool-registry │      │   distributed-lock  │      │   optimizer    │
└─────────────────┘      └──────────────────────┘      └─────────────────┘
        │                           │                           │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌────────▼────────┐
│  Performance   │      │  Storage & Cache    │      │  Security       │
│                 │      │                     │      │                 │
│ • terradev-     │      │ • terradev-cache-   │      │ • terradev-     │
│   gpu-discovery │      │   eviction          │      │   credential-  │
│ • terradev-     │      │ • terradev-         │      │   vault        │
│   gpu-topology  │      │   snapshot-manager  │      │ • terradev-     │
│ • terradev-     │      │ • terradev-         │      │   authentication│
│   vram-estimator│      │   artifact-verification│     └─────────────────┘
└─────────────────┘      └──────────────────────┘
        │
┌───────▼────────┐
│  Cost & Billing│
│                 │
│ • terradev-     │
│   cost-calculator│
│ • terradev-     │
│   cost-scaler   │
│ • terradev-     │
│   price-        │
│   intelligence  │
└─────────────────┘
```

### 5. Cluster Templates (`clusters/`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLUSTER TEMPLATES                                │
│                         (Production-Ready Patterns)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌────────▼────────┐
│  MoE Models     │      │  RAG Applications    │      │  Agentic AI     │
│                 │      │                     │      │                 │
│ • moe-template/ │      │ • rag-template/     │      │ • agentic-      │
│   - GLM-5       │      │   - vLLM + Qdrant   │      │   template/     │
│   - Qwen 3.5    │      │   - Embedding       │      │   - Multi-agent  │
│   - DeepSeek V4 │      │   - Redis           │      │   - Tool calling│
│                 │      │   - Phoenix/Guardrails│     │                 │
└─────────────────┘      └──────────────────────┘      └─────────────────┘
        │                           │
┌───────▼────────┐      ┌──────────▼──────────┐
│  LLM Deployment│      │  Specialized       │
│                 │      │  Workloads         │
│ • glm-5/        │      │ • llmd-template/   │
│   - vLLM config │      │   - Language Model │
│   - K8s manifests│      │     Deployment    │
│   - Monitoring  │      │                   │
└─────────────────┘      └──────────────────────┘
```

## Data Flow Architecture

```
┌──────────────┐
│   CLI Input  │
│  (terradev)  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLI Parser (cli.py)                         │
│                    557KB, 218 MCP Tools                          │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────────────────┐
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Rust DAG Executor                               │
│              (10x faster than Python)                            │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────────────────┐
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Provider Registry & Profiles                     │
│         23+ providers with quirk-aware routing                   │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────────────────┐
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Core Orchestration Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Provisioning│  │  Training   │  │  Inference  │              │
│  │   Engine    │  │ Orchestrator│  │   Router    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────────────────┐
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Optimization & Topology Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ NUMA/       │  │ KV Cache    │  │ Cost        │              │
│  │ Topology    │  │ Offloading  │  │ Optimizer   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────────────────┐
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Provider Adapters                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ AWS/GCP/    │  │ RunPod/     │  │ Emerging    │              │
│  │ Azure       │  │ Vast.ai     │  │ Providers   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Cloud Infrastructure                            │
│              (23+ GPU Cloud Providers)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Key Integration Points

### MCP Server Integration
```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Server (terradev_mcp.py)                  │
│                    ~6700 lines, 168 tools                         │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────────────────┐
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Rust MCP Codec & Optimizer                          │
│         (Fast tool call serialization)                           │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CLI Command Handlers                           │
│              (All 218 CLI commands)                              │
└─────────────────────────────────────────────────────────────────┘
```

### Kubernetes Integration
```
┌─────────────────────────────────────────────────────────────────┐
│              K8s Enhanced Service                               │
│         (NUMA-aware, GPUDirect RDMA)                             │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────────────────┐
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Helm Generator                                      │
│         (Dynamic K8s manifests)                                 │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Cluster Templates                                   │
│         (MoE, RAG, Agentic patterns)                             │
└─────────────────────────────────────────────────────────────────┘
```

## File Statistics

| Component | Files | Lines | Purpose |
|-----------|-------|-------|---------|
| **CLI Core** | 211 | ~500K+ | Main orchestration logic |
| **Rust Accelerators** | 30+ | ~50K+ | Performance-critical components |
| **Providers** | 33 | ~400K+ | Cloud provider integrations |
| **ML Services** | 23 | ~500K+ | ML/LLM service integrations |
| **Cluster Templates** | 37 | ~100K+ | Production deployment patterns |
| **Tests** | 34 | ~50K+ | Test coverage |
| **Documentation** | 13+ | ~100K+ | User guides and API docs |

## Technology Stack

- **Python 3.8+**: Main CLI and orchestration
- **Rust**: Performance-critical accelerators (30+ crates)
- **Kubernetes**: Container orchestration
- **Helm**: Package management
- **Terraform**: Infrastructure as code
- **MCP**: Model Context Protocol (agent integration)
- **OpenTelemetry**: Observability

## Dependencies

### Python Core
- Click (CLI framework)
- PyYAML (Configuration)
- Requests (HTTP clients)
- Asyncio (Async operations)
- Kubernetes Python client

### ML Services
- vLLM, SGLang, Ollama (Inference engines)
- Ray (Distributed computing)
- MLflow, DVC (MLOps)
- LangChain, LangGraph (LLM orchestration)
- Qdrant (Vector database)
- Phoenix, Langfuse (Observability)

### Rust Crates
- Tokio (Async runtime)
- Serde (Serialization)
- Clap (CLI parsing)
- Kubernetes Rust client
- Various performance libraries

## Entry Points

1. **CLI Entry**: `terradev_cli/__main__.py` → `cli.py`
2. **MCP Server**: `terradev_mcp/terradev_mcp.py`
3. **Rust Components**: Individual crates with `lib.rs`
4. **Cluster Templates**: `clusters/*/task.yaml` or `helm/values-*.yaml`

## Testing Strategy

- **Unit Tests**: Per-module Python tests
- **Integration Tests**: Provider-specific tests
- **E2E Tests**: Full workflow tests
- **Rust Tests**: Cargo test suite
- **Battle Tests**: Comprehensive validation (tests/battle_test_*.py)

## Deployment

- **PyPI**: `pip install terradev-cli`
- **Docker**: Multi-stage builds
- **Kubernetes**: Helm charts
- **Terraform**: Infrastructure provisioning
- **Elastic Beanstalk**: AWS deployment option
