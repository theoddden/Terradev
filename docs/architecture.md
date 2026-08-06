# Terradev Architecture — v5.1.5

## Overview

Terradev is an imperative CLI for AI workload orchestration. It provisions topology-optimized GPU instances across 21+ cloud providers, launches distributed training jobs with automatic FlashOptim injection, and deploys inference stacks (vLLM, MoE templates, disaggregated prefill/decode, RAG) — all orchestrated via a Rust-based MCP server with 218 tools for Claude Code and other AI agents.

## System Layers

```
┌─────────────────────────────────────────────────────┐
│  Claude Code / AI Agent (MCP client)                │
└──────────────────────┬──────────────────────────────┘
                       │ MCP protocol (JSON-RPC 2.0)
┌──────────────────────▼──────────────────────────────┐
│  Rust MCP Orchestrator  (terradev-mcp)               │
│  218 tools · DAG sequencing · idempotency guarantees │
└──────────────────────┬──────────────────────────────┘
                       │ Python interop / subprocess
┌──────────────────────▼──────────────────────────────┐
│  terradev-cli  (Python)                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │Providers │ │ Training │ │Inference │ │  K8s   │ │
│  │  21+     │ │Orchestr. │ │ vLLM/MoE │ │Cluster │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  Qdrant  │ │ Phoenix  │ │Guardrails│ │  LoRA  │ │
│  │  (RAG)   │ │ (Traces) │ │ (Safety) │ │Adapters│ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Rust MCP Orchestrator (`terradev-mcp`)

- **Protocol**: JSON-RPC 2.0 over stdio (MCP spec)
- **Tools**: 168 Tool() definitions with full JSON Schema validation
- **DAG execution**: Enforces correct sequencing and idempotency — agents can issue commands freely, orchestrator ensures safe execution
- **Performance**: Sub-millisecond overhead per tool call vs ~50ms for Python MCP servers

### 2. Provider Layer (21+ clouds)

Each provider implements the `BaseProvider` async interface: `list_instances()`, `provision()`, `terminate()`, `get_status()`. Providers with non-standard auth (Alibaba HMAC, OVHcloud HMAC) bypass `_make_request()` and use raw aiohttp to avoid auth header clobbering.

**Providers**: RunPod, Vast.ai, Lambda Labs, CoreWeave, AWS, GCP, Azure, Hyperstack, TensorDock, Latitude.sh (bare metal + VM), FluidStack, Alibaba Cloud, OVHcloud, Hetzner, SiliconFlow, Paperspace, Crusoe, Lepton, Voltage Park, Genesis Cloud, Nebius.

### 3. NUMA Topology Engine

Auto-applied on every `terradev provision` — no user config required:
- **NUMA alignment** — GPU and NIC forced to same NUMA node
- **GPUDirect RDMA** — `nvidia_peermem`, zero-copy GPU↔GPU transfers
- **CPU pinning** — static CPU manager policy, no core migration
- **SR-IOV** — virtual functions per GPU for isolated RDMA paths
- **NCCL tuning** — InfiniBand enabled, `GDR_LEVEL=PIX`, `GDR_READ=1`

### 4. Training Orchestrator (`core/training_orchestrator.py`)

- **Frameworks**: torchrun, DeepSpeed, Accelerate, Megatron-LM
- **FlashOptim** (Databricks): auto-injected when `bf16`/`fp16` detected and total VRAM ≥ 40GB. Injects `FLASHOPTIM_*` env vars, pre-installs package on all nodes
- **Checkpoint management**: save/restore/verify/repair with integrity checksums
- **Dataset staging**: parallel compression + multi-region S3 pre-placement
- **Straggler detection**: identifies slow nodes in distributed runs

### 5. vLLM / Inference Services (`ml_services/vllm_service.py`)

**Auto-applied for MoE templates** (zero config):
- KV cache offloading (`--kv-connector=offloading`) — up to 9x throughput
- MTP speculative decoding (`--speculative-config.method=mtp`) — up to 2.8x speed
- Sleep mode (`--enable-sleep-mode`) — 18–200x faster than cold restart

**User opt-in**: Multi-LoRA (`terradev lora add/list/remove`), vLLM Router for P/D disaggregation.

**Disaggregated Prefill/Decode**: prefill pool (compute-bound) + decode pool (memory-bound) connected via NIXL zero-copy RDMA with sticky routing for KV cache locality.

### 6. RAG Stack

| Service | File | Port | Auth |
|---------|------|------|------|
| Qdrant (vector DB) | `ml_services/qdrant_service.py` | 6333/6334 | `api-key` header |
| Arize Phoenix (traces) | `ml_services/phoenix_service.py` | 6006 | Bearer (cloud) / none (self-hosted) |
| NeMo Guardrails (safety) | `ml_services/guardrails_service.py` | 8000 | config_id |

RAG template (`clusters/rag-template/`) deploys all three + embedding model + Redis in one command.

### 7. Kubernetes Layer

`terradev_cli/kubernetes/` + `kubernetes_enhanced.py`

- **Karpenter NodePools**: NUMA-aligned kubelet Topology Manager, GPUDirect RDMA, PCIe locality enforcement
- **GPU Operator**: NVIDIA device plugin, MIG configuration, time-slicing
- **Monitoring stack**: DCGM exporter
- **API versions**: `karpenter.sh/v1`, `karpenter.k8s.aws/v1`

---

## Data Flows

### Training
```
terradev train
  → preflight (GPU / NCCL / RDMA validation)
  → stage (dataset compress + pre-place to S3)
  → _flashoptim_auto_config()   (inject if eligible)
  → _launch_native()            (torchrun / deepspeed)
  → monitor (live GPU util + cost)
  → checkpoint (periodic save)
```

### Inference (MoE)
```
terradev provision --task clusters/moe-template/task.yaml
  → NUMA topology applied
  → vLLM deployed with auto-flags:
      --kv-connector=offloading
      --speculative-config.method=mtp
      --enable-sleep-mode
  → LMCache (Redis KV sharing across replicas)
  → (opt-in) vLLM Router for P/D disaggregation
```

### MCP Tool Call
```
Agent: tool_call("provision_gpu", {...})
  → Rust MCP router
  → command_map["provision_gpu"]
  → Python CLI handler (async)
  → BaseProvider.provision() + NUMA topology
  → result JSON streamed to agent
```

---

## Credential Security

All credentials stored at `~/.terradev/credentials.json` — **never transmitted to Terradev servers**. SSH keys are auto-generated per-provision, encrypted at rest. BYOAPI means Terradev has zero visibility into your cloud accounts.

---

## Repository Structure

```
Terradev/
├── terradev_cli/
│   ├── cli.py                      — Main CLI entry (Click)
│   ├── providers/                  — 21+ cloud provider implementations
│   ├── core/
│   │   ├── training_orchestrator.py
│   │   ├── provision_orchestrator.py
│   │   └── trace_viewer.py
│   ├── ml_services/
│   │   ├── vllm_service.py
│   │   ├── qdrant_service.py
│   │   ├── phoenix_service.py
│   │   └── guardrails_service.py
│   └── kubernetes/
│       ├── kubernetes_service.py
│       └── kubernetes_enhanced.py
├── clusters/
│   ├── moe-template/               — MoE deployment (all opts auto-applied)
│   ├── rag-template/               — Qdrant + Phoenix + Guardrails
│   └── glm-5/
├── rust/                           — Rust accelerators + NVML bindings
├── terradev-mcp/
│   └── terradev_mcp.py             — 218-tool MCP server
└── docs/
    ├── USER_GUIDE.md
    └── architecture.md
```

