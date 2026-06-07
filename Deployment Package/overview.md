# Terradev Overview

**Plain language. No jargon. What this system actually is and how it works.**

---

## What Is Terradev?

Terradev is a command-line tool that lets you rent, manage, and use GPU compute from 21+ cloud providers through a single interface. Instead of logging into RunPod, then Vast.ai, then Lambda Labs separately — you run one command and Terradev finds the cheapest available GPU across all of them, provisions it, and sets it up correctly.

That last part — "sets it up correctly" — is the reason it exists.

Every GPU cloud hands you hardware in a suboptimal configuration by default. Your GPU and network card end up on different processor groups (NUMA nodes), which introduces a 30–50% bandwidth penalty on every distributed operation. You won't see this in `nvidia-smi`. You'll just notice your training is slower than it should be. Terradev fixes this automatically every time you provision.

---

## Who Is It For?

- **ML engineers** training large models who need to control costs across multiple clouds
- **Inference teams** serving models at scale who need sub-2-second cold starts and per-tenant adapter routing
- **Platform engineers** building GPU infrastructure with Kubernetes who need auto-scaling done properly
- **Researchers** who have local GPUs and want to mix them with cloud compute in a single pool
- **AI agents** (via MCP server) — Terradev exposes 218 tools to language models so they can autonomously manage GPU infrastructure

---

## The Core Problem It Solves

GPU infrastructure has three failure modes that all compound:

**1. Wrong topology.** Provisioning a GPU without NUMA alignment, GPUDirect RDMA, or correct CPU pinning gives you 30–50% less bandwidth on every distributed operation. This is silent — no errors, just slower training. Terradev fixes this at provision time, automatically, every time.

**2. Wrong price.** GPU prices fluctuate constantly across providers. Spot instances are often 60–80% cheaper than on-demand. A100s on RunPod are sometimes cheaper than on Vast.ai and sometimes not. Terradev quotes all configured providers before every provision and picks the cheapest option that meets your constraints.

**3. Wrong stack config.** vLLM's default settings leave 60–70% of throughput untapped. The six critical knobs (`max-num-batched-tokens`, `gpu-memory-utilization`, `enable-prefix-caching`, etc.) are all set to conservative defaults. Terradev has an auto-optimization layer for vLLM, SGLang, and distributed training that applies the right configuration for your specific workload and GPU.

---

## Key Concepts

### Providers
A provider is any GPU cloud you have credentials for. Terradev currently supports 21:

RunPod, Vast.ai, Lambda Labs, TensorDock, Crusoe, Baseten, CoreWeave, AWS, GCP, Azure, Oracle Cloud, Alibaba Cloud, OVHcloud, FluidStack, Hetzner, SiliconFlow, Hyperstack, DigitalOcean, InferX, Latitude.sh, and more.

Each provider has an async API adapter. When you run `terradev quote`, all configured providers are queried in parallel and results are returned in a unified table sorted cheapest-first.

### Credentials
All API keys are stored locally at `~/.terradev/credentials.json`. Nothing is sent to Terradev servers. The model is **BYOAPI** — you bring your own keys, Terradev routes them. Terradev has zero visibility into your cloud accounts.

### Compute Pool
A pool is everything Terradev knows about: cloud instances you've provisioned, spot markets, and local GPUs you've registered. When you provision, Terradev picks from the pool based on your constraints. The pool includes live pricing, availability, and current utilization across all configured providers.

### Topology Optimization
When Terradev provisions a node, five settings are applied automatically with no user configuration required:

- **NUMA alignment** — GPU and NIC forced onto the same processor group so GPU-to-NIC data transfer never crosses the CPU bus
- **GPUDirect RDMA** — `nvidia_peermem` loaded, enables zero-copy GPU-to-GPU transfers across the network
- **CPU pinning** — static CPU manager policy prevents process migration across cores during training
- **SR-IOV** — virtual functions created per GPU for isolated RDMA paths (important for multi-tenant)
- **NCCL tuning** — InfiniBand enabled, `GDR_LEVEL=PIX`, `GDR_READ=1` for maximum collective throughput

### Training Orchestration
Terradev's training layer wraps `torchrun`, DeepSpeed, Accelerate, and Megatron-LM. Point it at a script, specify nodes and GPUs, and it handles: inter-node SSH mesh setup, checkpoint save/restore with integrity verification, auto-recovery on failure (resumes from last checkpoint), and FlashOptim injection (a memory optimizer that reduces VRAM usage by 20–40% on bf16/fp16 workloads — auto-enabled when eligible).

### Inference Serving
The inference layer wraps vLLM and SGLang. Three optimizations are auto-applied on every `infer-deploy` with no flags required:

- **KV cache offloading** — KV cache spills to CPU DRAM when GPU VRAM is full, up to 9x throughput improvement
- **MTP speculative decoding** — draft tokens generated in parallel with verification, up to 2.8x faster generation
- **Sleep mode** — idle models hibernate to CPU RAM; wake-up is 18–200x faster than a cold restart from scratch

Multi-tenant serving uses LoRA adapters — multiple fine-tuned variants loaded simultaneously on a single base model, switched per-request based on routing headers.

### Kubernetes Integration
Terradev creates and destroys topology-optimized Kubernetes clusters with Karpenter for GPU node auto-provisioning. Every node that Karpenter spins up inherits the same NUMA-aware kubelet Topology Manager configuration as a manually provisioned instance — correct topology is enforced at the cluster level, not just per-job.

### MCP Server
Terradev runs a Rust-based MCP (Model Context Protocol) server exposing 218 tools to language models. Claude, Cursor, and Windsurf can call `terradev mcp serve` and then autonomously manage infrastructure through conversation — provision GPUs, launch training, deploy inference endpoints, manage costs. The Rust runtime processes tool calls with sub-millisecond overhead vs ~50ms for Python-based MCP servers, which compounds across complex multi-step agent workflows.

### ML Platform Integrations
Native integrations for Weights & Biases, MLflow, LangSmith, Langfuse, Arize Phoenix, Databricks, and HuggingFace Spaces. Each surfaces as a CLI command group and as MCP tools.

### RAG Stack
The `qdrant`, `phoenix`, and `guardrails` command groups form a complete production RAG pipeline:

- **Qdrant** — vector database for document retrieval (REST on 6333, gRPC on 6334)
- **Arize Phoenix** — LLM trace observability via OpenTelemetry
- **NeMo Guardrails** — output safety (topical filtering, jailbreak detection, PII masking, fact checking)

All three deploy to Kubernetes via a single `k8s` subcommand and are included in the RAG cluster template at `clusters/rag-template/`.

---

## What "3–5x Faster Provisioning" Means

Terradev provisions all nodes in parallel using a Rust DAG orchestrator. Sequential provisioning (the default in cloud UIs and most CLIs) starts node 2 after node 1 finishes. Parallel provisioning starts all nodes simultaneously. For a 4-node cluster that takes 3 minutes per node, sequential = 12 minutes, parallel = 3 minutes.

The DAG enforces correct sequencing only for operations with real dependencies (e.g., dataset staging must finish before training launches). Everything else runs simultaneously.

---

## What Terradev Does Not Do

- **Does not manage your cloud accounts.** You create accounts, add billing, generate API keys. Terradev just uses those keys.
- **Does not run a control plane.** No Terradev server your traffic routes through. Commands go from your machine directly to provider APIs.
- **Does not store your data.** Datasets, checkpoints, and model weights live wherever you put them. Terradev only knows paths.
- **Does not replace your cloud provider.** It is a management layer on top of them.

---

## Performance Reference

| Improvement | Source |
|---|---|
| 2–8x throughput | vLLM optimization (6 knobs) |
| 30–50% bandwidth penalty eliminated | NUMA topology alignment |
| Up to 90% cost savings | Automatic provider switching + spot |
| <2 minute spot recovery | KV cache checkpointing |
| 3.6x faster cold starts | Weight streaming |
| 57% VRAM cost savings | MLA-aware VRAM estimation |
| 9x throughput on MoE | KV cache offloading |
| 2.8x faster generation | MTP speculative decoding |
| 18–200x faster wake | vLLM sleep mode vs cold restart |

---

## Version Notes (v5.1.5)

- Apache 2.0 — free for commercial and personal use
- Tier/paywall system removed in v5.0.0
- Rust DAG orchestrator, local GPU scanning (NVML bindings), and MCP server introduced in v5.0.0
- FlashOptim auto-injection added in v5.1.x
- 21 providers as of v5.1.5

---

## Documentation Package

This package contains everything you need to run Terradev end-to-end. Each document serves a specific purpose:

| Document | What It Covers | When to Read It |
|---|---|---|
| `overview.md` | This file. Plain-language explanation of what Terradev is, who it's for, and the core problems it solves. | First. Read this to understand whether Terradev fits your use case. |
| `LIFECYCLES.md` | 19 complete end-to-end workflows covering nearly every command in the CLI. From first-time setup to advanced RAG pipelines. | Second. Pick the lifecycle that matches your goal and follow it step-by-step. |
| `architecture.md` | System diagrams, data flows, component detail, and file map. How the Rust MCP orchestrator, provider layer, NUMA engine, and all other pieces fit together. | When you need to understand how Terradev is built, or if you're extending it. |
| `security.md` | BYOAPI credential model, where keys are stored, network security, bare metal compliance, secrets management patterns, and a hardening checklist. | Before deploying to production, or if you have compliance requirements. |
| `troubleshooting.md` | 30+ common issues organized by command group (provisioning, training, inference, Kubernetes, ML services). Each with symptom, diagnosis command, and fix. | When something breaks. Search by command or error message. |
| `COMPLETE_COMMAND_REFERENCE.md` | Full reference of every CLI command and option as of v5.1.5. 60+ main commands, 200+ subcommands. | When you need to look up a specific flag or subcommand. |
| `BNF_GRAMMAR.md` | Formal grammar of the CLI command structure. | If you're building tooling that parses Terradev commands or generating commands programmatically. |

---

## Where to Go Next

| Goal | Document |
|---|---|
| Get started immediately | `LIFECYCLES.md` → Lifecycle 1 |
| See every command and option | `COMPLETE_COMMAND_REFERENCE.md` |
| Understand security and credentials | `security.md` |
| Understand how the system is built | `architecture.md` |
| Fix something broken | `troubleshooting.md` |
