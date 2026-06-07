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

1. **Wrong topology** — provisioning a GPU without NUMA alignment, GPUDirect RDMA, or correct CPU pinning. Terradev fixes this at provision time, automatically.

2. **Wrong price** — paying full on-demand rates when spot instances are available, or not knowing that another provider has the same GPU for 40% less right now. Terradev quotes all configured providers before provisioning.

3. **Wrong stack** — running vLLM with default settings that leave 60–70% of throughput on the table. Terradev has an auto-optimization layer for vLLM, SGLang, and distributed training frameworks that applies the right configuration for your workload.

---

## Key Concepts

### Providers
A provider is any GPU cloud you have credentials for. Terradev currently supports 21:

> RunPod, Vast.ai, Lambda Labs, TensorDock, Crusoe, Baseten, CoreWeave, AWS, GCP, Azure, Oracle Cloud, Alibaba Cloud, OVHcloud, FluidStack, Hetzner, SiliconFlow, Hyperstack, DigitalOcean, InferX, Latitude.sh

Each provider has an API adapter. When you run `terradev quote`, Terradev queries all configured providers in parallel and returns a unified price table sorted cheapest-first.

### Credentials
All API keys are stored locally at `~/.terradev/credentials.json`. Nothing is sent to Terradev servers. The model is BYOAPI — you bring your own keys, Terradev just routes them.

### Compute Pool
A pool is the collection of compute resources Terradev knows about — both cloud instances and local GPUs you've registered. The pool includes pricing, availability, and current utilization. When you provision, Terradev picks from the pool based on your constraints (GPU type, price ceiling, spot vs. on-demand, region).

### Topology Optimization
When Terradev provisions a node, it applies five settings automatically:

- **NUMA alignment** — forces GPU and NIC onto the same processor group
- **GPUDirect RDMA** — enables zero-copy GPU-to-GPU transfers across the network
- **CPU pinning** — prevents process migration across cores during training
- **SR-IOV** — creates isolated RDMA paths per GPU for multi-tenant deployments
- **NCCL tuning** — sets InfiniBand flags, GDR level, and GDR read for maximum collective throughput

You don't configure any of this. It runs at provision time.

### Training Orchestration
Terradev's training layer wraps `torchrun`, DeepSpeed, Accelerate, and Megatron. You point it at a script, tell it how many nodes and GPUs, and it handles: node-to-node SSH setup, checkpoint intervals and storage, auto-recovery on failure (resumes from last checkpoint), and FlashOptim injection (a memory optimizer from Databricks that reduces VRAM usage by 20–40% on bf16/fp16 workloads).

### Inference Serving
The inference layer wraps vLLM and SGLang. Auto-applied optimizations on every `infer-deploy`:

- **KV cache offloading** — spills KV cache to CPU DRAM, up to 9x throughput
- **MTP speculative decoding** — parallel draft/verify, up to 2.8x faster token generation
- **Sleep mode** — idle models hibernate to CPU RAM; wake-up is 18–200x faster than a cold restart

Multi-tenant serving uses LoRA adapters — multiple fine-tuned variants loaded on a single base model, switched per-request.

### Kubernetes Integration
Terradev can create and destroy topology-optimized Kubernetes clusters with Karpenter for GPU node auto-provisioning. The Karpenter NodePools are pre-configured with NUMA-aware kubelet Topology Manager settings so every node that spins up gets the same correct configuration as a manually provisioned instance.

### MCP Server
Terradev runs an MCP (Model Context Protocol) server that exposes all of its capabilities as tools to language models. Claude, Cursor, and Windsurf can call `terradev mcp serve` and then autonomously provision GPUs, launch training jobs, deploy inference endpoints, and manage infrastructure through a conversation. The MCP server is written in Rust for low-overhead tool call routing.

### ML Platform Integrations
Terradev has native integrations for: Weights & Biases, MLflow, LangSmith, Langfuse, Arize Phoenix, Databricks, and HuggingFace Spaces. These surface as CLI command groups (`terradev wandb`, `terradev langfuse`, etc.) and as MCP tools.

### RAG Stack
The `qdrant`, `phoenix`, and `guardrails` command groups form a complete RAG pipeline stack:

- **Qdrant** — vector database for document retrieval
- **Phoenix** — LLM trace observability (OpenTelemetry-based)
- **NeMo Guardrails** — output safety layer (topical filtering, jailbreak detection, PII masking, fact checking)

All three deploy onto Kubernetes via the same `k8s` subcommand pattern.

---

## What "3–5x Faster" Means

The headline refers to provisioning speed, not inference speed. Terradev provisions all nodes in parallel using a Rust DAG orchestrator. Sequential provisioning (the default when you use cloud UIs or most CLIs) starts node 2 after node 1 is ready. Parallel provisioning starts all nodes simultaneously. For a 4-node cluster, that's roughly 4x the wall-clock time saved.

The DAG enforces correct sequencing for operations that have actual dependencies (e.g., dataset staging must complete before training launches) while parallelizing everything else.

---

## What It Does Not Do

- **It does not manage your cloud accounts.** You create accounts, add billing, and generate API keys yourself. Terradev just uses those keys.
- **It does not run a control plane.** There is no Terradev server your traffic routes through. Commands go directly from your machine to provider APIs.
- **It does not store your data.** Datasets, checkpoints, and model weights live wherever you put them (S3, GCS, local disk). Terradev only knows paths.
- **It does not replace your cloud provider.** It's a management layer on top of them, not an alternative to them.

---

## Version Notes (v5.1.5)

- Apache 2.0 license — free for commercial and personal use
- Paywall removed in v5.0.0; tier system deprecated
- Rust acceleration introduced in v5.0.0 for DAG orchestration, local GPU scanning, and MCP server
- FlashOptim auto-injection added in v5.1.x
- 19 providers at v4.x; expanded to 21 in v5.x

---

## File Layout

```
terradev_cli/
  cli.py                    Main CLI entry point (~11,600 lines)
  cli_karpenter.py          Karpenter subcommands
  cli_hf_spaces.py          HuggingFace Spaces subcommands
  core/
    training_orchestrator.py    DeepSpeed/torchrun/FlashOptim
    evaluation_orchestrator.py  Model/endpoint evaluation
    event_system.py             Triggers, environments, lineage
    cost_tracker.py             Spend tracking
    price_intelligence.py       Spot market analytics
  ml_services/
    vllm_service.py         vLLM config + LoRA + sleep mode
    phoenix_service.py      Arize Phoenix tracing
    guardrails_service.py   NeMo Guardrails
    qdrant_service.py       Vector database
  providers/                One file per cloud provider
clusters/
  moe-template/             Mixture-of-Experts K8s templates
  rag-template/             RAG stack K8s templates
  glm-5/                    GLM-5 deployment templates
terradev-mcp/
  terradev_mcp.py           MCP server (~6,700 lines, 168 tools)
```

---

## Where to Go Next

| I want to... | Go to... |
|---|---|
| Get started fast | `LIFECYCLES.md` → Lifecycle 1 |
| Understand all commands | `COMPLETE_COMMAND_REFERENCE.md` |
| Deploy and trust it in production | `docs/security.md` |
| Understand how it's built | `docs/architecture.md` |
| Fix something broken | `docs/troubleshooting.md` |
