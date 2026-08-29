# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [6.2.4] - 2026-08-28

### Removed commands

- Removed the `integrations`, `cost-scaler`, `schedule`, and `observe` CLI command groups and their MCP tool schemas, handlers, and registry entries.
- Deleted `core/observe.py`, `core/schedule.py`, and the corresponding command and functional tests.
- Kept `core/cost_scaler.py` as it is still used by the model orchestrator.

## [6.2.3] - 2026-08-28

### 🚀 Provider consolidation — remove OVHcloud

- Version bumped to **6.2.3**.
- Removed **OVHcloud** from the active provider set and deleted the provider implementation, drift contract, and MCP/tooling references.
- Consolidated all CLI provider lists, provider factory, credential prompts, and MCP schemas to the **17 active GPU cloud and inference providers**.

### Full active provider list (17)

- `aws` — Amazon Web Services
- `azure` — Microsoft Azure
- `baseten` — Baseten
- `crusoe` — Crusoe Cloud
- `digitalocean` — DigitalOcean
- `e2enetworks` — E2E Networks
- `gcore` — Gcore
- `gcp` — Google Cloud Platform
- `huggingface` — Hugging Face
- `hyperstack` — Hyperstack
- `inferx` — InferX
- `latitude` — Latitude.sh
- `runpod` — RunPod
- `siliconflow` — SiliconFlow
- `tensordock` — TensorDock
- `vastai` — Vast.ai
- `yottalabs` — YottaLabs


## [6.2.2] - 2026-08-28

### 🚀 Provider consolidation and Gcore integration

- Version bumped to **6.2.2**.
- Added **Gcore** (`gcore`) as a new GPU cloud provider with full VM lifecycle support, API token authentication, project/region discovery, and drift monitoring.
- Removed **Jio Cloud** and **Oracle Cloud** from the active provider set; removed **Alibaba Cloud** references from the CLI UI.
- Consolidated all CLI provider lists (onboarding, `configure`, `setup`, `quote`, `credential_prompt`) to the 18 active providers.
- Updated MCP schemas and handlers to include `gcore` in provider enums and defaults.
- Full active provider list (18):
  - `aws` — Amazon Web Services
  - `azure` — Microsoft Azure
  - `baseten` — Baseten
  - `crusoe` — Crusoe Cloud
  - `digitalocean` — DigitalOcean
  - `e2enetworks` — E2E Networks
  - `gcore` — Gcore
  - `gcp` — Google Cloud Platform
  - `huggingface` — Hugging Face
  - `hyperstack` — Hyperstack
  - `inferx` — InferX
  - `latitude` — Latitude.sh
  - `ovhcloud` — OVHcloud
  - `runpod` — RunPod
  - `siliconflow` — SiliconFlow
  - `tensordock` — TensorDock
  - `vastai` — Vast.ai
  - `yottalabs` — YottaLabs


### 🚀 Multi-stage LLM training pipeline (`terradev train`)
- Added `terradev train sft`, `terradev train dpo`, `terradev train grpo`, and `terradev train pipeline` commands.
- Added `terradev_cli/core/training_stages.py` with declarative `StageConfig` / `PipelineConfig` and command builders for `unsloth`, `trl`, `openrlhf`, `axolotl`, `llama-factory`, and `ms-swift`.
- Added `terradev_cli/core/training_pipeline.py` with DAG-based orchestration using the existing Rust/Py `DAGExecutor`, provider-aware quote selection, auto-provisioning via `ParallelProvisioner`, checkpoint handoff, and optional teardown.
- Added `examples/training_pipeline.yaml` sample pipeline covering SFT → DPO → GRPO with provider-agnostic resource selection.
- Wrapped CLI-style training frameworks (`trl`, `openrlhf`, `axolotl`, `llama-factory`, `ms-swift`) in throw-away Python scripts so the existing `TrainingOrchestrator` can launch them with `torchrun`/`deepspeed`/`accelerate`.

## [6.1.0] - 2026-08-23

### 🚀 Drift monitor, no-auth REST providers, and CI scheduling
- Added `auth_required: false` support to provider drift contracts so public REST endpoints can be monitored without an API key.
- Scheduled the `provider_drift.yml` GitHub Actions workflow to run at 13:00 UTC (morning EDT/PDT).
- Added `REQUIRES_AUTH` flag to `BaseProvider` and `ProviderFactory` for provider auth-intent discovery.
- Fixed `_get_provider_quotes` so it no longer pre-emptively skips auth-required providers; each provider's `get_instance_quotes` handles missing credentials and returns cleanly (preserving the original resilience tests and avoiding a forced short-circuit).
- `REQUIRES_AUTH` remains available for drift/contract logic while public REST providers (HuggingFace, Oracle, Crusoe, Baseten) continue to return static quotes without credentials.
- Bumped version to **6.1.0**.

---

## [6.0.8] - 2026-08-08

### 🐛 **CI hardening for agent subcommands**
- Added Landlock ABI-version detection and masked access bits to only those supported by the running kernel, fixing the `landlock_add_rule(/)` failure on GitHub Actions runners.
- Landlock runtime availability now performs a real smoke test (ruleset creation + path rule) instead of only checking for the syscall, preventing auto-selection on broken or incompatible kernels.
- Fixed `test_libp2p_transport_lifecycle` to skip when the optional `p2pclient`/`p2pd` dependency is not installed.
- Bumped version to **6.0.8**.

---

## [6.0.7] - 2026-08-08

### 🚀 **Agent subcommands are now real, downloadable, and no longer placeholders**
- New `terradev agent sandbox` — hardware-isolated execution with real `bwrap`, `runsc`, `firecracker`, and Linux Landlock LSM runtimes, all discoverable/downloadable through `DependencyManager`.
- New `terradev agent mesh` — peer-to-peer agent mesh with real `libp2p` (`go-libp2p-daemon` + `p2pclient`), A2A HTTP, and `WireGuard` encrypted transports.
- New `terradev agent mcp` — dynamic Model Context Protocol bridge with stdio and HTTP transports and a multi-server bridge.
- Added `terradev_cli/commands/agent_infra/dependency_manager.py` to locate or download `p2pd`, `runsc`, `firecracker`, `bwrap`, `wg`, and `wireguard-go` binaries.
- All agent subcommands integrate with OpenTelemetry tracing via the existing TerradevTelemetry/NodeSpanStream pipeline.

---

## [5.7.3] - 2026-08-01

### 🔧 **Fixes & Polish**
- Expanded Python 3.9 support with a compatibility shim for the MCP SDK.
- Added `EnterpriseAuthManager` and `SAMLProvider` scaffolding for enterprise SSO.
- Hardened the test suite: provider conformance, Kubernetes CPU parsing, and major-provider error paths are now fully exercised.
- Lowered `requires-python` to `>=3.9` and gated the `mcp` dependency on Python `>=3.10`.
- Updated READMEs, command reference, and lifecycle docs to v5.7.2.

---

## [5.1.5] - 2026-06-06

### 🔧 **Fixes & Polish**
- UI audit: corrected all version references across docs, landing page, and READMEs
- `docs/index.html`: v3.4.0 → v5.1.5, BUSL 1.1 → Apache 2.0, 15 → 21+ providers, modernized feature tiles with stat row
- `docs/USER_GUIDE.md`: full rewrite — replaced dead SaaS web app guide with current CLI reference
- `docs/architecture.md`: full rewrite — replaced pre-v5 architecture with Rust MCP + CLI layer diagram
- `docs/sitemap.xml`: removed version-pinned PyPI URL, updated all lastmod dates
- `terradev_cli/README.md`: version sync to v5.1.5, 104 → 218 tools
- `demo/generate_gif.py`: "15 clouds" → "21+" in closing tagline

---

## [5.1.2] - 2026-05-20

### 🔧 **Fixes**
- K8s bughunt: 14 bugs fixed across `kubernetes_service.py`, `kubernetes_enhanced.py`, `cli.py`
- Critical: added 4 missing methods (`install_gpu_operator`, `configure_device_plugin`, `configure_mig`, `configure_time_slicing`)
- Critical: `EnhancedKubernetesService()` default config fix
- API version corrections: `karpenter.sh/v1beta1` → `karpenter.sh/v1`, `karpenter.k8s.aws/v1beta1` → `karpenter.k8s.aws/v1`
- CPU parsing bug: `.endswith('m')` after strip fixed to check before strip

---

## [5.1.0] - 2026-05-05

### 🚀 **RAG Stack Integration**
- **Qdrant** (Apache 2.0): vector DB client, collection management, similarity search, K8s deployment
- **Arize Phoenix** (ELv2): LLM trace observability, OTLP ingestion, span tree rendering, OTel env setup
- **NeMo Guardrails** (Apache 2.0): Colang 2.x config generation, topical/jailbreak/PII/factcheck rails, sidecar mode
- **RAG template** (`clusters/rag-template/`): vLLM + Qdrant + Embedding + Redis + optional Phoenix/Guardrails
- CLI: `terradev qdrant`, `terradev phoenix`, `terradev guardrails` command groups

---

## [5.0.5] - 2026-04-22

### 🚀 **FlashOptim Auto-Injection**
- FlashOptim (Databricks, arXiv:2602.23349) silently auto-applied to training jobs
- Auto-enabled when `bf16`/`fp16` detected in `script_args` + total VRAM ≥ 40GB
- OFF automatically for Megatron, no GPUs, or GPUs < 24GB
- Injects `FLASHOPTIM_*` env vars, pre-installs package on all nodes (fail-safe)
- `--flashoptim auto|on|off` flags for override
- `terradev preflight --flashoptim-check` for compatibility verification
- `terradev train` output shows FlashOptim status when auto-applied

---

## [5.0.3] - 2026-04-10

### 🚀 **5 New Cloud Providers**
- **Alibaba Cloud**: HMAC-SHA1 signed URLs, RFC 3986 percent-encoding
- **OVHcloud**: `X-Ovh-Signature` HMAC, raw aiohttp to avoid header clobbering
- **Hetzner**: Cloud + Robot API dual support
- **SiliconFlow**: Bearer auth, LLM inference endpoints
- Provider count: 16 → 21+

---

## [5.0.0] - 2026-03-20

### 🎉 **Open Source Release**
- **Removed paywall** — fully open source under Apache 2.0
- **Rust MCP Orchestrator**: DAG sequencing, idempotency guarantees, sub-ms tool routing
- **MCP Server expanded to 218 tools** (55 new in v5.0.0, further expanded in v5.1.x+):
  - HuggingFace Hub (8): list models, datasets, create/manage endpoints, inference
  - HF Smart Templates (3): hardware recommendation, comparison
  - LangChain/LangGraph/LangSmith (9): workflow creation, orchestrator-worker, evaluation
  - W&B Enhanced (7): dashboards, reports, alerts with Terradev-specific templates
  - Cost Optimizer Deep (4): analyze, recommend, simulate, budget-optimize
  - Data Governance (6): consent, OPA evaluation, data movement, compliance reports
  - K8s Enhanced (5): GPU operator, device plugin, MIG, time-slicing, monitoring stack
  - Training extras (4): config generate, distributed launch, snapshot, straggler detection
  - Price extras (3): trends, budget-optimize, spot-risk
  - Preflight extras (3): report, GPU check, network check
- **vLLM 5 auto-optimizations** for MoE templates:
  - KV cache offloading (`--kv-connector=offloading`) — up to 9x throughput
  - MTP speculative decoding (`--speculative-config.method=mtp`) — up to 2.8x speed
  - Sleep mode (`--enable-sleep-mode`) — 18–200x faster than cold restart
  - Multi-LoRA (`terradev lora add/list/remove`)
  - vLLM Router opt-in for P/D disaggregation
- **Local GPU discovery**: NVML Rust bindings, hybrid local+cloud compute pools
- **Disaggregated prefill/decode**: NIXL zero-copy RDMA, sticky routing
- License changed from BUSL 1.1 → **Apache 2.0**

---

## [4.0.12] - 2026-03-12

### 🚀 **New Latitude.sh Provider Integration**
- **Dual Instance Support**: Full support for both bare metal servers AND virtual machines with GPU
- **Premium GPU Access**: NVIDIA H100, A100, RTX 4090, RTX PRO 6000 Blackwell support
- **Bare Metal Performance**: Dedicated hardware with IPMI out-of-band management
- **VM Flexibility**: GPU-enabled virtual machines with dedicated GPU resources
- **JSON:API Compliant**: Complete API specification compliance with built-in rate limiting
- **SSH Access Patterns**: Direct SSH for bare metal, container SSH for VMs
- **Real-time Pricing**: Live quotes with stock levels and instant deployment information

### 🔧 **Provider Enhancements**
- **Provider Count**: Updated to 20 supported cloud providers
- **Instance Differentiation**: Clear categorization between bare metal and virtual machine instances
- **Rate Limiting**: Automatic exponential backoff with 429 response handling
- **Error Handling**: Graceful degradation with detailed error responses
- **Test Coverage**: 13 comprehensive tests with 100% pass rate

### 📊 **Configuration & Usage**
- **Easy Setup**: Simple `LATITUDE_API_KEY` environment variable configuration
- **CLI Integration**: Full integration with existing terradev CLI commands
- **Instance Types**: Support for both `latitude-bare-metal-*` and `latitude-vm-*` instance types
- **Regional Support**: Multi-region availability with real-time stock information

### 🎯 **Technical Improvements**
- **Async Support**: Full async/await implementation throughout provider
- **Type Safety**: Complete type annotations for better IDE support
- **Documentation**: Comprehensive integration guide and API reference
- **Factory Registration**: Automatic provider registration in provider factory

---

## [2.9.8] - 2026-02-20

### 🎯 **Documentation Corrections**
- **Fixed Tier Pricing**: Corrected GPU limits to 1/8/32 for Research/Research+/Enterprise tiers
- **Removed Emojis**: Cleaned up all emoji characters from README for professional appearance
- **Removed Project Structure**: Eliminated detailed project structure section from documentation
- **Updated Version**: Bumped to v2.9.8 with corrected information

### 📚 **Documentation Accuracy**
- **Tier Limits**: Fixed concurrent instance limits (1/8/32 GPUs)
- **Professional Formatting**: Removed all emoji characters throughout README
- **Streamlined Content**: Removed unnecessary project structure details
- **Corrected Information**: Ensured all pricing and tier information is accurate

### 🔧 **Technical Updates**
- **Version Sync**: Updated README version to match package version
- **Clean Documentation**: Professional-grade README without emojis
- **Accurate Pricing**: Corrected tier pricing table with proper GPU limits

---

## [2.9.7] - 2026-02-20

### 🎯 **Major Documentation Update**
- **Complete README Overhaul**: Comprehensive documentation with GitOps automation
- **HuggingFace Spaces Integration**: One-click model deployment documentation
- **BYOAPI Security Model**: Detailed authentication and security explanations
- **Enhanced Quick Start**: 17-step comprehensive getting started guide
- **Project Structure**: Complete architecture documentation
- **Integration Guides**: Jupyter, GitHub Actions, Docker workflows

### 📚 **Documentation Features**
- **GitOps Workflows**: Production-ready ArgoCD/Flux integration
- **HF Spaces Templates**: LLM, embedding, and image model deployment
- **CLI Command Reference**: Complete command documentation with examples
- **Pricing Tiers**: Clear feature comparison across tiers
- **Security Architecture**: Zero-trust credential management
- **Integration Matrix**: W&B setup guides

### 🚀 **Enhanced User Experience**
- **One-Command Deployments**: Simplified HF Spaces deployment
- **Template-Based Workflows**: Pre-configured model templates
- **Multi-Environment Support**: Dev, staging, production workflows
- **Policy as Code**: Gatekeeper/Kyverno integration
- **Manifest Cache**: Versioned deployment management

### 📖 **Documentation Structure**
- **Why Terradev**: Clear value proposition and use cases
- **Installation Guide**: Multiple installation options
- **Quick Start**: Comprehensive 17-step tutorial
- **CLI Commands**: Complete command reference
- **Integrations**: Detailed third-party service setup
- **Project Architecture**: Complete codebase structure

### 🔗 **External Links**
- **GitHub Repository**: Updated repository links
- **License Details**: Comprehensive license information
- **Integration Examples**: Real-world usage patterns
- **Security Documentation**: BYOAPI security model

---

## [2.9.6] - 2026-02-20

### 🚀 **Major Features**
- **InferX Serverless Integration**: Complete serverless AI inference platform
  - <2s cold starts with snapshot technology
  - 90% GPU utilization optimization
  - 30+ models per GPU capacity
  - OpenAI-compatible API support

### 🎯 **InferX Provider Features**
- Serverless deployment with pay-per-request pricing
- GPU slicing and multi-tenant isolation
- Snapshot technology for instant model loading
- AI-powered cost optimization with 70% savings potential
- Comprehensive Kubernetes platform deployment

### ⚡ **Performance Optimizations**
- SPDK blobstore for high-performance snapshot storage
- GPU-aware scheduling and resource pooling
- KEDA-based auto-scaling for model functions
- Custom resource definitions for model management

### 💰 **Cost Optimization**
- Spot-first GPU instance strategy (70% cost reduction)
- Resource pooling and sharing capabilities
- Tiered storage classes for different use cases
- AI-powered cost analysis and recommendations

### 🐳 **Kubernetes Platform**
- Complete InferX platform deployment automation
- GPU node pools with Karpenter integration
- Multi-tier storage classes (blobstore, cache, database)
- Network policies and security isolation
- Monitoring dashboards and metrics

### 🔧 **CLI Commands**
- `terradev inferx configure` - Provider setup
- `terradev inferx deploy` - Model deployment
- `terradev inferx status` - Deployment monitoring
- `terradev inferx list` - Model inventory
- `terradev inferx usage` - Usage statistics
- `terradev inferx quote` - Pricing information
- `terradev inferx optimize` - Cost optimization

### 📊 **Monitoring & Observability**
- Health checks and readiness probes
- Resource utilization tracking

### 🛡️ **Security & Isolation**
- Multi-tenant namespace isolation
- RBAC permissions for model controller
- Network policies for traffic control
- Pod security contexts and capabilities

### 📦 **Package Updates**
- Added InferX provider dependencies
- Updated Kubernetes client libraries
- Enhanced async/await support throughout
- Improved error handling and logging

### 🐛 **Bug Fixes**
- Fixed GPU resource allocation issues
- Resolved snapshot storage permissions
- Improved error messages for deployment failures
- Enhanced timeout handling for long-running operations

### 📚 **Documentation**
- Complete InferX integration guide
- Kubernetes deployment instructions
- Cost optimization best practices
- API reference documentation

---

## [2.9.5] - Previous Release

### 🔄 **Previous Features**
- Multi-cloud GPU provisioning
- GitOps automation
- HuggingFace Spaces deployment
- Cost optimization analytics
- Provider integrations (AWS, GCP, Azure, RunPod, VastAI, etc.)

---

## [Unreleased]

## [6.2.2] - 2026-08-28

### 🚀 Provider consolidation and Gcore integration

- Version bumped to **6.2.2**.
- Added **Gcore** (`gcore`) as a new GPU cloud provider with full VM lifecycle support, API token authentication, project/region discovery, and drift monitoring.
- Removed **Jio Cloud** and **Oracle Cloud** from the active provider set; removed **Alibaba Cloud** references from the CLI UI.
- Consolidated all CLI provider lists (onboarding, `configure`, `setup`, `quote`, `credential_prompt`) to the 18 active providers.
- Updated MCP schemas and handlers to include `gcore` in provider enums and defaults.
- Full active provider list (18):
  - `aws` — Amazon Web Services
  - `azure` — Microsoft Azure
  - `baseten` — Baseten
  - `crusoe` — Crusoe Cloud
  - `digitalocean` — DigitalOcean
  - `e2enetworks` — E2E Networks
  - `gcore` — Gcore
  - `gcp` — Google Cloud Platform
  - `huggingface` — Hugging Face
  - `hyperstack` — Hyperstack
  - `inferx` — InferX
  - `latitude` — Latitude.sh
  - `ovhcloud` — OVHcloud
  - `runpod` — RunPod
  - `siliconflow` — SiliconFlow
  - `tensordock` — TensorDock
  - `vastai` — Vast.ai
  - `yottalabs` — YottaLabs


### 🚀 **Upcoming Features**
- Additional provider integrations
- Enhanced monitoring capabilities
- Advanced cost optimization algorithms
- Multi-region deployment support
