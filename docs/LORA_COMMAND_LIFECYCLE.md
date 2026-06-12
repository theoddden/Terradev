# Command Lifecycle Spotlight: `terradev lora`

## Production-Grade Multi-Tenant LoRA Serving in One CLI

Serving multiple fine-tuned models on a single GPU is the holy grail of cost-efficient inference. With LoRA (Low-Rank Adaptation), you can serve N adapters on one base model—but managing them across replicas, versions, and tenants is a nightmare.

**Terradev v5.3.8** ships a complete production LoRA lifecycle: registry, versioning, rollback, drift detection, cost attribution, and adapter-aware routing—all from a single CLI.

---

## The Problem: LoRA at Scale

When you're serving 50+ customer adapters across 10 replicas, you hit operational chaos:

- **Which replica has which adapter loaded?** (manual spreadsheets)
- **How do I rollback a bad adapter version?** (copy-paste disasters)
- **Which tenant is driving GPU costs?** (no visibility)
- **Is adapter performance degrading?** (manual spot checks)
- **How do I sync loads across replicas?** (ssh loops and shell scripts)

Terradev's `lora` command group solves all of this with a declarative, production-ready workflow.

---

## The Lifecycle: From Register to Rollback

### 1. Register an Adapter Version

```bash
terradev lora register \
  --name customer-a-financial \
  --path /adapters/customer-a/v3 \
  --base-model meta-llama/Llama-2-7b-hf \
  --rank 64 \
  --tenant customer-a \
  --metadata '{"domain": "financial", "training_date": "2026-06-01"}'
```

**What happens:**
- Adapter metadata stored in SQLite registry (`~/.terradev/lora_registry.db`)
- Version ID auto-generated (e.g., `customer-a-financial-v3-20260601`)
- Baseline performance score captured from Phoenix traces
- Storage cost recorded in cost attribution service

**Output:**
```
✓ Registered adapter: customer-a-financial
  Version ID: customer-a-financial-v3-20260601
  Base model: meta-llama/Llama-2-7b-hf
  Rank: 64
  Tenant: customer-a
  Baseline score: 0.92
```

---

### 2. List Versions

```bash
terradev lora versions --name customer-a-financial
```

**What happens:**
- Queries registry for all versions of this adapter
- Shows version IDs, creation timestamps, baseline scores, and active status

**Output:**
```
Adapter: customer-a-financial
Versions:
  v1-20260515  [INACTIVE]  baseline: 0.89  created: 2026-05-15
  v2-20260528  [INACTIVE]  baseline: 0.91  created: 2026-05-28
  v3-20260601  [ACTIVE]    baseline: 0.92  created: 2026-06-01
```

---

### 3. Activate a Version

```bash
terradev lora activate \
  --name customer-a-financial \
  --version v3-20260601
```

**What happens:**
- Marks version as active in registry
- Triggers cross-replica sync (see step 4)
- Updates warm pool priorities

**Output:**
```
✓ Activated version: v3-20260601
  Syncing to 3 replicas...
    ✓ replica-1 (10.0.0.1:8000)
    ✓ replica-2 (10.0.0.2:8000)
    ✓ replica-3 (10.0.0.3:8000)
  Warm pool updated: priority=high
```

---

### 4. Sync Across Replicas

```bash
terradev lora sync \
  --directory /adapter-configs \
  --name customer-a-financial \
  --replicas 10.0.0.1:8000,10.0.0.2:8000,10.0.0.3:8000
```

**What happens:**
- Uses gossip protocol to discover all replicas in the cluster
- Broadcasts load command to each replica
- Tracks replica state in registry (`AdapterReplicaState`)
- Retries failed loads with exponential backoff

**Output:**
```
Syncing adapter: customer-a-financial
Discovered 3 replicas via Kubernetes
  Loading on replica-1... ✓
  Loading on replica-2... ✓
  Loading on replica-3... ✓
All replicas synced
```

---

### 5. Check for Drift

```bash
terradev lora drift-check \
  --name customer-a-financial \
  --version v3-20260601 \
  --threshold 0.05
```

**What happens:**
- Queries Phoenix traces for recent inference spans
- Computes current quality score vs baseline
- Triggers drift alert if degradation exceeds threshold
- Recommends action (monitor, retrain, rollback)

**Output:**
```
Drift check: customer-a-financial (v3-20260601)
  Baseline score: 0.92
  Current score: 0.87
  Drift magnitude: 0.054
  Status: ⚠️ DRIFT DETECTED
  Recommended action: retrain
  Samples analyzed: 1,247
```

---

### 6. Rollback to Previous Version

```bash
terradev lora rollback \
  --name customer-a-financial \
  --version v2-20260528 \
  --replicas 10.0.0.1:8000,10.0.0.2:8000,10.0.0.3:8000
```

**What happens:**
- Unloads current version from all replicas
- Loads previous version
- Updates registry with rollback event
- Records rollback in audit log

**Output:**
```
Rolling back: customer-a-financial
  From: v3-20260601
  To: v2-20260528
  Unloading from replicas...
    ✓ replica-1
    ✓ replica-2
    ✓ replica-3
  Loading v2-20260528...
    ✓ replica-1
    ✓ replica-2
    ✓ replica-3
  Rollback complete
  Audit log: ~/.terradev/lora_rollback_20260611.json
```

---

### 7. Cost Attribution Report

```bash
terradev lora cost-report \
  --days 30 \
  --adapter customer-a-financial
```

**What happens:**
- Aggregates GPU hours, tokens, and requests per adapter
- Calculates cost using instance-type pricing
- Shows cost breakdown by replica
- Provides warm pool recommendations

**Output:**
```
Cost report: customer-a-financial (last 30 days)
  Total GPU hours: 142.5
  Total tokens: 8.2M
  Total requests: 12,450
  Total cost: $213.75

  Cost by replica:
    replica-1: $71.25 (142.5 GPU hours)
    replica-2: $71.25 (142.5 GPU hours)
    replica-3: $71.25 (142.5 GPU hours)

  Warm pool recommendations:
    ✓ Keep warm (high cost adapter: $213.75/month)
```

---

## Under the Hood: What Makes This Production-Grade

### 1. **Adapter Registry (SQLite)**
- `AdapterVersion`: Version metadata, baseline scores, timestamps
- `AdapterReplicaState`: Which replica has which version loaded
- `AdapterRegistry`: Centralized SQLite database with ACID guarantees

### 2. **Cross-Replica Consistency**
- Gossip protocol for replica discovery
- Broadcast load/unload commands with retries
- Kubernetes pod discovery via label selectors
- Static replica registration for air-gapped environments

### 3. **Versioning & Rollback**
- Semantic versioning with auto-generated IDs
- One-click rollback to any previous version
- Audit trail of all version changes
- Baseline performance tracking per version

### 4. **Drift Detection**
- Integration with Arize Phoenix for trace analysis
- Per-adapter drift detection with configurable thresholds
- Automatic retrain triggers via `drift_retrain_service`
- Continuous monitoring loop

### 5. **Cost Attribution**
- Per-adapter GPU time, token, and request tracking
- Per-tenant cost aggregation
- Cost-aware warm pool recommendations
- Billing and chargeback support

### 6. **Adapter-Aware Routing**
- `AdapterEndpoint`: Extended model endpoint with adapter metadata
- `AdapterAwareRouter`: Routes requests to replicas with required adapters
- Warm pool tracking per adapter per replica
- Intelligent pre-warming based on adapter traffic

---

## Infrastructure: Helm & K8s Templates

Terradev ships production-ready Kubernetes manifests:

### lora-template Cluster
```bash
helm install lora-serving ./clusters/lora-template/helm \
  -f clusters/lora-template/helm/values-lora.yaml
```

**Features:**
- vLLM with LoRA enabled (max 8 adapters, rank 64)
- KV cache offloading to CPU
- MTP speculative decoding
- Sleep mode for idle savings
- LoRA registry PVC (10Gi RWO)
- Adapter storage PVC (100Gi RWX)
- HPA (2-10 replicas)
- Pod disruption budget

### moe-template Integration
```yaml
# clusters/moe-template/helm/values-moe.yaml
lora:
  enabled: true
  maxLoras: 8
  registry:
    enabled: true
    path: "/data/lora_registry.db"
    registryStorage:
      enabled: true
      size: 10Gi
```

---

## MCP Integration: Agentic Control

The Terradev MCP server exposes 4 LoRA tools for agentic workflows:

```python
Tool("lora_register", ...)
Tool("lora_versions", ...)
Tool("lora_activate", ...)
Tool("lora_sync", ...)
```

**Use case:** An AI agent can autonomously:
1. Detect drift via Phoenix traces
2. Trigger retraining via drift service
3. Register new adapter version
4. Activate and sync across replicas
5. Monitor cost attribution

All without human intervention.

---

## The Result: LoRA at Scale, Simplified

**Before Terradev:**
- Manual adapter management across replicas
- No versioning or rollback capability
- No cost visibility per adapter
- Drift detection via manual spot checks
- Shell scripts for cross-replica sync

**After Terradev:**
- Declarative adapter lifecycle
- One-click rollback to any version
- Per-adapter and per-tenant cost attribution
- Automatic drift detection with retrain triggers
- Gossip-based cross-replica consistency
- Production-ready Helm/K8s templates
- Agentic control via MCP

**Cost savings:** Serve 50 adapters on 8 GPUs instead of 50 GPUs → **84% cost reduction**

**Operational efficiency:** Rollback from bad adapter in 30 seconds instead of 2 hours → **240x faster recovery**

---

## Get Started

```bash
pip install terradev-cli==5.3.8

# Register your first adapter
terradev lora register \
  --name my-adapter \
  --path /path/to/adapter \
  --base-model meta-llama/Llama-2-7b-hf

# Deploy the lora-template cluster
helm install lora-serving ./clusters/lora-template/helm \
  -f clusters/lora-template/helm/values-lora.yaml
```

---

## LoRAX Integration (Predibase LoRA eXchange)

Terradev includes full support for LoRAX, the multi-LoRA inference server from Predibase that serves thousands of fine-tuned models on a single GPU.

### Deploy LoRAX Server

**Docker:**
```bash
terradev lora lorax deploy \
  -m mistralai/Mistral-7B-Instruct-v0.1 \
  --docker
```

**Kubernetes:**
```bash
terradev lora lorax deploy \
  -m mistralai/Mistral-7B-Instruct-v0.1 \
  --k8s \
  --namespace lorax
```

### Manage LoRAX Adapters

```bash
# Test LoRAX server
terradev lora lorax test --host localhost --port 8080

# List loaded adapters
terradev lora lorax list-adapters

# Load an adapter
terradev lora lorax load-adapter \
  -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k

# Unload an adapter
terradev lora lorax unload-adapter -a my-adapter

# Generate text
terradev lora lorax generate \
  -p "What is 2+2?" \
  -a my-adapter

# Sync Terradev registry with LoRAX state
terradev lora lorax sync-registry
```

### LoRAX Features

- **Dynamic Adapter Loading:** Load adapters just-in-time without blocking requests
- **Heterogeneous Continuous Batching:** Pack requests for different adapters together
- **Adapter Exchange Scheduling:** Async prefetch and offload between GPU/CPU memory
- **Production Ready:** Docker images, Helm charts, Prometheus metrics, OpenTelemetry tracing
- **OpenAI Compatible API:** Multi-turn chat conversations with adapters

**Documentation:** https://loraexchange.ai/
**GitHub:** https://github.com/predibase/lorax

---

## HuggingFace PEFT Import

Import LoRA adapters directly from HuggingFace using the PEFT library.

### Import Adapters

```bash
# Import from HuggingFace
terradev lora peft import \
  -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k

# Import with custom local name
terradev lora peft import \
  -a username/adapter \
  --local-name my-adapter

# Import and register in Terradev registry
terradev lora peft import \
  -a username/adapter \
  --local-name my-adapter \
  --register \
  --base-model mistralai/Mistral-7B-Instruct-v0.1

# Import from private repo
terradev lora peft import \
  -a username/private-adapter \
  --token hf_xxx
```

### Manage Local Adapters

```bash
# List imported adapters
terradev lora peft list

# Validate adapter structure
terradev lora peft validate -p ~/.terradev/peft_adapters/username--adapter

# Delete adapter
terradev lora peft delete -a username/adapter
```

### PEFT Features

- **Auto-detection:** Automatically extracts rank, alpha, target modules from adapter_config.json
- **Validation:** Checks for required files (adapter_config.json, adapter_model.bin/safetensors)
- **Registry Integration:** One-step import and register in Terradev LoRA registry
- **Private Repo Support:** Auth token support for private HuggingFace repositories

---

## Complete Workflow Example

```bash
# 1. Import adapter from HuggingFace
terradev lora peft import \
  -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k \
  --local-name gsm8k-adapter \
  --register \
  --base-model mistralai/Mistral-7B-Instruct-v0.1

# 2. Deploy LoRAX server
terradev lora lorax deploy \
  -m mistralai/Mistral-7B-Instruct-v0.1 \
  --docker

# 3. Load adapter onto LoRAX
terradev lora lorax load-adapter -a gsm8k-adapter

# 4. Test generation
terradev lora lorax generate \
  -p "Natalia sold clips to 48 of her friends in April..." \
  -a gsm8k-adapter

# 5. Sync registry state
terradev lora lorax sync-registry
```

---

**Documentation:** https://github.com/theoddden/Terradev

**PyPI:** https://pypi.org/project/terradev-cli/5.3.8/

---

*Built for production. Open source. Apache 2.0 licensed.*
