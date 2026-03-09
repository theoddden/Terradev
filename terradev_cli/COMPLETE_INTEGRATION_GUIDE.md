# 🚀 Complete Terradev Integration Guide
**Production-ready ML infrastructure with enterprise-grade integrations**

---

## 🎯 **SGLang Optimization Stack**

**Revolutionary workload-specific auto-optimization for SGLang serving with 7 workload types:**

### 🚀 **SGLang Workload Optimizations**
```bash
# Auto-optimize any model for workload type
terradev sglang optimize deepseek-ai/DeepSeek-V3

# Detect workload from description
terradev sglang detect meta-llama/Llama-2-7b-hf --user-description "Real-time API"

# Deploy with workload-specific optimizations
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload agentic-chat

# Monitor SGLang performance
terradev sglang monitor --endpoint http://localhost:30000

# Benchmark different workloads
terradev sglang benchmark --model meta-llama/Llama-2-7b-hf --workloads all
```

### **Workload Types & Optimizations**

#### **1. Agentic/Multi-turn Chat**
```bash
# Deploy for agentic applications
terradev sglang deploy --model deepseek-ai/DeepSeek-V3 --workload agentic-chat

# Features auto-applied:
- LPM (Long Prompt Management) + RadixAttention
- Cache-aware routing (75-90% cache hit rate)
- Conversation state management
- Context window optimization
```

#### **2. High-Throughput Batch**
```bash
# Deploy for batch processing
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload high-throughput

# Features auto-applied:
- FCFS (First-Come-First-Served) scheduling
- CUDA graphs for maximum throughput
- FP8 quantization
- Batch size optimization
```

#### **3. Low-Latency/Real-Time**
```bash
# Deploy for real-time applications
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload low-latency

# Features auto-applied:
- EAGLE3 speculative decoding
- Speculative Decoding v2
- Capped concurrency for consistency
- 30-50% TTFT (Time To First Token) improvement
```

#### **4. MoE Models**
```bash
# Deploy Mixture of Experts models
terradev sglang deploy --model mistralai/Mixtral-8x7B --workload moe

# Features auto-applied:
- DeepEP auto-optimization
- TBO/SBO (Token/Batch-level Orchestration)
- EPLB (Expert Load Balancing)
- Redundant experts (up to 2x throughput)
```

#### **5. PD (Prefill/Decode) Disaggregated**
```bash
# Deploy with prefill/decode separation
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload pd-disaggregated

# Features auto-applied:
- Separate prefill/decode configurations
- Production-optimized scheduling
- Resource allocation optimization
- Load balancing between phases
```

#### **6. Structured Output/RAG**
```bash
# Deploy for structured data applications
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload structured-output

# Features auto-applied:
- xGrammar optimization
- FSM (Finite State Machine) optimization
- 10x faster structured output generation
- JSON schema validation
```

#### **7. Hardware-Specific**
```bash
# Deploy with hardware-specific optimizations
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload hardware-specific --gpu-type H100

# Supported hardware optimizations:
- H100/H200: Tensor cores, FP8 support
- H20: China-specific optimizations
- GB200: Blackwell architecture
- AMD MI300X: CDNA optimizations
```

### **SGLang Management Commands**
```bash
# Status and monitoring
terradev sglang status --endpoint http://localhost:30000
terradev sglang health --endpoint http://localhost:30000

# Configuration management
terradev sglang config --show
terradev sglang config --set max_tokens=4096
terradev sglang config --set temperature=0.7

# Performance tuning
terradev sglang tune --endpoint http://localhost:30000 --objective latency
terradev sglang tune --endpoint http://localhost:30000 --objective throughput

# Scaling operations
terradev sglang scale --endpoint http://localhost:30000 --replicas 4
terradev sglang autoscale --endpoint http://localhost:30000 --min-replicas 2 --max-replicas 10
```

---

## 1. **GitOps Foundation for Production**

**Start with GitOps - every production deployment should be under version control.**

```bash
# Initialize GitOps repository
terradev gitops init --provider github --repo my-org/infra --tool argocd --cluster prod

# Bootstrap GitOps tool on cluster
terradev gitops bootstrap --tool argocd --cluster production

# Validate before applying
terradev gitops validate --dry-run --cluster production

# Sync changes
terradev gitops sync --cluster production

# Check GitOps status
terradev gitops status --cluster production

# Rollback if needed
terradev gitops rollback --cluster production --revision previous
```

**What happens behind the scenes:**
- **Repository structure** — Automated GitOps repo with clusters/, apps/, infra/, policies/
- **Tool integration** — ArgoCD or Flux CD bootstrapped with RBAC
- **Policy templates** — Gatekeeper/Kyverno policies for security
- **Multi-environment** — Dev, staging, production support

### **GitOps Advanced Features**
```bash
# Multi-cluster GitOps
terradev gitops init --provider github --repo my-org/infra --tool flux --clusters dev,staging,prod

# Policy as Code
terradev gitops add-policy --cluster prod --policy security --template gatekeeper-restrict-external-ips

# Secret management
terradev gitops add-secrets --cluster prod --provider vault --path secret/terradev

# Progressive delivery
terradev gitops rollout --cluster prod --app my-app --strategy blue-green
terradev gitops rollout --cluster prod --app my-app --strategy canary --steps 10%,30%,60%,100%
```

---

## 2. **Set Up Observability Stack**

**Deploy monitoring before you need it.**

```bash
# Weights & Biases integration
terradev configure --provider wandb --api-key $WANDB_KEY
terradev ml wandb --test

# Prometheus + Grafana
terradev configure --provider prometheus --api-key $PROMETHEUS_URL
terradev integrations --export-grafana

# Datadog integration
terradev configure --provider datadog --api-key $DD_API_KEY
terradev datadog --test

# Complete observability stack
terradev integrations --deploy --stack full --cluster production
```

**Auto-injection:** All WANDB_*, PROMETHEUS_*, DD_* environment variables automatically injected into provisioned containers.

### **Advanced Observability**
```bash
# Custom dashboards
terradev integrations --export-grafana --dashboard gpu-utilization
terradev integrations --export-grafana --dashboard training-metrics
terradev integrations --export-grafana --dashboard cost-analysis

# Alert management
terradev integrations --alert --name gpu-memory-high --threshold 90%
terradev integrations --alert --name training-failure --condition job.status=failed

# Log aggregation
terradev integrations --logs --provider elasticsearch --cluster production
terradev integrations --logs --provider loki --cluster production

# Distributed tracing
terradev integrations --tracing --provider jaeger --cluster production
terradev integrations --tracing --provider tempo --cluster production
```

---

## 3. **HuggingFace Spaces - Public Demo**

**Deploy public demos to Spaces with one command.**

```bash
# Install HF Spaces support
pip install terradev-cli[hf]
export HF_TOKEN=your_huggingface_token

# LLM Template (A10G GPU)
terradev hf-space my-llama --model-id meta-llama/Llama-2-7b-hf --template llm

# Embedding Model (CPU Upgrade)
terradev hf-space my-embeddings --model-id sentence-transformers/all-MiniLM-L6-v2 --template embedding

# Image Model (T4 GPU)
terradev hf-space my-image --model-id runwayml/stable-diffusion-v1-5 --template image

# Custom Hardware and SDK
terradev hf-space my-model --model-id microsoft/DialoGPT-medium \
  --hardware a10g-large --sdk gradio --private

# Multi-space deployment
terradev hf-space batch --config spaces-config.yaml

# Space management
terradev hf-space list
terradev hf-space status --space my-llama
terradev hf-space update --space my-llama --model-id meta-llama/Llama-2-13b-hf
terradev hf-space delete --space my-llama
```

**Available Templates:**
- **llm** — vLLM server with auto-optimization
- **embedding** — FastAPI serving with batch processing
- **image** — Diffusers pipeline with memory optimization
- **custom** — Your choice of SDK (Gradio, Streamlit, FastAPI)

### **Advanced Spaces Features**
```bash
# Custom domain
terradev hf-space update --space my-llama --domain chat.mycompany.com

# Organization spaces
terradev hf-space create --org mycompany --space my-llama --template llm

# Space monitoring
terradev hf-space monitor --space my-llama --metrics gpu,requests,errors

# A/B testing
terradev hf-space ab-test --space my-llama --model-a meta-llama/Llama-2-7b-hf --model-b meta-llama/Llama-2-13b-hf

# Analytics integration
terradev hf-space analytics --space my-llama --provider wandb
```

---

## 4. **LoRA Adapter Deploy**

**Deploy your core model with multi-tenant adapter support.**

```bash
# Deploy MoE model with all optimizations auto-applied
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3.5-397B-A17B

# Deploy LoRA-enabled endpoint
terradev lora deploy --model-id meta-llama/Llama-2-70b-hf --endpoint multi-tenant-llm

# Auto-applied optimizations:
# - KV cache offloading (up to 9x throughput)
# - MTP speculative decoding (up to 2.8x faster)
# - Sleep mode (18-200x faster than cold restart)
# - Expert load balancing
# - NUMA topology optimization from Part 1
```

**Hot-load customer adapters without restarting the server:**

```bash
# Load Customer Adapter
terradev lora add -e http://<endpoint>:8000 \
  -n customer-a -p /adapters/customer-a

# Load Multiple Adapters
terradev lora add -e http://<endpoint>:8000 \
  -n customer-b -p /adapters/customer-b
terradev lora add -e http://<endpoint>:8000 \
  -n customer-c -p /adapters/customer-c

# List Loaded Adapters
terradev lora list -e http://<endpoint>:8000

# Remove Adapter
terradev lora remove -e http://<endpoint>:8000 -n customer-b

# Adapter management
terradev lora status -e http://<endpoint>:8000
terradev lora update -e http://<endpoint>:8000 -n customer-a -p /adapters/customer-a-v2
terradev lora benchmark -e http://<endpoint>:8000 -n customer-a
```

**Use in OpenAI API:** Each adapter becomes a model name in the OpenAI-compatible endpoint:

```bash
curl http://<endpoint>:8000/v1/chat/completions \
  -d '{"model": "customer-a", "messages": [{"role": "user", "content": "Hello"}]}'

# Performance: 454% higher output tokens/sec, 87% lower TTFT vs separate deployments
```

### **Advanced LoRA Features**
```bash
# Batch adapter operations
terradev lora batch-add -e http://<endpoint>:8000 --config adapters.yaml

# Adapter versioning
terradev lora version -e http://<endpoint>:8000 -n customer-a --version v2.0

# Adapter A/B testing
terradev lora ab-test -e http://<endpoint>:8000 -n customer-a --baseline customer-b

# Adapter analytics
terradev lora analytics -e http://<endpoint>:8000 --metrics throughput,latency,memory

# Adapter optimization
terradev lora optimize -e http://<endpoint>:8000 -n customer-a --objective throughput
```

---

## 5. **InferX Serverless for Burst Traffic**

**Handle traffic spikes without managing GPU instances.**

```bash
# Deploy Serverless Endpoint
terradev inferx deploy --endpoint burst-api \
  --model-id meta-llama/Llama-2-7b-hf \
  --hardware a10g \
  --max-concurrency 100

# Check Endpoint Health
terradev inferx status --endpoint burst-api

# Run Failover Tests
terradev inferx failover --endpoint burst-api --test-load 1000

# Cost Analysis
terradev inferx cost-analysis --days 30 --endpoint burst-api

# Scale serverless endpoints
terradev inferx scale --endpoint burst-api --min-concurrency 10 --max-concurrency 1000

# Update endpoint
terradev inferx update --endpoint burst-api --model-id meta-llama/Llama-2-13b-hf
```

**InferX Features:**
- **Sub-2s cold starts** via GPU slicing
- **90%+ GPU utilization** with multi-tenant isolation
- **30+ models per GPU** via snapshotting
- **OpenAI-compatible API endpoint**
- **Pay-per-request pricing** (no hourly cost)

### **Advanced InferX Features**
```bash
# Multi-model endpoints
terradev inferx deploy --endpoint multi-model \
  --models meta-llama/Llama-2-7b-hf,mistralai/Mistral-7B \
  --hardware a10g

# Custom scaling policies
terradev inferx autoscale --endpoint burst-api \
  --policy cpu-based --threshold 70% --scale-factor 2

# Traffic routing
terradev inferx route --endpoint burst-api --strategy round-robin
terradev inferx route --endpoint burst-api --strategy least-connections

# Performance monitoring
terradev inferx monitor --endpoint burst-api --metrics latency,throughput,errors
terradev inferx benchmark --endpoint burst-api --load-test duration=5m
```

---

## 6. **Arize Phoenix - LLM Observability**

**Understand what's happening inside your inference pipelines.**

```bash
# Install Phoenix Support
pip install terradev-cli[phoenix]

# Deploy Phoenix (Kubernetes)
terradev phoenix k8s --namespace observability \
  --project my-inference --replicas 2

# Generate OTLP Environment Variables
terradev phoenix otlp-env --endpoint http://phoenix:6006

# Output:
# export OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006
# export OTEL_EXPORTER_OTLP_HEADERS=api_key=your_key
# export OTEL_PROJECT_NAME=my-inference
# export OTEL_SERVICE_NAME=vllm-inference
```

**View and Analyze Traces:**

```bash
# List Projects
terradev phoenix projects

# View Recent Traces
terradev phoenix spans --project my-inference --limit 100

# Filter Traces
terradev phoenix spans --project my-inference \
  --filter "span_kind == 'RETRIEVER' and status_code == 'ERROR'"

# View Specific Trace
terradev phoenix trace --trace-id 123e4567-e89b-12d3-a456-426614174000

# Performance analysis
terradev phoenix analyze --project my-inference --metric latency
terradev phoenix analyze --project my-inference --metric error-rate

# Export traces
terradev phoenix export --project my-inference --format json --output traces.json
```

### **Advanced Phoenix Features**
```bash
# Custom dashboards
terradev phoenix dashboard --create --name performance --project my-inference
terradev phoenix dashboard --add-chart --dashboard performance --metric latency

# Alerting
terradev phoenix alert --create --name high-latency --threshold 1000ms --project my-inference
terradev phoenix alert --create --name error-spike --condition "error_rate > 5%"

# Data retention
terradev phoenix retention --project my-inference --days 30
terradev phoenix archive --project my-inference --older-than 30d

# Integration with other tools
terradev phoenix integrate --provider grafana --project my-inference
terradev phoenix integrate --provider slack --project my-inference --webhook $SLACK_WEBHOOK
```

---

## 7. **Qdrant - Vector Database for RAG**

**High-performance vector storage for RAG applications.**

```bash
# Install Qdrant Support
pip install terradev-cli[qdrant]

# Deploy Qdrant (Production Kubernetes)
terradev qdrant k8s --namespace vector-db \
  --replicas 3 --storage-class ssd \
  --embedding-model text-embedding-3-large

# Create Collection
terradev qdrant create-collection --name documents \
  --vector-size 1536 --distance Cosine \
  --hnsw-m 16 --hnsw-ef 100

# Add Documents
terradev qdrant upsert --name documents \
  --file documents.json --batch-size 1000

# Search Vectors
terradev qdrant search --name documents \
  --query "What is machine learning?" \
  --limit 10 --score-threshold 0.7

# Collection management
terradev qdrant list-collections
terradev qdrant info --collection documents
terradev qdrant delete --collection documents
```

### **Advanced Qdrant Features**
```bash
# Batch operations
terradev qdrant batch-upsert --name documents --directory ./documents/
terradev qdrant batch-search --name documents --queries-file queries.json

# Performance optimization
terradev qdrant optimize --collection documents --rebuild-index
terradev qdrant benchmark --collection documents --test-size 10000

# Replication and sharding
terradev qdrant replicate --collection documents --shards 3 --replicas 2

# Hybrid search
terradev qdrant hybrid-search --name documents \
  --query "machine learning" \
  --filter '{"category": "AI"}' \
  --limit 10

# Monitoring
terradev qdrant monitor --collection documents --metrics performance
terradev qdrant backup --name documents --backup-path ./backups/
```

---

## 8. **NeMo Guardrails - Output Safety**

**Add safety layers to prevent harmful content and PII leakage.**

```bash
# Install Guardrails Support
pip install terradev-cli[guardrails]

# Deploy Guardrails (Sidecar Mode)
terradev guardrails sidecar --llm-endpoint http://vllm:8000/v1 \
  --deployment-mode sidecar --memory-backend redis

# Generate Default Configurations
terradev guardrails generate-config --output ./guardrails \
  --enable-topical --enable-jailbreak --enable-pii

# Test Guardrails
terradev guardrails chat --config-id my-safety-config \
  --message "How do I hack a system?"

# Deploy standalone guardrails
terradev guardrails deploy --config-path ./guardrails --port 8080
```

**Configuration Types:**
- **Topical** — Restrict topics (finance, health, legal)
- **Jailbreak** — Prevent prompt injection attacks
- **PII** — Filter personal information (email, phone, SSN)
- **Fact-check** — Ensure factual consistency

### **Advanced Guardrails Features**
```bash
# Custom policies
terradev guardrails add-policy --config my-safety-config --type topical --topics finance,health
terradev guardrails add-policy --config my-safety-config --type pii --fields email,phone

# Policy testing
terradev guardrails test --config my-safety-config --test-suite jailbreaks
terradev guardrails benchmark --config my-safety-config --test-size 1000

# Integration with LLMs
terradev guardrails integrate --provider openai --api-key $OPENAI_KEY
terradev guardrails integrate --provider vllm --endpoint http://vllm:8000/v1

# Monitoring and analytics
terradev guardrails monitor --config my-safety-config --metrics violations,blocks,modifications
terradev guardrails analytics --config my-safety-config --period 7d
```

---

## 9. **Dataset Smart-Staging**

**Intelligent dataset preparation and caching for optimal training performance.**

```bash
# Stage local dataset near compute
terradev stage -d ./my-dataset --target-regions us-east-1,eu-west-1

# Cache a HuggingFace dataset near target regions
terradev stage --hf-dataset allenai/C4 --target-regions us-east-1,eu-west-1

# Stage with specific split and configuration
terradev stage --hf-dataset HuggingFaceH4/llava-instruct-mistral-7b \
  --split train --target-regions us-west-2,eu-central-1

# Cache multiple datasets in parallel
terradev stage --hf-dataset "allenai/C4,mozilla/common-voice,bookcorpus/openwebtext" \
  --target-regions us-east-1,eu-west-1,ap-southeast-1
```

**Advanced staging with preprocessing:**

```bash
# Filter, deduplicate, and compress in one pass
terradev stage --hf-dataset allenai/C4 --target-regions us-east-1 \
  --process "filter english,remove duplicates" --format parquet --compression zstd

# Stage with size limits and sampling
terradev stage --hf-dataset mozilla/common-voice --target-regions us-east-1 \
  --max-size 100GB --sample-rate 0.1

# Stage with full preprocessing pipeline
terradev stage --hf-dataset HuggingFaceH4/ultrachat_200k --target-regions us-east-1 \
  --preprocess "tokenize,truncate_length=2048,remove_pii"
```

**Dataset management:**

```bash
# List cached datasets
terradev stage --list-cached --region us-east-1

# Check dataset status
terradev stage --status --dataset-id <dataset-id>

# Validate dataset integrity
terradev stage --validate --dataset-id <dataset-id>

# Remove cached dataset
terradev stage --remove --dataset-id <dataset-id>
```

---

## 10. **Advanced Monitoring and Cost Optimization**

**Track spend and optimize automatically.**

```bash
# Cost Analytics
terradev analytics --days 30

# View spend over last 30 days
terradev optimize

# Find cheaper alternatives for running instances
terradev price-discovery --gpu-type H100 --confidence 0.95

# Enhanced price discovery with confidence scoring
terradev price-discovery --gpu-type A100 --regions us-east-1,us-west-2 --confidence 0.99

# Cost optimization recommendations
terradev optimize --instance-id <instance-id> --auto-apply
terradev optimize --cluster production --scope all
```

**Real-Time Training Monitoring:**

```bash
# Real-Time Training Monitoring
terradev monitor --job my-training-job

# Monitor GPU utilization, memory, temperature, cost
terradev train-status

# Check all training jobs
terradev train-status --all

# Detailed job monitoring
terradev monitor --job my-training-job --metrics gpu,memory,temperature,cost
terradev monitor --job my-training-job --live --refresh 5s

# Checkpoint management
terradev checkpoint list --job my-job
terradev checkpoint create --job my-job --name manual-checkpoint
terradev checkpoint restore --job my-job --checkpoint-id <checkpoint-id>
```

**Advanced Cost Features:**

```bash
# Budget management
terradev budget set --amount 1000 --period monthly --alert-threshold 80
terradev budget status
terradev budget history --days 90

# Cost forecasting
terradev forecast --days 30 --confidence 0.95
terradev forecast --instance-id <instance-id> --period weekly

# Resource optimization
terradev optimize --scope storage --cleanup-policy keep-latest-3
terradev optimize --scope networking --egress-optimization
terradev optimize --scope compute --right-sizing

# Cost alerts
terradev alerts create --name budget-warning --condition "cost > 0.8*budget"
terradev alerts create --name unusual-spend --condition "daily_cost > 3*avg_daily_cost"
```

---

## 🎯 **Integration Summary**

### **Key Integrations for User Acquisition**

| Integration | Use Case | Value Proposition |
|-------------|----------|------------------|
| **SGLang** | Workload-optimized inference | 7 workload types, auto-optimization |
| **GitOps** | Production deployment | Version control, automated rollouts |
| **HuggingFace Spaces** | Public demos | One-click deployment, social sharing |
| **LoRA Support** | Multi-tenant serving | Hot-swappable adapters, cost efficiency |
| **InferX** | Serverless inference | Burst traffic, pay-per-request |
| **Arize Phoenix** | LLM observability | Debugging, performance analysis |
| **Qdrant** | Vector database | RAG applications, similarity search |
| **NeMo Guardrails** | Output safety | Content filtering, compliance |
| **Dataset Staging** | Training optimization | Smart caching, preprocessing |
| **Cost Optimization** | Spend management | Real-time tracking, auto-optimization |

### **Installation Options**

```bash
# Core installation
pip install terradev-cli

# With all integrations
pip install terradev-cli[all]

# Specific integrations
pip install terradev-cli[sglang,gitops,hf,phoenix,qdrant,guardrails]

# Development installation
pip install terradev-cli[dev]
```

### **Quick Start**

```bash
# 1. Configure providers
terradev configure --provider runpod
terradev configure --provider wandb

# 2. Deploy optimized inference
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload low-latency

# 3. Set up monitoring
terradev integrations --deploy --stack monitoring

# 4. Deploy public demo
terradev hf-space my-demo --model-id meta-llama/Llama-2-7b-hf --template llm

# 5. Monitor costs
terradev analytics --days 7
```

---

## 🚀 **Get Started Today**

**PyPI:** [pypi.org/project/terradev-cli](https://pypi.org/project/terradev-cli)

**GitHub:** [github.com/theoddden/Terradev](https://github.com/theoddden/Terradev)

**Complete integration stack for production ML infrastructure with enterprise-grade features.**
