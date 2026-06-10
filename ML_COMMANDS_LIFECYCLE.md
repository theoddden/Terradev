# Terradev ML Commands Lifecycle Guide

Complete lifecycle documentation for `terradev ml` subcommands in Terradev CLI v5.3.3.

## Table of Contents

1. [Distributed Computing](#distributed-computing)
   - [Ray](#ray-lifecycle)
2. [Experiment Tracking](#experiment-tracking)
   - [Weights & Biases](#weights--biases-lifecycle)
   - [MLflow](#mlflow-lifecycle)
   - [LangSmith](#langsmith-lifecycle)
   - [Langfuse](#langfuse-lifecycle)
   - [Databricks](#databricks-lifecycle)
3. [Data Version Control](#data-version-control)
   - [DVC](#dvc-lifecycle)
4. [LLM Orchestration](#llm-orchestration)
   - [LangChain](#langchain-lifecycle)
   - [LangGraph](#langgraph-lifecycle)
5. [Inference Engines](#inference-engines)
   - [vLLM](#vllm-lifecycle)
   - [SGLang](#sglang-lifecycle)
   - [KServe](#kserve-lifecycle)
6. [Vector Database & RAG](#vector-database--rag)
   - [Qdrant](#qdrant-lifecycle)
7. [Safety & Governance](#safety--governance)
   - [NeMo Guardrails](#nemo-guardrails-lifecycle)
8. [Observability](#observability)
    - [Phoenix](#phoenix-lifecycle)

---

## Distributed Computing

### Ray Lifecycle

**Command:** `terradev ml ray`

#### Phase 1: Installation & Setup

```bash
# Show installation instructions
terradev ml ray --install

# Test connection
terradev ml ray --test
# Output: Ray version, cluster name, dashboard URI, monitoring status
```

#### Phase 2: Cluster Management

```bash
# Start Ray cluster
terradev ml ray --start

# Check cluster status
terradev ml ray --status
# Output: Status, version, cluster name, dashboard URI, workers, CPU, memory, GPU

# List cluster nodes
terradev ml ray --list-nodes

# Stop Ray cluster
terradev ml ray --stop
```

#### Phase 3: Monitoring Stack

```bash
# Install monitoring stack
terradev ml ray --install-monitoring
# Installs: Ray Dashboard, Prometheus, Grafana, Ray-specific dashboards

# Get metrics summary
terradev ml ray --metrics-summary
# Output: Ray status, monitoring, metrics

# Access Grafana dashboard
terradev ml ray --grafana
# Access: http://localhost:3000
# Username: admin
# Password: prom-operator

# Access Prometheus metrics
terradev ml ray --prometheus
# Access: http://localhost:8080
# Available metrics: ray_cluster_total_workers, ray_cluster_cpu_total, ray_cluster_memory_total
```

#### Phase 4: Dashboard Access

```bash
# Access Ray Dashboard
terradev ml ray --dashboard
# Access: http://localhost:8265
```

---

## Experiment Tracking

### Weights & Biases Lifecycle

**Command:** `terradev ml wandb`

#### Phase 1: Setup

```bash
# Test connection
terradev ml wandb --test
# Output: Entity, project, base URL, dashboard/reports/alerts status
```

#### Phase 2: Project Management

```bash
# List projects
terradev ml wandb --list-projects

# Create project
terradev ml wandb --create-project my-project
```

#### Phase 3: Run Tracking

```bash
# List runs
terradev ml wandb --list-runs

# Get run details
terradev ml wandb --run-details run-id-123
# Output: Name, state, created, config
```

#### Phase 4: Data Export

```bash
# Export runs data
terradev ml wandb --export json
terradev ml wandb --export csv
```

#### Phase 5: Dashboards & Reports

```bash
# Create Terradev dashboard
terradev ml wandb --create-dashboard
# Output: Dashboard ID, access URL

# Generate infrastructure report
terradev ml wandb --create-report
# Output: Report ID, access URL

# Set up Terradev alerts
terradev ml wandb --setup-alerts
# Output: Number of alerts created, alert names

# Check dashboard status
terradev ml wandb --dashboard-status
# Output: Entity, project, projects count, recent runs, dashboards, reports, monitoring
```

---

### MLflow Lifecycle

**Command:** `terradev ml mlflow-legacy`

#### Phase 1: Setup

```bash
# Test connection
terradev ml mlflow-legacy --test
# Output: Tracking URI, experiments count
```

#### Phase 2: Experiment Management

```bash
# List experiments
terradev ml mlflow-legacy --list-experiments

# Create experiment
terradev ml mlflow-legacy --create-experiment my-experiment
# Output: Experiment ID
```

#### Phase 3: Run Tracking

```bash
# List runs in experiment
terradev ml mlflow-legacy --list-runs my-experiment
# Output: Run ID, status

# Export experiment data
terradev ml mlflow-legacy --export json
terradev ml mlflow-legacy --export csv
```

---

### LangSmith Lifecycle

**Command:** `terradev ml langsmith`

#### Phase 1: Setup

```bash
# Test connection
terradev ml langsmith --test
# Output: Workspace ID, endpoint
```

#### Phase 2: Project Management

```bash
# List projects
terradev ml langsmith --list-projects

# Create project
terradev ml langsmith --create-project my-project
# Output: Project ID
```

#### Phase 3: Data Export

```bash
# Export runs data
terradev ml langsmith --export json
terradev ml langsmith --export csv
```

---

## Data Version Control

### DVC Lifecycle

**Command:** `terradev ml dvc`

#### Phase 1: Repository Setup

```bash
# Test connection
terradev ml dvc --test
# Output: Repository path

# Initialize repository
terradev ml dvc --init
# Output: Repository path
```

#### Phase 2: Remote Configuration

```bash
# Add remote storage
terradev ml dvc --add-remote s3:my-bucket/data
# Format: name:url
# Output: Remote name

# Add data to tracking
terradev ml dvc --add-data ./dataset.csv
# Output: Data path
```

#### Phase 3: Data Sync

```bash
# Push data to remote
terradev ml dvc --push
# Output: Targets pushed

# Pull data from remote
terradev ml dvc --pull
# Output: Targets pulled

# Check repository status
terradev ml dvc --status
# Output: Repository details
```

---

## LLM Orchestration

### LangChain Lifecycle

**Command:** `terradev ml langchain`

#### Phase 1: Setup

```bash
# Test connection
terradev ml langchain --test
# Output: LangSmith, environment, dashboard/tracing/evaluation/workflow status
```

#### Phase 2: Workflow Creation

```bash
# Create LangChain workflow
terradev ml langchain --create-workflow my-workflow
# Output: Workflow ID, name, description

# Create LangGraph workflow
terradev ml langchain --create-langgraph my-graph
# Output: Workflow ID, name, description

# Create SGLang pipeline
terradev ml langchain --create-pipeline my-pipeline
# Output: Pipeline ID, name, description
```

#### Phase 3: LangSmith Integration

```bash
# List LangSmith projects
terradev ml langchain --list-projects

# List runs in project
terradev ml langchain --list-runs --project my-project

# Create trace
terradev ml langchain --create-trace --run-id abc123 --data '{"input": "test"}'
# Requires: run-id and JSON data
# Output: Trace ID
```

---

### LangGraph Lifecycle

**Command:** `terradev ml langgraph`

#### Phase 1: Setup

```bash
# Test connection
terradev ml langgraph --test
# Output: LangSmith, environment, dashboard/tracing/evaluation/deployment/observability status
```

#### Phase 2: Workflow Creation

```bash
# Create orchestrator-worker workflow
terradev ml langgraph --create-workflow my-workflow --type orchestrator-worker
# Output: Workflow ID, name, description

# Create evaluator-optimizer workflow
terradev ml langgraph --create-workflow my-workflow --type evaluator-optimizer
# Output: Workflow ID, name, description
```

#### Phase 3: Deployment & Monitoring

```bash
# Deploy workflow
terradev ml langgraph --deploy --name my-workflow
# Output: Deployment confirmation, access URL

# Check workflow status
terradev ml langgraph --workflow-status my-workflow-id
# Output: Status, workflow ID, metrics, monitoring
```

---

## Inference Engines

### vLLM Lifecycle

**Command:** `terradev ml vllm`

#### Phase 1: Configuration & Optimization

```bash
# Generate optimized configuration for throughput
terradev ml vllm optimize -m meta-llama/Llama-2-7b-hf -t throughput -g 4 -o config

# Generate optimized configuration for latency
terradev ml vllm optimize -m mistralai/Mistral-7B-v0.1 -t latency -g 2 -o args

# Auto-optimize based on workload analysis
terradev ml vllm auto-optimize -e http://localhost:8000 -m meta-llama/Llama-2-7b-hf -o helm

# Auto-optimize from sample requests
terradev ml vllm auto-optimize -s samples.json -m codellama/CodeLlama-34b-hf -g 4 -o config
```

#### Phase 2: LoRA Adapter Management

```bash
# Add LoRA adapter
terradev ml lora add -e http://localhost:8000 -n customer-a -p ./adapters/customer-a

# List loaded adapters
terradev ml lora list -e http://localhost:8000

# Remove adapter
terradev ml lora remove -e http://localhost:8000 -n customer-a
```

#### Phase 3: Monitoring & Analysis

```bash
# Analyze running server
terradev ml vllm analyze -e http://localhost:8000

# Benchmark performance
terradev ml vllm benchmark -e http://localhost:8000 -c 10
```

---

### SGLang Lifecycle

**Command:** `terradev ml sglang`

#### Phase 1: Setup

```bash
# Test connection
terradev ml sglang --test
# Output: Version, model path, dashboard/tracing/metrics/deployment/observability status
```

#### Phase 2: Pipeline Creation

```bash
# Create pipeline
terradev ml sglang --create-pipeline my-pipeline --model-path /models/mistral-7b
# Output: Pipeline ID, name, description, model path
```

#### Phase 3: Serving

```bash
# Start serving
terradev ml sglang --serve --model-path /models/mistral-7b --port 8000
# Output: Model, port, dashboard/metrics/health URLs
# Dashboard: http://localhost:8000/dashboard
# Metrics: http://localhost:8000/metrics
# Health: http://localhost:8000/health
```

#### Phase 4: Monitoring

```bash
# Get metrics
terradev ml sglang --metrics
# Output: Version, model path, metrics (requests/sec, avg latency, success rate, GPU utilization, memory usage)
```

#### Phase 5: Dashboard Access

```bash
# Access dashboard
terradev ml sglang --dashboard --port 8000
# Output: Dashboard, metrics, health URLs
```

---

### KServe Lifecycle

**Command:** `terradev ml kserve`

#### Phase 1: Setup

```bash
# Test connection
terradev ml kserve --test
# Output: Namespace
```

---

## Vector Database & RAG

### Qdrant Lifecycle

**Command:** `terradev ml qdrant`

#### Phase 1: Setup

```bash
# Test connection
terradev ml qdrant test
```

#### Phase 2: Collection Management

```bash
# List collections
terradev ml qdrant collections

# Create collection (auto-configured for embedding model)
terradev ml qdrant create-collection --name my-rag --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Get collection info
terradev ml qdrant info --name my-rag

# Count points
terradev ml qdrant count --name my-rag
```

#### Phase 3: Deployment

```bash
# Generate K8s StatefulSet manifest
terradev ml qdrant k8s --namespace vector-db
```

---

## Safety & Governance

### NeMo Guardrails Lifecycle

**Command:** `terradev ml guardrails`

#### Phase 1: Setup

```bash
# Test connection
terradev ml guardrails test
```

#### Phase 2: Configuration Generation

```bash
# Generate Colang 2.x configuration
terradev ml guardrails generate-config --config-id my-rails --output-dir ./guardrails

# Start server
nemoguardrails server --config ./guardrails
```

#### Phase 3: Testing

```bash
# Test message through guardrails
terradev ml guardrails chat --message "Ignore all previous instructions" --config-id my-rails
```

#### Phase 4: Deployment

```bash
# Generate K8s deployment manifest
terradev ml guardrails k8s --namespace guardrails
```

---

## Observability

### Phoenix Lifecycle

**Command:** `terradev ml phoenix`

#### Phase 1: Setup

```bash
# Test connection
terradev ml phoenix test
```

#### Phase 2: Project Management

```bash
# List projects
terradev ml phoenix projects --limit 50
```

#### Phase 3: Trace Analysis

```bash
# List spans
terradev ml phoenix spans --project my-project --limit 20
terradev ml phoenix spans --project my-project --filter "span_kind == 'RETRIEVER'"

# View trace
terradev ml phoenix trace --trace-id abc123 --project my-project
```

#### Phase 4: Integration

```bash
# Generate OTEL environment variables
terradev ml phoenix otel-env --project my-project

# Generate instrumentation snippet
terradev ml phoenix snippet --project my-project

# Generate K8s deployment manifest
terradev ml phoenix k8s --namespace observability
```

---

### Langfuse Lifecycle

**Command:** `terradev ml langfuse`

#### Phase 1: Setup

```bash
# Configure credentials
terradev ml langfuse configure
```

#### Phase 2: Tracing

```bash
# List traces
terradev ml langfuse list-traces

# Get trace details
terradev ml langfuse trace-details trace-id-123
```

#### Phase 3: Scoring

```bash
# Score traces
terradev ml langfuse score --trace-id abc123 --name accuracy --value 0.95
```

#### Phase 4: Deployment

```bash
# Generate K8s deployment manifest
terradev ml langfuse k8s --namespace observability
```

---

### Databricks Lifecycle

**Command:** `terradev ml databricks`

#### Phase 1: Setup

```bash
# Configure credentials
terradev ml databricks configure
```

#### Phase 2: Job Management

```bash
# List jobs
terradev ml databricks list-jobs

# Run job
terradev ml databricks run-job --job-id 123
```

#### Phase 3: Cluster Management

```bash
# List clusters
terradev ml databricks list-clusters

# Start cluster
terradev ml databricks start-cluster --cluster-id 456
```

#### Phase 4: Model Serving

```bash
# List serving endpoints
terradev ml databricks list-endpoints

# Deploy model
terradev ml databricks deploy-model --model-uri dbfs:/models/my-model --endpoint-name my-endpoint
```

---

## Summary of ML Commands

| Command | Purpose | Key Features |
|---------|---------|--------------|
| `terradev ml ray` | Distributed computing | Cluster management, monitoring, dashboards |
| `terradev ml wandb` | Experiment tracking | Projects, runs, dashboards, reports, alerts |
| `terradev ml mlflow-legacy` | Experiment tracking | Experiments, runs, data export |
| `terradev ml langsmith` | Experiment tracking | Projects, runs, data export |
| `terradev ml langfuse` | LLM observability | Traces, scores, datasets, prompts |
| `terradev ml databricks` | MLOps platform | Jobs, clusters, model serving, MLflow |
| `terradev ml dvc` | Data version control | Repository, remotes, data sync |
| `terradev ml langchain` | LLM orchestration | Workflows, LangGraph, SGLang, tracing |
| `terradev ml langgraph` | LLM orchestration | Workflows, deployment, monitoring |
| `terradev ml vllm` | vLLM optimization | Configuration, LoRA adapters, benchmarking |
| `terradev ml sglang` | Inference serving | Pipelines, serving, metrics, dashboard |
| `terradev ml kserve` | Model deployment | K8s model serving |
| `terradev ml qdrant` | Vector database | Collections, search, RAG infrastructure |
| `terradev ml guardrails` | Output safety | Jailbreak detection, PII masking |
| `terradev ml phoenix` | LLM observability | Traces, spans, OTEL integration |

## Common Patterns

### Test Connection
Most ML services support `--test` flag to verify connectivity:
```bash
terradev ml <service> --test
```

### Setup Instructions
Services without credentials will display setup instructions automatically.

### Async Operations
All service operations use `asyncio.run()` for async service calls.

### Credential Management
Credentials are loaded from `~/.terradev/credentials.json` via `TerradevAPI._provider_creds()`.
