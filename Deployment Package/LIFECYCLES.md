# Terradev CLI Lifecycles (v5.7.10)

Complete end-to-end workflows covering nearly every command in the CLI. These have been updated to match the current CLI signatures and remove deprecated commands.

---

## Lifecycle 1 — First-Time Setup

**Goal:** Install Terradev, configure credentials, and run your first GPU provision.

### Phase 1: Installation

```bash
# Install via pip
pip install terradev

# Verify installation
terradev --version
```

### Phase 2: Interactive Onboarding

```bash
# Run interactive onboarding (auto-detects and configures providers)
terradev onboarding
```

Onboarding will:
- Scan for existing cloud credentials in `~/.aws`, `~/.config/gcloud`, `~/.kube`, etc.
- Prompt for missing credentials
- Configure provider API keys
- Set up cost tracking

### Phase 3: Manual Provider Configuration

If onboarding skipped a provider, configure it manually:

```bash
# Configure AWS
terradev configure --provider aws
# Follow prompts for AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

# Configure GCP
terradev configure --provider gcp
# Follow prompts for GOOGLE_APPLICATION_CREDENTIALS path

# Configure Azure
terradev configure --provider azure
# Follow prompts for AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID

# Configure RunPod
terradev configure --provider runpod
# Follow prompts for RUNPOD_API_KEY

# Configure Vast.ai
terradev configure --provider vastai
# Follow prompts for VAST_API_KEY

# Configure Lambda Labs
terradev configure --provider lambda
# Follow prompts for LAMBDA_API_KEY

# Configure Crusoe
terradev configure --provider crusoe
# Follow prompts for CRUSOE_API_KEY, CRUSOE_API_SECRET

# Configure CoreWeave
terradev configure --provider coreweave
# Follow prompts for KUBECONFIG path

# Configure Oracle Cloud
terradev configure --provider oracle
# Follow prompts for OCI_CONFIG_FILE path

# Configure Alibaba Cloud
terradev configure --provider alibaba
# Follow prompts for ALIBABA_ACCESS_KEY_ID, ALIBABA_ACCESS_KEY_SECRET, ALIBABA_REGION

# Configure OVHcloud
terradev configure --provider ovh
# Follow prompts for OVH_APPLICATION_KEY, OVH_APPLICATION_SECRET, OVH_CONSUMER_KEY, OVH_ENDPOINT

# Configure FluidStack
terradev configure --provider fluidstack
# Follow prompts for FLUIDSTACK_API_KEY

# Configure Hetzner
terradev configure --provider hetzner
# Follow prompts for HETZNER_API_TOKEN

# Configure SiliconFlow
terradev configure --provider siliconflow
# Follow prompts for SILICONFLOW_API_KEY

# Configure Hyperstack
terradev configure --provider hyperstack
# Follow prompts for HYPERSTACK_API_KEY

# Configure DigitalOcean
terradev configure --provider digitalocean
# Follow prompts for DIGITALOCEAN_API_TOKEN

# Configure InferX
terradev configure --provider inferx
# Follow prompts for INFERX_API_KEY
```

### Phase 4: Verify Configuration

```bash
# List configured providers
terradev configure --list

# Test provider connectivity
terradev status --providers
```

### Phase 5: First Quote

```bash
# Get pricing for A100 across all configured providers
terradev quote --gpu a100

# Get pricing for specific provider
terradev quote --providers runpod --gpu a100

# Get pricing with spot instances
terradev quote --gpu a100 --spot

# Get pricing with on-demand only
terradev quote --gpu a100 --on-demand
```

### Phase 6: First Provision

```bash
# Provision a single A100 on RunPod
terradev provision --providers runpod --gpu a100 --count 1

# Provision with specific instance type
terradev provision --providers aws --gpu a100 --instance-type p3.2xlarge --count 1

# Provision with spot pricing
terradev provision --providers vastai --gpu a100 --spot --count 1
```

### Phase 7: Check Status

```bash
# Check all provisions
terradev status

# Check specific provision
terradev status --instance-id <instance-id>

# Check provider status
terradev status --providers runpod
```

### Phase 8: SSH Into Instance

```bash
# SSH into provisioned instance
terradev ssh <instance-id>

# SSH with custom command
terradev ssh <instance-id> --command "nvidia-smi"
```

### Phase 9: Terminate Instance

```bash
# Terminate specific instance
terradev manage --terminate <instance-id>

# Terminate all instances
terradev manage --terminate-all
```

---

## Lifecycle 2 — Distributed Training with DeepSpeed

**Goal:** Launch a multi-node, multi-GPU training job with DeepSpeed, checkpointing, and auto-recovery.

### Phase 1: Prepare Training Script

```bash
# Create training script
cat > train.py << 'EOF'
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    model = torch.nn.Linear(1000, 1000).cuda()
    model = DDP(model)
    
    # Training loop with checkpointing
    for epoch in range(100):
        for batch in dataloader:
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'checkpoint_epoch_{epoch}.pt')
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
EOF
```

### Phase 2: Launch Distributed Training

```bash
# Launch 4-node, 8-GPU per node training job
terradev train \
  --script train.py \
  --nodes 4 \
  --gpus-per-node 8 \
  --gpu-type a100 \
  --provider runpod \
  --deepstage \
  --checkpoint-dir ./checkpoints \
  --checkpoint-interval 10 \
  --auto-recovery \
  --env LR=0.001 \
  --env BATCH_SIZE=32
```

### Phase 3: Monitor Training

```bash
# Check training status
terradev status --job-id <job-id>

# View logs
terradev logs --job-id <job-id> --follow

# Check GPU utilization
terradev ssh <instance-id> --command "nvidia-smi"
```

### Phase 4: Manage Checkpoints

```bash
# List checkpoints
terradev checkpoint list --job-id <job-id>

# Create manual checkpoint
terradev checkpoint create --job-id <job-id> --name manual-checkpoint

# Restore from checkpoint
terradev checkpoint restore --checkpoint-id <checkpoint-id> --job-id <job-id>

# Validate checkpoint integrity
terradev checkpoint validate <checkpoint-id> --detailed
```

### Phase 5: Handle Failures

```bash
# If training fails, auto-recovery will:
# 1. Detect failure
# 2. Find latest checkpoint
# 3. Relaunch from checkpoint
# 4. Resume training

# Manual recovery
terradev train \
  --script train.py \
  --nodes 4 \
  --gpus-per-node 8 \
  --gpu-type a100 \
  --provider runpod \
  --deepstage \
  --checkpoint-dir ./checkpoints \
  --restore-from <checkpoint-id>
```

### Phase 6: Cleanup

```bash
# Terminate training cluster
terradev manage --terminate-all

# Delete old checkpoints
terradev checkpoint delete <checkpoint-id> --force
```

---

## Lifecycle 3 — Inference Deployment with vLLM

**Goal:** Deploy a model for inference with vLLM, optimize for throughput, and manage multi-tenant serving.

### Phase 1: Prepare Model

```bash
# Download model
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/llama-2-7b
```

### Phase 2: Deploy Inference Endpoint

```bash
# Deploy with vLLM (auto-applies KV offloading, speculative decoding, sleep mode)
terradev infer-deploy \
  --model ./models/llama-2-7b \
  --gpu-type a100 \
  --provider runpod \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --port 8000
```

### Phase 3: Check Inference Status

```bash
# Check endpoint status
terradev infer-status --endpoint <endpoint-id>

# View logs
terradev logs --endpoint <endpoint-id> --follow

# Test endpoint
curl -X POST http://<endpoint-ip>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 100}'
```

### Phase 4: Optimize Performance

```bash
# Analyze workload
terradev vllm analyze http://<endpoint-ip>:8000 --duration 300 --metrics latency,throughput

# Auto-optimize
terradev vllm auto-optimize http://<endpoint-ip>:8000 --duration 300 --objective throughput --apply

# Benchmark
terradev vllm benchmark http://<endpoint-ip>:8000 --concurrent-requests 10 --duration 300
```

### Phase 5: Manage LoRA Adapters

```bash
# Add LoRA adapter
terradev lora add \
  --endpoint http://<endpoint-ip>:8000 \
  --name adapter-1 \
  --path ./adapters/adapter-1 \
  --priority high

# List adapters
terradev lora list --endpoint http://<endpoint-ip>:8000 --detailed

# Remove adapter
terradev lora remove --endpoint http://<endpoint-ip>:8000 --name adapter-1
```

### Phase 6: Sleep/Wake for Cost Savings

```bash
# Put endpoint to sleep (instant wake)
terradev infer-deploy --sleep <endpoint-id>

# Wake endpoint
terradev infer-deploy --wake <endpoint-id>
```

### Phase 7: Cleanup

```bash
# Terminate endpoint
terradev manage --terminate <endpoint-id>
```

---

## Lifecycle 4 — RAG Pipeline with Qdrant + Phoenix

**Goal:** Build a RAG pipeline with Qdrant vector DB and Phoenix observability.

### Phase 1: Deploy Qdrant

```bash
# Deploy Qdrant on Kubernetes
terradev qdrant k8s \
  --namespace vector-db \
  --replicas 3 \
  --storage-class ssd \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

### Phase 2: Create Collection

```bash
# Create collection
terradev qdrant create-collection \
  --name documents \
  --vector-size 384 \
  --distance Cosine \
  --hnsw-m 16 \
  --hnsw-ef 100
```

### Phase 3: Upsert Documents

```bash
# Prepare documents
cat > documents.json << 'EOF'
[
  {"id": "1", "text": "Document 1 content"},
  {"id": "2", "text": "Document 2 content"}
]
EOF

# Upsert documents
terradev qdrant upsert \
  --name documents \
  --file documents.json \
  --batch-size 100 \
  --format json
```

### Phase 4: Deploy Phoenix

```bash
# Deploy Phoenix on Kubernetes
terradev phoenix k8s \
  --namespace observability \
  --project rag-pipeline \
  --replicas 2
```

### Phase 5: Generate OTLP Environment

```bash
# Generate OTLP environment variables
terradev phoenix otlp-env \
  --endpoint http://phoenix:6006 \
  --project rag-pipeline \
  --service-name rag-service \
  --export-format env
```

### Phase 6: Search Vectors

```bash
# Search collection
terradev qdrant search \
  --name documents \
  --query "search query" \
  --limit 10 \
  --score-threshold 0.7
```

### Phase 7: View Traces

```bash
# List Phoenix projects
terradev phoenix projects

# View spans
terradev phoenix spans \
  --project rag-pipeline \
  --limit 100 \
  --filter "name = 'rag_query'"

# View specific trace
terradev phoenix trace <trace-id> --project rag-pipeline --detailed
```

### Phase 8: Cleanup

```bash
# Delete collection
terradev qdrant delete-collection --name documents
```

---

## Lifecycle 5 — Guardrails for Output Safety

**Goal:** Deploy NeMo Guardrails for output safety and jailbreak detection.

### Phase 1: Generate Guardrails Config

```bash
# Generate topical filtering config
terradev guardrails generate-config \
  --config-type topical \
  --output ./guardrails/topical.colang

# Generate jailbreak detection config
terradev guardrails generate-config \
  --config-type jailbreak \
  --output ./guardrails/jailbreak.colang

# Generate PII filtering config
terradev guardrails generate-config \
  --config-type pii \
  --output ./guardrails/pii.colang

# Generate fact-checking config
terradev guardrails generate-config \
  --config-type factcheck \
  --output ./guardrails/factcheck.colang
```

### Phase 2: Deploy Guardrails

```bash
# Deploy in standalone mode
terradev guardrails deploy \
  --config-path ./guardrails \
  --port 8080 \
  --mode standalone \
  --memory-backend memory
```

### Phase 3: Deploy in Sidecar Mode

```bash
# Deploy as sidecar to LLM endpoint
terradev guardrails sidecar \
  --llm-endpoint http://localhost:8000 \
  --deployment-mode sidecar \
  --memory-backend redis
```

### Phase 4: Test Guardrails

```bash
# Test connection
terradev guardrails test

# Test with chat interface
terradev guardrails chat \
  --config-id topical \
  --message "Test message" \
  --interactive
```

### Phase 5: Cleanup

```bash
# Terminate guardrails service
terradev manage --terminate <instance-id>
```

---

## Lifecycle 6 — Kubernetes Cluster with Karpenter

**Goal:** Set up a Kubernetes cluster with Karpenter for auto-provisioning GPU nodes.

### Phase 1: Create Kubernetes Cluster

```bash
# Create cluster
terradev k8s create production-cluster \
  --provider aws \
  --region us-east-1 \
  --node-type t3.medium \
  --nodes 3
```

### Phase 2: Install Karpenter

```bash
# Install Karpenter
terradev karpenter install --cluster-name production-cluster
```

### Phase 3: Create GPU NodePool

```bash
# Create nodepool for H100 GPUs
terradev karpenter create-nodepool \
  --gpu-type H100 \
  --cpu-limit 1000 \
  --memory-limit 1000Gi

# Create nodepool for A100 GPUs
terradev karpenter create-nodepool \
  --gpu-type A100 \
  --cpu-limit 500 \
  --memory-limit 500Gi
```

### Phase 4: Check Karpenter Status

```bash
# Check Karpenter status
terradev karpenter status

# List nodepools
terradev karpenter nodepools

# View GPU nodes
terradev karpenter gpu-nodes
```

### Phase 5: View Events

```bash
# View Karpenter events
terradev karpenter events

# View logs
terradev karpenter logs
```

### Phase 6: Delete NodePool

```bash
# Delete nodepool
terradev karpenter delete-nodepool gpu-nodes
```

### Phase 7: Cleanup

```bash
# Destroy cluster
terradev k8s destroy production-cluster
```

---

## Lifecycle 7 — HuggingFace Spaces Deployment

**Goal:** Deploy a model to HuggingFace Spaces with one-click deployment.

### Phase 1: Configure HuggingFace

```bash
# Configure HuggingFace credentials
terradev configure --provider huggingface
# Follow prompts for HF_TOKEN
```

### Phase 2: Create Space

```bash
# Create space (space_name is positional argument)
terradev hf-spaces create my-space \
  --model-id meta-llama/Llama-2-7b-chat-hf \
  --template llm \
  --hardware cpu-upgrade \
  --sdk gradio \
  --private \
  --org my-org
```

### Phase 3: List Spaces

```bash
# List all spaces
terradev hf-spaces list

# List spaces in organization
terradev hf-spaces list --org my-org
```

### Phase 4: Get Space Info

```bash
# Get space info (space_name is positional argument)
terradev hf-spaces info my-space
```

### Phase 5: Manage Hardware

```bash
# Upgrade hardware (space_name is positional argument)
terradev hf-spaces hardware my-space --hardware gpu-a10g-large
```

### Phase 6: Restart Space

```bash
# Restart space (space_name is positional argument)
terradev hf-spaces restart my-space
```

### Phase 7: Pause/Resume Space

```bash
# Pause space (space_name is positional argument)
terradev hf-spaces pause my-space

# Resume space (space_name is positional argument)
terradev hf-spaces resume my-space
```

### Phase 8: View Logs

```bash
# View logs (space_name is positional argument)
terradev hf-spaces logs my-space --follow
```

### Phase 9: Delete Space

```bash
# Delete space (space_name is positional argument)
terradev hf-spaces delete my-space
```

---

## Lifecycle 8 — Cost Optimization

**Goal:** Optimize GPU costs with spot instances, arbitrage, and budget management.

### Phase 1: Get Cost Analysis

```bash
# Analyze costs
terradev cost --analyze --days 30

# Analyze by provider
terradev cost --analyze --provider runpod --days 30
```

### Phase 2: Get Optimization Recommendations

```bash
# Get recommendations
terradev cost --optimize --recommend

# Apply recommendations
terradev cost --optimize --apply
```

### Phase 3: Simulate Changes

```bash
# Simulate cost changes
terradev cost --simulate --gpu a100 --spot --days 30
```

### Phase 4: Set Budget

```bash
# Set monthly budget
terradev cost --budget 1000 --monthly

# Set daily budget
terradev cost --budget 50 --daily
```

### Phase 5: Monitor Budget

```bash
# Check budget status
terradev cost --budget-status

# Set alerts
terradev cost --alert 80 --email admin@example.com
```

### Phase 6: Spot Arbitrage

```bash
# Find cheapest spot instances
terradev quote --gpu a100 --spot --arbitrage

# Auto-migrate to cheaper provider
terradev migrate --from runpod --to vastai --instance-id <instance-id>
```

---

## Lifecycle 9 — SGLang Optimization

**Goal:** Deploy and optimize SGLang for high-throughput inference.

### Phase 1: Install SGLang

```bash
# Install SGLang with optimization stack
terradev sglang install \
  --version 0.2.0 \
  --gpu-type a100 \
  --cuda-version 12.1 \
  --force
```

### Phase 2: Deploy SGLang

```bash
# Deploy SGLang endpoint
terradev infer-deploy \
  --model ./models/llama-2-7b \
  --gpu-type a100 \
  --provider runpod \
  --engine sglang \
  --port 8000
```

### Phase 3: Test SGLang

```bash
# Test SGLang endpoint
terradev sglang test \
  --endpoint http://<endpoint-ip>:8000 \
  --workload chat \
  --test-file prompts.json \
  --benchmark
```

### Phase 4: Generate Router

```bash
# Generate cache-aware router
terradev sglang router \
  --replicas 3 \
  --endpoint http://<endpoint-ip>:8000 \
  --output-file router.sh \
  --cache-type redis
```

### Phase 5: Cleanup

```bash
# Terminate endpoint
terradev manage --terminate <endpoint-id>
```

---

## Lifecycle 10 — MLflow Integration

**Goal:** Track experiments with MLflow.

### Phase 1: Configure MLflow

```bash
# Configure MLflow
terradev ml configure \
  --tracking-uri http://mlflow:5000 \
  --registry-uri http://mlflow:5000
```

### Phase 2: Track Training

```bash
# Launch training with MLflow tracking
terradev train \
  --script train.py \
  --nodes 2 \
  --gpus-per-node 4 \
  --gpu-type a100 \
  --provider runpod \
  --mlflow \
  --mlflow-experiment-name my-experiment
```

### Phase 3: View Experiments

```bash
# List experiments
terradev ml experiments

# View runs
terradev ml runs --experiment-id <experiment-id>
```

### Phase 4: Register Model

```bash
# Register model
terradev ml register \
  --model-name my-model \
  --run-id <run-id> \
  --model-version 1
```

### Phase 5: Deploy Model

```bash
# Deploy model from MLflow
terradev infer-deploy \
  --model-uri mlflow://my-model/1 \
  --gpu-type a100 \
  --provider runpod
```

---

## Lifecycle 11 — Weights & Biases Integration

**Goal:** Track experiments with Weights & Biases.

### Phase 1: Configure W&B

```bash
# Configure W&B
terradev wandb configure \
  --api-key <wandb-api-key> \
  --entity my-entity \
  --project my-project
```

### Phase 2: Track Training

```bash
# Launch training with W&B tracking
terradev train \
  --script train.py \
  --nodes 2 \
  --gpus-per-node 4 \
  --gpu-type a100 \
  --provider runpod \
  --wandb \
  --wandb-entity my-entity \
  --wandb-project my-project
```

### Phase 3: View Runs

```bash
# View runs
terradev wandb runs --project my-project

# View specific run
terradev wandb run <run-id>
```

### Phase 4: Create Dashboard

```bash
# Create dashboard
terradev wandb create-dashboard \
  --project my-project \
  --name my-dashboard

# Create Terradev-specific dashboard
terradev wandb create-terradev-dashboard \
  --project my-project
```

### Phase 5: Setup Alerts

```bash
# Setup alerts
terradev wandb setup-alerts \
  --metric loss \
  --threshold 0.1 \
  --operator greater-than

# Create Terradev-specific alerts
terradev wandb create-terradev-alerts \
  --project my-project
```

---

## Lifecycle 12 — LangSmith Integration

**Goal:** Track LLM traces with LangSmith.

### Phase 1: Configure LangSmith

```bash
# Configure LangSmith
terradev langsmith configure \
  --api-key <langsmith-api-key> \
  --project my-project
```

### Phase 2: Track Traces

```bash
# Launch inference with LangSmith tracking
terradev infer-deploy \
  --model ./models/llama-2-7b \
  --gpu-type a100 \
  --provider runpod \
  --langsmith \
  --langsmith-project my-project
```

### Phase 3: View Traces

```bash
# View traces
terradev langsmith traces --project my-project

# View specific trace
terradev langsmith trace <trace-id>
```

### Phase 4: Evaluate

```bash
# Evaluate traces
terradev langsmith evaluate \
  --project my-project \
  --dataset my-dataset
```

---

## Lifecycle 13 — Agentic Serving

**Goal:** Deploy multi-backend inference serving with model routing.

### Phase 1: Configure Agentic Serving

```bash
# Configure serving settings
terradev agentic-serving configure \
  --backend vllm \
  --cache-backend redis \
  --enable-prefix-caching
```

### Phase 2: Show Configuration

```bash
# Show current configuration
terradev agentic-serving show-config
```

### Phase 3: Generate Launch Args

```bash
# Show launch arguments
terradev agentic-serving launch-args
```

### Phase 4: Generate LMCache Environment

```bash
# Generate LMCache environment
terradev agentic-serving lmcache-env
```

### Phase 5: Deploy on Kubernetes

```bash
# Deploy on Kubernetes
terradev agentic-serving k8s \
  --namespace inference \
  --replicas 3
```

### Phase 6: Generate Helm Values

```bash
# Generate Helm values
terradev agentic-serving helm-values --output values.yaml
```

### Phase 7: Configure Model Router

```bash
# Configure model router
terradev model-router configure \
  --strong-model gpt-4 \
  --weak-model gpt-3.5-turbo \
  --cost-threshold 0.01
```

### Phase 8: Show Router Configuration

```bash
# Show router configuration
terradev model-router show-config
```

---

## Lifecycle 14 — Drift-Triggered Retraining

**Goal:** Automatically retrain models when drift is detected.

### Phase 1: Configure Retraining

```bash
# Configure drift detection
terradev retrain detect \
  --model my-model \
  --source phoenix-traces \
  --threshold 0.1 \
  --min-samples 1000
```

### Phase 2: Trigger Retraining

```bash
# Trigger retraining
terradev retrain trigger \
  --model my-model \
  --method lora \
  --eval-threshold 0.85 \
  --deploy canary \
  --auto-swap
```

### Phase 3: Deploy Retrained Model

```bash
# Deploy retrained model
terradev retrain deploy \
  --model my-model \
  --endpoint http://localhost:8000
```

---

## Lifecycle 15 — Advanced Workflow Orchestration

**Goal:** Record live workflows, manage lineage, set up triggers, and handle environment promotions.

### Phase 1: Register Artifacts

```bash
# Register dataset
terradev lineage register dataset my-dataset s3://my-bucket/dataset

# Register model
terradev lineage register model my-model s3://my-bucket/model

# Register checkpoint
terradev lineage register checkpoint my-checkpoint s3://my-bucket/checkpoint

# Register metrics
terradev lineage register metrics my-metrics s3://my-bucket/metrics

# Register config
terradev lineage register config my-config s3://my-bucket/config
```

### Phase 2: View Lineage Graph

```bash
# View lineage graph for artifact
terradev lineage graph my-model --direction both

# View upstream only
terradev lineage graph my-model --direction up

# View downstream only
terradev lineage graph my-model --direction down
```

### Phase 3: Show Production Artifacts

```bash
# Show all production artifacts
terradev lineage production

# Show production models only
terradev lineage production --type model
```

### Phase 4: Show Artifact Details

```bash
# Show artifact details
terradev lineage show my-model --env prod
```

### Phase 5: Compare Versions

```bash
# Compare two versions
terradev lineage diff v1.0 v2.0
```

### Phase 6: Trace Artifacts

```bash
# Trace from checkpoint
terradev lineage trace --checkpoint my-checkpoint

# Trace from execution
terradev lineage trace --execution my-execution
```

### Phase 7: Auto-Track Lineage

```bash
# Auto-track lineage for pipeline
terradev lineage auto \
  --pipeline my-pipeline \
  --env dev \
  --triggered-by manual
```

### Phase 8: Add Input to Execution

```bash
# Add input artifact
terradev lineage add-input my-execution dataset my-dataset

# Add model input
terradev lineage add-input my-execution model my-model

# Add config input
terradev lineage add-input my-execution config my-config

# Add checkpoint input
terradev lineage add-input my-execution checkpoint my-checkpoint
```

### Phase 9: Add Output to Execution

```bash
# Add model output
terradev lineage add-output my-execution model my-new-model

# Add checkpoint output
terradev lineage add-output my-execution checkpoint my-new-checkpoint

# Add metrics output
terradev lineage add-output my-execution metrics my-metrics

# Add evaluation output
terradev lineage add-output my-execution evaluation my-evaluation
```

### Phase 10: Complete Execution

```bash
# Complete execution
terradev lineage complete my-execution --status completed

# Mark as failed
terradev lineage complete my-execution --status failed
```

### Phase 11: Export Lineage

```bash
# Export lineage as JSON
terradev lineage export --format json --model my-model --env prod

# Export lineage as CSV
terradev lineage export --format csv --model my-model --env prod
```

### Phase 12: Record Workflow

```bash
# Start recording workflow
terradev record start --name my-workflow --output-dir ./recordings

# Run your commands...

# Stop recording and export as pipeline
terradev record stop --name my-workflow --export final.yaml --output-dir ./recordings
```

### Phase 13: Create Triggers

```bash
# Create event-based trigger
terradev triggers create my-trigger my-pipeline \
  --type event \
  --event dataset_landed \
  --env dev

# Create schedule-based trigger
terradev triggers create my-trigger my-pipeline \
  --type schedule \
  --schedule "0 0 * * 0" \
  --env staging

# Create condition-based trigger
terradev triggers create my-trigger my-pipeline \
  --type condition \
  --condition "drift_score > 0.1" \
  --env prod
```

### Phase 14: List Triggers

```bash
# List all triggers
terradev triggers list
```

### Phase 15: Enable/Disable Triggers

```bash
# Enable trigger
terradev triggers enable my-trigger

# Disable trigger
terradev triggers disable my-trigger
```

### Phase 16: Fire Event

```bash
# Fire event manually
terradev triggers fire dataset_landed --data '{"path": "s3://bucket/dataset"}' --source manual
```

### Phase 17: List Environments

```bash
# List all environments
terradev environments list

# List specific environment
terradev environments list --env staging
```

### Phase 18: Promote Artifact

```bash
# Promote artifact
terradev environments promote my-model --from staging --to prod --user admin
```

### Phase 19: Approve Promotion

```bash
# Approve promotion
terradev environments approve <promotion-id> --user admin
```

### Phase 20: View Promotion History

```bash
# View promotion history
terradev environments history --artifact my-model
```

---

## Lifecycle 16 — Local GPU Scanning (NEW)

**Goal:** Discover and pool on-prem/workstation GPUs alongside cloud providers.

### Phase 1: Scan Local GPUs

```bash
# Scan local GPUs
terradev local scan

# Scan with detailed output
terradev local scan --detailed
```

### Phase 2: Scan Remote Host via SSH

```bash
# Scan remote host
terradev local scan \
  --host 192.168.1.50 \
  --user ubuntu \
  --key ~/.ssh/id_rsa

# Scan multiple hosts
terradev local scan \
  --host 192.168.1.50 \
  --user ubuntu \
  --key ~/.ssh/id_rsa \
  --host 192.168.1.51 \
  --user ubuntu \
  --key ~/.ssh/id_rsa
```

### Phase 3: Register GPU

```bash
# Scan and register GPU
terradev local scan --register --name workstation-4090
```

---

## Lifecycle 17 — LoRAX Multi-LoRA Inference (NEW)

**Goal:** Deploy LoRAX server for serving thousands of fine-tuned models on a single GPU with dynamic adapter loading.

### Phase 1: Import Adapter from HuggingFace

```bash
# Import LoRA adapter from HuggingFace
terradev lora peft import \
  -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k \
  --local-name gsm8k-adapter

# Import and register in Terradev registry
terradev lora peft import \
  -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k \
  --local-name gsm8k-adapter \
  --register \
  --base-model mistralai/Mistral-7B-Instruct-v0.1

# Import from private repo
terradev lora peft import \
  -a username/private-adapter \
  --token hf_xxx
```

### Phase 2: List Local Adapters

```bash
# List all imported adapters
terradev lora peft list

# Validate adapter structure
terradev lora peft validate -p ~/.terradev/peft_adapters/username--adapter
```

### Phase 3: Deploy LoRAX Server (Docker)

```bash
# Deploy LoRAX with Docker
terradev lora lorax deploy \
  -m mistralai/Mistral-7B-Instruct-v0.1 \
  --docker

# Deploy with quantization
terradev lora lorax deploy \
  -m mistralai/Mistral-7B-Instruct-v0.1 \
  --quantization bitsandbytes \
  --max-loras 16 \
  --docker
```

### Phase 4: Deploy LoRAX Server (Kubernetes)

```bash
# Deploy LoRAX to Kubernetes
terradev lora lorax deploy \
  -m mistralai/Mistral-7B-Instruct-v0.1 \
  --k8s \
  --namespace lorax

# Install via Helm
helm install lorax ./clusters/lorax-template/helm \
  -f clusters/lorax-template/helm/values-lorax.yaml \
  --set model.id=mistralai/Mistral-7B-Instruct-v0.1 \
  --set maxLoras=8
```

### Phase 5: Test LoRAX Server

```bash
# Test server connectivity
terradev lora lorax test --host localhost --port 8080

# Test with custom host/port
terradev lora lorax test --host 10.0.0.1 --port 8080
```

### Phase 6: Load Adapters

```bash
# Load adapter from HuggingFace
terradev lora lorax load-adapter \
  -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k

# Load local adapter with custom name
terradev lora lorax load-adapter \
  -a ~/.terradev/peft_adapters/gsm8k-adapter \
  --adapter-name gsm8k-adapter \
  --host localhost --port 8080
```

### Phase 7: List Loaded Adapters

```bash
# List all loaded adapters
terradev lora lorax list-adapters

# List from specific server
terradev lora lorax list-adapters --host 10.0.0.1 --port 8080
```

### Phase 8: Generate Text

```bash
# Generate with base model
terradev lora lorax generate \
  -p "What is 2+2?" \
  --max-tokens 64

# Generate with adapter
terradev lora lorax generate \
  -p "Natalia sold clips to 48 of her friends..." \
  -a gsm8k-adapter \
  --temperature 0.7
```

### Phase 9: Unload Adapter

```bash
# Unload adapter
terradev lora lorax unload-adapter -a gsm8k-adapter
```

### Phase 10: Sync Registry

```bash
# Sync Terradev registry with LoRAX state
terradev lora lorax sync-registry

# Sync specific adapter
terradev lora lorax sync-registry -a gsm8k-adapter
```

### Phase 11: Delete Local Adapter

```bash
# Delete imported adapter
terradev lora peft delete -a username/adapter
```

### Phase 12: Cleanup

```bash
# Terminate LoRAX server
docker stop <container-id>

# Or delete Kubernetes deployment
helm uninstall lorax -n lorax
```

### Phase 4: View Hybrid Pool

```bash
# View hybrid pool (local + cloud)
terradev local pool
```

### Phase 5: Provision from Hybrid Pool

```bash
# Provision from hybrid pool (will use local if available)
terradev provision --gpu rtx4090 --count 1 --pool hybrid
```

---

## Lifecycle 18 — HuggingFace PEFT Import (NEW)

**Goal:** Import, validate, and manage LoRA adapters from HuggingFace using the PEFT library.

### Phase 1: Import Adapter from HuggingFace

```bash
# Import public adapter
terradev lora peft import \
  -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k

# Import with custom local name
terradev lora peft import \
  -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k \
  --local-name gsm8k-adapter

# Import from private repository
terradev lora peft import \
  -a username/private-adapter \
  --token hf_xxx
```

**What happens:**
- Downloads adapter from HuggingFace Hub using PEFT library
- Auto-detects rank, alpha, and target modules from adapter config
- Stores adapter in `~/.terradev/peft_adapters/username--adapter/`
- Validates adapter structure and compatibility

### Phase 2: Import and Register in Terradev Registry

```bash
# Import and register in one step
terradev lora peft import \
  -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k \
  --local-name gsm8k-adapter \
  --register \
  --base-model mistralai/Mistral-7B-Instruct-v0.1

# Import with custom rank override
terradev lora peft import \
  -a username/adapter \
  --register \
  --base-model meta-llama/Llama-2-7b-hf \
  --rank 128
```

**What happens:**
- Downloads and validates adapter (same as Phase 1)
- Registers adapter in Terradev LoRA registry (`~/.terradev/lora_registry.db`)
- Creates version entry with auto-generated version ID
- Associates with base model for compatibility tracking
- Captures metadata for cost attribution

### Phase 3: List Local Adapters

```bash
# List all imported adapters
terradev lora peft list
```

**Output:**
```
Imported PEFT Adapters:
  vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k
    Local name: gsm8k-adapter
    Base model: mistralai/Mistral-7B-Instruct-v0.1
    Rank: 64
    Alpha: 16
    Target modules: q_proj, v_proj
    Path: ~/.terradev/peft_adapters/vineetsharma--qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k
    Registered: Yes
```

### Phase 4: Validate Adapter Structure

```bash
# Validate adapter
terradev lora peft validate \
  -p ~/.terradev/peft_adapters/vineetsharma--qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k
```

**What happens:**
- Checks adapter_config.json exists and is valid
- Verifies adapter weights are present
- Validates target modules match base model
- Checks rank and alpha values are consistent
- Reports any structural issues

**Output:**
```
Validation: PASSED
  Config: ✓ Valid
  Weights: ✓ Present (8 files)
  Target modules: ✓ Compatible with base model
  Rank: 64
  Alpha: 16
```

### Phase 5: Use with LoRAX

```bash
# Deploy LoRAX server
terradev lora lorax deploy \
  -m mistralai/Mistral-7B-Instruct-v0.1 \
  --docker

# Load imported adapter
terradev lora lorax load-adapter \
  -a ~/.terradev/peft_adapters/gsm8k-adapter \
  --adapter-name gsm8k-adapter

# Generate with adapter
terradev lora lorax generate \
  -p "Solve: Natalia sold clips to 48 of her friends..." \
  -a gsm8k-adapter
```

### Phase 6: Use with vLLM

```bash
# Deploy vLLM with LoRA support
terradev ml vllm --start \
  --instance-ip <ip> \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --enable-lora \
  --lora-modules gsm8k=~/.terradev/peft_adapters/gsm8k-adapter

# Generate with adapter
curl -X POST http://<ip>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.1",
    "prompt": "What is 2+2?",
    "max_tokens": 64
  }'
```

### Phase 7: Delete Local Adapter

```bash
# Delete imported adapter
terradev lora peft delete -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k

# Delete by local name
terradev lora peft delete -a gsm8k-adapter
```

**What happens:**
- Removes adapter from `~/.terradev/peft_adapters/`
- Removes from registry if previously registered
- Frees disk space
- Updates local adapter index

### Phase 8: Batch Import Multiple Adapters

```bash
# Import multiple adapters for a tenant
terradev lora peft import \
  -a customer-a/adapter-1 \
  --local-name customer-a-adapter-1 \
  --register \
  --base-model meta-llama/Llama-2-7b-hf

terradev lora peft import \
  -a customer-a/adapter-2 \
  --local-name customer-a-adapter-2 \
  --register \
  --base-model meta-llama/Llama-2-7b-hf

terradev lora peft import \
  -a customer-a/adapter-3 \
  --local-name customer-a-adapter-3 \
  --register \
  --base-model meta-llama/Llama-2-7b-hf

# List all tenant adapters
terradev lora peft list
```

### Phase 9: Troubleshooting

```bash
# If import fails due to missing token
terradev lora peft import \
  -a username/private-adapter \
  --token hf_xxx

# If adapter is incompatible with base model
terradev lora peft validate -p ~/.terradev/peft_adapters/username--adapter

# If rank detection fails, specify manually
terradev lora peft import \
  -a username/adapter \
  --rank 64

# If disk space is low, delete unused adapters
terradev lora peft list
terradev lora peft delete -a unused-adapter
```

---

## Lifecycle 17 — Langfuse Integration (NEW)

**Goal:** Track LLM observability with Langfuse.

### Phase 1: Configure Langfuse

```bash
# Configure Langfuse
terradev langfuse configure \
  --public-key <public-key> \
  --secret-key <secret-key> \
  --host https://cloud.langfuse.com
```

### Phase 2: Test Connection

```bash
# Test connection
terradev langfuse test
```

### Phase 3: View Traces

```bash
# List traces
terradev langfuse traces

# View specific trace
terradev langfuse trace <trace-id>
```

### Phase 4: Manage Scores

```bash
# List scores
terradev langfuse scores

# Add score
terradev langfuse score <trace-id> --name accuracy --value 0.95
```

### Phase 5: Manage Datasets

```bash
# List datasets
terradev langfuse datasets

# Export training data
terradev langfuse export-training-data --dataset my-dataset
```

### Phase 6: Quality Metrics

```bash
# View quality metrics
terradev langfuse quality
```

### Phase 7: Generate OTLP Environment

```bash
# Generate OTLP environment
terradev langfuse otel-env
```

### Phase 8: Deploy on Kubernetes

```bash
# Deploy Langfuse on Kubernetes
terradev langfuse k8s --namespace observability
```

---

## Lifecycle 18 — Databricks Integration (NEW)

**Goal:** Integrate with Databricks MLOps platform.

### Phase 1: Configure Databricks

```bash
# Configure Databricks
terradev databricks configure
# Follow prompts for DATABRICKS_HOST, DATABRICKS_TOKEN
```

### Phase 2: Test Connection

```bash
# Test connection
terradev databricks test
```

### Phase 3: List Jobs

```bash
# List jobs
terradev databricks jobs
```

### Phase 4: Run Job

```bash
# Run job
terradev databricks run --job-id <job-id>
```

### Phase 5: Check Run Status

```bash
# Check run status
terradev databricks run-status <run-id>
```

### Phase 6: List Clusters

```bash
# List clusters
terradev databricks clusters
```

### Phase 7: List Serving Endpoints

```bash
# List serving endpoints
terradev databricks serving-endpoints
```

### Phase 8: Deploy Model

```bash
# Deploy model to serving endpoint
terradev databricks deploy-model \
  --endpoint-name my-endpoint \
  --model-name my-model \
  --model-version 1 \
  --workload-size Medium \
  --scale-to-zero
```

### Phase 9: Query Endpoint

```bash
# Query serving endpoint
terradev databricks query --endpoint-name my-endpoint
```

### Phase 10: MLflow Operations

```bash
# MLflow operations
terradev databricks mlflow
```

---

## Lifecycle 19 — MCP Server (NEW)

**Goal:** Run Terradev MCP server for Claude Desktop/Cursor/Windsurf.

### Phase 1: Install MCP Server

```bash
# Install MCP server
terradev mcp install
```

### Phase 2: Start MCP Server

```bash
# Start MCP server
terradev mcp serve --port 3000
```

### Phase 3: List Tools

```bash
# List available tools
terradev mcp list-tools
```

### Phase 4: Configure Claude Desktop

```bash
# Add to Claude Desktop config
# ~/.config/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "terradev": {
      "command": "terradev",
      "args": ["mcp", "serve"]
    }
  }
}
```

---

## Summary

These 19 lifecycles cover the complete Terradev CLI v5.1.5 functionality:

- **Core Infrastructure:** Provisioning, management, execution
- **Training & Distributed:** DeepSpeed, checkpointing, auto-recovery
- **Inference & Serving:** vLLM, SGLang, LoRA adapters, sleep mode
- **ML Integrations:** MLflow, W&B, LangSmith, Langfuse, Databricks
- **RAG & Observability:** Qdrant, Phoenix, Guardrails
- **Kubernetes:** Cluster management, Karpenter, GPU operators
- **Cost Optimization:** Spot instances, arbitrage, budget management
- **Workflow Orchestration:** Lineage, triggers, environments, recording
- **Local & Hybrid:** On-prem GPU discovery, hybrid pooling
- **MCP Server:** Claude Desktop/Cursor/Windsurf integration

All commands are production-ready and fully documented in `COMPLETE_COMMAND_REFERENCE.md`.
