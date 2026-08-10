# Complete Terradev CLI Command Reference

**All commands and subcommands for Terradev CLI v6.0.6**

---

## Main Commands (217+ MCP Tools)

### **Core Infrastructure Commands**

#### **provision** - Provision GPU instances across multiple providers

```bash
terradev provision [OPTIONS]
Options:
  -g, --gpu-type TEXT             GPU type (required: A100, H100, RTX4090, L40S,
                                  etc.)  [required]
  -n, --count INTEGER             Number of instances to provision (default: 1)
  --max-price FLOAT               Maximum price per hour in USD (e.g., 2.50)
  -p, --providers TEXT            Filter to specific providers (multiple
                                  allowed, e.g., runpod,vastai)
  --parallel INTEGER              Max parallel deploy threads (default: 6)
  --dry-run                       Show allocation plan without launching
                                  instances
  --type [training|inference]     Workload type (affects spot/on-demand auto-
                                  selection)
  --model-name TEXT               Model to deploy (for inference workloads)
  --endpoint-name TEXT            Endpoint name (for inference workloads)
  --min-workers INTEGER           Minimum workers for auto-scaling (inference)
  --max-workers INTEGER           Maximum workers for auto-scaling (inference)
  --spot                          Force spot instances (60-80% savings, 2-min
                                  termination notice)
  --on-demand                     Force on-demand instances (guaranteed
                                  availability, higher cost)
  --spot-strategy [aggressive|cheapest|balanced|conservative|safe]
                                  Spot instance strategy: aggressive/cheapest,
                                  balanced, conservative/safe (most stable)
  --backend [vllm|sglang|dynamo|tensorrt_llm|llmd]
                                  Inference backend: vllm (default), sglang,
                                  dynamo, tensorrt_llm, llmd
  --prefer-local                  Prefer local GPUs from your pool over cloud
                                  providers
  --agents INTEGER                Number of concurrent agents. Triggers multi-
                                  agent KV VRAM planner.
  --context TEXT                  Context window per agent (e.g. 32k, 128k).
                                  Used with --agents.
  --sharing-topology [broadcast|star|chain|none]
                                  KV cache sharing topology between agents
                                  (default: broadcast).
  --dtype [fp16|fp8]              KV cache dtype. fp8 halves KV VRAM
                                  requirement.
  --select TEXT                   Select instance by number or keyword: 1-N,
                                  cheapest, cheapest-spot, cheapest-secure,
                                  SXM4-40GB, SXM4-80GB, 80GB PCIe
  --auto                          Auto-select cheapest instance without
                                  prompting (CI/CD mode)
  --help                          Show this message and exit.
```



#### **status** - Show current status of all instances and usage

```bash
terradev status [OPTIONS]
Options:
  -f, --format [table|json]  Output format: table (default) or json
  --live                     Query providers for live instance status (slower
                             but accurate)
  --help                     Show this message and exit.
```



#### **quote** - Get real-time quotes from all providers

```bash
terradev quote [OPTIONS]
Options:
  -g, --gpu-type TEXT   GPU type to quote (A100, H100, RTX4090, L40S, etc.)
  -p, --providers TEXT  Filter to specific providers (multiple allowed, e.g.,
                        runpod,vastai)
  --parallel INTEGER    Number of parallel queries (default: 6)
  -r, --region TEXT     Filter by region (e.g., us-east-1, eu-west-1)
  -q, --quick           Show quick provision command for best quote
  --include-local       Include local GPUs from your registered pool (priced at
                        $0/hr)
  --help                Show this message and exit.
```



#### **availability** - Show GPU availability/stock status

```bash
terradev availability [OPTIONS]
Options:
  -g, --gpu-type TEXT   GPU type filter (shows all if omitted)
  -w, --window INTEGER  Lookback window in hours (default: 24)
  --help                Show this message and exit.
```



#### **manage** - Manage provisioned instances via real-time APIs

```bash
terradev manage [OPTIONS]
Options:
  -i, --instance-id TEXT          Instance ID (from terradev status)  [required]
  -a, --action [status|stop|start|terminate]
                                  Action: status (default), stop, start,
                                  terminate
  --help                          Show this message and exit.
```



#### **execute** - Execute commands on provisioned instances

```bash
terradev execute [OPTIONS]
Options:
  -i, --instance-id TEXT  Instance ID (from terradev status)  [required]
  --cmd TEXT              Command to execute on the instance  [required]
  --async-exec            Run command asynchronously (returns immediately)
  --help                  Show this message and exit.
```



#### **cleanup** - Clean up unused resources and temporary files

```bash
terradev cleanup [OPTIONS]
Options:
  --help  Show this message and exit.
```



---

### **Training & Distributed Computing Commands**

#### **train** - Launch a distributed training job

```bash
terradev train [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



#### **train-resume** - Resume a training job from checkpoint
```bash
terradev train-resume [OPTIONS]

Options:
  --checkpoint-id TEXT          Checkpoint ID to resume from
  --job-id TEXT                 Job ID to resume
  --script-args TEXT            Additional script arguments
```

#### **train-status** - Show training job status, GPU-hours, cost
```bash
terradev train-status [OPTIONS]

Options:
  --job-id TEXT                 Filter by specific job ID
  --format [table|json]         Output format [default: table]
  --detailed                    Show detailed metrics
  --gpu-utilization             Show GPU utilization
  --cost-breakdown              Show cost breakdown
```

#### **train-stop** - Stop a running training job
```bash
terradev train-stop [OPTIONS]

Options:
  --job-id TEXT                 Job ID to stop (required)
  --force                       Force stop without confirmation
  --save-checkpoint             Save final checkpoint before stopping
```

#### **checkpoint** - Manage distributed checkpoints

```bash
terradev checkpoint [OPTIONS] {list|restore|promote|delete}
Options:
  -j, --job-id TEXT         Job ID  [required]
  --step INTEGER            Checkpoint step
  --checkpoint-id TEXT      Checkpoint ID
  --dest TEXT               Destination path (for promote)
  -f, --format [json|text]
  --help                    Show this message and exit.
```



#### **stage** - Compress, chunk, and pre-position datasets

```bash
terradev stage [OPTIONS]
Options:
  -d, --dataset TEXT              Dataset path, S3 URI, GCS URI, HTTP URL, or
                                  HuggingFace name  [required]
  --target-regions TEXT           Comma-separated target regions
  --compression [auto|zstd|gzip|none]
                                  Compression algorithm (default: auto)
  --plan-only                     Show staging plan without executing
  --help                          Show this message and exit.
```



#### **preflight** - Run preflight hardware validation on GPU nodes

```bash
terradev preflight [OPTIONS]
Options:
  -n, --nodes TEXT          Node IPs (multiple allowed, empty = localhost)
  --ssh-user TEXT           SSH user (default: root)
  --ssh-key TEXT            SSH key path
  --from-provision TEXT     Use nodes from a provision group. "latest" = most
                            recent.
  --quick                   Quick GPU-only check (skip storage/NCCL)
  -f, --format [json|text]
  --help                    Show this message and exit.
```



---

### **Inference & Model Serving Commands**

#### **infer** - Deploy and manage inference endpoints

```bash
terradev infer [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



#### **infer-deploy** - Deploy inference endpoint
```bash
terradev infer-deploy [OPTIONS] MODEL_PATH

Options:
  -n, --name TEXT               Endpoint name (required)
  -p, --provider [runpod|vastai|lambda|baseten]
                                Provider
  -g, --gpu-type TEXT           GPU type (A100|H100|RTX4090)
  --min-workers INTEGER         Minimum workers
  --max-workers INTEGER         Maximum workers
  --idle-timeout INTEGER        Idle timeout in seconds
  --cost-optimize               Enable cost optimization
  --dry-run                     Show deployment plan without deploying
```

#### **infer-status** - Show inference endpoint health, latency, and failover status
```bash
terradev infer-status [OPTIONS]

Options:
  --check                       Run live health probes before showing status
  --endpoint TEXT               Filter by specific endpoint
  --detailed                    Show detailed metrics
  --format [table|json]         Output format [default: table]
```

#### **infer-failover** - Run health checks and auto-failover for inference endpoints
```bash
terradev infer-failover [OPTIONS]

Options:
  --dry-run                     Show what would happen without executing failover
  --endpoint TEXT               Specific endpoint to test
  --test-load INTEGER           Load test with N requests
```

#### **infer-route** - Find the best inference endpoint using routing strategies
```bash
terradev infer-route [OPTIONS]

Options:
  -m, --model TEXT              Filter by model name
  -s, --strategy [latency|cost|score]
                                Routing strategy [default: latency]
  --region TEXT                 Filter by region
  --provider TEXT               Filter by provider
```

#### **inferx** - InferX serverless inference platform

```bash
terradev inferx [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



#### **vllm** - vLLM optimization and management commands
```bash
terradev vllm [OPTIONS] COMMAND

Commands:
  optimize                      Generate optimized vLLM configurations
  auto-optimize                 Automatically optimize vLLM configuration
  analyze                       Analyze current vLLM server workload
  benchmark                     Benchmark vLLM endpoint performance
```

#### **sglang** - SGLang optimization and management with workload types
```bash
terradev sglang [OPTIONS] COMMAND

Commands:
  detect                        Auto-detect workload type and show optimization recommendations
  install                       Install SGLang with optimization stack
  optimize                      Auto-optimize SGLang configuration for workload type
  router                        Generate cache-aware router command for multi-replica deployments
  start                         Start optimized SGLang server
  test                          Test SGLang installation and configuration
```

#### **lora** - Manage LoRA adapters on a running vLLM server

```bash
terradev lora [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



---

### **Model Orchestration Commands**

#### **orchestrator-start** - Start the model orchestrator for multi-model serving
```bash
terradev orchestrator-start [OPTIONS]

Options:
  --port INTEGER                Orchestrator port [default: 8080]
  --models TEXT                 Comma-separated list of models to load
  --gpu-memory-fraction FLOAT   GPU memory fraction per model
  --auto-scale                  Enable auto-scaling
```

#### **orchestrator-status** - Get orchestrator and model status
```bash
terradev orchestrator-status [OPTIONS]

Options:
  --detailed                    Show detailed model information
  --format [table|json]         Output format [default: table]
```

#### **orchestrator-load** - Load a model into GPU memory
```bash
terradev orchestrator-load [OPTIONS] MODEL_PATH

Options:
  --name TEXT                   Model name (required)
  --priority [high|medium|low] Loading priority [default: medium]
  --gpu-memory-fraction FLOAT   GPU memory fraction
```

#### **orchestrator-infer** - Test inference with a model
```bash
terradev orchestrator-infer [OPTIONS]

Options:
  --model TEXT                  Model name (required)
  --prompt TEXT                 Input prompt
  --max-tokens INTEGER          Maximum tokens to generate
  --temperature FLOAT           Temperature for sampling
```

#### **orchestrator-evict** - Evict a model from GPU memory
```bash
terradev orchestrator-evict [OPTIONS]

Options:
  --model TEXT                  Model name to evict (required)
  --force                       Force eviction without confirmation
```

#### **orchestrator-register** - Register a model with the orchestrator
```bash
terradev orchestrator-register [OPTIONS] MODEL_PATH

Options:
  --name TEXT                   Model name (required)
  --type [llm|embedding|vision] Model type
  --description TEXT            Model description
```

---

### **Warm Pool Management Commands**

#### **warm-pool-start** - Start the warm pool manager for intelligent pre-warming
```bash
terradev warm-pool-start [OPTIONS]

Options:
  --config-file TEXT            Configuration file path
  --models TEXT                 Comma-separated list of models to pre-warm
  --gpu-types TEXT              GPU types to pre-warm
  --regions TEXT                Target regions
```

#### **warm-pool-status** - Get warm pool manager status
```bash
terradev warm-pool-status [OPTIONS]

Options:
  --detailed                    Show detailed warm pool information
  --format [table|json]         Output format [default: table]
```

#### **warm-pool-register** - Register a model with the warm pool manager
```bash
terradev warm-pool-register [OPTIONS] MODEL_PATH

Options:
  --name TEXT                   Model name (required)
  --priority [high|medium|low] Pre-warming priority [default: medium]
  --regions TEXT                Target regions
  --gpu-types TEXT              Preferred GPU types
```

---

### **Optimization & Cost Management Commands**

#### **optimize** - Multi-dimensional optimization: cost + performance

```bash
terradev optimize [OPTIONS]
Options:
  --instance-id TEXT  Optimize specific instance ID
  --auto-apply        Automatically apply all recommended optimizations
  --help              Show this message and exit.
```



#### **analytics** - Show cost analytics from the cost tracking database

```bash
terradev analytics [OPTIONS]
Options:
  -d, --days INTEGER         Number of days to analyze (default: 7)
  -f, --format [table|json]  Output format
  --help                     Show this message and exit.
```



#### **reliability** - Show provider reliability scores and error rates

```bash
terradev reliability [OPTIONS]
Options:
  -p, --provider TEXT   Filter to a single provider
  -w, --window INTEGER  Lookback window in hours (default: 720 = 30d)
  --ranking             Show ranked leaderboard
  --help                Show this message and exit.
```



#### **budget-optimize** - Find optimal deployment under budget constraints

```bash
terradev budget-optimize [OPTIONS]
Options:
  --gpu-type TEXT      GPU type  [required]
  --budget FLOAT       Budget constraint ($/hr)  [required]
  --gpu-count INTEGER  Number of GPUs
  --hours FLOAT        Estimated runtime in hours
  --region TEXT        Preferred region
  --workload TEXT      Workload type
  --help               Show this message and exit.
```



#### **cost-scaler-start** - Start the cost-aware scaling manager
```bash
terradev cost-scaler-start [OPTIONS]

Options:
  --config-file TEXT            Configuration file
  --budget FLOAT                Budget limit
  --scale-down-threshold FLOAT Scale down threshold
  --scale-up-threshold FLOAT    Scale up threshold
```

#### **cost-scaler-status** - Get cost scaler status and recommendations
```bash
terradev cost-scaler-status [OPTIONS]

Options:
  --detailed                    Show detailed recommendations
  --format [table|json]         Output format [default: table]
```

#### **cost-scaler-model-details** - Get cost details for a specific model
```bash
terradev cost-scaler-model-details [OPTIONS] MODEL_NAME

Options:
  --region TEXT                 Region
  --gpu-type TEXT               GPU type
  --include-spot                Include spot pricing
```

---

### **ML Platform Integration Commands**

#### **ml** - ML Platform Integration Commands

```bash
terradev ml [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



#### **phoenix** - Arize Phoenix LLM trace observability
```bash
terradev phoenix [OPTIONS] COMMAND

Commands:
  projects                      List Phoenix projects
  spans                         View and filter traces
  trace                         View specific trace
  analyze                       Analyze traces
  export                        Export traces
  dashboard                     Manage dashboards
  alert                         Manage alerts
  integrate                     Integrate with other tools
  k8s                           Deploy Phoenix on Kubernetes
  otlp-env                      Generate OTLP environment variables
```

#### **qdrant** - Qdrant vector database — collections, embeddings
```bash
terradev qdrant [OPTIONS] COMMAND

Commands:
  create-collection              Create new collection
  list-collections              List all collections
  info                          Get collection information
  delete                        Delete collection
  upsert                        Add documents to collection
  search                        Search vectors
  batch-upsert                  Batch add documents
  batch-search                  Batch search
  hybrid-search                 Hybrid search with filters
  optimize                      Optimize collection
  benchmark                     Benchmark performance
  replicate                     Configure replication
  monitor                       Monitor performance
  backup                        Backup collection
```

#### **guardrails** - NeMo Guardrails — LLM output safety
```bash
terradev guardrails [OPTIONS] COMMAND

Commands:
  deploy                        Deploy guardrails service
  sidecar                       Deploy in sidecar mode
  generate-config               Generate configuration files
  test                          Test guardrails
  chat                          Test with chat interface
  add-policy                    Add custom policy
  test-suite                    Run test suite
  benchmark                     Benchmark performance
  integrate                     Integrate with LLM providers
  monitor                       Monitor guardrails
  analytics                     Analytics and reporting
```

---

### **Kubernetes & Container Orchestration Commands**

#### **k8s** - Kubernetes cluster management with GPU operators

```bash
terradev k8s [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



#### **k8s create** - Create multi-cloud Kubernetes GPU cluster

```bash
terradev k8s create [OPTIONS] CLUSTER_NAME
Options:
  -g, --gpu TEXT                  GPU type (H100, A100, L40)  [required]
  -n, --count INTEGER             Number of GPU nodes  [required]
  --max-price FLOAT               Maximum price per hour
  --multi-cloud                   Use multi-cloud provisioning
  --prefer-spot                   Prefer spot instances
  --aws-region TEXT               AWS region
  --gcp-region TEXT               GCP region
  --control-plane [eks|gke|self-hosted]
                                  Control plane type
  --help                          Show this message and exit.
```



#### **k8s destroy** - Destroy Kubernetes cluster

```bash
terradev k8s destroy [OPTIONS] CLUSTER_NAME
Options:
  --help  Show this message and exit.
```



#### **helm-generate** - Generate Helm charts from Terradev workloads

```bash
terradev helm-generate [OPTIONS]
Options:
  --workload TEXT      Workload type (training, inference, cost-optimized, high-
                       performance, moe-inference, rag, vllm-optimized)
  --gpu-type TEXT      GPU type (A100, H100, V100, L4, L40S, RTX 4090, T4, etc.)
  --image TEXT         Docker image  [required]
  --gpu-count INTEGER  Number of GPUs
  --memory INTEGER     Memory in GB
  --storage INTEGER    Storage in GB
  --budget FLOAT       Budget constraint ($/hr)
  --region TEXT        Preferred region
  --port INTEGER       Expose port(s) via Service (repeatable)
  -s, --stack TEXT     Stack integrations: qdrant, phoenix, guardrails
                       (repeatable)
  -o, --output TEXT    Output directory
  --name TEXT          Chart name
  --dry-run            Show chart config without generating
  --help               Show this message and exit.
```



#### **gitops** - GitOps automation and infrastructure as code

```bash
terradev gitops [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



---

### **Enterprise & Security Commands**

#### **configure** - Configure cloud provider credentials

```bash
terradev configure [OPTIONS]
Options:
  -p, --provider TEXT  Configure specific provider (e.g., runpod, vastai, aws)
  --help               Show this message and exit.
```



#### **sso** - Enterprise SSO authentication (Enterprise tier)

```bash
terradev sso [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



#### **integrations** - Show status of observability & ML integrations

```bash
terradev integrations [OPTIONS]
Options:
  --export-wandb-script  Print a W&B setup script for remote instances
  --help                 Show this message and exit.
```



#### **datadog** - Datadog FinOps monitoring — metrics, dashboards
```bash
terradev datadog [OPTIONS] COMMAND

Commands:
  test                          Test Datadog integration
  dashboard                     Manage dashboards
  monitor                       Manage monitors
  alert                         Manage alerts
  export                        Export configurations
```

#### **upgrade** - Upgrade your Terradev subscription via Stripe
```bash
terradev upgrade [OPTIONS]

Options:
  -t, --tier [research_plus|enterprise|enterprise_plus]
                                Target tier
  --annual                      Annual billing (discount)
  --preview                     Show upgrade preview
```

---

### **Monitoring & Observability Commands**

#### **monitor** - Monitor GPU utilization, training metrics, costs

```bash
terradev monitor [OPTIONS]
Options:
  -j, --job-id TEXT         Job ID to monitor
  -n, --nodes TEXT          Node IPs
  --ssh-user TEXT           SSH user
  --ssh-key TEXT            SSH key path
  --from-provision TEXT     Use nodes from a provision group. "latest" = most
                            recent.
  -l, --log-path TEXT       Training log file to parse
  -i, --interval FLOAT      Snapshot interval in seconds
  --count INTEGER           Number of snapshots (0 = continuous)
  --cost-rate FLOAT         Cost per GPU-hour in USD
  -f, --format [json|text]
  --help                    Show this message and exit.
```



#### **manifests** - List cached manifests and versions
```bash
terradev manifests [OPTIONS]

Options:
  --list                        List all cached manifests
  --version TEXT                Filter by version
  --provider TEXT               Filter by provider
  --cleanup                     Clean old manifests
```

#### **setup** - Get setup instructions for any cloud provider

```bash
terradev setup [OPTIONS] {runpod|vastai|lambda_labs|tensordock|crusoe|bas
Options:
  -q, --quick  Show quick setup summary
  --help       Show this message and exit.
```



#### **smart-deploy** - Smart deployment with automatic optimization

```bash
terradev smart-deploy [OPTIONS]
Options:
  -w, --workload [training|inference|cost-optimized|high-performance]
                                  Workload type (maps to Karpenter provisioner)
  --image TEXT                    Docker image (e.g. pytorch/pytorch:latest)
                                  [required]
  --cmd TEXT                      Command to run inside the container
  -G, --gpu-count INTEGER         Number of GPUs (default: per workload profile)
  -b, --budget FLOAT              Max $/hr budget  forces spot if < $2/hr
  -n, --namespace TEXT            Kubernetes namespace
  --name TEXT                     Job/Deployment name (auto-generated if
                                  omitted)
  -e, --env TEXT                  Environment variables KEY=VALUE
  --mount TEXT                    Volume mounts host:container
  -o, --option INTEGER            Deployment option index from smart-deploy
  --memory INTEGER                Memory in GB
  -s, --storage INTEGER           Storage in GB
  --hours FLOAT                   Estimated runtime in hours
  --region TEXT                   Preferred region
  --dry-run                       Show recommendation without deploying
  --help                          Show this message and exit.
```



---

### **Utility Commands**

#### **job** - Run Terradev job from YAML configuration

```bash
terradev job [OPTIONS] JOB_FILE
Options:
  --optimize TEXT  Optimization criteria (cost, latency, balanced)
  --help           Show this message and exit.
```



#### **run** - Provision a GPU instance, deploy a Docker container

```bash
terradev run [OPTIONS]
Options:
  -g, --gpu TEXT     GPU type (required: A100, H100, RTX4090, L40S, etc.)
                     [required]
  --image TEXT       Docker image (required: e.g., pytorch/pytorch:latest)
                     [required]
  --cmd TEXT         Command to run inside the container (e.g., "python
                     train.py")
  -m, --mount TEXT   Mount local path:container path (multiple allowed, e.g.,
                     ./data:/workspace/data)
  --port INTEGER     Ports to expose (multiple allowed, e.g., 8000 for HTTP)
  -e, --env TEXT     Environment variables KEY=VALUE (multiple allowed, e.g.,
                     WANDB_KEY=xxx)
  --max-price FLOAT  Maximum price per hour in USD (e.g., 2.50)
  --providers TEXT   Filter to specific providers (multiple allowed, e.g.,
                     runpod,vastai)
  --keep-alive       Keep instance running after command completes (for serving)
  --dry-run          Show deployment plan without executing
  --help             Show this message and exit.
```



#### **rollback** - EXPLICIT ROLLBACK (versioned manifests)
```bash
terradev rollback [OPTIONS]

Options:
  --revision TEXT               Target revision
  --cluster TEXT                Target cluster
  --force                       Force rollback
  --dry-run                     Show rollback plan
```

#### **up** - CLI-native provisioning with manifest cache
```bash
terradev up [OPTIONS] WORKLOAD_FILE

Options:
  --cache                       Use cached manifests
  --refresh-cache               Refresh manifest cache
  --dry-run                     Show deployment plan
```

#### **onboarding** - Run the interactive onboarding flow

```bash
terradev onboarding [OPTIONS]
Options:
  --force  Force onboarding even if already configured
  --help   Show this message and exit.
```



---

##  **ML Subcommands**

### **ml wandb** - Weights & Biases integration

```bash
terradev ml wandb [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



### **ml mlflow** - MLflow integration
```bash
terradev ml mlflow [OPTIONS] COMMAND

Commands:
  test                          Test MLflow integration
  list-experiments              List experiments
  create-experiment             Create new experiment
  runs                          List runs
  models                        List models
  deploy                        Deploy model
```

### **ml phoenix** - Phoenix integration

```bash
terradev ml phoenix [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



### **ml qdrant** - Qdrant integration

```bash
terradev ml qdrant [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



### **ml guardrails** - Guardrails integration

```bash
terradev ml guardrails [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



---

##  **SGLang Subcommands**

### **sglang optimize** - Auto-optimize SGLang configuration
```bash
terradev sglang optimize [OPTIONS] MODEL

Options:
  --workload [agentic-chat|high-throughput|low-latency|moe|pd-disaggregated|structured-output|hardware-specific]
                                Workload type
  --gpu-type TEXT               GPU type
  --output-file TEXT            Save configuration to file
  --dry-run                     Show optimization plan
```

### **sglang detect** - Auto-detect workload type
```bash
terradev sglang detect [OPTIONS] MODEL

Options:
  --description TEXT            User description of use case
  --sample-prompts TEXT         Sample prompts
  --output-format [json|table]  Output format
```

### **sglang start** - Start optimized SGLang server
```bash
terradev sglang start [OPTIONS] MODEL

Options:
  --config-file TEXT            Configuration file
  --port INTEGER               Server port [default: 30000]
  --host TEXT                   Server host [default: 0.0.0.0]
  --workload TEXT               Workload type
  --gpu-type TEXT               GPU type
```

### **sglang test** - Test SGLang installation and configuration
```bash
terradev sglang test [OPTIONS]

Options:
  --endpoint TEXT               SGLang endpoint
  --workload TEXT               Test specific workload
  --test-file TEXT              Test file with prompts
  --benchmark                   Run benchmark tests
```

### **sglang install** - Install SGLang with optimization stack
```bash
terradev sglang install [OPTIONS]

Options:
  --version TEXT                SGLang version
  --gpu-type TEXT               GPU type for optimizations
  --cuda-version TEXT           CUDA version
  --force                       Force reinstallation
```

### **sglang router** - Generate cache-aware router command
```bash
terradev sglang router [OPTIONS]

Options:
  --replicas INTEGER            Number of replicas
  --endpoint TEXT               SGLang endpoint
  --output-file TEXT            Save router script
  --cache-type [redis|memory]   Cache backend
```

---

##  **vLLM Subcommands**

### **vllm optimize** - Generate optimized vLLM configurations
```bash
terradev vllm optimize [OPTIONS] MODEL

Options:
  --workload [chat|completion|embedding|batch]
                                Workload type
  --gpu-type TEXT               GPU type
  --max-batch-size INTEGER      Maximum batch size
  --tensor-parallel-size INTEGER Tensor parallel size
  --output-file TEXT            Save configuration
```

### **vllm auto-optimize** - Automatically optimize vLLM configuration
```bash
terradev vllm auto-optimize [OPTIONS] ENDPOINT

Options:
  --duration INTEGER            Analysis duration in seconds
  --objective [latency|throughput|memory]
                                Optimization objective
  --apply                       Apply optimizations automatically
```

### **vllm analyze** - Analyze current vLLM server workload
```bash
terradev vllm analyze [OPTIONS] ENDPOINT

Options:
  --duration INTEGER            Analysis duration
  --metrics TEXT                Metrics to analyze
  --output-file TEXT            Save analysis report
```

### **vllm benchmark** - Benchmark vLLM endpoint performance
```bash
terradev vllm benchmark [OPTIONS] ENDPOINT

Options:
  --concurrent-requests INTEGER Concurrent requests
  --duration INTEGER            Benchmark duration
  --prompts-file TEXT           File with test prompts
  --output-format [json|csv]     Output format
```

---

##  **LoRA Subcommands**

### **lora add** - Add LoRA adapter

```bash
terradev lora add [OPTIONS]
Options:
  -e, --endpoint TEXT  vLLM endpoint  [required]
  -n, --name TEXT      Adapter name (becomes the model name in API requests)
                       [required]
  --path TEXT          Path to adapter weights  [required]
  --api-key TEXT       vLLM API key
  --register           Also register in central registry
  --base-model TEXT    Base model (required with --register)
  --rank INTEGER       LoRA rank (default: 64)
  --help               Show this message and exit.
```



### **lora remove** - Remove LoRA adapter

```bash
terradev lora remove [OPTIONS]
Options:
  -e, --endpoint TEXT  vLLM endpoint  [required]
  -n, --name TEXT      Adapter name to unload  [required]
  --api-key TEXT       vLLM API key
  --help               Show this message and exit.
```



### **lora list** - List loaded adapters

```bash
terradev lora list [OPTIONS]
Options:
  -e, --endpoint TEXT  vLLM endpoint (e.g. http://10.0.0.1:8000)  [required]
  --api-key TEXT       vLLM API key
  --registry           Show registry state instead of live endpoint
  --help               Show this message and exit.
```



### **lora status** - Show adapter status
```bash
terradev lora status [OPTIONS]

Options:
  -e, --endpoint TEXT           vLLM endpoint (required)
  -name TEXT                    Filter by adapter name
  --metrics                     Show performance metrics
```

### **lora update** - Update adapter configuration
```bash
terradev lora update [OPTIONS]

Options:
  -e, --endpoint TEXT           vLLM endpoint (required)
  -n, --name TEXT               Adapter name (required)
  -p, --path TEXT               New adapter path
  --priority TEXT               New priority
```

### **lora benchmark** - Benchmark adapter performance
```bash
terradev lora benchmark [OPTIONS]

Options:
  -e, --endpoint TEXT           vLLM endpoint (required)
  -n, --name TEXT               Adapter name
  --prompts-file TEXT           Test prompts file
  --duration INTEGER            Benchmark duration
```

---

##  **Checkpoint Subcommands**

### **checkpoint list** - List all checkpoints
```bash
terradev checkpoint list [OPTIONS]

Options:
  --job-id TEXT                 Filter by job ID
  --format [table|json]         Output format
  --detailed                    Show detailed information
  --verify                      Verify checkpoint integrity
```

### **checkpoint create** - Create new checkpoint
```bash
terradev checkpoint create [OPTIONS]

Options:
  --job-id TEXT                 Job ID (required)
  --name TEXT                   Checkpoint name
  --force                       Force checkpoint creation
  --compress                    Compress checkpoint
```

### **checkpoint restore** - Restore from checkpoint
```bash
terradev checkpoint restore [OPTIONS]

Options:
  --checkpoint-id TEXT          Checkpoint ID (required)
  --job-id TEXT                 Job ID
  --verify                      Verify before restore
```

### **checkpoint validate** - Validate checkpoint integrity
```bash
terradev checkpoint validate [OPTIONS] CHECKPOINT_ID

Options:
  --detailed                    Show detailed validation results
  --repair                      Attempt repair if corrupted
```

### **checkpoint delete** - Delete checkpoint
```bash
terradev checkpoint delete [OPTIONS] CHECKPOINT_ID

Options:
  --force                       Force deletion
  --backup                      Backup before deletion
```

### **checkpoint status** - Show checkpoint status
```bash
terradev checkpoint status [OPTIONS]

Options:
  --job-id TEXT                 Job ID
  --checkpoint-id TEXT          Checkpoint ID
  --detailed                    Show detailed status
```

---

##  **InferX Subcommands**

### **inferx deploy** - Deploy serverless endpoint

```bash
terradev inferx deploy [OPTIONS]
Options:
  --model TEXT                    Model ID or HuggingFace model name  [required]
  --image TEXT                    Docker image for model
  --gpu-type TEXT                 GPU type
  --gpu-memory INTEGER            GPU memory in GB
  --max-concurrency INTEGER       Maximum concurrent requests
  --framework TEXT                Model framework
  --openai-compatible / --no-openai-compatible
                                  OpenAI-compatible API
  --timeout INTEGER               Request timeout in seconds
  --help                          Show this message and exit.
```



### **inferx status** - Check endpoint health
```bash
terradev inferx status [OPTIONS]

Options:
  --endpoint TEXT               Endpoint name
  --detailed                    Show detailed status
  --metrics                     Show performance metrics
```

### **inferx failover** - Run failover tests
```bash
terradev inferx failover [OPTIONS]

Options:
  --endpoint TEXT               Endpoint name
  --test-load INTEGER           Test load
  --duration INTEGER            Test duration
```

### **inferx cost-analysis** - Analyze costs
```bash
terradev inferx cost-analysis [OPTIONS]

Options:
  --endpoint TEXT               Endpoint name
  --days INTEGER                Number of days [default: 30]
  --format [table|json]         Output format
```

---

##  **Phoenix Subcommands**

### **phoenix projects** - List Phoenix projects
```bash
terradev phoenix projects [OPTIONS]

Options:
  --format [table|json]         Output format
  --limit INTEGER               Limit results
```

### **phoenix spans** - View and filter traces
```bash
terradev phoenix spans [OPTIONS]

Options:
  --project TEXT                 Project name (required)
  --limit INTEGER               Limit results [default: 100]
  --filter TEXT                 Filter expression
  --format [table|json]         Output format
```

### **phoenix trace** - View specific trace
```bash
terradev phoenix trace [OPTIONS] TRACE_ID

Options:
  --project TEXT                 Project name
  --format [table|json]         Output format
  --detailed                    Show detailed trace
```

### **phoenix analyze** - Analyze traces
```bash
terradev phoenix analyze [OPTIONS]

Options:
  --project TEXT                 Project name (required)
  --metric [latency|error-rate|throughput]
                                Metric to analyze
  --time-range TEXT             Time range
```

### **phoenix k8s** - Deploy Phoenix on Kubernetes
```bash
terradev phoenix k8s [OPTIONS]

Options:
  --namespace TEXT              Namespace [default: observability]
  --project TEXT                Project name
  --replicas INTEGER            Number of replicas [default: 2]
  --storage-class TEXT          Storage class
```

### **phoenix otlp-env** - Generate OTLP environment variables
```bash
terradev phoenix otlp-env [OPTIONS]

Options:
  --endpoint TEXT               Phoenix endpoint [default: http://phoenix:6006]
  --project TEXT                Project name
  --service-name TEXT           Service name
  --export-format [env|yaml]    Export format
```

---

##  **Qdrant Subcommands**

### **qdrant create-collection** - Create new collection
```bash
terradev qdrant create-collection [OPTIONS]

Options:
  --name TEXT                   Collection name (required)
  --vector-size INTEGER         Vector size (required)
  --distance [Cosine|Euclidean|DotProduct]
                                Distance metric [default: Cosine]
  --hnsw-m INTEGER              HNSW M parameter
  --hnsw-ef INTEGER             HNSW ef parameter
```

### **qdrant upsert** - Add documents to collection
```bash
terradev qdrant upsert [OPTIONS]

Options:
  --name TEXT                   Collection name (required)
  --file TEXT                   File with documents
  --batch-size INTEGER          Batch size [default: 1000]
  --format [json|csv]          File format
```

### **qdrant search** - Search vectors
```bash
terradev qdrant search [OPTIONS]

Options:
  --name TEXT                   Collection name (required)
  --query TEXT                  Search query
  --limit INTEGER               Limit results [default: 10]
  --score-threshold FLOAT       Score threshold
  --filter TEXT                 Filter expression
```

### **qdrant k8s** - Deploy Qdrant on Kubernetes
```bash
terradev qdrant k8s [OPTIONS]

Options:
  --namespace TEXT              Namespace [default: vector-db]
  --replicas INTEGER            Number of replicas [default: 3]
  --storage-class TEXT          Storage class [default: ssd]
  --embedding-model TEXT        Embedding model
```

---

##  **Guardrails Subcommands**

### **guardrails deploy** - Deploy guardrails service
```bash
terradev guardrails deploy [OPTIONS]

Options:
  --config-path TEXT            Configuration path
  --port INTEGER                Service port [default: 8080]
  --mode [standalone|sidecar]   Deployment mode
  --memory-backend [memory|redis] Memory backend
```

### **guardrails sidecar** - Deploy in sidecar mode
```bash
terradev guardrails sidecar [OPTIONS]

Options:
  --llm-endpoint TEXT           LLM endpoint (required)
  --deployment-mode TEXT        Deployment mode [default: sidecar]
  --memory-backend TEXT         Memory backend [default: redis]
```

### **guardrails generate-config** - Generate configuration files
```bash
terradev guardrails generate-config [OPTIONS]

Options:
  --output TEXT                 Output directory [default: ./guardrails]
  --enable-topical              Enable topical filtering
  --enable-jailbreak            Enable jailbreak detection
  --enable-pii                  Enable PII filtering
  --enable-factcheck            Enable fact checking
```

### **guardrails test** - Test guardrails
```bash
terradev guardrails test [OPTIONS]

Options:
  --config-id TEXT              Configuration ID
  --message TEXT                Test message
  --test-suite TEXT             Test suite
```

### **guardrails chat** - Test with chat interface
```bash
terradev guardrails chat [OPTIONS]

Options:
  --config-id TEXT              Configuration ID (required)
  --message TEXT                Message to test
  --interactive                 Interactive chat mode
```

---

##  **GitOps Subcommands**

### **gitops init** - Initialize GitOps repository

```bash
terradev gitops init [OPTIONS]
Options:
  --provider [github|gitlab|bitbucket|azure_devops]
                                  Git provider  [required]
  --repo, --repository TEXT       Repository name (format: owner/repo)
                                  [required]
  --tool [argocd|flux]            GitOps tool
  --cluster TEXT                  Cluster name  [required]
  --git-url TEXT                  Git repository URL (auto-generated if not
                                  provided)
  --git-token TEXT                Git access token
  --namespace TEXT                Namespace for GitOps tools
  --auto-sync / --no-auto-sync    Enable automatic synchronization
  --prune / --no-prune            Enable resource pruning
  --help                          Show this message and exit.
```



### **gitops bootstrap** - Bootstrap GitOps tool on cluster

```bash
terradev gitops bootstrap [OPTIONS]
Options:
  --tool [argocd|flux]  GitOps tool  [required]
  --cluster TEXT        Cluster name  [required]
  --namespace TEXT      Namespace for GitOps tools
  --help                Show this message and exit.
```



### **gitops validate** - Validate GitOps configuration

```bash
terradev gitops validate [OPTIONS]
Options:
  --dry-run / --apply  Dry run validation or apply changes
  --cluster TEXT       Cluster name for validation
  --environment TEXT   Environment to validate
  --help               Show this message and exit.
```



### **gitops sync** - Sync changes to cluster

```bash
terradev gitops sync [OPTIONS]
Options:
  --cluster TEXT        Cluster name  [required]
  --environment TEXT    Environment to sync
  --tool [argocd|flux]  GitOps tool
  --help                Show this message and exit.
```



### **gitops rollback** - Rollback to previous revision
```bash
terradev gitops rollback [OPTIONS]

Options:
  --cluster TEXT                Cluster name
  --revision TEXT               Target revision
  --force                       Force rollback
```

---

##  **Qdrant Subcommands**

### **qdrant** - Vector database for RAG
```bash
terradev qdrant [SUBCOMMAND]

Subcommands:
  test           Test Qdrant connection
  collections    List all collections
  create-collection  Create a new collection
  info           Get collection info
  count          Count points in collection
  k8s            Deploy Qdrant on Kubernetes
```

### **qdrant create-collection**
```bash
terradev qdrant create-collection [OPTIONS]

Options:
  -n, --name TEXT               Collection name
  -e, --embedding-model TEXT    Embedding model (e.g., sentence-transformers/all-MiniLM-L6-v2)
  --dimension INTEGER          Vector dimension (auto-detected from model)
```

---

##  **Phoenix Subcommands**

### **phoenix** - Arize Phoenix LLM tracing
```bash
terradev phoenix [SUBCOMMAND]

Subcommands:
  test           Test Phoenix connection
  projects       List Phoenix projects
  spans          Query spans
  trace          Get specific trace
  otel-env       Generate OTLP environment variables
  snippet        Generate tracing code snippet
  k8s            Deploy Phoenix on Kubernetes
```

### **phoenix spans**
```bash
terradev phoenix spans [OPTIONS]

Options:
  -p, --project TEXT           Project name
  -f, --filter TEXT            SpanQuery DSL filter
  -l, --limit INTEGER          Max spans to return
```

---

##  **Guardrails Subcommands**

### **guardrails** - NeMo Guardrails output safety
```bash
terradev guardrails [SUBCOMMAND]

Subcommands:
  test           Test Guardrails connection
  chat           Chat with guardrails
  generate-config  Generate Colang config
  k8s            Deploy Guardrails on Kubernetes
```

### **guardrails generate-config**
```bash
terradev guardrails generate-config [OPTIONS]

Options:
  -c, --config-type TEXT       Config type (topical, jailbreak, pii, factcheck)
  -o, --output PATH            Output file path
```

---

##  **Langfuse Subcommands**

### **langfuse** - LLM observability
```bash
terradev langfuse [SUBCOMMAND]

Subcommands:
  configure      Set Langfuse credentials
  test           Test connection
  traces         List traces
  trace          Get specific trace
  scores         List scores
  score          Add a score
  datasets       List datasets
  export-training-data  Export training data
  quality        Quality metrics
  otel-env       Generate OTLP environment
  k8s            Deploy on Kubernetes
```

### **langfuse configure**
```bash
terradev langfuse configure [OPTIONS]

Options:
  --public-key TEXT            Langfuse public key
  --secret-key TEXT            Langfuse secret key
  --host TEXT                  Langfuse host URL
```

---

##  **Databricks Subcommands**

### **databricks** - Databricks MLOps integration
```bash
terradev databricks [SUBCOMMAND]

Subcommands:
  configure      Set Databricks credentials
  test           Test connection
  jobs           List jobs
  run            Run a job
  run-status     Get job run status
  clusters       List clusters
  serving-endpoints  List serving endpoints
  deploy-model   Deploy model to serving endpoint
  query          Query a serving endpoint
  mlflow         MLflow operations
```

### **databricks deploy-model**
```bash
terradev databricks deploy-model [OPTIONS]

Options:
  --endpoint-name TEXT         Endpoint name (required)
  --model-name TEXT            Model name (required)
  --model-version TEXT         Model version
  --workload-size TEXT         Small|Medium|Large
  --scale-to-zero              Enable scale-to-zero
  --no-scale-to-zero           Disable scale-to-zero
```

---

##  **Retrain Subcommands**

### **retrain** - Automated model retraining

```bash
terradev retrain [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



### **retrain detect** - Check for drift without triggering a retrain.

```bash
terradev retrain detect [OPTIONS]
Options:
  -m, --model TEXT          Model identifier  [required]
  --phoenix-endpoint TEXT
  --phoenix-project TEXT
  --baseline FLOAT
  --threshold FLOAT
  --min-samples INTEGER
  -f, --format [json|text]
  --help                    Show this message and exit.
```



---

##  **Migrate Subcommands**

### **migrate** - Cross-cloud migration

```bash
terradev migrate [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



---

##  **Eval Subcommands**

### **eval** - Model evaluation

```bash
terradev eval [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



---

##  **Triggers Subcommands**

### **triggers** - Event-driven automation

```bash
terradev triggers [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



---

##  **Environments Subcommands**

### **environments** - Environment management

```bash
terradev environments [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



---

##  **Lineage Subcommands**

### **lineage** - Data/model lineage tracking

```bash
terradev lineage [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```



---

##  **HF Spaces Subcommands**

### **huggingface spaces** - Create, list, manage, and delete HuggingFace Spaces
```bash
terradev huggingface spaces [SUBCOMMAND]

Subcommands:
  create     Create a new Space with auto-generated app
  list       List HuggingFace Spaces
  info       Get Space details
  delete     Delete a Space
  restart    Restart a Space (factory reboot)
  pause      Pause a running Space (stops billing)
  resume     Resume a paused Space
  hardware   Show or change hardware tier
  logs       Show Space build/run logs
```

---


---

##  **New & Updated in v6.0.4**

### **agent** - Provision and manage heterogeneous agent fleets.

```bash
terradev agent [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```

### **agent agentic-serving** - Agentic inference serving  KV cache TTL, prefix caching, LMCache, priority

```bash
terradev agent agentic-serving [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```

### **agent agentic-serving configure** - Configure agentic inference serving settings.

```bash
terradev agent agentic-serving configure [OPTIONS]
Options:
  --engine [vllm|sglang]          Inference engine
  --model TEXT
  --tp INTEGER                    Tensor parallel size
  --max-model-len INTEGER
  --gpu-mem FLOAT
  --lmcache / --no-lmcache        Enable LMCache KV offload
  --lmcache-backend [cpu|disk|redis]
  --disaggregation / --no-disaggregation
                                  Prefill-decode disaggregation
  --help                          Show this message and exit.
```

### **agent agentic-serving helm-values** - Print Helm values for agentic inference deployment.

```bash
terradev agent agentic-serving helm-values [OPTIONS]
Options:
  -f, --format [json|yaml]
  --help                    Show this message and exit.
```

### **agent agentic-serving k8s** - Print K8s deployment manifests for agentic inference.

```bash
terradev agent agentic-serving k8s [OPTIONS]
Options:
  -n, --namespace TEXT  K8s namespace
  --help                Show this message and exit.
```

### **agent agentic-serving launch-args** - Print engine launch arguments for copy-paste.

```bash
terradev agent agentic-serving launch-args [OPTIONS]
Options:
  --help  Show this message and exit.
```

### **agent agentic-serving lmcache-env** - Print LMCache environment variables.

```bash
terradev agent agentic-serving lmcache-env [OPTIONS]
Options:
  --help  Show this message and exit.
```

### **agent agentic-serving show-config** - Show current agentic serving configuration.

```bash
terradev agent agentic-serving show-config [OPTIONS]
Options:
  -f, --format [json|text]
  --help                    Show this message and exit.
```

### **agent cost** - Show real-time cost breakdown for a fleet by tier.

```bash
terradev agent cost [OPTIONS]
Options:
  --fleet-id TEXT        Fleet ID  [required]
  --format [table|json]
  --help                 Show this message and exit.
```

### **agent deploy** - Provision a heterogeneous agent fleet across all tiers simultaneously.

```bash
terradev agent deploy [OPTIONS]
Options:
  -n, --agents INTEGER            Number of concurrent agent loops
  -m, --model TEXT                Model to serve
  --reasoning [instant|thinking]
  --topology PATH                 Path to agent-fleet.yaml spec file
  --planner-gpu TEXT              Reasoning tier GPU type
  --planner-count INTEGER         Reasoning tier instance count
  --worker-gpu TEXT               Decode tier GPU type
  --worker-count INTEGER          Decode tier instance count
  --cpu-cores INTEGER             vCPU count for CPU tools tier
  -p, --providers TEXT            Cloud providers to use (e.g. runpod vastai)
  --max-price FLOAT               Max price per GPU/hr in USD
  --dry-run                       Show allocation plan without provisioning
  --format [table|json]
  --help                          Show this message and exit.
```

### **agent langchain** - LangChain integration with workflows, LangGraph, and SGLang.

```bash
terradev agent langchain [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```

### **agent langchain create-langgraph** - Create a LangGraph workflow.

```bash
terradev agent langchain create-langgraph [OPTIONS] GRAPH_NAME
Options:
  --help  Show this message and exit.
```

### **agent langchain create-pipeline** - Create an SGLang pipeline.

```bash
terradev agent langchain create-pipeline [OPTIONS] PIPELINE_NAME
Options:
  --help  Show this message and exit.
```

### **agent langchain create-workflow** - Create a LangChain workflow.

```bash
terradev agent langchain create-workflow [OPTIONS] WORKFLOW_NAME
Options:
  --help  Show this message and exit.
```

### **agent langchain test** - Test connection to LangChain service.

```bash
terradev agent langchain test [OPTIONS]
Options:
  --help  Show this message and exit.
```

### **agent langgraph** - LangGraph workflow orchestration with monitoring.

```bash
terradev agent langgraph [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```

### **agent langgraph create-workflow** - Create a LangGraph workflow.

```bash
terradev agent langgraph create-workflow [OPTIONS] WORKFLOW_NAME
Options:
  -t, --type [orchestrator-worker|evaluator-optimizer]
                                  Workflow type  [required]
  --help                          Show this message and exit.
```

### **agent langgraph deploy** - Deploy a workflow.

```bash
terradev agent langgraph deploy [OPTIONS] WORKFLOW_NAME
Options:
  --help  Show this message and exit.
```

### **agent langgraph status** - Get workflow status.

```bash
terradev agent langgraph status [OPTIONS] WORKFLOW_ID
Options:
  --help  Show this message and exit.
```

### **agent langgraph test** - Test connection to LangGraph service.

```bash
terradev agent langgraph test [OPTIONS]
Options:
  --help  Show this message and exit.
```

### **agent letta** - Letta stateful agents with long-horizon memory management.

```bash
terradev agent letta [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```

### **agent letta chat** - Send a message to a Letta agent.

```bash
terradev agent letta chat [OPTIONS]
Options:
  -a, --agent-id TEXT          Agent ID  [required]
  -m, --message TEXT           Message to send  [required]
  --environment [cloud|local]  Letta environment
  -f, --format [json|text]
  --help                       Show this message and exit.
```

### **agent letta create** - Create a new stateful Letta agent.

```bash
terradev agent letta create [OPTIONS]
Options:
  -n, --name TEXT              Agent name  [required]
  -m, --model TEXT             Model to use
  --human TEXT                 Human memory block value
  --persona TEXT               Persona memory block value
  --memory-blocks TEXT         JSON list of memory blocks [{"label": ...,
                               "value": ...}]
  --vector-db TEXT             Vector DB connection string or JSON config for
                               agent memory
  --skill FILE                 Path to a skill.md to embed as an agent memory
                               block
  --environment [cloud|local]  Letta environment
  -f, --format [json|text]
  --help                       Show this message and exit.
```

### **agent letta delete** - Delete a Letta agent.

```bash
terradev agent letta delete [OPTIONS]
Options:
  -a, --agent-id TEXT          Agent ID  [required]
  --environment [cloud|local]  Letta environment
  --help                       Show this message and exit.
```

### **agent letta list** - List Letta agents.

```bash
terradev agent letta list [OPTIONS]
Options:
  --environment [cloud|local]  Letta environment
  -f, --format [json|text]
  --help                       Show this message and exit.
```

### **agent letta remember** - Teach a Letta agent a durable fact.

```bash
terradev agent letta remember [OPTIONS]
Options:
  -a, --agent-id TEXT          Agent ID  [required]
  -t, --text TEXT              Fact to remember  [required]
  -l, --label TEXT             Memory block label
  --environment [cloud|local]  Letta environment
  --help                       Show this message and exit.
```

### **agent letta status** - Show the state of a Letta agent.

```bash
terradev agent letta status [OPTIONS]
Options:
  -a, --agent-id TEXT          Agent ID  [required]
  --environment [cloud|local]  Letta environment
  -f, --format [json|text]
  --help                       Show this message and exit.
```

### **agent list** - List all known agent fleets.

```bash
terradev agent list [OPTIONS]
Options:
  --format [table|json]
  --help                 Show this message and exit.
```

### **agent plan** - Plan a heterogeneous agent fleet without provisioning.

```bash
terradev agent plan [OPTIONS]
Options:
  -n, --agents INTEGER            Number of concurrent agent loops to provision
                                  for  [required]
  -m, --model TEXT                Model to serve across the fleet
  --reasoning [instant|thinking]  Reasoning mode: instant (faster) or thinking
                                  (extended CoT, 45-67% more output tokens)
  --planner-gpu TEXT              Override reasoning tier GPU type (e.g.
                                  H100_SXM)
  --planner-count INTEGER         Override reasoning tier instance count
  --worker-gpu TEXT               Override decode tier GPU type (e.g.
                                  A100_SXM_80)
  --worker-count INTEGER          Override decode tier instance count
  --cpu-cores INTEGER             vCPU count for CPU tools tier instances
  --format [table|json]           Output format
  --help                          Show this message and exit.
```

### **agent scale** - Scale a single fleet tier up or down without affecting other tiers.

```bash
terradev agent scale [OPTIONS]
Options:
  --fleet-id TEXT                 Fleet ID  [required]
  --tier [reasoning|decode|cpu_tools]
                                  Tier to scale  [required]
  --count INTEGER                 New instance count for this tier  [required]
  -p, --providers TEXT            Providers to use for scale-out instances
  --help                          Show this message and exit.
```

### **agent skill** - Manage skill.md files and attach them to Letta agents.

```bash
terradev agent skill [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```

### **agent skill attach** - Attach a skill.md to a Letta agent as a durable memory block.

```bash
terradev agent skill attach [OPTIONS]
Options:
  -a, --agent-id TEXT             Letta agent ID  [required]
  -s, --skill FILE                Path to skill.md  [required]
  -l, --label TEXT                Memory block label
  -e, --environment [cloud|local]
                                  Letta environment
  --help                          Show this message and exit.
```

### **agent skill init** - Create a skill.md template for an agent.

```bash
terradev agent skill init [OPTIONS]
Options:
  -n, --name TEXT         Skill name  [required]
  -o, --output TEXT       Output path (default: <name>.skill.md)
  -d, --description TEXT  Short description
  --tools TEXT            Comma-separated tool names
  --help                  Show this message and exit.
```

### **agent status** - Show live status of a fleet — tier health, KV hit rate, queue depth, cost.

```bash
terradev agent status [OPTIONS]
Options:
  --fleet-id TEXT        Fleet ID returned by 'terradev agent deploy'
                         [required]
  --format [table|json]
  --help                 Show this message and exit.
```

### **agent teardown** - Terminate all fleet instances and remove fleet state.

```bash
terradev agent teardown [OPTIONS]
Options:
  --fleet-id TEXT  Fleet ID to destroy  [required]
  --yes            Skip confirmation prompt
  --help           Show this message and exit.
```

### **agent vector-db** - Provision vector databases for agent memory and retrieval.

```bash
terradev agent vector-db [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```

### **agent vector-db down** - Teardown a vector database provisioned for an agent fleet.

```bash
terradev agent vector-db down [OPTIONS]
Options:
  -n, --name TEXT                 Vector DB name
  -a, --adapter [qdrant|weaviate]
                                  Vector DB adapter
  -c, --config TEXT               JSON adapter config
  -m, --manifest PATH             Path to universal manifest
  --help                          Show this message and exit.
```

### **agent vector-db up** - Provision a vector database for an agent fleet.

```bash
terradev agent vector-db up [OPTIONS]
Options:
  -n, --name TEXT                 Vector DB name
  -a, --adapter [qdrant|weaviate]
                                  Vector DB adapter
  -c, --config TEXT               JSON adapter config
  -m, --manifest PATH             Path to universal manifest
  --help                          Show this message and exit.
```

### **ml vllm lora** - LoRA adapter management for vLLM serving engines.

```bash
terradev ml vllm lora [OPTIONS] COMMAND [ARGS]...
Options:
  --help  Show this message and exit.

```

### **ml vllm lora link** - Load the active registry version of an adapter onto a vLLM server.

```bash
terradev ml vllm lora link [OPTIONS]
Options:
  -e, --endpoint TEXT  vLLM endpoint  [required]
  -n, --name TEXT      Registered adapter name  [required]
  --api-key TEXT       vLLM API key
  --help               Show this message and exit.
```

### **ml vllm lora list** - List LoRA adapters currently loaded on a vLLM server.

```bash
terradev ml vllm lora list [OPTIONS]
Options:
  -e, --endpoint TEXT  vLLM endpoint  [required]
  --api-key TEXT       vLLM API key
  --help               Show this message and exit.
```

### **ml vllm lora load** - Hot-load a LoRA adapter onto a running vLLM server.

```bash
terradev ml vllm lora load [OPTIONS]
Options:
  -e, --endpoint TEXT  vLLM endpoint  [required]
  -n, --name TEXT      Adapter name  [required]
  --path TEXT          Local path to adapter weights  [required]
  --api-key TEXT       vLLM API key
  --register           Register in LoRA registry before loading
  --base-model TEXT    Base model name (required with --register)
  --rank INTEGER       LoRA rank (default: 64)
  --help               Show this message and exit.
```

### **ml vllm lora sync** - Synchronize an adapter from the registry across multiple vLLM replicas.

```bash
terradev ml vllm lora sync [OPTIONS]
Options:
  -n, --name TEXT  Registered adapter name  [required]
  --replicas TEXT  Comma-separated host:port list  [required]
  --help           Show this message and exit.
```

### **ml vllm lora unload** - Hot-unload a LoRA adapter from a running vLLM server.

```bash
terradev ml vllm lora unload [OPTIONS]
Options:
  -e, --endpoint TEXT  vLLM endpoint  [required]
  -n, --name TEXT      Adapter name to unload  [required]
  --api-key TEXT       vLLM API key
  --help               Show this message and exit.
```

##  **Complete Command Summary**

### **Total Commands: 60+ Main Commands + 200+ Subcommands**

| Category | Commands | Key Features |
|----------|----------|-------------|
| **Core Infrastructure** | 15 | Provisioning, management, execution |
| **Training & Distributed** | 10 | Distributed training, checkpoints, staging |
| **Inference & Serving** | 12 | Model deployment, routing, optimization |
| **Model Orchestration** | 6 | Multi-model serving, memory management |
| **Warm Pool** | 3 | Pre-warming, intelligent scaling |
| **Optimization & Cost** | 10 | Cost optimization, analytics, arbitrage |
| **ML Integrations** | 40+ | External ML platform integrations |
| **Kubernetes** | 10+ | K8s cluster management, GPU operators |
| **Enterprise** | 8 | SSO, security, billing |
| **Monitoring** | 6 | Observability, metrics, status |
| **Utilities** | 10 | Job management, deployment, setup |

**All commands are production-ready and fully documented with comprehensive options and examples.** 