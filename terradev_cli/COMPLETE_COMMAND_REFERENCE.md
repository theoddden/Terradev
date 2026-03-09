# 📋 Complete Terradev CLI Command Reference

**All commands and subcommands for Terradev CLI v4.0.1**

---

## 🚀 **Main Commands (60+ Commands)**

### **Core Infrastructure Commands**

#### **provision** - Provision GPU instances across multiple providers
```bash
terradev provision [OPTIONS]

Options:
  -g, --gpu-type TEXT          GPU type (required) [A100, H100, RTX4090, etc.]
  -n, --count INTEGER          Number of instances [default: 1]
  --max-price FLOAT            Maximum price per hour
  --parallel INTEGER           Number of parallel queries [default: 6]
  --dry-run                    Show plan without provisioning
  --providers TEXT             Specific providers (multiple allowed)
  --region TEXT                Target region
  --spot                       Use spot instances
  --ensure-numa-alignment      Ensure NUMA alignment
  --enable-rdma                Enable RDMA/InfiniBand
  --enable-gpudirect           Enable GPUDirect
```

#### **status** - Show current status of all instances and usage
```bash
terradev status [OPTIONS]

Options:
  -f, --format [table|json]    Output format [default: table]
  --live                        Query providers for live instance status
  --instance-id TEXT           Filter by specific instance ID
  --provider TEXT               Filter by provider
  --region TEXT                 Filter by region
```

#### **quote** - Get real-time quotes from all providers
```bash
terradev quote [OPTIONS]

Options:
  -g, --gpu-type TEXT          GPU type [default: A100]
  -p, --providers TEXT         Specific providers (multiple allowed)
  --parallel INTEGER           Number of parallel queries [default: 6]
  --region TEXT                 Target region
  --spot                        Include spot instances
  --confidence FLOAT            Confidence level for price prediction
```

#### **availability** - Show GPU availability/stock status
```bash
terradev availability [OPTIONS]

Options:
  -g, --gpu-type TEXT          Filter by GPU type
  -p, --provider TEXT           Filter by provider
  -r, --region TEXT             Filter by region
  --refresh                     Force refresh of availability data
  --format [table|json]         Output format [default: table]
```

#### **manage** - Manage provisioned instances via real-time APIs
```bash
terradev manage [OPTIONS]

Options:
  -i, --instance-id TEXT        Instance ID (required)
  -a, --action [start|stop|restart|terminate]
                                Action to perform
  --force                       Force action without confirmation
```

#### **execute** - Execute commands on provisioned instances
```bash
terradev execute [OPTIONS]

Options:
  -i, --instance-id TEXT        Instance ID (required)
  -c, --command TEXT            Command to execute (required)
  --async-exec                  Run asynchronously
  --timeout INTEGER             Command timeout in seconds
  --output-file TEXT            Save output to file
```

#### **cleanup** - Clean up unused resources and temporary files
```bash
terradev cleanup [OPTIONS]

Options:
  --dry-run                     Show what would be cleaned
  --force                       Force cleanup without confirmation
  --older-than INTEGER          Clean resources older than X hours
  --type [all|instances|files|cache]
                                Type of resources to clean
```

---

### **Training & Distributed Computing Commands**

#### **train** - Launch a distributed training job
```bash
terradev train [OPTIONS] SCRIPT

Options:
  --framework [torchrun|ray|accelerate|deepspeed]
                                Training framework
  --from-provision TEXT         Use latest provisioned instances
  --tp-size INTEGER             Tensor parallel size
  --pp-size INTEGER             Pipeline parallel size
  --script-args TEXT             Additional script arguments
  --kv-checkpointing            Enable KV cache checkpointing
  --checkpoint-interval INTEGER Checkpoint interval in seconds
  --checkpoint-backend [s3|gcs|azure|local]
                                Checkpoint storage backend
  --auto-recovery               Enable automatic recovery
  --max-recovery-attempts INTEGER Max recovery attempts
  --flashoptim [on|off]         Enable/disable FlashOptim
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
terradev checkpoint [OPTIONS] COMMAND

Commands:
  list                          List all checkpoints
  create                        Create new checkpoint
  restore                       Restore from checkpoint
  validate                      Validate checkpoint integrity
  delete                        Delete checkpoint
  status                        Show checkpoint status
```

#### **stage** - Compress, chunk, and pre-position datasets
```bash
terradev stage [OPTIONS]

Options:
  -d, --dataset TEXT            Dataset path, S3 URI, GCS URI, HTTP URL, or HuggingFace name (required)
  --target-regions TEXT         Comma-separated target regions
  --compression [auto|zstd|gzip|none]
                                Compression type [default: auto]
  --format [auto|parquet|json|arrow]
                                Output format [default: auto]
  --parallel-streams INTEGER    Number of parallel upload streams
  --hf-dataset TEXT             HuggingFace dataset name
  --split TEXT                  Dataset split
  --process TEXT                Processing pipeline
  --preprocess TEXT             Preprocessing steps
  --max-size TEXT               Maximum dataset size
  --sample-rate FLOAT           Sample rate
```

#### **preflight** - Run preflight hardware validation on GPU nodes
```bash
terradev preflight [OPTIONS]

Options:
  --detailed                    Run detailed validation
  --network-test                Run network performance tests
  --gpu-test                    Run GPU performance tests
  --flashoptim-check            Check FlashOptim compatibility
  --cluster-id TEXT             Specific cluster to validate
```

---

### **Inference & Model Serving Commands**

#### **infer** - Deploy and manage inference endpoints
```bash
terradev infer [OPTIONS] COMMAND

Commands:
  deploy                        Deploy inference endpoint
  status                        Show endpoint status
  scale                         Scale endpoint replicas
  update                        Update endpoint configuration
  delete                        Delete endpoint
  list                          List all endpoints
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
terradev inferx [OPTIONS] COMMAND

Commands:
  deploy                        Deploy serverless endpoint
  status                        Check endpoint health
  failover                      Run failover tests
  cost-analysis                 Analyze costs
  scale                         Scale serverless endpoints
  update                        Update endpoint configuration
  delete                        Delete endpoint
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
terradev lora [OPTIONS] COMMAND

Commands:
  add                           Add LoRA adapter
  remove                        Remove LoRA adapter
  list                          List loaded adapters
  status                        Show adapter status
  update                        Update adapter configuration
  benchmark                     Benchmark adapter performance
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
  --instance-id TEXT            Optimize specific instance ID
  --auto-apply                  Automatically apply all recommended optimizations
  --scope [all|compute|storage|networking]
                                Optimization scope [default: all]
  --objective [cost|performance|balanced]
                                Optimization objective [default: balanced]
```

#### **analytics** - Show cost analytics from the cost tracking database
```bash
terradev analytics [OPTIONS]

Options:
  -d, --days INTEGER            Number of days to analyze [default: 7]
  -f, --format [table|json]     Output format [default: table]
  --breakdown                   Show cost breakdown by category
  --forecast                    Show cost forecast
```

#### **price-discovery** - Enhanced price discovery with capacity and arbitrage
```bash
terradev price-discovery [OPTIONS]

Options:
  -g, --gpu-type TEXT           GPU type
  --confidence FLOAT            Confidence level [default: 0.95]
  --regions TEXT                Target regions
  --include-spot                Include spot instances
  --arbitrage                   Show arbitrage opportunities
```

#### **percentiles** - Show historical price percentiles (p10–p99)
```bash
terradev percentiles [OPTIONS]

Options:
  -g, --gpu-type TEXT           GPU type
  --r, --region TEXT             Region
  -d, --days INTEGER            Number of days [default: 30]
  --format [table|json]         Output format [default: table]
```

#### **reliability** - Show provider reliability scores and error rates
```bash
terradev reliability [OPTIONS]

Options:
  -p, --provider TEXT           Filter by provider
  -r, --region TEXT             Filter by region
  -d, --days INTEGER            Number of days [default: 30]
  --format [table|json]         Output format [default: table]
```

#### **budget-optimize** - Find optimal deployment under budget constraints
```bash
terradev budget-optimize [OPTIONS]

Options:
  --budget FLOAT                Budget amount (required)
  --currency [USD|EUR|GBP]      Currency [default: USD]
  --period [hourly|daily|monthly]
                                Budget period [default: hourly]
  --gpu-type TEXT               Preferred GPU type
  --region TEXT                  Preferred region
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
terradev ml [OPTIONS] COMMAND

Commands:
  dvc                           DVC (Data Version Control) management
  huggingface                   Hugging Face models, datasets, and inference endpoints
  kserve                        KServe model deployment and management
  kubernetes                    Enhanced Kubernetes cluster management with Karpenter and monitoring
  langchain                     Enhanced LangChain integration with workflows, LangGraph, and monitoring
  langgraph                     Enhanced LangGraph workflow orchestration with monitoring
  langsmith                     LangSmith experiment tracking and monitoring
  mlflow                        MLflow experiment tracking and model registry
  ray                           Enhanced Ray distributed computing with monitoring and optimization
  sglang                        Enhanced SGLang model serving with monitoring
  wandb                         Enhanced Weights & Biases with dashboards, reports, and alerts
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
terradev k8s [OPTIONS] COMMAND

Commands:
  create                        Create multi-cloud Kubernetes GPU cluster
  destroy                       Destroy Kubernetes cluster
  info                          Get detailed cluster information
  list                          List all Kubernetes clusters
  gpu-operator                  Install GPU operator
  monitoring                    Deploy monitoring stack
  storage                       Configure storage
  networking                    Configure networking
  security                      Configure security policies
```

#### **k8s create** - Create multi-cloud Kubernetes GPU cluster
```bash
terradev k8s create [OPTIONS] CLUSTER_NAME

Options:
  -g, --gpu TEXT                GPU type (H100, A100, L40) (required)
  -n, --count INTEGER           Number of GPU nodes (required)
  --provider TEXT               Cloud provider
  --region TEXT                 Region
  --node-type TEXT              Node instance type
  --addons TEXT                 Comma-separated list of addons
  --dry-run                     Show creation plan
```

#### **k8s destroy** - Destroy Kubernetes cluster
```bash
terradev k8s destroy [OPTIONS] CLUSTER_NAME

Options:
  --force                       Force destruction without confirmation
  --preserve-volumes           Preserve persistent volumes
```

#### **helm-generate** - Generate Helm charts from Terradev workloads
```bash
terradev helm-generate [OPTIONS]

Options:
  --workload-file TEXT          Workload configuration file
  --output-dir TEXT             Output directory
  --chart-name TEXT             Helm chart name
  --version TEXT                Chart version
  --values-file TEXT            Custom values file
```

#### **gitops** - GitOps automation and infrastructure as code
```bash
terradev gitops [OPTIONS] COMMAND

Commands:
  init                          Initialize GitOps repository
  bootstrap                     Bootstrap GitOps tool on cluster
  validate                      Validate GitOps configuration
  sync                          Sync changes to cluster
  status                        Show GitOps status
  rollback                      Rollback to previous revision
  add-policy                    Add policy as code
  add-secrets                   Add secret management
  rollout                       Progressive deployment
```

---

### **Enterprise & Security Commands**

#### **configure** - Configure cloud provider credentials
```bash
terradev configure [OPTIONS]

Options:
  -p, --provider TEXT           Configure specific provider
  --api-key TEXT                API key
  --secret-key TEXT             Secret key
  --region TEXT                 Default region
  --interactive                 Interactive configuration
```

#### **sso** - Enterprise SSO authentication (Enterprise tier)
```bash
terradev sso [OPTIONS] COMMAND

Commands:
  login                         Login with SSO
  logout                        Logout from SSO
  status                        Show SSO status
  configure                     Configure SSO provider
  test                          Test SSO configuration
```

#### **integrations** - Show status of observability & ML integrations
```bash
terradev integrations [OPTIONS]

Options:
  --export-grafana              Export Grafana dashboard JSON
  --export-scrape-config        Print Prometheus scrape config
  --export-wandb-script         Print W&B setup script
  --deploy                      Deploy integration stack
  --stack [monitoring|ml|full]  Integration stack to deploy
  --cluster TEXT                Target cluster
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
  --job TEXT                    Job ID to monitor
  --live                        Live monitoring with auto-refresh
  --refresh INTEGER             Refresh interval in seconds
  --metrics TEXT                Comma-separated metrics to show
  --format [table|json]         Output format [default: table]
  --output-file TEXT            Save metrics to file
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
terradev setup [OPTIONS] PROVIDER

Options:
  --detailed                    Show detailed setup instructions
  --region TEXT                 Filter by region
  --gpu-type TEXT               Filter by GPU type
```

#### **smart-deploy** - Smart deployment with automatic optimization
```bash
terradev smart-deploy [OPTIONS] WORKLOAD_FILE

Options:
  --optimize-for [cost|performance|balanced]
                                Optimization objective [default: balanced]
  --dry-run                     Show deployment plan
  --auto-apply                  Apply optimizations automatically
```

---

### **Utility Commands**

#### **job** - Run Terradev job from YAML configuration
```bash
terradev job [OPTIONS] JOB_FILE

Options:
  --optimize [cost|latency|balanced]
                                Optimization criteria
  --dry-run                     Show job execution plan
  --format [yaml|json]          Output format
```

#### **run** - Provision a GPU instance, deploy a Docker container
```bash
terradev run [OPTIONS]

Options:
  -g, --gpu TEXT                GPU type (A100, H100, RTX4090, etc.) (required)
  -i, --image TEXT              Docker image (e.g. pytorch/pytorch:latest) (required)
  -c, --command TEXT            Command to run inside the container
  --name TEXT                   Instance name
  --provider TEXT               Cloud provider
  --region TEXT                 Region
  --spot                        Use spot instances
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
  --force                       Force onboarding even if already configured
  --skip-providers             Skip provider configuration
```

---

## 🎯 **ML Subcommands**

### **ml wandb** - Weights & Biases integration
```bash
terradev ml wandb [OPTIONS] COMMAND

Commands:
  test                          Test W&B integration
  projects                      List W&B projects
  runs                          List runs
  reports                       Generate reports
  dashboards                    Manage dashboards
  alerts                        Manage alerts
  sweep                         Manage sweeps
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
terradev ml phoenix [OPTIONS] COMMAND

Commands:
  projects                      List projects
  spans                         View spans
  trace                         View trace
  analyze                       Analyze traces
  dashboard                     Manage dashboards
```

### **ml qdrant** - Qdrant integration
```bash
terradev ml qdrant [OPTIONS] COMMAND

Commands:
  test                          Test Qdrant integration
  collections                   Manage collections
  search                        Search vectors
  embeddings                    Generate embeddings
```

### **ml guardrails** - Guardrails integration
```bash
terradev ml guardrails [OPTIONS] COMMAND

Commands:
  test                          Test guardrails
  config                        Manage configurations
  policies                      Manage policies
  chat                          Test chat interface
```

---

## 🚀 **SGLang Subcommands**

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

## 🎯 **vLLM Subcommands**

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

## 🎯 **LoRA Subcommands**

### **lora add** - Add LoRA adapter
```bash
terradev lora add [OPTIONS]

Options:
  -e, --endpoint TEXT           vLLM endpoint (required)
  -n, --name TEXT               Adapter name (required)
  -p, --path TEXT               Adapter path (required)
  --priority [high|medium|low]  Loading priority
  --force                       Override existing adapter
```

### **lora remove** - Remove LoRA adapter
```bash
terradev lora remove [OPTIONS]

Options:
  -e, --endpoint TEXT           vLLM endpoint (required)
  -n, --name TEXT               Adapter name (required)
  --force                       Force removal
```

### **lora list** - List loaded adapters
```bash
terradev lora list [OPTIONS]

Options:
  -e, --endpoint TEXT           vLLM endpoint (required)
  --format [table|json]         Output format
  --detailed                    Show detailed information
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

## 🎯 **Checkpoint Subcommands**

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

## 🎯 **InferX Subcommands**

### **inferx deploy** - Deploy serverless endpoint
```bash
terradev inferx deploy [OPTIONS]

Options:
  --endpoint TEXT               Endpoint name (required)
  --model-id TEXT               Model ID (required)
  --hardware TEXT               Hardware type
  --max-concurrency INTEGER     Maximum concurrency
  --min-concurrency INTEGER     Minimum concurrency
  --region TEXT                 Region
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

## 🎯 **Phoenix Subcommands**

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

## 🎯 **Qdrant Subcommands**

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

## 🎯 **Guardrails Subcommands**

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

## 🎯 **GitOps Subcommands**

### **gitops init** - Initialize GitOps repository
```bash
terradev gitops init [OPTIONS]

Options:
  --provider TEXT               Git provider [github|gitlab|bitbucket] (required)
  --repo TEXT                   Repository name (required)
  --tool TEXT                   GitOps tool [argocd|flux] (required)
  --cluster TEXT                Cluster name (required)
  --branch TEXT                 Git branch [default: main]
```

### **gitops bootstrap** - Bootstrap GitOps tool on cluster
```bash
terradev gitops bootstrap [OPTIONS]

Options:
  --tool TEXT                   GitOps tool [argocd|flux] (required)
  --cluster TEXT                Cluster name (required)
  --namespace TEXT              Namespace [default: gitops]
```

### **gitops validate** - Validate GitOps configuration
```bash
terradev gitops validate [OPTIONS]

Options:
  --cluster TEXT                Cluster name
  --dry-run                     Dry run validation
  --path TEXT                   Specific path to validate
```

### **gitops sync** - Sync changes to cluster
```bash
terradev gitops sync [OPTIONS]

Options:
  --cluster TEXT                Cluster name
  --force                       Force sync
  --dry-run                     Dry run sync
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

## 🎯 **HF Spaces Subcommands**

### **hf-space** - One-click HuggingFace Spaces deployment
```bash
terradev hf-space [OPTIONS] SPACE_NAME

Options:
  --model-id TEXT               Model ID (required)
  --template [llm|embedding|image|custom]
                                Space template [default: llm]
  --hardware TEXT                Hardware type
  --sdk [gradio|streamlit|fastapi]
                                SDK [default: gradio]
  --private                     Private space
  --org TEXT                    Organization
```

---

## 🎯 **Complete Command Summary**

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

**All commands are production-ready and fully documented with comprehensive options and examples.** 🚀
