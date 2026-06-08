# Terradev CLI BNF Grammar
# Complete syntax specification for all commands and options — v6.0.0

<terradev-cli> ::= <global-options> <command>

<global-options> ::=
    | --config <string>
    | --verbose
    | --skip-onboarding
    | <global-options> <global-options>

<command> ::=
    # ─ Infrastructure ─
    | <onboarding-command>
    | <configure-command>
    | <quote-command>
    | <provision-command>
    | <instance-command>
    | <status-command>
    | <stage-command>
    | <exec-command>
    | <analytics-command>
    | <optimize-command>
    | <integrations-command>
    | <cleanup-command>
    | <job-command>
    | <deploy-command>
    # ─ Price Intelligence ─
    | <price-discovery-command>
    | <budget-optimize-command>
    | <percentiles-command>
    | <availability-command>
    | <reliability-command>
    # ─ Smart Deploy / Helm ─
    | <smart-deploy-command>
    | <helm-generate-command>
    # ─ Training Pipeline ─
    | <preflight-command>
    | <train-command>
    | <train-status-command>
    | <train-stop-command>
    | <train-resume-command>
    | <monitor-command>
    | <checkpoint-command>
    # ─ Manifest / Drift ─
    | <up-command>
    | <rollback-command>
    | <manifests-command>
    | <export-command>
    | <import-command>
    | <record-commands>
    # ─ Model Orchestration ─
    | <orchestrator-start-command>
    | <orchestrator-register-command>
    | <orchestrator-load-command>
    | <orchestrator-evict-command>
    | <orchestrator-status-command>
    | <orchestrator-infer-command>
    | <warm-pool-start-command>
    | <warm-pool-register-command>
    | <warm-pool-status-command>
    | <cost-scaler-start-command>
    | <cost-scaler-status-command>
    | <cost-scaler-model-details-command>
    # ─ HuggingFace Spaces ─
    | <hf-space-command>
    # ─ MCP ─
    | <mcp-command>
    # ─ Local GPU Pool ─
    | <local-commands>
    # ─ Command Groups ─
    | <k8s-commands>
    | <ml-commands>
    | <vllm-commands>
    | <lora-commands>
    | <sglang-commands>
    | <sso-commands>
    | <gitops-commands>
    | <inferx-commands>
    | <phoenix-commands>
    | <guardrails-commands>
    | <qdrant-commands>
    | <retrain-commands>
    | <langfuse-commands>
    | <databricks-commands>
    | <agentic-serving-commands>
    | <model-router-commands>
    | <migrate-commands>
    | <eval-commands>
    | <triggers-commands>
    | <environments-commands>
    | <lineage-commands>
    | <agent-commands>


# ════════════════════════════════════════════════════════════════════
# TOP-LEVEL SINGLE COMMANDS
# ════════════════════════════════════════════════════════════════════

<onboarding-command> ::= onboarding [--force]

<configure-command>  ::= configure [--provider <provider>]

<quote-command> ::= quote
    [-g <gpu-type>]
    [-p <provider>]...
    [--parallel <integer>]
    [-r <string>]
    [-q | --quick]
    [--include-local]

<provision-command> ::= provision
    -g <gpu-type>
    [-n <integer>]
    [--max-price <float>]
    [--providers <provider>]...
    [--parallel <integer>]
    [--dry-run]
    [--type <provision-type>]
    [--model-name <string>]
    [--endpoint-name <string>]
    [--min-workers <integer>]
    [--max-workers <integer>]
    [--spot]
    [--on-demand]
    [--spot-strategy <spot-strategy>]
    [--backend <inference-backend>]
    [--prefer-local]

<provision-type>   ::= inference | training | jupyter | custom
<spot-strategy>    ::= cheapest | most-available | balanced

<instance-command> ::= instance
    -i <string>
    [-a <instance-action>]

<instance-action> ::= status | stop | start | terminate

<status-command> ::= status
    [-f <table-or-json>]
    [--live]

<stage-command> ::= stage
    -d <string>
    [--target-regions <string>]
    [--compression <compression-type>]

<compression-type> ::= auto | zstd | gzip | none

<exec-command> ::= exec
    -i <string>
    --cmd <string>
    [--async-exec]

<analytics-command> ::= analytics
    [-d <integer>]
    [-f <table-or-json>]

<optimize-command> ::= optimize
    [--instance-id <string>]
    [--auto-apply]

<integrations-command> ::= integrations
    [--export-grafana]
    [--export-scrape-config]
    [--export-wandb-script]

<cleanup-command> ::= cleanup

<job-command> ::= job <path> [--optimize <string>]

<deploy-command> ::= deploy
    -m <string>
    [-t <model-type>]
    [-p <provider>]
    [--gpu-type <gpu-type>]
    [--gpu-count <integer>]
    [--min-workers <integer>]
    [--max-workers <integer>]

<model-type> ::= llm | embedding | vision


# ── Price Intelligence ──

<price-discovery-command> ::= price-discovery
    [--gpu-type <gpu-type>]
    [--region <string>]
    [--hours <integer>]
    [--trends]

<budget-optimize-command> ::= budget-optimize
    --gpu-type <gpu-type>
    --budget <float>
    [--gpu-count <integer>]
    [--hours <float>]
    [--region <string>]
    [--workload <workload-type>]

<percentiles-command> ::= percentiles
    -g <gpu-type>
    [-p <provider>]
    [--spot]
    [-w <string>]

<availability-command> ::= availability
    [-g <gpu-type>]
    [-w <string>]

<reliability-command> ::= reliability
    [-p <provider>]
    [-w <string>]
    [--ranking]


# ── Smart Deploy / Helm ──

<smart-deploy-command> ::= smart-deploy
    --image <string>
    [-w <workload-type>]
    [--cmd <string>]
    [--gpu-count <integer>]
    [--budget <float>]
    [--namespace <string>]
    [--name <string>]
    [-e <string>]...
    [--mount <string>]...
    [--option <string>]...
    [--memory <string>]
    [--storage <string>]
    [--hours <float>]
    [--region <string>]
    [--dry-run]

<helm-generate-command> ::= helm-generate
    --image <string>
    [--workload <workload-type>]
    [--gpu-type <gpu-type>]
    [--gpu-count <integer>]
    [--memory <string>]
    [--storage <string>]
    [--budget <float>]
    [--region <string>]
    [--port <integer>]...
    [--stack <string>]...
    [--output <path>]
    [--name <string>]
    [--dry-run]


# ════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ════════════════════════════════════════════════════════════════════

<preflight-command> ::= preflight
    [-n <string>]...
    [--ssh-user <string>]
    [--ssh-key <path>]
    [--from-provision <string>]
    [--quick]
    [-f <json-or-text>]

<train-command> ::= train
    ([-c <path>] | [-s <path>])
    [--framework <training-framework>]
    [--backend <training-backend>]
    [-n <string>]...
    [--from-provision <string>]
    [--pool <string>]
    [--overflow-to-cloud]
    [--gpus-per-node <integer>]
    [--tp <integer>]
    [--pp <integer>]
    [--total-steps <integer>]
    [--skip-preflight]
    [-f <json-or-text>]
    [-- <script-args>...]

<training-framework> ::= torchrun | deepspeed | accelerate | megatron
<training-backend>   ::= native | ray

<train-status-command> ::= train-status
    [-j <string>]
    [-f <json-or-text>]

<train-stop-command> ::= train-stop
    -j <string>
    [-f <json-or-text>]

<train-resume-command> ::= train-resume
    -j <string>
    [--checkpoint-id <string>]
    [-f <json-or-text>]

<monitor-command> ::= monitor
    [-j <string>]
    [-n <string>]...
    [--ssh-user <string>]
    [--ssh-key <path>]
    [--from-provision <string>]
    [-l <path>]
    [-i <float>]
    [--count <integer>]
    [--prometheus <url>]
    [--cost-rate <float>]
    [-f <json-or-text>]

<checkpoint-command> ::= checkpoint
    <checkpoint-action>
    -j <string>
    [--step <integer>]
    [--checkpoint-id <string>]
    [--dest <path>]
    [-f <json-or-text>]

<checkpoint-action> ::= list | restore | promote | delete


# ════════════════════════════════════════════════════════════════════
# MANIFEST / DRIFT / PIPELINE
# ════════════════════════════════════════════════════════════════════

<up-command> ::= up
    -j <string>
    [--cache-dir <path>]
    [--fix-drift]
    [--gpu-type <gpu-type>]
    [--gpu-count <integer>]
    [--hours <float>]
    [--budget <float>]
    [--region <string>]
    [--dataset <path>]
    [--ttl <string>]

<rollback-command> ::= rollback <job-version> [--cache-dir <path>]
<job-version> ::= <string>

<manifests-command> ::= manifests
    [--job <string>]
    [--cache-dir <path>]
    [--show-imported]
    [--show-recordings]

<export-command> ::= export
    -o <path>
    [-j <string>]
    [--cache-dir <path>]
    [--format <pipeline-format>]

<pipeline-format> ::= argo | native

<import-command> ::= import <path>
    [-n <string>]
    [--force]
    [--validate-only]
    [--cache-dir <path>]

<record-commands> ::= record <record-subcommand>

<record-subcommand> ::=
    | start -n <string> [--output-dir <path>]
    | stop  -n <string> [--export <path>] [--output-dir <path>]


# ════════════════════════════════════════════════════════════════════
# MODEL ORCHESTRATION
# ════════════════════════════════════════════════════════════════════

<orchestrator-start-command> ::= orchestrator-start
    [--gpu-id <integer>]
    [--memory-gb <float>]
    [--policy <orchestrator-policy>]

<orchestrator-policy> ::= billing_optimized | latency_optimized | hybrid

<orchestrator-register-command> ::= orchestrator-register
    <string>
    <path>
    [--framework <orch-framework>]
    [--priority <integer>]
    [--tags <string>]

<orch-framework> ::= pytorch | vllm | sglang

<orchestrator-load-command>   ::= orchestrator-load   <string> [--force]
<orchestrator-evict-command>  ::= orchestrator-evict  <string>
<orchestrator-status-command> ::= orchestrator-status [--model-id <string>]
<orchestrator-infer-command>  ::= orchestrator-infer  <string>

<warm-pool-start-command> ::= warm-pool-start
    [--strategy <warm-strategy>]
    [--max-warm <integer>]
    [--min-warm <integer>]

<warm-strategy> ::= traffic_based | time_based | priority_based | cost_optimized | latency_optimized

<warm-pool-register-command> ::= warm-pool-register <string> [--priority <integer>]
<warm-pool-status-command>   ::= warm-pool-status

<cost-scaler-start-command> ::= cost-scaler-start
    [--strategy <cost-strategy>]
    [--budget <float>]
    [--cost-per-gb <float>]

<cost-strategy> ::= minimize_cost | balance_cost_latency | latency_critical | budget_constrained

<cost-scaler-status-command>       ::= cost-scaler-status
<cost-scaler-model-details-command>::= cost-scaler-model-details <string>


# ── HuggingFace Spaces ──

<hf-space-command> ::= hf-space <string>
    --model-id <string>
    [--hardware <hf-hardware>]
    [--sdk <hf-sdk>]
    [--private]
    [--template <hf-template>]
    [-e <string>]...
    [-s <string>]...

<hf-hardware> ::= cpu-basic | cpu-upgrade | t4-medium | a10g-large | a100-large
<hf-sdk>      ::= gradio | streamlit | docker
<hf-template> ::= llm | embedding | image


# ── MCP Server ──

<mcp-command> ::= mcp <mcp-action>
    [--client <mcp-client>]
    [--transport <mcp-transport>]

<mcp-action>    ::= serve | install | list-tools
<mcp-client>    ::= claude-desktop | cursor | windsurf | continue | cline
<mcp-transport> ::= stdio | sse | http


# ════════════════════════════════════════════════════════════════════
# K8S COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<k8s-commands> ::= k8s <k8s-subcommand>

<k8s-subcommand> ::=
    | create   <k8s-create-options>
    | destroy  <k8s-destroy-options>
    | list
    | info     <string>

<k8s-create-options> ::=
    <string>
    -g <gpu-type>
    -n <integer>
    [--max-price <float>]
    [--multi-cloud]
    [--prefer-spot]
    [--aws-region <string>]
    [--gcp-region <string>]
    [--control-plane <string>]

<k8s-destroy-options> ::=
    <string>
    [--force]
    [--keep-volumes]


# ════════════════════════════════════════════════════════════════════
# ML COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<ml-commands> ::= ml <ml-subcommand>

<ml-subcommand> ::=
    | kubernetes   <ml-k8s-options>
    | wandb        <ml-wandb-options>
    | langchain    <ml-lc-options>
    | langgraph    <ml-lg-options>
    | sglang       <ml-sglang-options>
    | huggingface  <ml-hf-options>
    | kserve       <ml-kserve-options>
    | langsmith    <ml-ls-options>
    | dvc          <ml-dvc-options>
    | mlflow       <ml-mlflow-options>
    | ray          <ml-ray-options>

<ml-k8s-options> ::=
    [--test] [--gpu-nodes] [--install-karpenter] [--create-provisioner]
    [--gpu-type <gpu-type>] [--cpu-limit <string>] [--memory-limit <string>]
    [--resources] [--install-monitoring] [--metrics-summary] [--dashboard]

<ml-wandb-options> ::=
    [--test] [--create-dashboard] [--create-report] [--setup-alerts]
    [--project <string>] [--entity <string>]

<ml-lc-options> ::=
    [--test] [--create-workflow] [--create-sglang-pipeline]
    [--workflow-name <string>] [--model <string>]

<ml-lg-options> ::=
    [--test] [--create-workflow] [--orchestrator-worker] [--evaluation-workflow]
    [--workflow-name <string>]

<ml-sglang-options> ::=
    [--test] [--create-workflow] [--model <string>]

<ml-hf-options> ::=
    [--test] [--list-models] [--list-datasets] [--model-info <string>]
    [--create-endpoint] [--list-endpoints] [--endpoint-infer]
    [--endpoint-id <string>] [--model-id <string>]

<ml-kserve-options> ::=
    [--test] [--deploy] [--list-models]
    [--model-name <string>] [--model-uri <string>]

<ml-ls-options> ::=
    [--test] [--create-project] [--create-trace] [--project <string>]

<ml-dvc-options> ::=
    [--test] [--init] [--add <path>] [--push] [--pull]

<ml-mlflow-options> ::=
    [--test] [--create-experiment] [--list-experiments] [--experiment-name <string>]

<ml-ray-options> ::=
    [--test] [--install-monitoring] [--metrics-summary] [--grafana] [--prometheus]
    [--status] [--list-nodes] [--start] [--stop] [--dashboard]
    [--gpu-type <gpu-type>]


# ════════════════════════════════════════════════════════════════════
# VLLM COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<vllm-commands> ::= vllm <vllm-subcommand>

<vllm-subcommand> ::=
    | optimize      <vllm-optimize-options>
    | auto-optimize <vllm-auto-optimize-options>
    | analyze       <vllm-analyze-options>
    | benchmark     <vllm-benchmark-options>

<vllm-optimize-options> ::=
    -m <string>
    [-t <vllm-opt-type>]
    [-G <integer>]
    [-o <vllm-output-format>]

<vllm-opt-type>      ::= throughput | latency
<vllm-output-format> ::= args | config | helm

<vllm-auto-optimize-options> ::=
    -m <string>
    [-e <url>]
    [-s <path>]
    [-G <integer>]
    [-o <vllm-output-format>]
    [--apply]

<vllm-analyze-options> ::=
    -e <url>
    [-d <integer>]

<vllm-benchmark-options> ::=
    -e <url>
    [--api-key <string>]
    [--prompt <string>]
    [-c <integer>]


# ════════════════════════════════════════════════════════════════════
# LORA COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<lora-commands> ::= lora <lora-subcommand>

<lora-subcommand> ::=
    | list   -e <url> [--api-key <string>]
    | add    -e <url> -n <string> --path <path> [--api-key <string>]
    | remove -e <url> -n <string> [--api-key <string>]


# ════════════════════════════════════════════════════════════════════
# SGLANG COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<sglang-commands> ::= sglang <sglang-subcommand>

<sglang-subcommand> ::=
    | optimize <model-path> <sglang-optimize-options>
    | router   <model-path> <sglang-router-options>
    | detect   <model-path> <sglang-detect-options>
    | install  <sglang-install-options>
    | start    <model-path> <sglang-start-options>
    | test

<sglang-optimize-options> ::=
    [--workload-type <workload-type>]
    [--user-description <string>]
    [--host <string>]
    [--port <integer>]
    [--dry-run]

<sglang-router-options> ::=
    [--dp-size <integer>]
    [--workload-type <workload-type>]

<sglang-detect-options> ::=
    [--workload-type <workload-type>]
    [--user-description <string>]

<sglang-install-options> ::=
    [--instance-ip <string>]
    [--ssh-user <string>]
    [--ssh-key <path>]

<sglang-start-options> ::=
    [--instance-ip <string>]
    [--ssh-user <string>]
    [--ssh-key <path>]
    [--workload-type <workload-type>]
    [--port <integer>]


# ════════════════════════════════════════════════════════════════════
# SSO COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<sso-commands> ::= sso <sso-subcommand>

<sso-subcommand> ::=
    | status
    | configure <sso-configure-options>
    | test      [--provider <sso-provider>]

<sso-configure-options> ::=
    -p <sso-provider>
    [--client-id <string>]
    [--client-secret <string>]
    [--domain <string>]
    [--tenant-id <string>]
    [--entity-id <string>]
    [--sso-url <url>]
    [--certificate <string>]

<sso-provider> ::= azure_ad | okta | google_workspace | auth0


# ════════════════════════════════════════════════════════════════════
# GITOPS COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<gitops-commands> ::= gitops <gitops-subcommand>

<gitops-subcommand> ::=
    | init       <gitops-init-options>
    | bootstrap  <gitops-bootstrap-options>
    | sync       <gitops-sync-options>
    | validate   <gitops-validate-options>

<gitops-init-options> ::=
    --provider <git-provider>
    --repo <string>
    --cluster <string>
    [--tool <gitops-tool>]
    [--git-url <url>]
    [--git-token <string>]
    [--namespace <string>]
    [--auto-sync | --no-auto-sync]
    [--prune | --no-prune]

<gitops-bootstrap-options> ::=
    --tool <gitops-tool>
    --cluster <string>
    [--namespace <string>]

<gitops-sync-options> ::=
    --cluster <string>
    [--environment <string>]
    [--tool <gitops-tool>]

<gitops-validate-options> ::=
    [--dry-run | --apply]
    [--cluster <string>]
    [--environment <string>]

<git-provider> ::= github | gitlab | bitbucket | azure_devops
<gitops-tool>  ::= argocd | flux


# ════════════════════════════════════════════════════════════════════
# INFERX COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<inferx-commands> ::= inferx <inferx-subcommand>

<inferx-subcommand> ::=
    | configure <inferx-configure-options>
    | deploy    <inferx-deploy-options>
    | status    --model-id <string>
    | delete    --model-id <string>
    | list
    | usage
    | quote     <inferx-quote-options>
    | optimize  <inferx-optimize-options>

<inferx-configure-options> ::=
    --api-key <string>
    [--endpoint <url>]
    [--region <string>]
    [--snapshot | --no-snapshot]
    [--gpu-slicing | --no-gpu-slicing]
    [--multi-tenant | --no-multi-tenant]

<inferx-deploy-options> ::=
    --model <string>
    [--image <string>]
    [--gpu-type <gpu-type>]
    [--gpu-memory <integer>]
    [--max-concurrency <integer>]
    [--framework <string>]
    [--openai-compatible | --no-openai-compatible]
    [--timeout <integer>]

<inferx-quote-options> ::=
    [--gpu-type <gpu-type>]
    [--region <string>]

<inferx-optimize-options> ::=
    [--cluster-config <path>]
    [--usage-metrics <path>]
    [--tier <cost-tier>]
    [--output <path>]
    [--implement]

<cost-tier> ::= economy | balanced | performance


# ════════════════════════════════════════════════════════════════════
# PHOENIX COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<phoenix-commands> ::= phoenix <phoenix-subcommand>

<phoenix-subcommand> ::=
    | test
    | projects  [-l <integer>]
    | spans     [-p <string>] [-f <string>] [-l <integer>]
    | trace     -t <string>  [-p <string>]
    | otel-env  [-p <string>]
    | snippet   [-p <string>]
    | k8s       [-n <string>]


# ════════════════════════════════════════════════════════════════════
# GUARDRAILS COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<guardrails-commands> ::= guardrails <guardrails-subcommand>

<guardrails-subcommand> ::=
    | test
    | chat            -m <string> [-c <string>]
    | generate-config [-c <string>] [-o <path>]
    | k8s             [-n <string>]


# ════════════════════════════════════════════════════════════════════
# QDRANT COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<qdrant-commands> ::= qdrant <qdrant-subcommand>

<qdrant-subcommand> ::=
    | test
    | collections
    | create-collection  [-n <string>] [-e <string>]
    | info               [-n <string>]
    | count              [-n <string>]
    | k8s                [-n <string>]


# ════════════════════════════════════════════════════════════════════
# RETRAIN COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<retrain-commands> ::= retrain <retrain-subcommand>

<retrain-subcommand> ::=
    | drift   <retrain-drift-options>
    | detect  <retrain-detect-options>
    | deploy  <retrain-deploy-options>
    | history [-n <integer>] [-f <json-or-text>]

<retrain-drift-options> ::=
    -m <string>
    [--source phoenix-traces]
    [--method lora]
    [--eval-threshold <float>]
    [--deploy <deploy-strategy>]
    [--auto-swap]
    [--phoenix-endpoint <url>]
    [--phoenix-project <string>]
    [-e <url>]
    [--vllm-api-key <string>]
    [--baseline <float>]
    [--threshold <float>]
    [--min-samples <integer>]
    [-f <json-or-text>]

<retrain-detect-options> ::=
    -m <string>
    [--phoenix-endpoint <url>]
    [--phoenix-project <string>]
    [--baseline <float>]
    [--threshold <float>]
    [--min-samples <integer>]
    [-f <json-or-text>]

<retrain-deploy-options> ::=
    --cycle-id <string>
    -e <url>
    [--vllm-api-key <string>]
    [-f <json-or-text>]

<deploy-strategy> ::= canary | direct


# ════════════════════════════════════════════════════════════════════
# LANGFUSE COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<langfuse-commands> ::= langfuse <langfuse-subcommand>

<langfuse-subcommand> ::=
    | configure          [--public-key <string>] [--secret-key <string>] [--host <url>]
    | test
    | traces             [-n <integer>] [--name <string>] [-f <json-or-text>]
    | trace              <string>       [-f <json-or-text>]
    | scores             [--trace-id <string>] [--name <string>] [-n <integer>] [-f <json-or-text>]
    | score              --trace-id <string> --name <string> --value <float>
                         [--observation-id <string>] [--comment <string>]
    | datasets           [-n <integer>] [-f <json-or-text>]
    | export-training-data
                         [-n <integer>] [--name <string>] [--min-score <float>]
                         [--score-name <string>] [-o <path>]
    | quality            [--score-name <string>] [-n <integer>] [-f <json-or-text>]
    | otel-env           [-p <string>]
    | k8s                [-n <string>]


# ════════════════════════════════════════════════════════════════════
# DATABRICKS COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<databricks-commands> ::= databricks <databricks-subcommand>

<databricks-subcommand> ::=
    | configure          [--host <url>] [--token <string>]
    | test
    | jobs               [-n <integer>] [-f <json-or-text>]
    | run                <integer> [-f <json-or-text>]
    | run-status         <integer> [-f <json-or-text>]
    | clusters           [-f <json-or-text>]
    | serving-endpoints  [-f <json-or-text>]
    | deploy-model       <databricks-deploy-model-options>
    | query              --endpoint <string> --prompt <string> [-f <json-or-text>]
    | mlflow             <databricks-mlflow-subcommand>

<databricks-deploy-model-options> ::=
    --endpoint-name <string>
    --model-name <string>
    [--model-version <string>]
    [--workload-size <db-workload-size>]
    [--scale-to-zero | --no-scale-to-zero]
    [-f <json-or-text>]

<db-workload-size> ::= Small | Medium | Large

<databricks-mlflow-subcommand> ::=
    | experiments  [-n <integer>] [-f <json-or-text>]
    | models       [-n <integer>] [-f <json-or-text>]


# ════════════════════════════════════════════════════════════════════
# AGENTIC-SERVING COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<agentic-serving-commands> ::= agentic-serving <agentic-serving-subcommand>

<agentic-serving-subcommand> ::=
    | configure     <agentic-serving-configure-options>
    | show-config   [-f <json-or-text>]
    | launch-args
    | lmcache-env
    | k8s           [-n <string>]
    | helm-values   [-f <json-or-yaml>]

<agentic-serving-configure-options> ::=
    [--engine <inference-engine>]
    [--model <string>]
    [--tp <integer>]
    [--max-model-len <integer>]
    [--gpu-mem <float>]
    [--lmcache | --no-lmcache]
    [--lmcache-backend <lmcache-backend>]
    [--disaggregation | --no-disaggregation]

<inference-engine>  ::= vllm | sglang
<lmcache-backend>   ::= cpu | disk | redis


# ════════════════════════════════════════════════════════════════════
# MODEL-ROUTER COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<model-router-commands> ::= model-router <model-router-subcommand>

<model-router-subcommand> ::=
    | configure  <model-router-configure-options>
    | test       [-p <string>] [-f <json-or-text>]
    | classify   <string>
    | stats      [-f <json-or-text>]
    | llmd-config [-f <json-or-yaml>]

<model-router-configure-options> ::=
    [--strong-url <url>]
    [--strong-model <string>]
    [--strong-api-key <string>]
    [--weak-url <url>]
    [--weak-model <string>]
    [--weak-api-key <string>]
    [--strategy <router-strategy>]
    [--cost-threshold <float>]

<router-strategy> ::= step_type | threshold | cascade | strong_only | weak_only


# ════════════════════════════════════════════════════════════════════
# MIGRATE COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<migrate-commands> ::= migrate <migrate-subcommand>

<migrate-subcommand> ::=
    | migration      <migration-options>
    | list-workloads <list-workloads-options>

<migration-options> ::=
    --from <provider>
    --to <provider>
    [--instance-id <string>]
    [--workload <string>]
    [--dry-run]

<list-workloads-options> ::=
    [--provider <provider>]
    [--format <table-or-json>]


# ════════════════════════════════════════════════════════════════════
# EVAL COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<eval-commands> ::= eval <eval-subcommand>

<eval-subcommand> ::=
    | evaluation <evaluation-options>
    | compare    <compare-options>

<evaluation-options> ::=
    [--model <path>]
    [--endpoint <url>]
    [--dataset <path>]
    [--metrics <metric>]...
    [--baseline <path>]
    [--workload-type <string>]
    [--duration <integer>]
    [--output <path>]
    [--format <table-or-json>]

<compare-options> ::=
    <string>
    <string>
    --dataset <path>
    [--metrics <metric>]...
    [--output <path>]


# ════════════════════════════════════════════════════════════════════
# TRIGGERS COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<triggers-commands> ::= triggers <triggers-subcommand>

<triggers-subcommand> ::=
    | create  <trigger-create-options>
    | list
    | enable  <string>
    | disable <string>
    | fire    <string> [--data <string>] [--source <string>]

<trigger-create-options> ::=
    <string>
    <string>
    [--type <trigger-type>]
    [--event <string>]
    [--schedule <string>]
    [--condition <string>]
    [--env <environment>]

<trigger-type> ::= event | schedule | condition


# ════════════════════════════════════════════════════════════════════
# ENVIRONMENTS COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<environments-commands> ::= environments <environments-subcommand>

<environments-subcommand> ::=
    | list     [--env <environment>]
    | promote  <string> --from <environment> --to <environment> [--user <string>]
    | approve  <string> [--user <string>]
    | history  [--artifact <string>]

<environment> ::= dev | staging | prod


# ════════════════════════════════════════════════════════════════════
# LINEAGE COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<lineage-commands> ::= lineage <lineage-subcommand>

<lineage-subcommand> ::=
    | register    <artifact-type> <string> <string>
                  [--env <environment>] [--hash <string>] [--size <integer>]
                  [--user <string>] [--parent <string>]
    | graph       <string> [--direction <lineage-direction>]
    | production  [--type <artifact-type>]
    | show        <string> [--env <environment>]
    | diff        <string> <string>
    | export      [--format <json-or-csv>] [--model <string>]
                  [--env <environment>] [-o <path>]
    | trace       [--checkpoint <string>] [--execution <string>]
    | auto        --pipeline <string> [--env <environment>] [--triggered-by <string>]
    | add-input   <string> <lineage-input-type>  <string>
    | add-output  <string> <lineage-output-type> <string>
    | complete    <string> [--status <string>]

<artifact-type>       ::= dataset | model | checkpoint | metrics | config
<lineage-input-type>  ::= dataset | model | config | checkpoint
<lineage-output-type> ::= model | checkpoint | metrics | evaluation
<lineage-direction>   ::= up | down | both


# ════════════════════════════════════════════════════════════════════
# AGENT FLEET COMMAND GROUP  (v6.0.0)
# ════════════════════════════════════════════════════════════════════

<agent-commands> ::= agent <agent-subcommand>

<agent-subcommand> ::=
    | plan      <agent-plan-options>
    | deploy    <agent-deploy-options>
    | status    --fleet-id <string>
    | scale     <agent-scale-options>
    | cost      --fleet-id <string>
    | list
    | teardown  --fleet-id <string> [--force]

<agent-plan-options> ::=
    --agents <integer>
    [--model <string>]
    [--reasoning <agent-reasoning-mode>]
    [--planner-gpu <gpu-type>]
    [--worker-gpu <gpu-type>]
    [--planner-count <integer>]
    [--worker-count <integer>]
    [--output <path>]

<agent-deploy-options> ::=
    --agents <integer>
    [--model <string>]
    [--reasoning <agent-reasoning-mode>]
    [--planner-gpu <gpu-type>]
    [--worker-gpu <gpu-type>]
    [--planner-count <integer>]
    [--worker-count <integer>]
    [--providers <provider>]...
    [--max-price <float>]
    [--dry-run]

<agent-scale-options> ::=
    --fleet-id <string>
    --tier <agent-tier>
    --count <integer>
    [--providers <provider>]...

<agent-reasoning-mode> ::= instant | deep | mixed
<agent-tier>           ::= reasoning | decode | cpu_tools


# ════════════════════════════════════════════════════════════════════
# LOCAL GPU POOL COMMAND GROUP
# ════════════════════════════════════════════════════════════════════

<local-commands> ::= local <local-subcommand>

<local-subcommand> ::=
    | scan      <local-scan-options>
    | register  <local-register-options>
    | pool      [--format <table-or-json>] [--remove <string>]

<local-scan-options> ::=
    [--host <string>]
    [--user <string>]
    [--key <path>]
    [--detailed]
    [--register]
    [--name <string>]

<local-register-options> ::=
    --name <string>
    [--host <string>]
    [--user <string>]
    [--key <path>]


# ════════════════════════════════════════════════════════════════════
# TERMINALS
# ════════════════════════════════════════════════════════════════════

<provider> ::=
    | aws | gcp | azure | runpod | vastai | lambda | coreweave
    | tensordock | baseten | oracle | crusoe | hyperstack
    | digitalocean | alibaba | ovhcloud | fluidstack | hetzner
    | siliconflow | inferx | demo

<gpu-type> ::=
    | H100 | A100 | A40 | L40 | L40S | A10G
    | RTX4090 | RTX3090 | RTX3080
    | T4 | V100

<workload-type> ::=
    | agentic_chat | batch_inference | low_latency | moe_model
    | pd_disaggregated | structured_output | rag_workload | general

<metric> ::=
    | accuracy | perplexity | latency | throughput | cost_per_token | error_rate

<json-or-text>   ::= json | text
<json-or-csv>    ::= json | csv
<json-or-yaml>   ::= json | yaml
<table-or-json>  ::= table | json

<model-path> ::= <string>
<path>       ::= <string>
<url>        ::= <string>
<float>      ::= <digit-string> [ "." <digit-string> ]
<integer>    ::= <digit-string>
<string>     ::= <character-string>
