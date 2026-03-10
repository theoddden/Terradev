# Terradev CLI BNF Grammar
# Complete syntax specification for all commands and options

<terradev-cli> ::= <global-options> <command>

<global-options> ::= 
    | --config <string>
    | --verbose 
    | --skip-onboarding
    | <global-options> <global-options>

<command> ::= 
    | migrate-commands
    | eval-commands
    | train-commands
    | k8s-commands
    | sglang-commands
    | vllm-commands
    | sso-commands
    | quote-commands
    | status-commands
    | optimize-commands
    | configure-commands
    | cleanup-commands
    | upgrade-commands
    | onboarding-commands

# ── MIGRATE COMMANDS ──
<migrate-commands> ::= migrate <migrate-subcommand>

<migrate-subcommand> ::= 
    | migration <migration-options>
    | list-workloads <list-workloads-options>

<migration-options> ::= 
    --from <provider> --to <provider> 
    | --from <provider> --to <provider> --instance-id <string>
    | --from <provider> --to <provider> --workload <string>
    | --from <provider> --to <provider> --dry-run
    | --from <provider> --to <provider> --instance-id <string> --workload <string> --dry-run

<list-workloads-options> ::= 
    | --provider <provider>
    | --format <output-format>

# ── EVAL COMMANDS ──
<eval-commands> ::= eval <eval-subcommand>

<eval-subcommand> ::= 
    | evaluation <evaluation-options>
    | compare <compare-options>

<evaluation-options> ::= 
    | --model <path> --dataset <path>
    | --endpoint <url>
    | --model <path> --dataset <path> --metrics <metric-list>
    | --endpoint <url> --metrics <metric-list> --duration <integer>
    | --model <path> --dataset <path> --baseline <path> --output <path> --format <output-format>

<compare-options> ::= 
    | <model-path> <model-path> --dataset <path>
    | <model-path> <model-path> --dataset <path> --metrics <metric-list>
    | <model-path> <model-path> --dataset <path> --metrics <metric-list> --output <path>

# ── TRAIN COMMANDS ──
<train-commands> ::= train <train-options>

<train-options> ::= 
    | <config-path>
    | <config-path> --script <path>
    | <config-path> --framework <framework> --backend <backend>
    | <config-path> --nodes <integer> --gpus-per-node <integer>
    | <config-path> --tp <integer> --pp <integer> --total-steps <integer>
    | <config-path> --skip-preflight --format <output-format>

# ── K8S COMMANDS ──
<k8s-commands> ::= k8s <k8s-subcommand>

<k8s-subcommand> ::= 
    | create <k8s-create-options>
    | destroy <k8s-destroy-options>
    | status <k8s-status-options>
    | gpu-operator <k8s-gpu-options>
    | device-plugin <k8s-device-options>
    | mig-configure <k8s-mig-options>
    | time-slicing <k8s-time-options>

<k8s-create-options> ::= 
    | --name <string> --gpu-type <gpu-type>
    | --name <string> --gpu-type <gpu-type> --node-count <integer>
    | --name <string> --gpu-type <gpu-type> --region <string>

<k8s-destroy-options> ::= 
    | --name <string>
    | --name <string> --force

# ── SGLANG COMMANDS ──
<sglang-commands> ::= sglang <sglang-subcommand>

<sglang-subcommand> ::= 
    | deploy <sglang-deploy-options>
    | benchmark <sglang-benchmark-options>
    | optimize <sglang-optimize-options>
    | test <sglang-test-options>

<sglang-deploy-options> ::= 
    | <model-path>
    | <model-path> --workload-type <workload-type>
    | <model-path> --host <string> --port <integer>

<sglang-benchmark-options> ::= 
    | --endpoint <url> --duration <integer>
    | --endpoint <url> --workload <workload-type> --metrics <metric-list>

# ── VLLM COMMANDS ──
<vllm-commands> ::= vllm <vllm-subcommand>

<vllm-subcommand> ::= 
    | deploy <vllm-deploy-options>
    | benchmark <vllm-benchmark-options>
    | optimize <vllm-optimize-options>

<vllm-deploy-options> ::= 
    | <model-path>
    | <model-path> --gpu-type <gpu-type> --gpu-memory <integer>

<vllm-benchmark-options> ::= 
    | --endpoint <url>
    | --endpoint <url> --concurrent <integer> --prompt <string>

# ── SSO COMMANDS ──
<sso-commands> ::= sso <sso-subcommand>

<sso-subcommand> ::= 
    | configure <sso-configure-options>
    | test <sso-test-options>
    | status <sso-status-options>

<sso-configure-options> ::= 
    | --provider <sso-provider>
    | --provider <sso-provider> --client-id <string> --client-secret <string>

# ── QUOTE COMMANDS ──
<quote-commands> ::= quote <quote-options>

<quote-options> ::= 
    | <gpu-type>
    | <gpu-type> --providers <provider-list>
    | <gpu-type> --region <string> --quick

# ── STATUS COMMANDS ──
<status-commands> ::= status <status-options>

<status-options> ::= 
    | 
    | --format <output-format>
    | --live

# ── OPTIMIZE COMMANDS ──
<optimize-commands> ::= optimize <optimize-options>

<optimize-options> ::= 
    | 
    | --instance-id <string>
    | --instance-id <string> --auto-apply

# ── CONFIGURE COMMANDS ──
<configure-commands> ::= configure <configure-options>

<configure-options> ::= 
    | 
    | --provider <provider>
    | --provider <provider> --interactive

# ── BASIC COMMANDS ──
<cleanup-commands> ::= cleanup
<upgrade-commands> ::= upgrade <upgrade-options>
<onboarding-commands> ::= onboarding <onboarding-options>

<upgrade-options> ::= 
    | --tier <tier>
    | --activate --email <string>

<onboarding-options> ::= 
    | 
    | --force

# ── TERMINALS ──
<provider> ::= 
    | aws | gcp | azure | runpod | vastai | lambda | coreweave
    | tensordock | huggingface | baseten | oracle | crusoe
    | hyperstack | digitalocean | alibaba | ovhcloud
    | fluidstack | hetzner | siliconflow | inferx | demo

<gpu-type> ::= 
    | A100 | H100 | A40 | L40S | RTX-4090 | RTX-3090 | T4 | V100

<framework> ::= 
    | torchrun | deepspeed | accelerate | megatron

<backend> ::= 
    | native | ray

<output-format> ::= 
    | table | json | yaml | text

<workload-type> ::= 
    | agentic_chat | batch_inference | low_latency | moe_model
    | pd_disaggregated | structured_output | rag_workload | general

<metric-list> ::= 
    | <metric>
    | <metric> <metric-list>

<metric> ::= 
    | accuracy | perplexity | latency | throughput | cost_per_token | error_rate

<provider-list> ::= 
    | <provider>
    | <provider> <provider-list>

<tier> ::= 
    | research_plus | enterprise | enterprise_plus

<sso-provider> ::= 
    | google | microsoft | okta | auth0

<model-path> ::= <string>
<path> ::= <string>
<url> ::= <string>
<string> ::= <character-string>
<integer> ::= <digit-string>
