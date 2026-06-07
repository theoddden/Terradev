# Terradev Troubleshooting

Issues organized by command group. Each entry covers: symptom, diagnosis command, fix.

---

## Provisioning

### Provider returns no quotes

**Symptom:** `terradev quote --gpu a100` returns an empty table or skips a provider.

**Diagnosis:**
```bash
# Check which providers are configured
terradev configure --list

# Test a specific provider's credentials
terradev status --providers
```

**Fixes:**
- Run `terradev configure --provider <name>` and re-enter the API key
- Check that the provider account has billing set up and credits available — providers reject API calls from unfunded accounts without a clear error
- Some providers (Crusoe, CoreWeave) require account approval before the API works; check your account status on their dashboard

---

### Provision hangs or times out

**Symptom:** `terradev provision` runs but never returns a running instance.

**Diagnosis:**
```bash
# Check status of pending provisions
terradev status

# Check specific provider for availability issues
terradev quote --providers runpod --gpu a100
```

**Fixes:**
- GPU type may be out of stock on the chosen provider; try `--providers vastai` or remove the provider constraint to let Terradev pick from all available
- Spot instance was outbid before it launched; use `--on-demand` flag
- Region constraint too narrow; remove `--region` to allow any region
- RunPod specifically: ensure account has $10+ credit balance — provisioning fails silently below their minimum

---

### SSH connection refused after provision

**Symptom:** `terradev ssh <instance-id>` returns `Connection refused`.

**Diagnosis:**
```bash
# Check instance is in RUNNING state (not STARTING)
terradev status --instance-id <instance-id>

# Test port directly
nc -zv <instance-ip> 22
```

**Fixes:**
- Instance may still be initializing — wait 30–60 seconds after RUNNING state appears, then retry
- Provider firewall rules blocked port 22; check provider dashboard → instance → network settings
- SSH key mismatch; the auto-generated keypair is stored in `~/.terradev/keys/` — verify it wasn't deleted

---

### NUMA topology not applied

**Symptom:** Training is slower than expected; `numactl --hardware` shows GPU and NIC on different NUMA nodes.

**Diagnosis:**
```bash
# Verify topology on instance
terradev execute -i <instance-id> -c "numactl --hardware"
terradev execute -i <instance-id> -c "lstopo --of txt"
```

**Fixes:**
- Some providers don't expose topology control (Vast.ai community instances, spot instances with pre-configured VMs); use CoreWeave, Crusoe, or Latitude.sh bare metal where topology is enforced
- Re-provision with `--ensure-numa-alignment` flag:
```bash
terradev provision -g H100 -n 4 --ensure-numa-alignment
```

---

## Distributed Training

### NCCL all-reduce hangs

**Symptom:** Distributed training starts then hangs silently; `nvidia-smi` shows GPUs at 100% util initially then drops to 0.

**Diagnosis:**
```bash
# Run preflight before training
terradev preflight --detailed

# Test NCCL connectivity directly
terradev execute -i <instance-id> -c "nccl_test -b 8G -e 8G -s 1073741824"

# Check InfiniBand status
terradev execute -i <instance-id> -c "ibstat -v"
```

**Fixes:**
```bash
# Re-provision with explicit RDMA flags
terradev provision -g H100 -n 4 --ensure-rdma --enable-gpudirect

# If provider doesn't support InfiniBand, disable it for NCCL
terradev train --script train.py --env NCCL_IB_DISABLE=1
```

---

### CUDA out of memory (OOM) during training

**Symptom:** `RuntimeError: CUDA out of memory` or training crashes with OOM error.

**Diagnosis:**
```bash
# Check memory usage across nodes
terradev monitor --job <job-id> --memory-usage
terradev execute -i <instance-id> -c "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
```

**Fixes:**
```bash
# Reduce batch size
terradev train --script train.py --script-args "--batch-size 16"

# Enable gradient checkpointing
terradev train --script train.py --script-args "--gradient-checkpointing"

# Disable FlashOptim if it's conflicting
terradev train --script train.py --flashoptim off

# Enable mixed precision if not already
terradev train --script train.py --script-args "--bf16"
```

---

### FlashOptim crashes or is incompatible

**Symptom:** Training fails with FlashOptim errors, or performance is worse with FlashOptim enabled.

**Diagnosis:**
```bash
# Check FlashOptim status on current job
terradev train-status --job <job-id> | grep flashoptim

# Run preflight FlashOptim check
terradev preflight --flashoptim-check
```

**Fixes:**
```bash
# Disable FlashOptim entirely
terradev train --script train.py --flashoptim off

# Force specific configuration
terradev train --script train.py \
  --flashoptim on \
  --flashoptim-optimizer adamw \
  --flashoptim-master-weight-bits 8
```

FlashOptim is auto-disabled for Megatron-LM scripts, single-GPU jobs, and GPUs with <24GB VRAM. If it's being incorrectly applied, use `--flashoptim off`.

---

### Checkpoint restore fails

**Symptom:** `terradev checkpoint restore` errors or training doesn't resume from expected step.

**Diagnosis:**
```bash
# List checkpoints and verify integrity
terradev checkpoint list --job <job-id> --verify

# Validate specific checkpoint
terradev checkpoint validate <checkpoint-id> --detailed
```

**Fixes:**
```bash
# Repair corrupted checkpoint
terradev checkpoint validate <checkpoint-id> --repair

# Force new checkpoint before restore attempt
terradev checkpoint create --job <job-id> --name pre-restore-backup --force

# If checkpoint is unrecoverable, list remaining options
terradev checkpoint list --job <job-id> --format json
```

---

### Slow training speed (not OOM, not NCCL)

**Diagnosis:**
```bash
# Run bottleneck analysis
terradev monitor --job <job-id> --bottleneck-analysis

# Check GPU utilization per node
terradev execute -i <instance-id> -c "nvtop --interval 1"

# Check network bandwidth
terradev preflight --network-test
```

**Fixes:**
```bash
# 1. Enable mixed precision if not set
terradev train --script train.py --script-args "--bf16"

# 2. Optimize dataset loading — pre-stage with caching
terradev stage -d ./my-dataset --target-regions us-east-1 \
  --parallel-streams 64 --compression zstd

# 3. Increase node count for more parallelism
terradev provision -g H100 -n 8 --parallel 12

# 4. Check if dataset loading is the bottleneck — profile one epoch
terradev train --script train.py --script-args "--profile-data-loading"
```

---

## Inference / vLLM

### vLLM endpoint not responding

**Symptom:** `curl http://<ip>:8000/health` returns connection refused or timeout.

**Diagnosis:**
```bash
# Check endpoint status
terradev infer-status --endpoint <endpoint-id>

# View vLLM logs
terradev logs --endpoint <endpoint-id> --follow
```

**Fixes:**
- vLLM takes 2–5 minutes to load large models — wait and retry `infer-status`
- Model too large for available VRAM; use `--tensor-parallel-size 2` (or higher) to split across GPUs
- Port 8000 blocked by provider firewall; check instance network settings
- If endpoint was put to sleep: `terradev infer-deploy --wake <endpoint-id>`

---

### vLLM throughput lower than expected

**Diagnosis:**
```bash
# Analyze running server
terradev vllm analyze http://<endpoint-ip>:8000 --duration 300

# Benchmark current state
terradev vllm benchmark http://<endpoint-ip>:8000 \
  --concurrent-requests 10 --duration 300
```

**Fixes:**
```bash
# Auto-optimize for throughput
terradev vllm auto-optimize http://<endpoint-ip>:8000 \
  --duration 300 \
  --objective throughput \
  --apply
```

The six knobs that matter most:

| Knob | Conservative default | Optimized |
|---|---|---|
| `max-num-batched-tokens` | 2048 | 16384 |
| `gpu-memory-utilization` | 0.90 | 0.95 |
| `max-num-seqs` | 256 | 512–2048 |
| `enable-prefix-caching` | OFF | ON |
| `enable-chunked-prefill` | OFF | ON |
| CPU cores allocated | 2 + #GPUs | Workload-tuned |

---

### LoRA adapter fails to load

**Symptom:** `terradev lora add` returns an error or adapter appears in list but requests fail.

**Diagnosis:**
```bash
# Check adapter status
terradev lora status --endpoint http://<ip>:8000 --metrics

# List loaded adapters
terradev lora list --endpoint http://<ip>:8000 --detailed
```

**Fixes:**
- Ensure vLLM was deployed with `--enable-lora` — adapters cannot be added to an endpoint launched without this flag; redeploy with `terradev infer-deploy --enable-lora`
- Adapter path must be accessible from the instance (S3 URI or local path on the GPU node)
- Adapter architecture must match the base model — mismatched adapters fail silently on some vLLM versions

---

### KV cache offloading causing errors

**Symptom:** vLLM crashes or returns errors after KV cache offloading auto-applied.

**Fix:**
```bash
# Redeploy without KV offloading
terradev infer-deploy \
  --model ./my-model \
  --gpu-type a100 \
  --provider runpod \
  --no-kv-offloading
```

KV offloading requires sufficient CPU DRAM headroom. If the instance has limited RAM (< 64GB), offloading can cause host OOM. Use an instance with more system RAM or disable offloading.

---

## Kubernetes / Karpenter

### Karpenter not provisioning nodes

**Symptom:** Pods stuck in `Pending` after deploying workloads to the cluster.

**Diagnosis:**
```bash
# Check Karpenter events
terradev karpenter events

# Check Karpenter logs
terradev karpenter logs

# Check nodepool configuration
terradev karpenter nodepools
```

**Fixes:**
- GPU type in nodepool doesn't match pod resource requests — the pod requests `nvidia.com/gpu: 1` but the nodepool is configured for a GPU type not available in the region
- Karpenter API version mismatch — Terradev uses `karpenter.sh/v1` and `karpenter.k8s.aws/v1`; older clusters may have `v1beta1` CRDs installed. Upgrade Karpenter to v1.0+
- Node budget exhausted (consolidation or drift controls) — check Karpenter disruption budget settings

---

### GPU nodes not joining cluster

**Symptom:** Karpenter provisions nodes but they don't become Ready or GPUs aren't schedulable.

**Diagnosis:**
```bash
# Check GPU nodes
terradev karpenter gpu-nodes

# Check resources
terradev karpenter resources
```

**Fixes:**
- NVIDIA device plugin not installed:
```bash
terradev k8s gpu-operator install
```
- Node has GPU but `nvidia.com/gpu` resource not appearing — device plugin DaemonSet may have failed; check pod status in `gpu-operator` namespace
- Driver mismatch — the instance AMI/image has the wrong CUDA version for the GPU

---

## ML Services (Qdrant / Phoenix / Guardrails)

### Qdrant connection refused

**Diagnosis:**
```bash
# Test connection
terradev qdrant test

# Check collection status
terradev qdrant collections
```

**Fixes:**
- Self-hosted Qdrant on port 6333 (REST) / 6334 (gRPC) — ensure firewall allows traffic from your machine to the Qdrant node
- Qdrant Cloud: API key uses `api-key` header, not `Authorization: Bearer` — if you configured Qdrant manually, confirm the header format
- Collection name mismatch — names are case-sensitive

---

### Phoenix traces not appearing

**Diagnosis:**
```bash
# Test Phoenix connection
terradev phoenix test

# Check projects exist
terradev phoenix projects
```

**Fixes:**
- OTLP environment variables not set on the inference/training process:
```bash
# Generate correct env vars for your setup
terradev phoenix otlp-env \
  --endpoint http://phoenix:6006 \
  --project my-project \
  --service-name my-service
```
- Self-hosted Phoenix uses no auth by default — ensure `PHOENIX_COLLECTOR_ENDPOINT` does not have a token set unless you've enabled cloud auth
- Spans use cursor pagination; if spans appear but are stale, the cursor may have advanced past recent data:
```bash
terradev phoenix spans --project my-project --limit 50
```

---

### NeMo Guardrails rejecting valid requests

**Symptom:** Guardrails returns block/refusal for messages that should pass.

**Diagnosis:**
```bash
# Test specific message
terradev guardrails chat --config-id topical --message "Test message"
```

**Fixes:**
- Topical guardrail too strict — regenerate config with narrower topic scope:
```bash
terradev guardrails generate-config --config-type topical --output ./guardrails/
# Edit the Colang config to widen allowed topics
```
- Wrong `config-id` — must match an installed Colang config name exactly
- Memory backend issue (Redis unavailable in prod mode) — check Redis connection or switch to memory backend for testing:
```bash
terradev guardrails deploy --memory-backend memory
```

---

## Provider-Specific Issues

### AWS: `AccessDenied` on provision

Terradev needs these IAM permissions: `ec2:RunInstances`, `ec2:DescribeInstances`, `ec2:TerminateInstances`, `ec2:DescribeInstanceTypes`, `ec2:DescribeSpotPriceHistory`.

Create a minimal IAM policy and attach to your `terradev-sa` service account.

---

### GCP: `RESOURCE_EXHAUSTED` on provision

GPU quotas in GCP are per-region and require manual increase requests. Check Quotas in the GCP Console for `NVIDIA_A100_GPUS` in your target region. Request an increase if at 0.

---

### Azure: `AuthorizationFailed`

The Azure service principal needs `Contributor` role on the subscription:
```bash
az role assignment create \
  --assignee <app-id> \
  --role Contributor \
  --scope /subscriptions/<subscription-id>
```

---

### RunPod: Provision returns but instance never reaches RUNNING

RunPod spot instances can be claimed by another user during the window between quote and provision. Try:
```bash
# Use on-demand to avoid spot race
terradev provision --providers runpod --gpu a100 --on-demand --count 1
```

---

### Vast.ai: `bid too low`

Vast.ai uses a bidding model for some instances. Your bid (usually auto-set to the ask price) was undercut. Increase `--max-price` slightly above current spot rate:
```bash
terradev quote --providers vastai --gpu a100 --spot
# Note the current ask price, then:
terradev provision --providers vastai --gpu a100 --max-price <ask + 0.05>
```

---

### OVHcloud / Alibaba: Authentication errors

These providers use HMAC signature schemes, not Bearer tokens. Do not set `ALIBABA_API_KEY` or `OVH_API_KEY` as generic env vars — use the provider-specific variables:

**Alibaba:** `ALIBABA_ACCESS_KEY_ID`, `ALIBABA_ACCESS_KEY_SECRET`, `ALIBABA_REGION`

**OVHcloud:** `OVH_APPLICATION_KEY`, `OVH_APPLICATION_SECRET`, `OVH_CONSUMER_KEY`, `OVH_ENDPOINT`

---

## Local GPU Discovery

### `terradev local scan` finds no GPUs

**Diagnosis:**
```bash
# Verify NVIDIA driver is loaded
nvidia-smi

# Check NVML directly
python3 -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlDeviceGetCount())"
```

**Fixes:**
- Rust NVML bindings require `libcuda.so` to be in `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
- If NVML fails, Terradev falls back to `nvidia-smi` parsing automatically — if `nvidia-smi` returns results but `terradev local scan` doesn't, file an issue
- WSL2: NVML has partial support; use `nvidia-smi` fallback path

---

### Remote SSH scan fails

```bash
# Test SSH connectivity first
ssh -i ~/.ssh/id_rsa ubuntu@192.168.1.50 "nvidia-smi"

# Then scan
terradev local scan --host 192.168.1.50 --user ubuntu --key ~/.ssh/id_rsa
```

Ensure the SSH key has no passphrase, or use `ssh-agent` before running the scan.

---

## General

### `terradev` command not found after install

```bash
# Verify pip install succeeded
pip show terradev-cli

# Check that pip's bin directory is in PATH
python3 -m site --user-base
# Add <output>/bin to PATH if missing

# Or install with pipx for isolated PATH management
pipx install terradev-cli
```

---

### Import errors on startup

```bash
# Install all optional dependencies
pip install terradev-cli[all]

# Or install specific extras
pip install terradev-cli[aws]    # AWS SDK
pip install terradev-cli[gcp]    # GCP SDK
pip install terradev-cli[azure]  # Azure SDK
```

---

### Commands complete but nothing happens (silent failures)

Most commands print `ERROR: ...` on failure. If you're seeing empty output:

```bash
# Add --verbose flag (available on most commands)
terradev provision --gpu a100 --verbose

# Check credentials are configured
terradev configure --list
```

If a provider is configured but not responding, it's silently skipped in `quote` output. Use `--providers <name>` to force a specific provider and surface the error.

---

## Getting More Help

- **Full command reference:** `COMPLETE_COMMAND_REFERENCE.md`
- **All workflows:** `LIFECYCLES.md`
- **GitHub Issues:** [github.com/theoddden/Terradev/issues](https://github.com/theoddden/Terradev/issues)
- **GitHub Discussions:** [github.com/theoddden/Terradev/discussions](https://github.com/theoddden/Terradev/discussions)
