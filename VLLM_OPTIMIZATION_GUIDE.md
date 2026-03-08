# vLLM Optimization Guide - 6 Critical Knobs Most Teams Never Touch

This guide explains the 6 critical vLLM optimizations that can dramatically improve your inference performance and cost efficiency. These optimizations are automatically applied in Terradev's vLLM integration.

## 🔧 The 6 Critical Knobs

### 1. `--max-num-batched-tokens` 
**Default:** 2048 | **Optimized:** 16384 (throughput) / 4096 (latency)

The single biggest throughput lever most teams never touch. The default is optimized for ITL (inter-token latency) not throughput.

```bash
# Throughput-optimized
--max-num-batched-tokens 16384

# Latency-optimized  
--max-num-batched-tokens 4096
```

**Impact:** 8x throughput improvement for batch-heavy workloads

### 2. `--gpu-memory-utilization`
**Default:** 0.90 | **Optimized:** 0.95

This leaves 10% VRAM idle on single-instance prod for no reason.

```bash
--gpu-memory-utilization 0.95
```

**Impact:** 5% more VRAM available for larger models/batches

### 3. `--max-num-seqs`
**Default:** 256 (V0) / 1024 (V1) | **Optimized:** 1024 (throughput) / 512 (latency)

This hard caps your concurrency. Bursty traffic hits this ceiling and queues silently.

```bash
# Throughput-optimized
--max-num-seqs 1024

# Latency-optimized
--max-num-seqs 512
```

**Impact:** Prevents silent queuing under bursty traffic

### 4. `--enable-prefix-caching`
**Default:** OFF | **Optimized:** ON

This gives you free throughput win if any requests share long system prompts or RAG chunks. No downside.

```bash
--enable-prefix-caching
```

**Impact:** Free throughput improvement for shared prompts

### 5. `--enable-chunked-prefill`
**Default:** OFF in V0, ON in V1 | **Optimized:** ON

If you're on V0, turn it on. If you're on V1, verify it's actually on.

```bash
--enable-chunked-prefill
```

**Impact:** Better prefill performance, especially for long prompts

### 6. CPU Core Allocation
**Default:** Usually underprovisioned | **Optimized:** 2 + #GPUs physical cores

V1 runs a busy loop on the engine core. If you starve it of CPU then your GPU sits idle. You'll see 40% GPU utilisation and spend 3 days blaming the model.

```yaml
# For 8 GPUs: allocate 10+ CPU cores
resources:
  requests:
    cpu: "10"  # 2 + 8 GPUs
  limits:
    cpu: "16"  # Extra headroom
```

**Impact:** Prevents CPU starvation of GPU workloads

## 📊 Optimization Profiles

### Throughput-Heavy Production
```bash
--max-num-batched-tokens 16384
--gpu-memory-utilization 0.95
--enable-prefix-caching
--enable-chunked-prefill
--max-num-seqs 1024
```

### Latency-Sensitive Production
```bash
--max-num-batched-tokens 4096
--max-num-seqs 512
--enable-chunked-prefill
--gpu-memory-utilization 0.95
--enable-prefix-caching
```

## 🚀 Terradev CLI Usage

### Generate Optimized Configurations
```bash
# Throughput optimization
terradev vllm optimize -m meta-llama/Llama-2-7b-hf -t throughput

# Latency optimization with 4 GPUs
terradev vllm optimize -m mistralai/Mistral-7B-v0.1 -t latency -g 4

# Output Helm values
terradev vllm optimize -m meta-llama/Llama-2-70b-hf -t throughput -o helm

# Output JSON config
terradev vllm optimize -m codellama/CodeLlama-34b-hf -t latency -o config
```

### Benchmark Your Endpoint
```bash
# Basic benchmark
terradev vllm benchmark -e http://localhost:8000

# Concurrent load test
terradev vllm benchmark -e http://localhost:8000 -c 10

# With custom prompt
terradev vllm benchmark -e http://localhost:8000 -p "Write a Python function for fibonacci"
```

## 🐳 Docker Example

```dockerfile
FROM vllm/vllm-openai:nightly

# Throughput-optimized vLLM
CMD ["vllm", "serve", "meta-llama/Llama-2-7b-hf", \
     "--max-num-batched-tokens", "16384", \
     "--gpu-memory-utilization", "0.95", \
     "--max-num-seqs", "1024", \
     "--enable-prefix-caching", \
     "--enable-chunked-prefill"]
```

## ☸️ Kubernetes Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-optimized
spec:
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:nightly
        command: ["vllm", "serve", "/models/weights"]
        args:
        - --max-num-batched-tokens=16384
        - --gpu-memory-utilization=0.95
        - --max-num-seqs=1024
        - --enable-prefix-caching
        - --enable-chunked-prefill
        resources:
          requests:
            nvidia.com/gpu: "8"
            cpu: "10"  # 2 + #GPUs
          limits:
            nvidia.com/gpu: "8"
            cpu: "16"
```

## 📈 Performance Impact

| Optimization | Throughput Gain | Latency Impact | Notes |
|-------------|----------------|----------------|-------|
| max-num-batched-tokens (16384) | 8x | +10-20% | Biggest throughput lever |
| gpu-memory-utilization (0.95) | 5% | 0% | Free VRAM |
| max-num-seqs (1024) | 2-3x | -5% | Prevents queuing |
| prefix-caching | 1.5-3x | -10% | Shared prompts |
| chunked-prefill | 1.2-2x | -5% | Long prompts |
| CPU cores (2+#GPUs) | 1.5-2x | -15% | Prevents starvation |

## ⚠️ Important Notes

1. **Version Compatibility:** These optimizations work best with vLLM ≥0.15.0
2. **Memory Usage:** Higher `max-num-batched-tokens` uses more RAM
3. **CPU Allocation:** Always allocate 2+ physical cores per GPU
4. **Monitoring:** Watch GPU utilization - if <70%, increase CPU allocation
5. **Testing:** Always benchmark with your specific workload

## 🔍 Verification

Check that optimizations are applied:

```bash
# Check vLLM server logs for the flags
docker logs vllm-container | grep -E "(max-num-batched|gpu-memory|prefix-caching)"

# Monitor GPU utilization
nvidia-smi dmon -s u -c 10

# Check for queuing
curl http://localhost:8000/metrics | grep vllm
```

## 📚 Additional Resources

- [vLLM Performance Tuning Guide](https://docs.vllm.ai/en/latest/performance_tuning.html)
- [Terradev vLLM Integration](./terradev_cli/ml_services/vllm_service.py)
- [Helm Templates](./clusters/moe-template/helm/values-moe.yaml)

---

**Remember:** These optimizations are automatically applied when using Terradev's vLLM integration. Use the CLI commands to generate custom configurations for your specific needs.
