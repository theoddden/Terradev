# 🚀 SGLang Command Guide

**Revolutionary workload-specific auto-optimization for SGLang serving with 7 workload types**

---

## 🎯 **Quick Start**

```bash
# Install SGLang support
pip install terradev-cli[sglang]

# Auto-optimize any model for workload type
terradev sglang optimize deepseek-ai/DeepSeek-V3

# Deploy with workload-specific optimizations
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload agentic-chat

# Monitor performance
terradev sglang monitor --endpoint http://localhost:30000
```

---

## 🚀 **SGLang Workload Types & Commands**

### **1. Agentic/Multi-turn Chat**

**Optimized for conversational AI with long context and memory management**

```bash
# Deploy for agentic applications
terradev sglang deploy --model deepseek-ai/DeepSeek-V3 --workload agentic-chat \
  --endpoint http://localhost:30000 --gpu-type H100

# Features auto-applied:
# - LPM (Long Prompt Management) + RadixAttention
# - Cache-aware routing (75-90% cache hit rate)
# - Conversation state management
# - Context window optimization

# Deploy with custom configuration
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload agentic-chat \
  --max-context-length 32768 \
  --cache-size 4096 \
  --temperature 0.7 \
  --endpoint http://localhost:30001

# Test agentic chat workload
terradev sglang test --endpoint http://localhost:30000 --workload agentic-chat \
  --test-file conversations.json

# Benchmark agentic performance
terradev sglang benchmark --endpoint http://localhost:30000 --workload agentic-chat \
  --metrics cache-hit-rate,latency,throughput
```

**Example Usage:**
```python
import requests

# Chat with memory
response = requests.post("http://localhost:30000/v1/chat/completions", json={
    "model": "deepseek-ai/DeepSeek-V3",
    "messages": [
        {"role": "user", "content": "Remember that I like Python programming"},
        {"role": "assistant", "content": "I'll remember that you like Python programming!"},
        {"role": "user", "content": "What programming language should I use for AI?"}
    ]
})
```

---

### **2. High-Throughput Batch**

**Optimized for maximum tokens per second in batch processing**

```bash
# Deploy for batch processing
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload high-throughput \
  --endpoint http://localhost:30002 --gpu-type A100

# Features auto-applied:
# - FCFS (First-Come-First-Served) scheduling
# - CUDA graphs for maximum throughput
# - FP8 quantization
# - Batch size optimization

# Deploy with batch optimization
terradev sglang deploy --model mistralai/Mistral-7B --workload high-throughput \
  --batch-size 32 \
  --max-tokens 2048 \
  --quantization fp8 \
  --endpoint http://localhost:30003

# Test batch throughput
terradev sglang test --endpoint http://localhost:30002 --workload high-throughput \
  --batch-sizes 16,32,64 --concurrent-requests 10

# Benchmark batch performance
terradev sglang benchmark --endpoint http://localhost:30002 --workload high-throughput \
  --duration 300 --metrics tokens-per-second,gpu-utilization
```

**Example Usage:**
```python
import requests

# Batch processing
prompts = [
    "Summarize this article about AI",
    "Translate this text to French",
    "Extract key information from this document",
    "Generate a product description"
] * 8  # 32 requests

responses = []
for prompt in prompts:
    response = requests.post("http://localhost:30002/v1/completions", json={
        "model": "meta-llama/Llama-2-70b-hf",
        "prompt": prompt,
        "max_tokens": 500
    })
    responses.append(response.json())
```

---

### **3. Low-Latency/Real-Time**

**Optimized for real-time applications with minimal Time To First Token**

```bash
# Deploy for real-time applications
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload low-latency \
  --endpoint http://localhost:30004 --gpu-type H100

# Features auto-applied:
# - EAGLE3 speculative decoding
# - Speculative Decoding v2
# - Capped concurrency for consistency
# - 30-50% TTFT (Time To First Token) improvement

# Deploy with latency optimization
terradev sglang deploy --model tinyllama/TinyLlama-1.1B-Chat-v1.0 --workload low-latency \
  --max-concurrent-requests 100 \
  --speculative-decoding true \
  --eagle3 true \
  --endpoint http://localhost:30005

# Test latency performance
terradev sglang test --endpoint http://localhost:30004 --workload low-latency \
  --concurrent-requests 50 --measure ttft,tbt,total-latency

# Benchmark latency
terradev sglang benchmark --endpoint http://localhost:30004 --workload low-latency \
  --metrics ttft,p95-latency,p99-latency
```

**Example Usage:**
```python
import requests
import time

# Real-time chat
start_time = time.time()
response = requests.post("http://localhost:30004/v1/chat/completions", json={
    "model": "meta-llama/Llama-2-7b-hf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": True
})

# Measure Time To First Token
for line in response.iter_lines():
    if line:
        ttft = time.time() - start_time
        print(f"TTFT: {ttft*1000:.2f}ms")
        break
```

---

### **4. MoE Models**

**Optimized for Mixture of Experts models with expert routing**

```bash
# Deploy Mixture of Experts models
terradev sglang deploy --model mistralai/Mixtral-8x7B --workload moe \
  --endpoint http://localhost:30006 --gpu-type H100

# Features auto-applied:
# - DeepEP auto-optimization
# - TBO/SBO (Token/Batch-level Orchestration)
# - EPLB (Expert Load Balancing)
# - Redundant experts (up to 2x throughput)

# Deploy with MoE optimization
terradev sglang deploy --model mistralai/Mixtral-8x7B --workload moe \
  --expert-parallelism true \
  --load-balancing eplb \
  --redundant-experts true \
  --endpoint http://localhost:30007

# Test MoE performance
terradev sglang test --endpoint http://localhost:30006 --workload moe \
  --test-load 100 --expert-utilization

# Benchmark MoE
terradev sglang benchmark --endpoint http://localhost:30006 --workload moe \
  --metrics expert-utilization,throughput,latency
```

**Example Usage:**
```python
import requests

# MoE model inference
response = requests.post("http://localhost:30006/v1/chat/completions", json={
    "model": "mistralai/Mixtral-8x7B",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 1000
})

# Check expert usage (if available)
expert_info = requests.get("http://localhost:30006/v1/experts/stats")
print(f"Expert utilization: {expert_info.json()}")
```

---

### **5. PD (Prefill/Decode) Disaggregated**

**Optimized with separate prefill and decode configurations**

```bash
# Deploy with prefill/decode separation
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload pd-disaggregated \
  --endpoint http://localhost:30008 --gpu-type H100

# Features auto-applied:
# - Separate prefill/decode configurations
# - Production-optimized scheduling
# - Resource allocation optimization
# - Load balancing between phases

# Deploy with custom PD configuration
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload pd-disaggregated \
  --prefill-batch-size 8 \
  --decode-batch-size 32 \
  --prefill-gpu-count 4 \
  --decode-gpu-count 2 \
  --endpoint http://localhost:30009

# Test PD performance
terradev sglang test --endpoint http://localhost:30008 --workload pd-disaggregated \
  --test-long-context --context-lengths 4096,8192,16384

# Benchmark PD
terradev sglang benchmark --endpoint http://localhost:30008 --workload pd-disaggregated \
  --metrics prefill-throughput,decode-throughput,total-latency
```

**Example Usage:**
```python
import requests

# Long context generation
response = requests.post("http://localhost:30008/v1/completions", json={
    "model": "meta-llama/Llama-2-70b-hf",
    "prompt": "Write a detailed article about artificial intelligence" * 100,  # Long prompt
    "max_tokens": 2000
})

# Monitor prefill/decode phases
status = requests.get("http://localhost:30008/v1/pd/status")
print(f"Prefill queue: {status.json()['prefill_queue']}")
print(f"Decode queue: {status.json()['decode_queue']}")
```

---

### **6. Structured Output/RAG**

**Optimized for structured data generation and RAG applications**

```bash
# Deploy for structured data applications
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload structured-output \
  --endpoint http://localhost:30010 --gpu-type A100

# Features auto-applied:
# - xGrammar optimization
# - FSM (Finite State Machine) optimization
# - 10x faster structured output generation
# - JSON schema validation

# Deploy with structured output
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload structured-output \
  --json-schema validation.json \
  --grammar-mode strict \
  --endpoint http://localhost:30011

# Test structured output
terradev sglang test --endpoint http://localhost:30010 --workload structured-output \
  --test-json --schema validation.json

# Benchmark structured output
terradev sglang benchmark --endpoint http://localhost:30010 --workload structured-output \
  --metrics structured-throughput,validation-accuracy
```

**Example Usage:**
```python
import requests
import json

# JSON schema for structured output
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age"]
}

# Generate structured output
response = requests.post("http://localhost:30010/v1/chat/completions", json={
    "model": "meta-llama/Llama-2-7b-hf",
    "messages": [{"role": "user", "content": "Create a profile for a software engineer"}],
    "response_format": {"type": "json_object", "schema": schema}
})

result = response.json()
structured_data = json.loads(result['choices'][0]['message']['content'])
print(f"Structured output: {structured_data}")
```

---

### **7. Hardware-Specific**

**Optimized for specific GPU architectures**

```bash
# Deploy with H100 optimizations
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload hardware-specific \
  --gpu-type H100 --endpoint http://localhost:30012

# Deploy with H200 optimizations
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload hardware-specific \
  --gpu-type H200 --endpoint http://localhost:30013

# Deploy with GB200 optimizations
terradev sglang deploy --model nvidia/Nemotron-4-340B --workload hardware-specific \
  --gpu-type GB200 --endpoint http://localhost:30014

# Deploy with AMD MI300X optimizations
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload hardware-specific \
  --gpu-type MI300X --endpoint http://localhost:30015
```

**Hardware-Specific Features:**
- **H100/H200**: Tensor cores, FP8 support, NVLink optimization
- **H20**: China-specific optimizations, compliance features
- **GB200**: Blackwell architecture, transformer engine
- **AMD MI300X**: CDNA architecture, ROCm optimization

---

## 🔧 **SGLang Management Commands**

### **Status and Monitoring**

```bash
# Check SGLang status
terradev sglang status --endpoint http://localhost:30000

# Health check
terradev sglang health --endpoint http://localhost:30000

# Detailed monitoring
terradev sglang monitor --endpoint http://localhost:30000 \
  --metrics gpu-utilization,memory,cache-hit-rate,throughput

# Real-time monitoring
terradev sglang monitor --endpoint http://localhost:30000 --live --refresh 5s
```

### **Configuration Management**

```bash
# Show current configuration
terradev sglang config --show --endpoint http://localhost:30000

# Update configuration
terradev sglang config --set --endpoint http://localhost:30000 \
  max_tokens=4096 temperature=0.7 top_p=0.9

# Reset to defaults
terradev sglang config --reset --endpoint http://localhost:30000

# Export configuration
terradev sglang config --export --endpoint http://localhost:30000 --output config.json
```

### **Performance Tuning**

```bash
# Auto-tune for latency
terradev sglang tune --endpoint http://localhost:30000 --objective latency

# Auto-tune for throughput
terradev sglang tune --endpoint http://localhost:30000 --objective throughput

# Auto-tune for cost
terradev sglang tune --endpoint http://localhost:30000 --objective cost

# Manual tuning with parameters
terradev sglang tune --endpoint http://localhost:30000 \
  --batch-size 16 --max-concurrent-requests 32 --cache-size 2048
```

### **Scaling Operations**

```bash
# Scale replicas
terradev sglang scale --endpoint http://localhost:30000 --replicas 4

# Auto-scaling configuration
terradev sglang autoscale --endpoint http://localhost:30000 \
  --min-replicas 2 --max-replicas 10 --target-cpu 70

# Load balancing
terradev sglang load-balance --endpoint http://localhost:30000 --strategy round-robin

# Update endpoint
terradev sglang update --endpoint http://localhost:30000 --model-id new-model-id
```

### **Testing and Benchmarking**

```bash
# Comprehensive test suite
terradev sglang test --endpoint http://localhost:30000 --workload all

# Performance benchmark
terradev sglang benchmark --endpoint http://localhost:30000 \
  --duration 600 --metrics all

# Stress test
terradev sglang stress-test --endpoint http://localhost:30000 \
  --concurrent-requests 100 --duration 300

# Workload-specific benchmark
terradev sglang benchmark --endpoint http://localhost:30000 \
  --workload agentic-chat --test-file agentic_tests.json
```

---

## 📊 **Performance Examples**

### **Agentic Chat Performance**
```bash
# Deploy and benchmark
terradev sglang deploy --model deepseek-ai/DeepSeek-V3 --workload agentic-chat
terradev sglang benchmark --endpoint http://localhost:30000 --workload agentic-chat

# Expected results:
# - Cache hit rate: 75-90%
# - Context length: Up to 32K tokens
# - Memory efficiency: 40% improvement
```

### **Low-Latency Performance**
```bash
# Deploy and benchmark
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload low-latency
terradev sglang benchmark --endpoint http://localhost:30004 --workload low-latency

# Expected results:
# - TTFT improvement: 30-50%
# - P95 latency: <100ms
# - Concurrent requests: 100+
```

### **MoE Throughput**
```bash
# Deploy and benchmark
terradev sglang deploy --model mistralai/Mixtral-8x7B --workload moe
terradev sglang benchmark --endpoint http://localhost:30006 --workload moe

# Expected results:
# - Throughput: 2x baseline
# - Expert utilization: 80-95%
# - Load balancing: Optimal distribution
```

---

## 🎯 **Best Practices**

### **Choosing Workload Types**
```bash
# For conversational AI with memory
terradev sglang deploy --model deepseek-ai/DeepSeek-V3 --workload agentic-chat

# For batch processing
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload high-throughput

# For real-time applications
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload low-latency

# For MoE models
terradev sglang deploy --model mistralai/Mixtral-8x7B --workload moe

# For long context
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload pd-disaggregated

# For structured data
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload structured-output

# For hardware optimization
terradev sglang deploy --model meta-llama/Llama-2-70b-hf --workload hardware-specific --gpu-type H100
```

### **Performance Optimization**
```bash
# Start with auto-tuning
terradev sglang tune --endpoint http://localhost:30000 --objective latency

# Monitor key metrics
terradev sglang monitor --endpoint http://localhost:30000 --metrics cache-hit-rate,throughput

# Scale based on demand
terradev sglang autoscale --endpoint http://localhost:30000 --min-replicas 2 --max-replicas 10
```

### **Production Deployment**
```bash
# Deploy with monitoring
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload agentic-chat \
  --monitoring --logging --health-checks

# Set up alerts
terradev sglang alert --endpoint http://localhost:30000 --name high-latency --threshold 200ms

# Backup configuration
terradev sglang config --export --endpoint http://localhost:30000 --backup
```

---

## 🚀 **Getting Started**

```bash
# 1. Install SGLang support
pip install terradev-cli[sglang]

# 2. Deploy your first optimized endpoint
terradev sglang deploy --model meta-llama/Llama-2-7b-hf --workload low-latency

# 3. Test performance
terradev sglang test --endpoint http://localhost:30000 --workload low-latency

# 4. Monitor and optimize
terradev sglang monitor --endpoint http://localhost:30000 --live
terradev sglang tune --endpoint http://localhost:30000 --objective latency

# 5. Scale for production
terradev sglang autoscale --endpoint http://localhost:30000 --min-replicas 2 --max-replicas 10
```

**SGLang with Terradev provides automatic workload optimization, eliminating the need for manual tuning and delivering production-ready performance out of the box.** 🚀
