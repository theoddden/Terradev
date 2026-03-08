# RAG Infrastructure Template

Full Retrieval-Augmented Generation stack provisioned by Terradev.

## Components

| Component | Image | Port | Purpose |
|---|---|---|---|
| **vLLM** | `vllm/vllm-openai:latest` | 8000 | Generation model with auto-applied optimizations |
| **Qdrant** | `qdrant/qdrant:latest` | 6333 (REST), 6334 (gRPC) | Vector similarity search (Apache 2.0) |
| **Embedding** | `huggingface/text-embeddings-inference` | 8001 | Embedding model (BAAI/bge-large-en-v1.5) |
| **Redis** | `redis:7-alpine` | 6379 | LMCache distributed KV cache backend |
| **Phoenix** | `arizephoenix/phoenix:latest` | 6006 | LLM trace observability (optional, ELv2) |
| **Guardrails** | `nemo-guardrails:latest` | 8090 | Output safety layer (optional, Apache 2.0) |

## Auto-Applied Optimizations

- **FlashInfer** fused attention (~50% memory bandwidth recovery)
- **KV Cache Offloading** to CPU DRAM (up to 9x throughput)
- **MTP Speculative Decoding** (up to 2.8x speed)
- **Sleep Mode** (18-200x faster than full restart)
- **LMCache** distributed KV cache via Redis
- **NUMA-aware placement** — Qdrant + embedding model co-located

## Quick Start

```bash
# Provision the full RAG stack
terradev provision --template rag \
  --model zai-org/GLM-5-FP8 \
  --embedding-model BAAI/bge-large-en-v1.5 \
  --vector-db qdrant

# Or deploy with kubectl
kubectl apply -f clusters/rag-template/k8s/deployment.yaml

# Or deploy with Helm
helm install rag ./clusters/rag-template/helm -f clusters/rag-template/helm/values-rag.yaml

# Or deploy with Terraform
cd clusters/rag-template/terraform && terraform apply
```

## Pipeline Flow

```
User Query → Embedding Model (bge-large) → Qdrant Search → Top-K Documents
                                                                    ↓
                                            vLLM Generation ← Context + Query
                                                    ↓
                                            (Optional) NeMo Guardrails → Response
```

## Configuration

Edit `helm/values-rag.yaml` to customize:
- Model selection (`vllm.model`, `embedding.model`)
- Qdrant collection settings (`qdrant.defaultCollection`)
- Resource limits per component
- Enable/disable optional components (Phoenix, Guardrails)
