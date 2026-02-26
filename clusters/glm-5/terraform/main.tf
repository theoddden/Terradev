# GLM-5 Inference Cluster — Terraform Configuration
# Provisions 8× H100 SXM GPU nodes optimized for GLM-5 (744B MoE, TP=8)
#
# Usage:
#   terradev provision --task ../task.yaml
#   terraform init && terraform apply

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    terradev = {
      source  = "theoddden/terradev"
      version = "~> 3.0"
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# Provider Configuration
# ─────────────────────────────────────────────────────────────────────────────

provider "terradev" {
  provider_name = var.provider
  region        = var.region
}

# ─────────────────────────────────────────────────────────────────────────────
# GPU Node Pool — 8× H100 SXM per replica (NVLink required for TP=8)
# ─────────────────────────────────────────────────────────────────────────────

resource "terradev_gpu_node" "glm5_serving" {
  count = var.num_replicas

  name          = "${var.cluster_name}-node-${count.index}"
  gpu_type      = var.gpu_type
  gpu_count     = var.gpu_count
  cpu_count     = var.cpu_count
  memory_gb     = var.memory_gb
  disk_size_gb  = var.disk_size_gb
  region        = var.region
  use_spot      = var.use_spot

  # Topology constraints — critical for TP=8 performance
  topology {
    require_nvlink    = true
    require_same_node = true     # all 8 GPUs on single physical node
    prefer_numa_align = true     # NUMA-aware GPU placement
    min_pcie_gen      = 5        # PCIe Gen5 for CPU↔GPU transfers
  }

  # Network configuration for inference serving
  network {
    open_ports = [8000, 8080, 9090]  # vLLM/SGLang, metrics, prometheus
    bandwidth_gbps = 100              # minimum NIC bandwidth
  }

  # Container runtime
  container {
    image = var.serving_backend == "vllm" ? "vllm/vllm-openai:nightly" : "lmsysorg/sglang:glm5-hopper"

    env = {
      HF_TOKEN                = var.hf_token
      CUDA_VISIBLE_DEVICES    = "0,1,2,3,4,5,6,7"
      NCCL_P2P_DISABLE        = "0"
      NCCL_IB_DISABLE         = "0"
      VLLM_ATTENTION_BACKEND  = "FLASH_ATTN"
      TRANSFORMERS_CACHE      = "/models/cache"
    }

    # vLLM serving command
    command = var.serving_backend == "vllm" ? join(" ", [
      "vllm serve ${var.model_id}",
      "--tensor-parallel-size 8",
      "--gpu-memory-utilization 0.85",
      "--speculative-config.method mtp",
      "--speculative-config.num_speculative_tokens 1",
      "--tool-call-parser glm47",
      "--reasoning-parser glm45",
      "--enable-auto-tool-choice",
      "--served-model-name glm-5-fp8",
      "--host 0.0.0.0",
      "--port 8000",
    ]) : join(" ", [
      "python3 -m sglang.launch_server",
      "--model-path ${var.model_id}",
      "--tp-size 8",
      "--tool-call-parser glm47",
      "--reasoning-parser glm45",
      "--speculative-algorithm EAGLE",
      "--speculative-num-steps 3",
      "--speculative-eagle-topk 1",
      "--speculative-num-draft-tokens 4",
      "--mem-fraction-static 0.85",
      "--served-model-name glm-5-fp8",
      "--host 0.0.0.0",
      "--port 8000",
    ])

    # Model weight volume — shared NVMe cache
    volumes = {
      "/models" = {
        size_gb = 400
        type    = "nvme"
      }
    }

    resources {
      gpu_memory_gb = 80
      shm_size_gb   = 32  # shared memory for NCCL
    }
  }

  # Health check
  health_check {
    path                = "/health"
    port                = 8000
    interval_seconds    = 10
    timeout_seconds     = 5
    unhealthy_threshold = 3
    startup_grace_seconds = 180  # model loading takes ~2-3 min
  }

  tags = var.tags

  lifecycle {
    create_before_destroy = true
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# Load Balancer — distributes across serving replicas
# ─────────────────────────────────────────────────────────────────────────────

resource "terradev_load_balancer" "glm5_lb" {
  count = var.num_replicas > 1 ? 1 : 0

  name = "${var.cluster_name}-lb"

  backend_nodes = [for node in terradev_gpu_node.glm5_serving : node.id]

  listener {
    port     = 8000
    protocol = "http"
  }

  health_check {
    path     = "/health"
    port     = 8000
    interval = 10
  }

  tags = var.tags
}

# ─────────────────────────────────────────────────────────────────────────────
# Model Weight Pre-loading — download weights before serving starts
# ─────────────────────────────────────────────────────────────────────────────

resource "terradev_startup_script" "model_download" {
  count = var.num_replicas

  node_id = terradev_gpu_node.glm5_serving[count.index].id

  script = <<-EOT
    #!/bin/bash
    set -e

    echo "📦 Downloading GLM-5-FP8 model weights..."
    pip install huggingface-hub
    python3 -c "
    from huggingface_hub import snapshot_download
    import os
    token = os.environ.get('HF_TOKEN', '')
    snapshot_download(
        '${var.model_id}',
        local_dir='/models/GLM-5-FP8',
        token=token if token else None,
        max_workers=8
    )
    print('✅ Model weights downloaded successfully')
    "
  EOT

  run_before_container = true
}
