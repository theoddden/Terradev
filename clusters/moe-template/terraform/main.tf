# MoE Model Inference Cluster — Terraform Configuration
# Generalized template for deploying any MoE model (GLM-5, Qwen 3.5, Mistral Large 3, etc.)
#
# Usage:
#   terraform apply -var="model_id=zai-org/GLM-5-FP8" -var="gpu_count=8"
#   terraform apply -var="model_id=Qwen/Qwen3.5-397B-A17B" -var="provider=vastai"

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
# Locals — compute derived values
# ─────────────────────────────────────────────────────────────────────────────

locals {
  cuda_devices = join(",", [for i in range(var.gpu_count) : tostring(i)])

  vllm_command = join(" ", [
    "vllm serve /models/weights",
    "--tensor-parallel-size ${var.tp_size}",
    "--gpu-memory-utilization ${var.gpu_memory_utilization}",
    "--max-model-len ${var.max_model_len}",
    "--dtype ${var.precision == "fp8" ? "auto" : var.precision}",
    "--trust-remote-code",
    "--enable-auto-tool-choice",
    "--served-model-name ${var.model_name}",
    "--host 0.0.0.0",
    "--port 8000",
  ])

  sglang_command = join(" ", [
    "python3 -m sglang.launch_server",
    "--model-path /models/weights",
    "--tp-size ${var.tp_size}",
    "--mem-fraction-static ${var.gpu_memory_utilization}",
    "--trust-remote-code",
    "--served-model-name ${var.model_name}",
    "--host 0.0.0.0",
    "--port 8000",
  ])

  serve_command = var.serving_backend == "vllm" ? local.vllm_command : local.sglang_command

  container_image = var.serving_backend == "vllm" ? "vllm/vllm-openai:nightly" : "lmsysorg/sglang:latest"
}

# ─────────────────────────────────────────────────────────────────────────────
# GPU Node Pool — NVLink-connected GPUs for MoE tensor parallelism
# ─────────────────────────────────────────────────────────────────────────────

resource "terradev_gpu_node" "moe_serving" {
  count = var.num_replicas

  name          = "${var.cluster_name}-node-${count.index}"
  gpu_type      = var.gpu_type
  gpu_count     = var.gpu_count
  cpu_count     = var.cpu_count
  memory_gb     = var.memory_gb
  disk_size_gb  = var.disk_size_gb
  region        = var.region
  use_spot      = var.use_spot

  # MoE topology constraints — NVLink required for TP across GPUs
  topology {
    require_nvlink    = var.gpu_count > 1
    require_same_node = true
    prefer_numa_align = true
    min_pcie_gen      = var.gpu_type == "H200_SXM" ? 6 : 5
  }

  network {
    open_ports     = [8000, 8080, 9090]
    bandwidth_gbps = 100
  }

  container {
    image = local.container_image

    env = {
      HF_TOKEN               = var.hf_token
      CUDA_VISIBLE_DEVICES   = local.cuda_devices
      NCCL_P2P_DISABLE       = "0"
      NCCL_IB_DISABLE        = "0"
      VLLM_ATTENTION_BACKEND = "FLASH_ATTN"
      TRANSFORMERS_CACHE     = "/models/cache"
      MODEL_ID               = var.model_id
      MODEL_NAME             = var.model_name
    }

    command = local.serve_command

    volumes = {
      "/models" = {
        size_gb = var.disk_size_gb > 100 ? var.disk_size_gb - 100 : var.disk_size_gb
        type    = "nvme"
      }
    }

    resources {
      gpu_memory_gb = 80
      shm_size_gb   = var.shm_size_gb
    }
  }

  health_check {
    path                  = "/health"
    port                  = 8000
    interval_seconds      = 10
    timeout_seconds       = 5
    unhealthy_threshold   = 3
    startup_grace_seconds = 180
  }

  tags = merge(var.tags, {
    model     = var.model_id
    precision = var.precision
    backend   = var.serving_backend
    tp_size   = tostring(var.tp_size)
  })

  lifecycle {
    create_before_destroy = true
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# Load Balancer (multi-replica only)
# ─────────────────────────────────────────────────────────────────────────────

resource "terradev_load_balancer" "moe_lb" {
  count = var.num_replicas > 1 ? 1 : 0

  name          = "${var.cluster_name}-lb"
  backend_nodes = [for node in terradev_gpu_node.moe_serving : node.id]

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
# Model Weight Pre-loading
# ─────────────────────────────────────────────────────────────────────────────

resource "terradev_startup_script" "model_download" {
  count = var.num_replicas

  node_id = terradev_gpu_node.moe_serving[count.index].id

  script = <<-EOT
    #!/bin/bash
    set -e
    echo "Downloading model weights: ${var.model_id}"
    pip install -q huggingface-hub
    python3 -c "
    from huggingface_hub import snapshot_download
    import os
    token = os.environ.get('HF_TOKEN', '')
    snapshot_download(
        '${var.model_id}',
        local_dir='/models/weights',
        token=token if token else None,
        max_workers=8
    )
    print('Model weights downloaded successfully')
    "
  EOT

  run_before_container = true
}
