# MoE Cluster Template — Variables
# Parameterized for any Mixture-of-Experts model

variable "cluster_name" {
  description = "Name of the MoE inference cluster"
  type        = string
  default     = "moe-inference"
}

variable "model_id" {
  description = "HuggingFace model ID (e.g. zai-org/GLM-5-FP8, Qwen/Qwen3.5-397B-A17B)"
  type        = string
}

variable "model_name" {
  description = "Served model name for the API endpoint"
  type        = string
  default     = "moe-model"
}

variable "provider" {
  description = "Cloud provider for GPU nodes"
  type        = string
  default     = "runpod"
  validation {
    condition     = contains(["runpod", "vastai", "lambda", "aws", "coreweave"], var.provider)
    error_message = "Provider must be one of: runpod, vastai, lambda, aws, coreweave."
  }
}

variable "gpu_type" {
  description = "GPU type — must support HBM and NVLink for MoE tensor parallelism"
  type        = string
  default     = "H100_SXM"
  validation {
    condition     = contains(["H100_SXM", "H200_SXM", "A100_SXM_80GB"], var.gpu_type)
    error_message = "MoE models require H100 SXM, H200 SXM, or A100 SXM 80GB."
  }
}

variable "gpu_count" {
  description = "Number of GPUs per node (must match tp_size)"
  type        = number
  default     = 8
  validation {
    condition     = contains([1, 2, 4, 8], var.gpu_count)
    error_message = "GPU count must be 1, 2, 4, or 8."
  }
}

variable "tp_size" {
  description = "Tensor parallelism degree (must equal gpu_count)"
  type        = number
  default     = 8
}

variable "num_replicas" {
  description = "Number of serving replicas (each replica = 1 full GPU node)"
  type        = number
  default     = 1
}

variable "region" {
  description = "Preferred deployment region"
  type        = string
  default     = "us-east-1"
}

variable "use_spot" {
  description = "Use spot/preemptible instances (not recommended for serving)"
  type        = bool
  default     = false
}

variable "serving_backend" {
  description = "Inference serving backend"
  type        = string
  default     = "vllm"
  validation {
    condition     = contains(["vllm", "sglang"], var.serving_backend)
    error_message = "Backend must be vllm or sglang."
  }
}

variable "precision" {
  description = "Model precision"
  type        = string
  default     = "fp8"
  validation {
    condition     = contains(["fp8", "bf16", "fp16", "auto"], var.precision)
    error_message = "Precision must be fp8, bf16, fp16, or auto."
  }
}

variable "gpu_memory_utilization" {
  description = "Fraction of GPU memory to use (0.0-1.0)"
  type        = number
  default     = 0.85
}

variable "max_model_len" {
  description = "Maximum sequence length"
  type        = number
  default     = 32768
}

variable "disk_size_gb" {
  description = "Disk size in GB for model weights and KV cache"
  type        = number
  default     = 500
}

variable "cpu_count" {
  description = "Number of vCPUs per node"
  type        = number
  default     = 96
}

variable "memory_gb" {
  description = "System RAM in GB per node"
  type        = number
  default     = 512
}

variable "shm_size_gb" {
  description = "Shared memory size for NCCL (GB)"
  type        = number
  default     = 32
}

variable "hf_token" {
  description = "HuggingFace API token for gated model access"
  type        = string
  sensitive   = true
  default     = ""
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default = {
    project    = "moe-inference"
    managed_by = "terradev"
    arch       = "moe"
  }
}
