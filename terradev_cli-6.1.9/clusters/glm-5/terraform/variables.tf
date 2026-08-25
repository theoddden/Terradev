# GLM-5 Cluster Variables
# Configurable parameters for multi-cloud GPU provisioning

variable "cluster_name" {
  description = "Name of the GLM-5 inference cluster"
  type        = string
  default     = "glm-5-inference"
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
  description = "GPU type — must support 80GB HBM and NVLink for TP=8"
  type        = string
  default     = "H100_SXM"
  validation {
    condition     = contains(["H100_SXM", "H200_SXM", "A100_SXM_80GB"], var.gpu_type)
    error_message = "GLM-5 FP8 requires H100 SXM, H200 SXM, or A100 SXM 80GB for TP=8."
  }
}

variable "gpu_count" {
  description = "Number of GPUs per node (must be 8 for GLM-5 TP=8)"
  type        = number
  default     = 8
  validation {
    condition     = var.gpu_count == 8
    error_message = "GLM-5 requires exactly 8 GPUs for tensor parallelism."
  }
}

variable "num_replicas" {
  description = "Number of serving replicas (each replica = 1 full 8-GPU node)"
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

variable "model_id" {
  description = "HuggingFace model ID"
  type        = string
  default     = "zai-org/GLM-5-FP8"
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
    project     = "glm-5-inference"
    managed_by  = "terradev"
    model       = "GLM-5-FP8"
    precision   = "fp8"
  }
}
