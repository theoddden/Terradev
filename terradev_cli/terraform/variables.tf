variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
}

variable "gpu_type" {
  description = "Type of GPU to provision (e.g., H100, A100, L40)"
  type        = string
}

variable "total_nodes" {
  description = "Total number of GPU nodes to provision"
  type        = number
}

variable "max_price_per_hour" {
  description = "Maximum price per hour per GPU node"
  type        = number
  default     = 4.00
}

variable "prefer_spot" {
  description = "Prefer spot instances over on-demand"
  type        = bool
  default     = true
}

variable "control_plane_type" {
  description = "Type of control plane (eks, gke, self-hosted)"
  type        = string
  default     = "eks"
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "gcp_region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "gcp_project" {
  description = "GCP project ID"
  type        = string
  default     = ""
}

variable "tailscale_authkey" {
  description = "Tailscale authentication key"
  type        = string
  sensitive   = true
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "Override instance types per provider"
  type        = map(string)
  default = {
    aws     = ""
    vastai  = ""
    lambda  = ""
    hyperstack = ""
  }
}

variable "enable_monitoring" {
  description = "Enable cluster monitoring"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable cluster logging"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    "Project"     = "Terradev"
    "ManagedBy"   = "Terraform"
    "Environment" = "production"
  }
}
