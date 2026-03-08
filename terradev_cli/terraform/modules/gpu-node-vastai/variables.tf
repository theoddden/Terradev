variable "cluster_name" {
  description = "Name of the cluster"
  type        = string
}

variable "node_config" {
  description = "Node configuration object"
  type = object({
    provider      = string
    index         = number
    gpu_type      = string
    cost_hr       = number
    instance_type = string
    spot          = bool
  })
}

variable "tailscale_authkey" {
  description = "Tailscale authentication key"
  type        = string
  sensitive   = true
}

variable "tailscale_network" {
  description = "Tailscale network configuration"
  type        = any
  default     = {}
}

variable "k8s_join_script" {
  description = "Kubernetes join script path"
  type        = string
}

variable "kubeconfig_path" {
  description = "Path to kubeconfig file"
  type        = string
  default     = ""
}

variable "gpu_labels" {
  description = "GPU-specific labels"
  type        = map(string)
  default     = {}
}

variable "ssh_keys" {
  description = "SSH keys for instance access"
  type        = list(string)
  default     = []
}

variable "ssh_private_key" {
  description = "SSH private key for remote access"
  type        = string
  sensitive   = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
