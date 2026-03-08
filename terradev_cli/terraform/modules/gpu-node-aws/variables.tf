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

variable "ami_id" {
  description = "AMI ID for GPU instances"
  type        = string
  default     = "ami-0c02fb55956c7d316" # Ubuntu 22.04 with NVIDIA drivers
}

variable "subnet_id" {
  description = "Subnet ID for the instance"
  type        = string
}

variable "security_group_id" {
  description = "Security group ID"
  type        = string
}

variable "key_name" {
  description = "SSH key pair name"
  type        = string
}

variable "iam_instance_profile" {
  description = "IAM instance profile name"
  type        = string
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
