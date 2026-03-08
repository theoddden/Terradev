variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
}

variable "control_plane_type" {
  description = "Type of control plane (eks, gke, self-hosted)"
  type        = string
  default     = "eks"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
}

variable "gcp_project" {
  description = "GCP project ID"
  type        = string
}

variable "network_cidr" {
  description = "Network CIDR block"
  type        = string
}

variable "subnet_cidrs" {
  description = "List of subnet CIDR blocks"
  type        = list(string)
}

variable "vpc_id" {
  description = "VPC ID for AWS"
  type        = string
  default     = ""
}

variable "tailscale_network" {
  description = "Tailscale network configuration"
  type        = any
  default     = {}
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
