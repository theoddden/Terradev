variable "cluster_name" {
  description = "Name of the cluster"
  type        = string
}

variable "tailscale_authkey" {
  description = "Tailscale authentication key"
  type        = string
  sensitive   = true
}

variable "create_subnet" {
  description = "Whether to create subnets"
  type        = bool
  default     = true
}

variable "create_vpc" {
  description = "Whether to create VPC"
  type        = bool
  default     = true
}

variable "network_cidr" {
  description = "Network CIDR block"
  type        = string
  default     = ""
}

variable "cloud_provider" {
  description = "Cloud provider"
  type        = string
  default     = "aws"
}

variable "total_nodes" {
  description = "Total number of nodes expected"
  type        = number
  default     = 0
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
