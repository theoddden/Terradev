# AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = merge(var.tags, {
      "Cluster" = var.cluster_name
      "GPUType" = var.gpu_type
    })
  }
}

# Tailscale Provider
provider "tailscale" {
  api_key = var.tailscale_authkey
}

# Random Provider for unique identifiers
provider "random" {}

# Local Provider for local file operations
provider "local" {}
