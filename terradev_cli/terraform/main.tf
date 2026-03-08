terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tailscale = {
      source  = "tailscale/tailscale"
      version = "~> 0.13"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
}

# Get optimal allocation from price optimizer
data "external" "optimal_allocation" {
  program = ["python3", "${path.module}/price-optimizer/optimal-allocation.py"]
  
  query = {
    gpu_type     = var.gpu_type
    total_needed = var.total_nodes
    max_price    = var.max_price_per_hour
    prefer_spot  = var.prefer_spot
  }
}

# Parse allocation results
locals {
  allocation = jsondecode(data.external.optimal_allocation.result)
  providers = keys(local.allocation)
  
  # Create node configurations
  node_configs = flatten([
    for provider in local.providers : [
      for i in range(local.allocation[provider].count) : {
        provider = provider
        index    = i
        gpu_type = var.gpu_type
        cost_hr  = local.allocation[provider].cost_per_node
        instance_type = local.allocation[provider].instance_type
        spot = local.allocation[provider].spot
      }
    ]
  ])
}

# Networking module - Tailscale VPN
module "networking" {
  source = "./modules/networking"
  
  cluster_name = var.cluster_name
  tailscale_authkey = var.tailscale_authkey
  create_subnet = true
  
  providers = {
    tailscale = tailscale
    aws = aws
  }
}

# Kubernetes control plane
module "control_plane" {
  source = "./modules/k8s-control-plane"
  
  cluster_name = var.cluster_name
  control_plane_type = var.control_plane_type
  aws_region = var.aws_region
  gcp_region = var.gcp_region
  gcp_project = var.gcp_project
  
  network_cidr = module.networking.network_cidr
  subnet_cidrs = module.networking.subnet_cidrs
  
  tailscale_network = module.networking.tailscale_network
  
  providers = {
    aws = aws
  }
}

# GPU Node Modules - Provision in parallel
module "gpu_nodes" {
  source   = "./modules"
  for_each = { for idx, config in local.node_configs : "node-${idx}" => config }
  
  providers = {
    aws = aws
  }
  
  # Pass node configuration to provider-specific modules
  node_config = each.value
  
  # Network configuration
  tailscale_authkey = var.tailscale_authkey
  tailscale_network = module.networking.tailscale_network
  
  # Kubernetes configuration
  cluster_name = var.cluster_name
  kubeconfig_path = module.control_plane.kubeconfig_path
  
  # Bootstrap configuration
  k8s_join_script = module.control_plane.join_script
  gpu_labels = {
    "terradev.io/provider"     = each.value.provider
    "terradev.io/gpu-type"     = each.value.gpu_type
    "terradev.io/hourly-cost"  = tostring(each.value.cost_hr)
    "terradev.io/instance-type" = each.value.instance_type
    "terradev.io/spot"         = tostring(each.value.spot)
  }
}

# Generate cost breakdown
locals {
  cost_breakdown = {
    for provider in local.providers : provider => {
      nodes    = local.allocation[provider].count
      cost_hr  = local.allocation[provider].cost_per_node * local.allocation[provider].count
      cost_mo  = local.allocation[provider].cost_per_node * local.allocation[provider].count * 730
    }
  }
  
  total_cost_hr = sum([for breakdown in values(local.cost_breakdown) : breakdown.cost_hr])
  total_cost_mo = sum([for breakdown in values(local.cost_breakdown) : breakdown.cost_mo])
  
  # Calculate AWS-only cost for comparison
  aws_only_cost_hr = var.total_nodes * local.allocation["aws"].cost_per_node
  aws_only_cost_mo = aws_only_cost_hr * 730
  savings_hr = aws_only_cost_hr - local.total_cost_hr
  savings_mo = aws_only_cost_mo - local.total_cost_mo
  savings_percent = (local.savings_hr / aws_only_cost_hr) * 100
}
