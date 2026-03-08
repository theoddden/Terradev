# Cluster Information
output "cluster_name" {
  description = "Name of the created cluster"
  value       = var.cluster_name
}

output "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  value       = module.control_plane.kubeconfig_path
}

output "control_plane_endpoint" {
  description = "Kubernetes API endpoint"
  value       = module.control_plane.endpoint
}

# Node Information
output "node_allocation" {
  description = "Optimal node allocation across providers"
  value       = local.allocation
}

output "total_nodes" {
  description = "Total number of nodes provisioned"
  value       = var.total_nodes
}

output "node_details" {
  description = "Detailed information about each node"
  value = {
    for idx, config in local.node_configs : "node-${idx}" => {
      provider      = config.provider
      gpu_type      = config.gpu_type
      instance_type = config.instance_type
      cost_per_hour = config.cost_hr
      spot          = config.spot
    }
  }
}

# Cost Information
output "cost_breakdown" {
  description = "Cost breakdown by provider"
  value       = local.cost_breakdown
}

output "total_cost_per_hour" {
  description = "Total cost per hour for all nodes"
  value       = local.total_cost_hr
}

output "total_cost_per_month" {
  description = "Total cost per month (730 hours)"
  value       = local.total_cost_mo
}

output "savings_analysis" {
  description = "Savings analysis compared to AWS-only"
  value = {
    aws_only_cost_per_hour = local.aws_only_cost_hr
    aws_only_cost_per_month = local.aws_only_cost_mo
    multi_cloud_cost_per_hour = local.total_cost_hr
    multi_cloud_cost_per_month = local.total_cost_mo
    savings_per_hour = local.savings_hr
    savings_per_month = local.savings_mo
    savings_percentage = local.savings_percent
  }
}

# Network Information
output "tailscale_network" {
  description = "Tailscale network information"
  value       = module.networking.tailscale_network
}

output "network_cidr" {
  description = "Network CIDR block"
  value       = module.networking.network_cidr
}

output "subnet_cidrs" {
  description = "Subnet CIDR blocks"
  value       = module.networking.subnet_cidrs
}

# Provider Information
output "provider_credentials" {
  description = "Provider credential status"
  value = {
    aws_configured     = length(var.aws_region) > 0
    gcp_configured     = length(var.gcp_project) > 0
    tailscale_configured = length(var.tailscale_authkey) > 0
  }
}

# GPU Information
output "gpu_summary" {
  description = "Summary of GPU resources"
  value = {
    total_gpus     = var.total_nodes
    gpu_type       = var.gpu_type
    max_price      = var.max_price_per_hour
    prefer_spot    = var.prefer_spot
    actual_average = local.total_cost_hr / var.total_nodes
  }
}

# Commands for user
output "next_steps" {
  description = "Commands to run after provisioning"
  value = [
    "export KUBECONFIG=${module.control_plane.kubeconfig_path}",
    "kubectl get nodes -o wide",
    "kubectl get nodes -L terradev.io/provider,terradev.io/hourly-cost",
    "kubectl apply -f gpu-test-pod.yaml",
    "kubectl logs gpu-test"
  ]
}
