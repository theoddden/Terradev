# Node Information
output "node_name" {
  description = "Name of the GPU node"
  value       = local.node_name
}

output "instance_id" {
  description = "VastAI instance ID"
  value       = vastai_instance.gpu_node.id
}

output "public_ip" {
  description = "Public IP address"
  value       = vastai_instance.gpu_node.public_ip
}

output "private_ip" {
  description = "Private IP address"
  value       = vastai_instance.gpu_node.private_ip
}

output "gpu_type" {
  description = "GPU type"
  value       = var.node_config.gpu_type
}

output "instance_type" {
  description = "Instance type"
  value       = var.node_config.instance_type
}

output "cost_per_hour" {
  description = "Cost per hour"
  value       = var.node_config.cost_hr
}

output "spot" {
  description = "Whether this is a spot instance"
  value       = var.node_config.spot
}

output "tailscale_ip" {
  description = "Tailscale IP address"
  value       = tailscale_device.node.tailscale_ip
}

output "kubernetes_node_name" {
  description = "Kubernetes node name"
  value = var.kubeconfig_path != "" ? kubernetes_node.gpu_node[0].metadata[0].name : ""
}

output "status" {
  description = "Node status"
  value = {
    instance = vastai_instance.gpu_node.state
    kubernetes = var.kubeconfig_path != "" ? "Ready" : "Not joined"
    tailscale = tailscale_device.node.status
  }
}
