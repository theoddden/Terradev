# MoE Cluster Outputs

output "node_ids" {
  description = "IDs of provisioned GPU nodes"
  value       = terradev_gpu_node.moe_serving[*].id
}

output "node_ips" {
  description = "Public IPs of GPU nodes"
  value       = terradev_gpu_node.moe_serving[*].public_ip
}

output "inference_endpoints" {
  description = "OpenAI-compatible inference API endpoints"
  value = [
    for node in terradev_gpu_node.moe_serving :
    "http://${node.public_ip}:8000/v1"
  ]
}

output "load_balancer_url" {
  description = "Load balancer URL (single endpoint for all replicas)"
  value       = var.num_replicas > 1 ? "http://${terradev_load_balancer.moe_lb[0].public_ip}:8000/v1" : "http://${terradev_gpu_node.moe_serving[0].public_ip}:8000/v1"
}

output "health_check_urls" {
  description = "Health check endpoints"
  value = [
    for node in terradev_gpu_node.moe_serving :
    "http://${node.public_ip}:8000/health"
  ]
}

output "cluster_summary" {
  description = "Cluster configuration summary"
  value = {
    cluster_name    = var.cluster_name
    model_id        = var.model_id
    model_name      = var.model_name
    provider        = var.provider
    gpu_type        = var.gpu_type
    gpu_count       = var.gpu_count
    tp_size         = var.tp_size
    precision       = var.precision
    backend         = var.serving_backend
    num_replicas    = var.num_replicas
    total_gpus      = var.gpu_count * var.num_replicas
  }
}

output "estimated_cost_per_hour" {
  description = "Estimated cost per hour (varies by provider)"
  value = lookup({
    "runpod"    = 2.95
    "vastai"    = 2.63
    "lambda"    = 3.45
    "aws"       = 12.29
    "coreweave" = 4.41
  }, var.provider, 0) * var.gpu_count * var.num_replicas
}
