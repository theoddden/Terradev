# GLM-5 Cluster Outputs

output "node_ids" {
  description = "IDs of provisioned GPU nodes"
  value       = terradev_gpu_node.glm5_serving[*].id
}

output "node_ips" {
  description = "Public IPs of GPU nodes"
  value       = terradev_gpu_node.glm5_serving[*].public_ip
}

output "inference_endpoints" {
  description = "GLM-5 inference API endpoints"
  value = [
    for node in terradev_gpu_node.glm5_serving :
    "http://${node.public_ip}:8000/v1"
  ]
}

output "load_balancer_url" {
  description = "Load balancer URL (if multiple replicas)"
  value       = var.num_replicas > 1 ? "http://${terradev_load_balancer.glm5_lb[0].public_ip}:8000/v1" : "http://${terradev_gpu_node.glm5_serving[0].public_ip}:8000/v1"
}

output "health_check_urls" {
  description = "Health check endpoints"
  value = [
    for node in terradev_gpu_node.glm5_serving :
    "http://${node.public_ip}:8000/health"
  ]
}

output "provider" {
  description = "Cloud provider used"
  value       = var.provider
}

output "gpu_config" {
  description = "GPU configuration summary"
  value = {
    gpu_type       = var.gpu_type
    gpu_count      = var.gpu_count
    num_replicas   = var.num_replicas
    total_gpus     = var.gpu_count * var.num_replicas
    serving_backend = var.serving_backend
    model_id       = var.model_id
  }
}

output "estimated_cost_per_hour" {
  description = "Estimated cost per hour (varies by provider)"
  value = lookup({
    "runpod"    = 23.60
    "vastai"    = 21.00
    "lambda"    = 27.60
    "aws"       = 98.32
    "coreweave" = 35.28
  }, var.provider, 0) * var.num_replicas
}
