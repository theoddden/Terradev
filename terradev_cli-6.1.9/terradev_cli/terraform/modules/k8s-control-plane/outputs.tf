# Cluster Information
output "cluster_name" {
  description = "Name of the cluster"
  value       = var.cluster_name
}

output "endpoint" {
  description = "Kubernetes API endpoint"
  value = var.control_plane_type == "eks" ? aws_eks_cluster.main[0].endpoint : ""
}

output "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  value = var.control_plane_type == "eks" ? local_file.kubeconfig[0].filename : ""
}

output "join_script" {
  description = "Path to the node join script"
  value = var.control_plane_type == "eks" ? local_file.join_script[0].filename : ""
}

output "cluster_ca_data" {
  description = "Cluster certificate authority data"
  value = var.control_plane_type == "eks" ? aws_eks_cluster.main[0].certificate_authority[0].data : ""
}

output "control_plane_type" {
  description = "Type of control plane deployed"
  value       = var.control_plane_type
}

output "kubernetes_version" {
  description = "Kubernetes version"
  value       = var.kubernetes_version
}

output "region" {
  description = "Region where control plane is deployed"
  value = var.control_plane_type == "eks" ? var.aws_region : var.gcp_region
}

output "vpc_id" {
  description = "VPC ID"
  value = var.control_plane_type == "eks" ? aws_eks_cluster.main[0].vpc_id : ""
}

output "security_group_id" {
  description = "Security group ID"
  value = var.control_plane_type == "eks" ? aws_security_group.eks_cluster[0].id : ""
}
