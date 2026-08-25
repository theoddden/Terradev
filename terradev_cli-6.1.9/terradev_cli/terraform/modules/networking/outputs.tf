# Network Information
output "network_cidr" {
  description = "Network CIDR block"
  value       = local.network_cidr
}

output "subnet_cidrs" {
  description = "Subnet CIDR blocks"
  value       = local.subnet_cidrs
}

output "tailscale_network" {
  description = "Tailscale network information"
  value = {
    name        = tailscale_network.main.name
    id          = tailscale_network.main.id
    auth_keys   = tailscale_auth_key.nodes[*].key
  }
}

output "tailscale_auth_keys" {
  description = "Tailscale authentication keys for nodes"
  value       = tailscale_auth_key.nodes[*].key
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID"
  value = var.create_vpc && var.cloud_provider == "aws" ? aws_vpc.main[0].id : ""
}

output "subnet_ids" {
  description = "Subnet IDs"
  value = var.create_subnet && var.cloud_provider == "aws" ? aws_subnet.main[*].id : []
}

output "security_group_id" {
  description = "Security group ID"
  value = var.create_vpc && var.cloud_provider == "aws" ? aws_security_group.cluster[0].id : ""
}

output "internet_gateway_id" {
  description = "Internet Gateway ID"
  value = var.create_vpc && var.cloud_provider == "aws" ? aws_internet_gateway.main[0].id : ""
}

output "route_table_id" {
  description = "Route table ID"
  value = var.create_subnet && var.cloud_provider == "aws" ? aws_route_table.main[0].id : ""
}
