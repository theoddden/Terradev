# Networking Module - Tailscale VPN Integration

# Create Tailscale network
resource "tailscale_network" "main" {
  name        = "${var.cluster_name}-network"
  description = "Terradev Kubernetes cluster network"
  
  dns_nameservers = ["8.8.8.8", "8.8.4.4"]
  
  tags = [
    "terradev",
    "kubernetes",
    var.cluster_name
  ]
}

# Create Tailscale auth keys for nodes
resource "tailscale_auth_key" "nodes" {
  count = var.total_nodes > 0 ? var.total_nodes : 1
  
  lifecycle {
    create_before_destroy = true
  }
  
  reusable   = true
  ephemeral  = false
  preauthorized = true
  
  tags = [
    "terradev-node",
    "${var.cluster_name}-node-${count.index}"
  ]
  
  acl = [
    {
      action = "accept"
      src    = ["autogroup:members"]
      dst    = ["*:*"]
    }
  ]
}

# Network CIDR allocation
locals {
  network_cidr = var.network_cidr != "" ? var.network_cid : "10.200.0.0/16"
  subnet_bits = 8
  subnet_count = var.create_subnet ? pow(2, local.subnet_bits) : 1
  
  subnet_cidrs = var.create_subnet ? [
    for i in range(local.subnet_count) :
    cidrsubnet(local.network_cidr, local.subnet_bits, i)
  ] : [local.network_cidr]
}

# Create VPC if needed (AWS)
resource "aws_vpc" "main" {
  count = var.create_vpc && var.cloud_provider == "aws" ? 1 : 0
  
  cidr_block           = local.network_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(var.tags, {
    Name = "${var.cluster_name}-vpc"
    ManagedBy = "Terraform"
  })
}

# Create subnets if needed
resource "aws_subnet" "main" {
  count = var.create_subnet && var.cloud_provider == "aws" ? length(local.subnet_cidrs) : 0
  
  vpc_id            = aws_vpc.main[0].id
  cidr_block        = local.subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index % length(data.aws_availability_zones.available.names)]
  
  map_public_ip_on_launch = true
  
  tags = merge(var.tags, {
    Name = "${var.cluster_name}-subnet-${count.index}"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  count = var.create_vpc && var.cloud_provider == "aws" ? 1 : 0
  
  vpc_id = aws_vpc.main[0].id
  
  tags = merge(var.tags, {
    Name = "${var.cluster_name}-igw"
  })
}

# Route Table
resource "aws_route_table" "main" {
  count = var.create_subnet && var.cloud_provider == "aws" ? 1 : 0
  
  vpc_id = aws_vpc.main[0].id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[0].id
  }
  
  tags = merge(var.tags, {
    Name = "${var.cluster_name}-rt"
  })
}

# Route Table Associations
resource "aws_route_table_association" "main" {
  count = var.create_subnet && var.cloud_provider == "aws" ? length(aws_subnet.main) : 0
  
  subnet_id      = aws_subnet.main[count.index].id
  route_table_id = aws_route_table.main[0].id
}

# Security Group for cluster communication
resource "aws_security_group" "cluster" {
  count = var.create_vpc && var.cloud_provider == "aws" ? 1 : 0
  
  name        = "${var.cluster_name}-cluster-sg"
  description = "Security group for Terradev cluster"
  vpc_id      = aws_vpc.main[0].id
  
  # Allow all traffic within the cluster
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self       = true
  }
  
  # Allow Tailscale traffic
  ingress {
    from_port   = 41641
    to_port     = 41641
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Allow Kubernetes API
  ingress {
    from_port   = 6443
    to_port     = 6443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Allow all egress
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(var.tags, {
    Name = "${var.cluster_name}-cluster-sg"
  })
}

# Network ACL for additional security
resource "aws_network_acl" "main" {
  count = var.create_vpc && var.cloud_provider == "aws" ? 1 : 0
  
  vpc_id = aws_vpc.main[0].id
  
  # Allow all inbound and outbound
  ingress {
    rule_no    = 100
    action     = "allow"
    from_port  = 0
    to_port    = 0
    protocol   = "-1"
    cidr_block = "0.0.0.0/0"
  }
  
  egress {
    rule_no    = 100
    action     = "allow"
    from_port  = 0
    to_port    = 0
    protocol   = "-1"
    cidr_block = "0.0.0.0/0"
  }
  
  tags = merge(var.tags, {
    Name = "${var.cluster_name}-nacl"
  })
}
