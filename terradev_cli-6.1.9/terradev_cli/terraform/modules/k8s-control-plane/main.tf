# Kubernetes Control Plane Module
# Supports EKS, GKE, and self-hosted options

locals {
  control_plane_config = {
    eks = {
      type = "eks"
      cluster_class = "managed"
      version = var.kubernetes_version
      endpoint_type = "public"
    }
    gke = {
      type = "gke"
      cluster_class = "managed"
      version = var.kubernetes_version
      endpoint_type = "public"
    }
    self_hosted = {
      type = "self_hosted"
      cluster_class = "self_managed"
      version = var.kubernetes_version
      endpoint_type = "private"
    }
  }
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  name     = var.cluster_name
  role_arn  = aws_iam_role.eks_cluster[0].arn
  version  = var.kubernetes_version
  
  vpc_config {
    subnet_ids = var.subnet_cidrs
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }
  
  tags = merge(var.tags, {
    Name = "${var.cluster_name}-eks"
  })
}

# EKS Node Group (for control plane components)
resource "aws_eks_node_group" "main" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  cluster_name    = aws_eks_cluster.main[0].name
  node_group_name = "control-plane"
  node_role_arn   = aws_iam_role.eks_nodes[0].arn
  subnet_ids      = var.subnet_cidrs
  
  scaling_config {
    desired_size = 2
    max_size     = 3
    min_size     = 1
  }
  
  instance_types = ["t3.medium"]
  
  tags = merge(var.tags, {
    Name = "${var.cluster_name}-control-plane"
  })
}

# IAM Role for EKS Cluster
resource "aws_iam_role" "eks_cluster" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  name = "${var.cluster_name}-eks-cluster"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster[0].name
}

# IAM Role for EKS Nodes
resource "aws_iam_role" "eks_nodes" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  name = "${var.cluster_name}-eks-nodes"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_nodes[0].name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_nodes[0].name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_nodes[0].name
}

# Security Group for EKS
resource "aws_security_group" "eks_cluster" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  name        = "${var.cluster_name}-eks-sg"
  description = "Security group for EKS cluster"
  vpc_id      = var.vpc_id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(var.tags, {
    Name = "${var.cluster_name}-eks-sg"
  })
}

# Generate kubeconfig
resource "local_file" "kubeconfig" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  content = templatefile("${path.module}/kubeconfig.tpl", {
    cluster_name = aws_eks_cluster.main[0].name
    endpoint     = aws_eks_cluster.main[0].endpoint
    ca_data      = aws_eks_cluster.main[0].certificate_authority[0].data
    region       = var.aws_region
  })
  
  filename = "${path.module}/${var.cluster_name}-kubeconfig"
}

# Generate join script for worker nodes
resource "local_file" "join_script" {
  count = var.control_plane_type == "eks" ? 1 : 0
  
  content = templatefile("${path.module}/join-script.tpl", {
    cluster_name = aws_eks_cluster.main[0].name
    endpoint     = aws_eks_cluster.main[0].endpoint
    ca_data      = aws_eks_cluster.main[0].certificate_authority[0].data
    region       = var.aws_region
  })
  
  filename = "${path.module}/${var.cluster_name}-join.sh"
}
