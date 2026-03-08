# AWS GPU Node Module

locals {
  node_name = "${var.cluster_name}-aws-${var.node_config.index}"
  gpu_labels = merge(var.gpu_labels, {
    "terradev.io/provider" = "aws"
    "terradev.io/node-type" = "gpu"
    "terradev.io/created-by" = "terraform"
  })
}

# Create AWS GPU instance
resource "aws_instance" "gpu_node" {
  ami           = var.ami_id
  instance_type = var.node_config.instance_type
  subnet_id     = var.subnet_id
  vpc_security_group_ids = [var.security_group_id]
  
  # Spot or on-demand
  instance_market_options {
    market_type = var.node_config.spot ? "spot" : "on-demand"
  }
  
  # Root volume
  root_block_device {
    volume_type = "gp3"
    volume_size = 100
    iops        = 3000
    throughput = 125
    delete_on_termination = true
  }
  
  # Additional GPU storage
  ebs_block_device {
    device_name = "/dev/sdb"
    volume_type = "gp3"
    volume_size = 500
    iops        = 3000
    throughput = 125
    delete_on_termination = true
  }
  
  # SSH keys
  key_name = var.key_name
  
  # IAM instance profile
  iam_instance_profile = var.iam_instance_profile
  
  # Tags
  tags = merge(var.tags, {
    Name = local.node_name
    Cluster = var.cluster_name
    Provider = "aws"
    GPUType = var.node_config.gpu_type
    NodeType = "gpu"
  })
  
  # User data for GPU setup and Kubernetes join
  user_data = templatefile("${path.module}/bootstrap.sh", {
    cluster_name = var.cluster_name
    tailscale_authkey = var.tailscale_authkey
    k8s_join_script = var.k8s_join_script
    gpu_type = var.node_config.gpu_type
    labels = local.gpu_labels
    provider = "aws"
  })
  
  # Wait for instance to be ready
  provisioner "remote-exec" {
    connection {
      host        = self.public_ip
      user        = "ubuntu"
      private_key = var.ssh_private_key
      timeout     = "10m"
    }
    
    inline = [
      "echo 'Instance is ready'",
      "nvidia-smi",
      "tailscale status"
    ]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Add to Tailscale network — runs in parallel with k8s_node after instance is ready
resource "tailscale_device" "node" {
  name = local.node_name
  
  depends_on = [aws_instance.gpu_node]
}

# Create Kubernetes node (if control plane is ready)
resource "kubernetes_node" "gpu_node" {
  count = var.kubeconfig_path != "" ? 1 : 0
  
  metadata {
    name = local.node_name
    labels = local.gpu_labels
  }
  
  spec {
    provider_id = aws_instance.gpu_node.id
    
    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NoSchedule"
    }
    
    unschedulable = false
  }
  
}

# EBS volume created in parallel with Tailscale + K8s node registration
resource "aws_ebs_volume" "gpu_storage" {
  availability_zone = aws_instance.gpu_node.availability_zone
  size              = 500
  type              = "gp3"
  iops              = 3000
  throughput        = 125
  
  tags = merge(var.tags, {
    Name = "${local.node_name}-storage"
    Cluster = var.cluster_name
    Provider = "aws"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Attach storage volume — depends_on inferred from instance_id + volume_id refs
resource "aws_volume_attachment" "gpu_storage" {
  device_name = "/dev/sdc"
  instance_id = aws_instance.gpu_node.id
  volume_id   = aws_ebs_volume.gpu_storage.id
}
