# VastAI GPU Node Module

locals {
  node_name = "${var.cluster_name}-vastai-${var.node_config.index}"
  gpu_labels = merge(var.gpu_labels, {
    "terradev.io/provider" = "vastai"
    "terradev.io/node-type" = "gpu"
    "terradev.io/created-by" = "terraform"
  })
}

# Create VastAI instance
resource "vastai_instance" "gpu_node" {
  name        = local.node_name
  image       = "ubuntu-22.04"
  instance_type = var.node_config.instance_type
  
  # GPU configuration
  gpu_type    = var.node_config.gpu_type
  gpu_count   = 1
  
  # Spot or on-demand
  spot        = var.node_config.spot
  
  # Network configuration
  network {
    type = "public"
  }
  
  # Storage
  disk_size   = 100
  disk_type   = "ssd"
  
  # SSH keys
  ssh_keys    = var.ssh_keys
  
  # Tags
  tags = merge(var.tags, {
    Name = local.node_name
    Cluster = var.cluster_name
    Provider = "vastai"
    GPUType = var.node_config.gpu_type
  })
  
  # User data for GPU setup and Kubernetes join
  user_data = templatefile("${path.module}/bootstrap.sh", {
    cluster_name = var.cluster_name
    tailscale_authkey = var.tailscale_authkey
    k8s_join_script = var.k8s_join_script
    gpu_type = var.node_config.gpu_type
    labels = local.gpu_labels
    provider = "vastai"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Wait for instance to be ready
resource "null_resource" "wait_for_instance" {
  depends_on = [vastai_instance.gpu_node]
  
  provisioner "remote-exec" {
    connection {
      host        = vastai_instance.gpu_node.public_ip
      user        = "root"
      private_key = var.ssh_private_key
      timeout     = "10m"
    }
    
    inline = [
      "echo 'Instance is ready'",
      "nvidia-smi",
      "tailscale status"
    ]
  }
}

# Add to Tailscale network
resource "tailscale_device" "node" {
  name = local.node_name
  
  depends_on = [null_resource.wait_for_instance]
}

# Create Kubernetes node — runs in parallel with tailscale_device after wait
resource "kubernetes_node" "gpu_node" {
  count = var.kubeconfig_path != "" ? 1 : 0
  
  metadata {
    name = local.node_name
    labels = local.gpu_labels
  }
  
  spec {
    provider_id = vastai_instance.gpu_node.id
    
    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NoSchedule"
    }
  }
  
  depends_on = [null_resource.wait_for_instance]
}
