# Terraform Configuration for Parallel Provisioning
# This defines the infrastructure that the Python brain will control

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
}

# Configure AWS provider
provider "aws" {
  region = var.aws_region
  
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key
  
  default_tags {
    tags = {
      Project     = "Terradev-Arbitrage"
      ManagedBy   = "Python-Brain"
      Environment = var.environment
    }
  }
}

# Configure GCP provider
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  
  # Use service account credentials
  credentials = var.gcp_credentials_path != "" ? var.gcp_credentials_path : null
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "development"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = "terradev-arbitrage"
}

variable "gcp_credentials_path" {
  description = "Path to GCP service account credentials"
  type        = string
  default     = ""
}

variable "aws_access_key" {
  description = "AWS access key"
  type        = string
  sensitive   = true
}

variable "aws_secret_key" {
  description = "AWS secret key"
  type        = string
  sensitive   = true
}

variable "selected_provider" {
  description = "Provider selected by Python brain"
  type        = string
  default     = ""
}

variable "selected_instance_type" {
  description = "Instance type selected by Python brain"
  type        = string
  default     = ""
}

variable "gpu_type" {
  description = "GPU type requested"
  type        = string
  default     = "a100"
}

variable "gpu_count" {
  description = "Number of GPUs requested"
  type        = number
  default     = 1
}

variable "job_duration_hours" {
  description = "Expected job duration in hours"
  type        = number
  default     = 4
}

variable "max_budget_per_hour" {
  description = "Maximum budget per hour"
  type        = number
  default     = 5.0
}

# Local values for computed values
locals {
  timestamp = formatdate("YYYY-MM-DD-hhmm", timestamp())
  instance_name = "${var.environment}-${var.selected_provider}-${var.gpu_type}-${local.timestamp}"
  
  # Provider-specific configurations
  aws_config = {
    ami           = "ami-0c02fb55956c7d316" # Ubuntu 22.04 LTS
    instance_type = var.selected_instance_type
    subnet_id     = "subnet-12345678" # Replace with actual subnet
    security_groups = ["sg-12345678"] # Replace with actual security group
  }
  
  gcp_config = {
    machine_type = var.selected_instance_type
    zone         = "${var.gcp_region}-a"
    image        = "ubuntu-os-cloud/ubuntu-2204-lts"
  }
}

# AWS GPU Instance (conditional)
resource "aws_instance" "gpu_instance" {
  count = var.selected_provider == "aws" ? 1 : 0
  
  ami           = local.aws_config.ami
  instance_type = local.aws_config.instance_type
  subnet_id     = local.aws_config.subnet_id
  
  vpc_security_group_ids = local.aws_config.security_groups
  
  tags = {
    Name = local.instance_name
    JobDuration = var.job_duration_hours
    MaxBudget = var.max_budget_per_hour
  }
  
  # Instance market options for spot instances
  instance_market_options {
    market_type = "spot"
    spot_options {
      instance_interruption_behavior = "terminate"
      spot_instance_type = "persistent"
    }
  }
  
  # Root block device
  root_block_device {
    volume_size = 100
    volume_type = "gp3"
    delete_on_termination = true
  }
  
  # User data for GPU setup
  user_data = templatefile("${path.module}/user-data.sh", {
    gpu_type = var.gpu_type
    job_duration_hours = var.job_duration_hours
  })
  
  # Wait for instance to be ready
  provisioner "remote-exec" {
    connection {
      type        = "ssh"
      host        = self.public_ip
      user        = "ubuntu"
      private_key = file("/dev/null")  # Placeholder - would use actual key in production
    }
    
    inline = [
      "echo 'GPU instance is ready'",
      "nvidia-smi",
      "echo 'Instance provisioned successfully'"
    ]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# GCP GPU Instance (conditional)
resource "google_compute_instance" "gpu_instance" {
  count = var.selected_provider == "gcp" ? 1 : 0
  
  name         = local.instance_name
  machine_type = local.gcp_config.machine_type
  zone         = local.gcp_config.zone
  
  boot_disk {
    initialize_params {
      image = local.gcp_config.image
      size  = 100
      type  = "pd-balanced"
    }
  }
  
  network_interface {
    network = "default"
    access_config {
      # Ephemeral IP
    }
  }
  
  # GPU accelerator
  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }
  
  # Metadata
  metadata = {
    job-duration = var.job_duration_hours
    max-budget   = var.max_budget_per_hour
    gpu-setup    = "true"
  }
  
  # Service account
  service_account {
    email  = "default"
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
  
  # Tags
  tags = [
    "Terradev-Arbitrage",
    "Python-Brain",
    var.environment,
    "JobDuration-${var.job_duration_hours}",
    "MaxBudget-${var.max_budget_per_hour}"
  ]
  
  # Wait for instance to be ready
  provisioner "remote-exec" {
    connection {
      type        = "ssh"
      host        = self.network_interface[0].access_config[0].nat_ip
      user        = "ubuntu"
      private_key = file("/dev/null")  # Placeholder - would use actual key in production
    }
    
    inline = [
      "echo 'GPU instance is ready'",
      "nvidia-smi",
      "echo 'Instance provisioned successfully'"
    ]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# RunPod equivalent (handled by API, not Terraform)
resource "null_resource" "runpod_instance" {
  count = var.selected_provider == "runpod" ? 1 : 0
  
  triggers = {
    instance_id = timestamp()
  }
  
  provisioner "local-exec" {
    command = "python3 ${path.module}/provision_runpod.py '${var.selected_instance_type}' '${var.gpu_type}' '${var.job_duration_hours}'"
  }
}

# User data script for AWS instances
data "template_file" "user_data" {
  template = file("${path.module}/user-data.sh")
  
  vars = {
    gpu_type = var.gpu_type
    job_duration_hours = var.job_duration_hours
  }
}

# Outputs for Python brain to consume
output "aws_instance_id" {
  value = var.selected_provider == "aws" ? aws_instance.gpu_instance[0].id : null
}

output "aws_instance_public_ip" {
  value = var.selected_provider == "aws" ? aws_instance.gpu_instance[0].public_ip : null
}

output "gcp_instance_id" {
  value = var.selected_provider == "gcp" ? google_compute_instance.gpu_instance[0].id : null
}

output "gcp_instance_public_ip" {
  value = var.selected_provider == "gcp" ? google_compute_instance.gpu_instance[0].network_interface[0].access_config[0].nat_ip : null
}

output "runpod_instance_id" {
  value = var.selected_provider == "runpod" ? timestamp() : null
}

output "selected_provider" {
  value = var.selected_provider
}

output "selected_instance_type" {
  value = var.selected_instance_type
}

output "instance_name" {
  value = local.instance_name
}

output "provisioning_time" {
  value = timestamp()
}
