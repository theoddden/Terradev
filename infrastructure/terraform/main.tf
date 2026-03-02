# Terraform configuration for Terradev Microservices Infrastructure
# AWS EKS, RDS, Redis, and supporting services

# OCI Provider Configuration
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    oci = {
      source  = "oracle-oci/oci"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    tensordock = {
      source  = "terraform-providers/tensordock"
      version = "~> 1.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

provider "oci" {
  tenancy_ocid     = var.oci_tenancy_ocid
  user_ocid        = var.oci_user_ocid
  fingerprint      = var.oci_fingerprint
  private_key_path = var.oci_private_key_path
  region           = var.oci_region
}

provider "tensordock" {
  client_id = var.tensordock_client_id
  api_token = var.tensordock_api_token
  
  # Rate limiting to prevent throttling
  rate_limit {
    requests_per_second = 5
    burst = 10
  }
  
  # Retry configuration
  retry {
    max_attempts = 3
    min_delay = "2s"
    max_delay = "30s"
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.terradev.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.terradev.token
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "oci_region" {
  description = "OCI region"
  type        = string
  default     = "us-ashburn-1"
}

variable "oci_tenancy_ocid" {
  description = "OCI tenancy OCID"
  type        = string
}

variable "oci_user_ocid" {
  description = "OCI user OCID"
  type        = string
}

variable "oci_fingerprint" {
  description = "OCI API key fingerprint"
  type        = string
}

variable "oci_private_key_path" {
  description = "Path to OCI private key"
  type        = string
  default     = "~/.oci/oci_api_key.pem"
}

variable "aws_access_key_id" {
  description = "AWS access key ID"
  type        = string
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "AWS secret access key"
  type        = string
  sensitive   = true
}

variable "aws_kms_key_id" {
  description = "AWS KMS key ID"
  type        = string
}

# Azure variables
variable "azure_tenant_id" {
  description = "Azure tenant ID"
  type        = string
  sensitive   = true
}

variable "azure_client_id" {
  description = "Azure client ID"
  type        = string
  sensitive   = true
}

variable "azure_client_secret" {
  description = "Azure client secret"
  type        = string
  sensitive   = true
}

variable "azure_subscription_id" {
  description = "Azure subscription ID"
  type        = string
  sensitive   = true
}

# TensorDock variables
variable "tensordock_client_id" {
  description = "TensorDock Client ID"
  type        = string
  sensitive   = true
  default     = "6f71b631-5a32-42f1-94a5-d089adffdb24"
}

variable "tensordock_api_token" {
  description = "TensorDock API Token"
  type        = string
  sensitive   = true
  default     = "rxOI1pnIZ9UL3J8MZmMmYBkKk7EwPdyS"
}

variable "tensordock_region" {
  description = "TensorDock region/location"
  type        = string
  default     = "us-east"
}

variable "tensordock_ssh_public_key" {
  description = "SSH public key for TensorDock instances"
  type        = string
  default     = ""
}

variable "azure_vault_name" {
  description = "Azure Key Vault name"
  type        = string
}

variable "azure_resource_group" {
  description = "Azure resource group name"
  type        = string
}

variable "azure_key_name" {
  description = "Azure key name"
  type        = string
}

variable "azure_storage_account" {
  description = "Azure storage account name"
  type        = string
}

variable "azure_container_name" {
  description = "Azure blob container name"
  type        = string
}

# GCP variables
variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_service_account_key" {
  description = "GCP service account key"
  type        = string
  sensitive   = true
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_key_ring_name" {
  description = "GCP key ring name"
  type        = string
}

variable "gcp_key_name" {
  description = "GCP key name"
  type        = string
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "terradev-cluster"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}
# OCI Resources
module "oci_vcn" {
  source  = "oracle-terraform-modules/vcn/oci"
  version = "3.0.0"

  compartment_id = var.oci_tenancy_ocid
  vcn_cidr_blocks = ["10.0.0.0/16"]
  vcn_dns_label   = "terradevvcn"
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
  }
}

module "oci_subnets" {
  source  = "oracle-terraform-modules/subnets/oci"
  version = "2.0.0"

  compartment_id = var.oci_tenancy_ocid
  vcn_id         = module.oci_vcn.vcn_id
  
  subnet_cidrs                 = ["10.0.1.0/24", "10.0.2.0/24"]
  subnet_dns_labels            = ["terradevsub1", "terradevsub2"]
  subnet_prohibit_public_ip_on_vnic = false
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
  }
}

module "oci_compute" {
  source  = "oracle-terraform-modules/compute/oci"
  version = "0.9.0"

  compartment_id      = var.oci_tenancy_ocid
  subnet_id           = module.oci_subnets.subnets["terradevsub1"].subnet_id
  availability_domain = "Uocm:US-ASHBURN-AD-1"
  
  # GPU instances for ML workloads
  instance_shape   = "VM.GPU.A100_v2"
  instance_shape_config = {
    ocpus         = 8
    memory_in_gbs = 512
  }
  
  source_details = {
    source_type = "image"
    image_id    = data.oci_core_images.ubuntu.images[0].id
  }
  
  assign_public_ip = true
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Type        = "gpu-compute"
  }
}

# OCI Cloud Advisor Integration
data "oci_optimizer_recommendations" "terradev_recommendations" {
  compartment_id = var.oci_tenancy_ocid
  
  # Get recommendations for cost optimization
  category    = "COST_OPTIMIZATION"
  level       = "HIGH"
  status      = "ACTIVE"
}

# OCI Cloud Advisor Recommendations
resource "oci_optimizer_profile" "terradev_profile" {
  compartment_id = var.oci_tenancy_ocid
  name          = "terradev-optimization"
  
  description = "Terradev cost optimization profile"
  
  # Cost optimization settings
  cost_analysis_target {
    type        = "ABSOLUTE"
    target_type = "REDUCE_COST"
    value       = 30.0  # Target 30% cost reduction
  }
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
  }
}

# OCI Budget for cost tracking
resource "oci_budget_budget" "terradev_budget" {
  compartment_id = var.oci_tenancy_ocid
  display_name  = "Terradev ML Training Budget"
  amount        = 1000.0  # $1000 monthly budget
  start_date    = timeadd(timestamp(), "0h")
  
  budget_processing_period_start_offset = 0
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
  }
}

# OCI Budget Alert
resource "oci_budget_alert" "terradev_alert" {
  budget_id     = oci_budget_budget.terradev_budget.id
  display_name = "Terradev Budget Alert"
  type          = "ACTUAL"
  threshold     = 80.0  # Alert at 80% of budget
  threshold_type = "PERCENTAGE"
  
  description = "Alert when Terradev spending exceeds 80% of budget"
}

# OCI Data Sources
data "oci_core_images" "ubuntu" {
  compartment_id           = var.oci_tenancy_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = "VM.GPU.A100_v2"
  
  most_recent = true
}

data "oci_optimizer_recommendations" "cost_optimization" {
  compartment_id = var.oci_tenancy_ocid
  category       = "COST_OPTIMIZATION"
  level          = "HIGH"
}
# Oracle Database MultiCloud Data Plane Integration
resource "oci_database_multicloud_aws_connector" "terradev_aws" {
  compartment_id = var.oci_tenancy_ocid
  display_name = "Terradev AWS Connector"
  description = "AWS connector for Terradev multi-cloud database"
  
  aws_credentials {
    access_key_id = var.aws_access_key_id
    secret_access_key = var.aws_secret_access_key
    region = var.aws_region
  }
  
  connector_type = "IDENTITY"
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "aws"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "oci_database_multicloud_azure_connector" "terradev_azure" {
  compartment_id = var.oci_tenancy_ocid
  display_name = "Terradev Azure Connector"
  description = "Azure connector for Terradev multi-cloud database"
  
  azure_credentials {
    tenant_id       = var.azure_tenant_id
    client_id       = var.azure_client_id
    client_secret   = var.azure_client_secret
    subscription_id = var.azure_subscription_id
  }
  
  connector_type = "IDENTITY"
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "azure"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "oci_database_multicloud_gcp_connector" "terradev_gcp" {
  compartment_id = var.oci_tenancy_ocid
  display_name = "Terradev GCP Connector"
  description = "GCP connector for Terradev multi-cloud database"
  
  gcp_credentials {
    project_id        = var.gcp_project_id
    service_account_key = var.gcp_service_account_key
    region            = var.gcp_region
  }
  
  connector_type = "IDENTITY"
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "gcp"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Azure Blob Container for Terradev
resource "oci_database_multicloud_azure_blob_container" "terradev_container" {
  compartment_id = var.oci_tenancy_ocid
  connector_id   = oci_database_multicloud_azure_connector.terradev_azure.id
  display_name   = "Terradev Azure Container"
  description    = "Azure blob container for Terradev data storage"
  
  storage_account_name = var.azure_storage_account
  container_name      = var.azure_container_name
  access_tier         = "Hot"
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "azure"
    Type        = "storage"
  }
}

# Mount Azure Blob Container on Exadata
resource "oci_database_multicloud_azure_blob_mount" "terradev_mount" {
  compartment_id    = var.oci_tenancy_ocid
  connector_id      = oci_database_multicloud_azure_connector.terradev_azure.id
  blob_container_id = oci_database_multicloud_azure_blob_container.terradev_container.id
  display_name     = "Terradev Azure Mount"
  description      = "Mount point for Azure blob container on Exadata"
  
  mount_path = "/mnt/terradev-azure"
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "azure"
    Type        = "mount"
  }
}

# Azure Key Vault for Terradev
resource "oci_database_multicloud_azure_vault" "terradev_vault" {
  compartment_id = var.oci_tenancy_ocid
  connector_id   = oci_database_multicloud_azure_connector.terradev_azure.id
  display_name   = "Terradev Azure Vault"
  description    = "Azure vault for Terradev encryption keys"
  
  vault_name      = var.azure_vault_name
  resource_group = var.azure_resource_group
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "azure"
    Type        = "vault"
  }
}

# Azure Key for Encryption
resource "oci_database_multicloud_azure_key" "terradev_azure_key" {
  compartment_id = var.oci_tenancy_ocid
  vault_id       = oci_database_multicloud_azure_vault.terradev_vault.id
  display_name   = "Terradev Azure Encryption Key"
  description    = "Azure key for Terradev data encryption"
  
  key_name  = var.azure_key_name
  key_type  = "ENCRYPTION"
  key_size  = 2048
  
  key_operations = ["encrypt", "decrypt", "sign", "verify"]
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "azure"
    Type        = "key"
  }
}

# GCP Key Ring for Terradev
resource "oci_database_multicloud_gcp_key_ring" "terradev_key_ring" {
  compartment_id = var.oci_tenancy_ocid
  connector_id   = oci_database_multicloud_gcp_connector.terradev_gcp.id
  display_name   = "Terradev GCP Key Ring"
  description    = "GCP key ring for Terradev encryption keys"
  
  key_ring_name = var.gcp_key_ring_name
  project_id    = var.gcp_project_id
  location      = var.gcp_region
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "gcp"
    Type        = "key_ring"
  }
}

# GCP Key for Encryption
resource "oci_database_multicloud_gcp_key" "terradev_gcp_key" {
  compartment_id = var.oci_tenancy_ocid
  key_ring_id   = oci_database_multicloud_gcp_key_ring.terradev_key_ring.id
  display_name   = "Terradev GCP Encryption Key"
  description    = "GCP key for Terradev data encryption"
  
  key_name         = var.gcp_key_name
  key_type         = "ENCRYPTION"
  algorithm        = "RSA_SIGN_PKCS1_2048_SHA256"
  protection_level = "SOFTWARE"
  
  key_operations = ["encrypt", "decrypt", "sign", "verify"]
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "gcp"
    Type        = "key"
  }
}

# AWS Key for Encryption
resource "oci_database_multicloud_aws_key" "terradev_aws_key" {
  compartment_id = var.oci_tenancy_ocid
  connector_id   = oci_database_multicloud_aws_connector.terradev_aws.id
  display_name   = "Terradev AWS Encryption Key"
  description    = "AWS key for Terradev data encryption"
  
  key_id  = var.aws_kms_key_id
  key_type = "ENCRYPTION"
  region  = var.aws_region
  
  freeform_tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "aws"
    Type        = "key"
  }
}
# AWS VPC Configuration
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.1.0.0/16"  # Different CIDR to avoid conflict with OCI

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.1.1.0/24", "10.1.2.0/24", "10.1.3.0/24"]
  public_subnets  = ["10.1.101.0/24", "10.1.102.0/24", "10.1.103.0/24"]

  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Environment = var.environment
    Project     = "terradev"
    Provider    = "aws"
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"
  
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  node_groups = {
    terradev_nodes = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 2

      instance_types = ["m5.large", "m5.xlarge"]
      
      k8s_labels = {
        Environment = var.environment
        Project     = "terradev"
        NodeGroup   = "general"
      }
    }
    
    terradev_gpu_nodes = {
      desired_capacity = 0
      max_capacity     = 5
      min_capacity     = 0

      instance_types = ["p3.2xlarge", "g4dn.xlarge"]
      
      k8s_labels = {
        Environment = var.environment
        Project     = "terradev"
        NodeGroup   = "gpu"
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = "terradev"
  }
}

# EKS Cluster Auth
data "aws_eks_cluster_auth" "terradev" {
  name = module.eks.cluster_name
}

# RDS for PostgreSQL
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${var.cluster_name}-postgres"

  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"

  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 500
  storage_encrypted     = true

  db_name  = "terradev"
  username = "terradev_admin"
  port     = 5432

  vpc_security_group_ids = [module.rds_security_group.security_group_id]
  db_subnet_group_name   = module.vpc.database_subnet_group_name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true

  tags = {
    Environment = var.environment
    Project     = "terradev"
  }
}

# RDS Security Group
resource "aws_security_group" "rds_security_group" {
  name_prefix = "${var.cluster_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Environment = var.environment
    Project     = "terradev"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Redis Cluster
module "elasticache" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  cluster_id = "${var.cluster_name}-redis"

  engine = "redis"
  engine_version = "7.0"

  node_type = "cache.t3.micro"
  num_cache_nodes = 3
  parameter_group_name = "default.redis7"

  port = 6379

  subnet_group_name = module.vpc.redis_subnet_group_name
  security_group_ids = [module.redis_security_group.security_group_id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = "your-redis-auth-token"

  tags = {
    Environment = var.environment
    Project     = "terradev"
  }
}

# Redis Security Group
resource "aws_security_group" "redis_security_group" {
  name_prefix = "${var.cluster_name}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Environment = var.environment
    Project     = "terradev"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# S3 Bucket for storage
resource "aws_s3_bucket" "terradev_storage" {
  bucket = "${var.cluster_name}-${var.environment}-storage"

  tags = {
    Environment = var.environment
    Project     = "terradev"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "terradev_storage_versioning" {
  bucket = aws_s3_bucket.terradev_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "terradev_storage_encryption" {
  bucket = aws_s3_bucket.terradev_storage.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# IAM Roles for Service Accounts
resource "aws_iam_policy" "rds_access" {
  name        = "${var.cluster_name}-rds-access"
  description = "Policy for RDS access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "rds:Describe*",
          "rds:List*"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_policy" "s3_access" {
  name        = "${var.cluster_name}-s3-access"
  description = "Policy for S3 access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.terradev_storage.arn,
          "${aws_s3_bucket.terradev_storage.arn}/*"
        ]
      }
    ]
  })
}

# Kubernetes Namespaces
resource "kubernetes_namespace" "terradev_system" {
  metadata {
    name = "terradev-system"
    labels = {
      Environment = var.environment
      Project     = "terradev"
    }
  }
}

resource "kubernetes_namespace" "terradev_workloads" {
  metadata {
    name = "terradev-workloads"
    labels = {
      Environment = var.environment
      Project     = "terradev"
    }
  }
}

# ConfigMaps for microservices
resource "kubernetes_config_map" "database_config" {
  metadata {
    name      = "database-config"
    namespace = kubernetes_namespace.terradev_system.metadata.name
  }

  data = {
    POSTGRES_HOST = module.rds.db_instance_address
    POSTGRES_PORT = tostring(module.rds.db_instance_port)
    POSTGRES_DB   = module.rds.db_instance_name
    REDIS_HOST    = module.elasticache.cache_nodes[0].address
    REDIS_PORT    = tostring(module.elasticache.cache_nodes[0].port)
  }
}

resource "kubernetes_secret" "database_credentials" {
  metadata {
    name      = "database-credentials"
    namespace = kubernetes_namespace.terradev_system.metadata.name
  }

  data = {
    POSTGRES_USER = module.rds.db_instance_username
    POSTGRES_PASSWORD = module.rds.db_instance_password
    REDIS_PASSWORD = "your-redis-auth-token"
  }
}

# Helm Releases for microservices
resource "helm_release" "nginx_ingress" {
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  namespace  = "ingress-nginx"

  create_namespace = true

  set {
    name  = "controller.replicaCount"
    value = "2"
  }

  set {
    name  = "controller.service.type"
    value = "LoadBalancer"
  }
}

resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = "cert-manager"

  create_namespace = true

  set {
    name  = "installCRDs"
    value = "true"
  }

  set {
    name  = "replicaCount"
    value = "1"
  }
}

resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"

  create_namespace = true

  set {
    name  = "prometheus.prometheusSpec.replicas"
    value = "2"
  }

  set {
    name  = "grafana.adminPassword"
    value = "admin123"
  }

  # Removed depends_on — prometheus and nginx_ingress are independent helm charts
  # Terraform parallelizes their installation for ~2x faster cluster bootstrap
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "database_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.db_instance_address
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.elasticache.cache_nodes[0].address
}

output "storage_bucket" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.terradev_storage.bucket
}

# Multi-Cloud Outputs
output "aws_connector_id" {
  description = "AWS connector ID"
  value       = oci_database_multicloud_aws_connector.terradev_aws.id
}

output "azure_connector_id" {
  description = "Azure connector ID"
  value       = oci_database_multicloud_azure_connector.terradev_azure.id
}

# TensorDock Resources
resource "tensordock_instance" "ml_worker" {
  count = 2  # Create 2 ML worker instances
  
  name = "terradev-ml-worker-${count.index + 1}"
  image = "ubuntu2404"
  
  resources {
    vcpu_count = 8
    ram_gb = 32
    storage_gb = 200
    
    gpus = {
      "geforcertx4090-pcie-24gb" = {
        count = 1
      }
    }
  }
  
  location_id = var.tensordock_region
  use_dedicated_ip = true
  
  # SSH key configuration
  ssh_key = var.tensordock_ssh_public_key != "" ? var.tensordock_ssh_public_key : null
  
  # Cloud-init for ML environment
  cloud_init = {
    package_update = true
    package_upgrade = true
    packages = [
      "python3", "python3-pip", "python3-venv", "git", "curl", "wget",
      "build-essential", "cuda-toolkit", "nvidia-driver-535",
      "docker.io", "docker-compose", "nginx", "htop"
    ]
    runcmd = [
      "systemctl enable docker",
      "usermod -aG docker ubuntu",
      "pip3 install --upgrade pip",
      "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
      "pip3 install tensorflow-gpu==2.13.0",
      "pip3 install jupyterlab pandas numpy scikit-learn matplotlib seaborn",
      "echo '🚀 Terradev TensorDock ML Worker Ready!' > /etc/motd"
    ]
    write_files = [
      {
        path = "/etc/motd"
        content = "🚀 Welcome to Terradev TensorDock ML Worker!\n🔥 GPU Accelerated Machine Learning Ready\n💡 Instance: terradev-ml-worker-${count.index + 1}\n📊 Connected to EKS Cluster"
        owner = "root:root"
        permissions = "0644"
      }
    ]
  }
  
  # Tags
  tags = {
    Project = "Terradev"
    Environment = "production"
    InstanceType = "ml-worker"
    ManagedBy = "Terraform"
    Cluster = module.eks.cluster_name
  }
  
  # Wait for instance to be ready
  wait_for_ready = true
  
  # Timeouts to prevent throttling
  timeouts {
    create = "20m"
    delete = "15m"
    update = "15m"
  }
}

# Kubernetes ConfigMap for TensorDock instances
resource "kubernetes_config_map" "tensordock_config" {
  metadata {
    name = "tensordock-instances"
    namespace = "default"
  }
  
  data = {
    "instances.json" = jsonencode([
      for instance in tensordock_instance.ml_worker : {
        id = instance.id
        name = instance.name
        ip_address = instance.ip_address
        gpu_count = 1
        gpu_model = "geforcertx4090-pcie-24gb"
        status = "running"
      }
    ])
  }
}

# Kubernetes Service for TensorDock instances
resource "kubernetes_service" "tensordock_ml" {
  metadata {
    name = "tensordock-ml-service"
    namespace = "default"
    labels = {
      app = "tensordock-ml"
    }
  }
  
  spec {
    type = "ExternalName"
    external_name = tensordock_instance.ml_worker[0].ip_address
    port {
      port = 80
      target_port = 8080
    }
  }
}

# TensorDock Monitoring
resource "kubernetes_config_map" "tensordock_monitoring" {
  metadata {
    name = "tensordock-monitoring"
    namespace = "monitoring"
  }
  
  data = {
    "prometheus-tensordock.yml" = <<-EOT
      global:
        scrape_interval: 30s
      
      scrape_configs:
        - job_name: 'tensordock-ml-workers'
          static_configs:
            - targets: ${jsonencode([for instance in tensordock_instance.ml_worker : "${instance.ip_address}:8080"])}
          metrics_path: /metrics
          scrape_interval: 30s
          scrape_timeout: 10s
        
        - job_name: 'tensordock-gpu-metrics'
          static_configs:
            - targets: ${jsonencode([for instance in tensordock_instance.ml_worker : "${instance.ip_address}:9400"])}
          metrics_path: /metrics
          scrape_interval: 30s
          scrape_timeout: 10s
    EOT
  }
}

# Auto-scaling based on GPU utilization
resource "kubernetes_horizontal_pod_autoscaler" "tensordock_hpa" {
  metadata {
    name = "tensordock-ml-hpa"
    namespace = "default"
  }
  
  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind = "Deployment"
      name = "terradev-ml-deployment"
    }
    
    min_replicas = 1
    max_replicas = 5
    
    metric {
      type = "External"
      external {
        metric {
          name = "tensordock_gpu_utilization"
        }
        target {
          type = "Value"
          value = "70"
        }
      }
    }
    
    behavior {
      scale_up {
        stabilization_window_seconds = 60
        policy {
          type = "Percent"
          value = 50
          period_seconds = 30
        }
      }
      
      scale_down {
        stabilization_window_seconds = 300
        policy {
          type = "Percent"
          value = 25
          period_seconds = 60
        }
      }
    }
  }
}

# TensorDock Outputs
output "tensordock_instance_ids" {
  description = "TensorDock ML worker instance IDs"
  value       = [for instance in tensordock_instance.ml_worker : instance.id]
}

output "tensordock_instance_ips" {
  description = "TensorDock ML worker IP addresses"
  value       = [for instance in tensordock_instance.ml_worker : instance.ip_address]
}

output "tensordock_instance_names" {
  description = "TensorDock ML worker instance names"
  value       = [for instance in tensordock_instance.ml_worker : instance.name]
}

output "tensordock_hourly_cost" {
  description = "Total hourly cost for TensorDock instances"
  value       = length(tensordock_instance.ml_worker) * 1.0  # $1/hour per instance
}

output "gcp_connector_id" {
  description = "GCP connector ID"
  value       = oci_database_multicloud_gcp_connector.terradev_gcp.id
}

output "azure_container_id" {
  description = "Azure blob container ID"
  value       = oci_database_multicloud_azure_blob_container.terradev_container.id
}

output "azure_mount_id" {
  description = "Azure blob mount ID"
  value       = oci_database_multicloud_azure_blob_mount.terradev_mount.id
}

output "azure_vault_id" {
  description = "Azure vault ID"
  value       = oci_database_multicloud_azure_vault.terradev_vault.id
}

output "azure_key_id" {
  description = "Azure key ID"
  value       = oci_database_multicloud_azure_key.terradev_azure_key.id
}

output "gcp_key_ring_id" {
  description = "GCP key ring ID"
  value       = oci_database_multicloud_gcp_key_ring.terradev_key_ring.id
}

output "gcp_key_id" {
  description = "GCP key ID"
  value       = oci_database_multicloud_gcp_key.terradev_gcp_key.id
}

output "aws_key_id" {
  description = "AWS key ID"
  value       = oci_database_multicloud_aws_key.terradev_aws_key.id
}
output "oci_vcn_id" {
  description = "OCI VCN ID"
  value       = module.oci_vcn.vcn_id
}

output "oci_subnet_id" {
  description = "OCI subnet ID"
  value       = module.oci_subnets.subnets["terradevsub1"].subnet_id
}

output "oci_instance_id" {
  description = "OCI compute instance ID"
  value       = module.oci_compute.instance_id
}

output "oci_budget_id" {
  description = "OCI budget ID"
  value       = oci_budget_budget.terradev_budget.id
}

output "oci_recommendations" {
  description = "OCI Cloud Advisor recommendations"
  value       = data.oci_optimizer_recommendations.terradev_recommendations
}
