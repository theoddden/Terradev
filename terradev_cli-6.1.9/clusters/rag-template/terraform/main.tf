# RAG Infrastructure Template — Terraform
# Provisions: Qdrant + Embedding + vLLM + Redis on Kubernetes

variable "namespace" {
  default = "rag"
}

variable "qdrant_storage_size" {
  default = "100Gi"
}

variable "vllm_model" {
  default = "zai-org/GLM-5-FP8"
}

variable "embedding_model" {
  default = "BAAI/bge-large-en-v1.5"
}

resource "kubernetes_namespace" "rag" {
  metadata { name = var.namespace }
}

resource "helm_release" "qdrant" {
  name       = "qdrant"
  repository = "https://qdrant.github.io/qdrant-helm"
  chart      = "qdrant"
  namespace  = var.namespace
  depends_on = [kubernetes_namespace.rag]

  set {
    name  = "persistence.size"
    value = var.qdrant_storage_size
  }
  set {
    name  = "resources.requests.memory"
    value = "2Gi"
  }
  set {
    name  = "resources.limits.memory"
    value = "8Gi"
  }
}

resource "kubernetes_deployment" "redis" {
  metadata {
    name      = "redis"
    namespace = var.namespace
  }
  spec {
    replicas = 1
    selector { match_labels = { app = "redis" } }
    template {
      metadata { labels = { app = "redis" } }
      spec {
        container {
          name  = "redis"
          image = "redis:7-alpine"
          port { container_port = 6379 }
          resources {
            requests = { cpu = "250m", memory = "512Mi" }
            limits   = { cpu = "1", memory = "2Gi" }
          }
        }
      }
    }
  }
  depends_on = [kubernetes_namespace.rag]
}

resource "kubernetes_service" "redis" {
  metadata {
    name      = "redis-svc"
    namespace = var.namespace
  }
  spec {
    selector = { app = "redis" }
    port {
      port        = 6379
      target_port = 6379
    }
  }
  depends_on = [kubernetes_deployment.redis]
}
