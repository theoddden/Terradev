# Kubernetes Operator for Terradev
# Essential for ML teams running on Kubernetes

resource "local_file" "kubernetes_operator" {
  filename = "${path.module}/k8s-operator/terradev-operator.yaml"
  content = <<-EOT
  apiVersion: apiextensions.k8s.io/v1
  kind: CustomResourceDefinition
  metadata:
    name: gpujobs.terradev.io
  spec:
    group: terradev.io
    versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                gpuType:
                  type: string
                  enum: ["a100", "h100", "a10g"]
                hours:
                  type: integer
                  minimum: 1
                confidenceThreshold:
                  type: number
                  minimum: 0
                  maximum: 1
                  default: 0.7
                maxRiskScore:
                  type: number
                  minimum: 0
                  maximum: 1
                  default: 0.5
                trainingScript:
                  type: string
                dataset:
                  type: string
                model:
                  type: string
            status:
              type: object
              properties:
                phase:
                  type: string
                  enum: ["Pending", "FindingGPU", "Deploying", "Running", "Completed", "Failed"]
                provider:
                  type: string
                deploymentId:
                  type: string
                actualCost:
                  type: number
                savings:
                  type: number
                confidence:
                  type: number
                riskScore:
                  type: number
                startTime:
                  type: string
                completionTime:
                  type: string
    scope: Namespaced
    names:
      plural: gpujobs
      singular: gpujob
      kind: GPUJob
      shortNames:
      - gj
  ---
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: terradev-operator
    namespace: terradev-system
  spec:
    replicas: 1
    selector:
      matchLabels:
        app: terradev-operator
    template:
      metadata:
        labels:
          app: terradev-operator
      spec:
        serviceAccountName: terradev-operator
        containers:
        - name: operator
          image: terradev/operator:latest
          env:
          - name: TERRADEV_API_KEY
            valueFrom:
              secretKeyRef:
                name: terradev-secrets
                key: api-key
          - name: TERRADEV_API_ENDPOINT
            value: "https://api.terradev.io/v1"
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 256Mi
  ---
  apiVersion: v1
  kind: ServiceAccount
  metadata:
    name: terradev-operator
    namespace: terradev-system
  ---
  apiVersion: rbac.authorization.k8s.io/v1
  kind: ClusterRole
  metadata:
    name: terradev-operator
  rules:
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps", "secrets"]
    verbs: ["create", "get", "list", "watch", "update", "patch", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["create", "get", "list", "watch", "update", "patch", "delete"]
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "watch", "update", "patch", "delete"]
  - apiGroups: ["terradev.io"]
    resources: ["gpujobs"]
    verbs: ["create", "get", "list", "watch", "update", "patch", "delete"]
  - apiGroups: ["terradev.io"]
    resources: ["gpujobs/status"]
    verbs: ["update", "patch"]
  ---
  apiVersion: rbac.authorization.k8s.io/v1
  kind: ClusterRoleBinding
  metadata:
    name: terradev-operator
  roleRef:
    apiGroup: rbac.authorization.k8s.io
    kind: ClusterRole
    name: terradev-operator
  subjects:
  - kind: ServiceAccount
    name: terradev-operator
    namespace: terradev-system
  ---
  apiVersion: v1
  kind: Secret
  metadata:
    name: terradev-secrets
    namespace: terradev-system
  type: Opaque
  data:
    api-key: <base64-encoded-api-key>
  EOT
}

# Example GPUJob CRD Usage
resource "local_file" "kubernetes_gpujob_example" {
  filename = "${path.module}/k8s-operator/examples/gpujob-example.yaml"
  content = <<-EOT
  apiVersion: terradev.io/v1
  kind: GPUJob
  metadata:
    name: llama-training-job
    namespace: ml-workloads
  spec:
    gpuType: a100
    hours: 8
    confidenceThreshold: 0.8
    maxRiskScore: 0.3
    trainingScript: |
      #!/usr/bin/env python3
      import torch
      from transformers import LlamaForCausalLM, LlamaTokenizer
      
      # Load model and tokenizer
      model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
      tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
      
      # Training code here
      print("Starting LLaMA training...")
      
    dataset: "coco"
    model: "llama-2-7b"
  ---
  # Simple GPUJob for quick testing
  apiVersion: terradev.io/v1
  kind: GPUJob
  metadata:
    name: quick-test
    namespace: ml-workloads
  spec:
    gpuType: a10g
    hours: 2
    trainingScript: |
      #!/usr/bin/env python3
      print("Testing GPU deployment...")
      import torch
      print(f"CUDA available: {torch.cuda.is_available()}")
      if torch.cuda.is_available():
          print(f"GPU: {torch.cuda.get_device_name()}")
  EOT
}

# Helm Chart for easy installation
resource "local_file" "helm_chart" {
  filename = "${path.module}/helm/terradev-operator/Chart.yaml"
  content = yamlencode({
    apiVersion = "v2"
    name = "terradev-operator"
    description = "Terradev GPU Arbitrage Operator for Kubernetes"
    type = "application"
    version = "1.0.0"
    appVersion = "1.0.0"
    
    keywords = ["gpu", "ml", "arbitrage", "kubernetes"]
    maintainers = [
      {
        name = "Terradev Team"
        email = "team@terradev.io"
      }
    ]
    
    dependencies = []
  })
}

resource "local_file" "helm_values" {
  filename = "${path.module}/helm/terradev-operator/values.yaml"
  content = yamlencode({
    operator = {
      image = {
        repository = "terradev/operator"
        tag = "latest"
        pullPolicy = "IfNotPresent"
      }
      resources = {
        requests = {
          cpu = "100m"
          memory = "128Mi"
        }
        limits = {
          cpu = "500m"
          memory = "256Mi"
        }
      }
    }
    
    terradev = {
      apiEndpoint = "https://api.terradev.io/v1"
      apiKey = {
        secretName = "terradev-secrets"
        secretKey = "api-key"
      }
    }
    
    rbac = {
      create = true
    }
    
    serviceAccount = {
      create = true
      name = "terradev-operator"
    }
    
    namespace = {
      create = true
      name = "terradev-system"
    }
  })
}
