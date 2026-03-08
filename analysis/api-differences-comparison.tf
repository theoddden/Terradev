# API Differences: Training vs Inference Arbitrage
# Specific APIs and integration requirements

resource "local_file" "api_differences_comparison" {
  filename = "${path.module}/analysis/api-differences.md"
  content = <<-EOT
  # API Differences: Training vs Inference Arbitrage
  
  ## Training Arbitrage APIs (GPU-Centric)
  
  ### 1. GPU Pricing APIs
  ```python
  # AWS EC2 Pricing API
  GET https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/index.json
  Response: {
    "products": {
      "p4d.24xlarge": {
        "attributes": {"instanceType": "p4d.24xlarge"},
        "terms": {"OnDemand": {...}, "Spot": {...}}
      }
    }
  }
  
  # GCP Cloud Billing API
  GET https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus
  Response: {
    "skus": [
      {
        "description": "A2 Instance Core running in Americas",
        "pricingInfo": [{"pricingExpression": {...}}]
      }
    ]
  }
  
  # Azure Retail Prices API
  GET https://prices.azure.com/api/retail/prices
  Response: {
    "Items": [
      {
        "productName": "Virtual Machines",
        "skuName": "Standard_ND96asr_v4",
        "unitPrice": 4.29,
        "armSkuName": "Standard_ND96asr_v4"
      }
    ]
  }
  ```
  
  ### 2. GPU Instance Management APIs
  ```python
  # AWS EC2 API
  POST https://ec2.amazonaws.com/
  Action=RunInstances
  &ImageId=ami-12345678
  &InstanceType=p4d.24xlarge
  &InstanceMarketOptions.MarketType=spot
  
  # GCP Compute Engine API
  POST https://compute.googleapis.com/compute/v1/projects/my-project/zones/us-central1-a/instances
  {
    "name": "gpu-training-instance",
    "machineType": "zones/us-central1-a/machineTypes/a2-highgpu-8g",
    "guestAccelerators": [{"acceleratorType": "nvidia-tesla-a100", "acceleratorCount": 8}]
  }
  
  # Azure Resource Manager API
  PUT https://management.azure.com/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Compute/virtualMachines/{vmName}
  {
    "location": "eastus",
    "properties": {
      "hardwareProfile": {"vmSize": "Standard_ND96asr_v4"},
      "priority": "Spot"
    }
  }
  ```
  
  ## Inference Arbitrage APIs (Service-Centric)
  
  ### 1. Inference Service APIs
  ```python
  # AWS SageMaker Runtime API
  POST https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-endpoint/invoke
  {
    "input": "{\"prompt\": \"Hello, world!\"}"
  }
  Response: {
    "output": "{\"generated_text\": \"Hello! How can I help you?\"}",
    "contentType": "application/json"
  }
  
  # GCP Vertex AI Endpoint API
  POST https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/my-endpoint:predict
  {
    "instances": [{"content": "Hello, world!"}]
  }
  Response: {
    "predictions": [{"content": "Hello! How can I help you?"}]
  }
  
  # Azure ML Endpoint API
  POST https://my-region.api.azureml.ms/score
  {
    "input_data": {"prompt": "Hello, world!"}
  }
  Response: {
    "output": "Hello! How can I help you?"
  }
  
  # RunPod Serverless API
  POST https://api.runpod.io/v1/serverless/llama-2-7b-chat/runs
  {
    "input": {"prompt": "Hello, world!"}
  }
  Response: {
    "output": {"text": "Hello! How can I help you?"},
    "usage": {"prompt_tokens": 10, "completion_tokens": 15}
  }
  
  # Replicate API
  POST https://api.replicate.com/v1/predictions
  {
    "version": "a21c011e6d8087fc953d0841a1a9d6b6e0c8e1b3a5c7d9e2f4a6b8c0d2e4f6a8",
    "input": {"prompt": "Hello, world!"}
  }
  Response: {
    "id": "pred_123",
    "output": "Hello! How can I help you?",
    "metrics": {"predict_time": 0.5}
  }
  
  # Hugging Face Inference API
  POST https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf
  {
    "inputs": "Hello, world!",
    "parameters": {"max_new_tokens": 100}
  }
  Response: [
    {"generated_text": "Hello! How can I help you?"}
  ]
  ```
  
  ### 2. Real-Time Pricing APIs
  ```python
  # RunPod Pricing API
  GET https://api.runpod.io/v1/gpu/pricing
  Response: {
    "gpu_types": {
      "A100-80GB": {
        "price_per_hour": 2.99,
        "price_per_request": 0.001,
        "price_per_token": 0.000001
      }
    }
  }
  
  # Replicate Pricing API
  GET https://api.replicate.com/v1/models/meta-llama/Llama-2-7b-chat-hf/pricing
  Response: {
    "pricing": {
      "per_second": 0.0001,
      "per_token": 0.000001,
      "per_request": 0.002
    }
  }
  
  # Hugging Face Pricing API
  GET https://api-inference.huggingface.co/pricing
  Response: {
    "models": {
      "meta-llama/Llama-2-7b-chat-hf": {
        "per_request": 0.001,
        "per_token": 0.000001
      }
    }
  }
  ```
  
  ### 3. Edge Computing APIs
  ```python
  # Cloudflare Workers AI API
  POST https://gateway.ai.cloudflare.com/v1/accounts/{account_id}/models/@cf/meta/llama-2-7b-chat
  {
    "prompt": "Hello, world!"
  }
  Response: {
    "response": "Hello! How can I help you?",
    "success": true
  }
  
  # Fastly Compute@Edge API
  POST https://api.fastly.com/compute/edge-functions/inference
  {
    "model": "llama-2-7b",
    "input": {"prompt": "Hello, world!"}
  }
  Response: {
    "output": "Hello! How can I help you!",
    "latency_ms": 45
  }
  ```
  
  ### 4. Performance & Latency APIs
  ```python
  # AWS CloudWatch Metrics API
  GET https://monitoring.us-east-1.amazonaws.com/
  Action=GetMetricStatistics
  &Namespace=AWS/SageMaker
  &MetricName=Invocations
  &Dimensions=Name=EndpointName,Value=my-endpoint
  
  # GCP Cloud Monitoring API
  GET https://monitoring.googleapis.com/v3/projects/my-project/timeSeries
  Filter: metric.type="aiplatform.googleapis.com/prediction/online/prediction_count"
  
  # Azure Monitor Metrics API
  GET https://management.azure.com/subscriptions/{subscriptionId}/providers/microsoft.insights/metrics
  Filter: (Microsoft.MachineLearningServices/onlineEndpoints/name eq 'my-endpoint')
  ```
  
  ## Key API Differences Summary
  
  ### Training APIs:
  - **Focus**: GPU instance pricing and management
  - **Frequency**: Periodic (hourly/daily updates)
  - **Complexity**: Lower (single dimension: GPU cost)
  - **Data Source**: Cloud provider pricing APIs
  - **Decision Making**: Batch optimization
  
  ### Inference APIs:
  - **Focus**: Service pricing, performance, latency
  - **Frequency**: Real-time (sub-second decisions)
  - **Complexity**: Higher (cost, latency, geography, load)
  - **Data Source**: Inference service APIs, edge APIs
  - **Decision Making**: Real-time routing optimization
  
  ## Integration Complexity Comparison
  
  ### Training Integration:
  ```python
  # Simple - just need GPU pricing
  class TrainingArbitrage:
      def get_cheapest_gpu(self, gpu_type, hours):
          prices = self.fetch_gpu_prices()
          return min(prices, key=lambda x: x['spot_price'])
  ```
  
  ### Inference Integration:
  ```python
  # Complex - multiple factors to consider
  class InferenceArbitrage:
      def route_request(self, request):
          # 1. Get real-time pricing from multiple services
          prices = self.fetch_inference_prices(request.model)
          
          # 2. Check latency requirements
          viable = [p for p in prices if p['latency'] <= request.max_latency]
          
          # 3. Check geographic constraints
          if request.user_location:
              viable = self.filter_by_geography(viable, request.user_location)
          
          # 4. Check provider load
          viable = self.filter_by_load(viable)
          
          # 5. Optimize cost vs latency tradeoff
          return self.optimize_cost_latency(viable, request)
  ```
  
  ## New API Integrations Required for Inference
  
  ### 1. Inference Service Providers:
  - **AWS SageMaker Runtime**
  - **GCP Vertex AI Endpoints**
  - **Azure ML Endpoints**
  - **RunPod Serverless**
  - **Replicate**
  - **Hugging Face Inference API**
  - **Together AI**
  - **Anyscale**
  
  ### 2. Edge Computing Providers:
  - **Cloudflare Workers AI**
  - **Fastly Compute@Edge**
  - **AWS Lambda@Edge**
  - **Google Cloud Functions**
  
  ### 3. Performance Monitoring:
  - **CloudWatch Metrics**
  - **Cloud Monitoring**
  - **Azure Monitor**
  - **Custom latency tracking**
  
  ### 4. Geographic Services:
  - **IP Geolocation APIs**
  - **CDN APIs**
  - **Regional availability APIs**
  
  ## Conclusion: Different APIs, Different Complexity
  
  Training arbitrage is primarily about **GPU pricing APIs** - relatively simple, single-dimensional optimization.
  
  Inference arbitrage requires **service APIs** - much more complex, multi-dimensional optimization including:
  - Real-time pricing
  - Latency measurements
  - Geographic routing
  - Load balancing
  - Performance monitoring
  
  This is why inference arbitrage is both more valuable and more complex than training arbitrage.
  EOT
}
