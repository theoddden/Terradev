# Free APIs for Latency Arbitrage Integration
# Complete stack for studying and arbitraging inference latency

resource "local_file" "free_apis_latency_arbitrage" {
  filename = "${path.module}/apis/free-apis-latency-arbitrage.md"
  content = <<-EOT
  # Free APIs for Latency Arbitrage Integration
  
  ## 🎯 Core Latency Measurement APIs (Free)
  
  ### 1. Teledyne / Network Latency APIs
  ```python
  # IP Geolocation APIs (Free tiers)
  GET https://ipapi.co/json/
  Response: {
    "ip": "8.8.8.8",
    "city": "Mountain View",
    "region": "California",
    "country": "US",
    "latitude": 37.4056,
    "longitude": -122.0775,
    "timezone": "America/Los_Angeles"
  }
  
  # Alternative: ip-api.com (Free)
  GET http://ip-api.com/json/8.8.8.8
  Response: {
    "query": "8.8.8.8",
    "country": "United States",
    "region": "CA",
    "city": "Mountain View",
    "lat": 37.4056,
    "lon": -122.0775
  }
  
  # Alternative: ipgeolocation.io (Free 1000 requests/month)
  GET https://api.ipgeolocation.io/ipgeo?apiKey=FREE_API_KEY&ip=8.8.8.8
  ```
  
  ### 2. Network Distance & Latency APIs
  ```python
  # Cloudflare Radar (Free)
  GET https://api.cloudflare.com/client/v4/radar/quality
  Response: {
    "result": {
      "networks": [
        {
          "asn": 8075,
          "name": "Microsoft Corporation",
          "country": "US",
          "latency": 45.2
        }
      ]
    }
  }
  
  # Pingdom / SpeedTest APIs (Free tiers)
  GET https://api.pingdom.com/api/3.1/checks
  # Requires free account, provides latency metrics
  
  # WebPageTest API (Free)
  GET https://www.webpagetest.org/testinfo.php?test=123456&f=json
  # Provides network latency, TTFB, and performance metrics
  ```
  
  ### 3. CDN & Edge Location APIs
  ```python
  # Cloudflare Edge Locations (Free)
  GET https://api.cloudflare.com/client/v4/ips
  Response: {
    "result": {
      "ipv4_cidrs": ["173.245.48.0/20"],
      "ipv6_cidrs": ["2400:cb00::/32"],
      "colo": ["SFO", "LAX", "IAD"]
    }
  }
  
  # Fastly Edge Locations (Free)
  GET https://api.fastly.com/datacenters
  Response: {
    "datacenters": [
      {"code": "iad", "name": "Washington DC", "coordinates": [38.9, -77.0]},
      {"code": "sfo", "name": "San Francisco", "coordinates": [37.8, -122.4]}
    ]
  }
  
  # AWS CloudFront Locations (Free)
  GET https://cloudfront.amazonaws.com/2020-05-31/distribution?Id=E123456789
  # Provides edge location information
  ```
  
  ## 🚀 Cloud Provider APIs (Free Tiers)
  
  ### AWS APIs (Free Tier)
  ```python
  # AWS CloudWatch Metrics (Free 10 custom metrics)
  GET https://monitoring.us-east-1.amazonaws.com/
  Action=GetMetricStatistics
  &Namespace=AWS/SageMaker
  &MetricName=ModelLatency
  &Dimensions=Name=EndpointName,Value=my-endpoint
  &Statistics=Average
  &Period=60
  
  # AWS Route 53 Health Checks (Free 50 health checks)
  GET https://route53.amazonaws.com/2013-04-01/healthcheck
  # Provides latency-based routing metrics
  
  # AWS X-Ray Tracing (Free 100K traces/month)
  GET https://xray.us-east-1.amazonaws.com/GetTraceSummaries
  # Provides detailed latency and performance tracing
  ```
  
  ### Google Cloud APIs (Free Tier)
  ```python
  # Google Cloud Monitoring (Free tier with quotas)
  GET https://monitoring.googleapis.com/v3/projects/my-project/timeSeries
  Filter: metric.type="aiplatform.googleapis.com/prediction/online/prediction_count"
  
  # Google Cloud Trace (Free tier)
  GET https://cloudtrace.googleapis.com/v1/projects/my-project/traces
  # Provides distributed tracing for latency analysis
  
  # Google Cloud DNS (Free tier)
  GET https://dns.googleapis.com/dns/v1/projects/my-project/managedZones
  # Can be used for geographic routing analysis
  ```
  
  ### Azure APIs (Free Tier)
  ```python
  # Azure Monitor Metrics (Free tier)
  GET https://management.azure.com/subscriptions/{subscriptionId}/providers/microsoft.insights/metrics
  Filter: (Microsoft.MachineLearningServices/onlineEndpoints/name eq 'my-endpoint')
  
  # Azure Application Insights (Free tier)
  GET https://api.applicationinsights.io/v1/apps/my-app/metrics/performanceCounters/requestDuration
  # Provides detailed performance metrics
  
  # Azure Front Door (Free tier)
  GET https://management.azure.com/subscriptions/{subscriptionId}/resourceGroups/{rg}/providers/Microsoft.Network/frontDoors/{fd}
  # Provides edge routing and latency metrics
  ```
  
  ## 🔧 Specialized Provider APIs (Free Tiers)
  
  ### RunPod API (Free tier)
  ```python
  # RunPod Serverless Pricing (Free)
  GET https://api.runpod.io/v1/serverless/llama-2-7b-chat
  Response: {
    "input_price": 0.0001,
    "output_price": 0.0002,
    "unit": "token"
  }
  
  # RunPod GPU Status (Free)
  GET https://api.runpod.io/v1/gpu
  Response: {
    "secure": {
      "a100-80gb": {"available": true, "price": 2.99}
    }
  }
  
  # RunPod Pod Metrics (Free)
  GET https://api.runpod.io/v1/pod/{pod_id}
  Response: {
    "runtime": {
      "uptimeInSeconds": 3600,
      "ports": [{"ip": "123.45.67.89", "privatePort": 8080}]
    }
  }
  ```
  
  ### Lambda Labs API (Free tier)
  ```python
  # Lambda Labs Instance Types (Free)
  GET https://api.labs.lambda.cloud/instance-types
  Response: {
    "data": [
      {"name": "gpu_1x_a100", "price": "1.60", "gpu": "A100"}
    ]
  }
  
  # Lambda Labs Regions (Free)
  GET https://api.labs.lambda.cloud/regions
  Response: {
    "data": [
      {"name": "us-west", "display_name": "US West"}
    ]
  }
  ```
  
  ### CoreWeave API (Free tier)
  ```python
  # CoreWeave GPU Inventory (Free)
  GET https://api.coreweave.com/v1/gpu-inventory
  Response: {
    "nvidia": {
      "a100-80gb": {"available": 42, "regions": ["LAX1", "NYC1"]}
    }
  }
  
  # CoreWeave Regions (Free)
  GET https://api.coreweave.com/v1/regions
  Response: {
    "regions": [
      {"name": "LAX1", "location": "Los Angeles"},
      {"name": "NYC1", "location": "New York"}
    ]
  }
  ```
  
  ## 📊 DevOps & Monitoring APIs (Free Tiers)
  
  ### Kubernetes API (Free)
  ```python
  # Kubernetes Metrics Server (Free)
  GET https://kubernetes.default.svc/apis/metrics.k8s.io/v1beta1/nodes
  Response: {
    "items": [
      {
        "metadata": {"name": "node-1"},
        "usage": {"cpu": "500m", "memory": "1Gi"}
      }
    ]
  }
  
  # Kubernetes Custom Metrics (Free)
  GET https://kubernetes.default.svc/apis/custom.metrics.k8s.io/v1beta1
  # Can track custom latency metrics
  
  # Kubernetes Events (Free)
  GET https://kubernetes.default.svc/api/v1/events
  # Can track deployment and scaling events
  ```
  
  ### Grafana API (Free)
  ```python
  # Grafana Query API (Free)
  POST https://grafana.example.com/api/datasources/proxy/1/api/v1/query_range
  {
    "query": "avg_over_time(inference_latency_seconds[5m])",
    "start": 1640995200,
    "end": 1641081600,
    "step": 60
  }
  
  # Grafana Dashboard API (Free)
  GET https://grafana.example.com/api/dashboards/uid/terradev-inference
  # Provides pre-built latency dashboards
  
  # Grafana Alerting API (Free)
  POST https://grafana.example.com/api/alerts
  # Can set up latency-based alerts
  ```
  
  ### Jenkins API (Free)
  ```python
  # Jenkins Build API (Free)
  GET https://jenkins.example.com/job/inference-tests/api/json
  Response: {
    "builds": [
      {"number": 123, "duration": 45000, "result": "SUCCESS"}
    ]
  }
  
  # Jenkins Metrics API (Free)
  GET https://jenkins.example.com/metrics/api/json
  # Provides build performance and timing metrics
  
  # Jenkins Pipeline API (Free)
  GET https://jenkins.example.com/job/inference-pipeline/api/json
  # Can track deployment and performance testing
  ```
  
  ### Databricks API (Free tier)
  ```python
  # Databricks Clusters API (Free tier)
  GET https://databricks.example.com/api/2.0/clusters/list
  Response: {
    "clusters": [
      {
        "cluster_id": "1234-567890-abc123",
        "state": "RUNNING",
        "driver": {"node_type_id": "Standard_NC6s_v3"}
      }
    ]
  }
  
  # Databricks Jobs API (Free tier)
  GET https://databricks.example.com/api/2.1/jobs/list
  # Can track inference job performance
  
  # Databricks MLflow API (Free tier)
  GET https://databricks.example.com/api/2.0/mlflow/experiments/list
  # Can track model performance and latency
  ```
  
  ## 🤝 Integration & Communication APIs (Free)
  
  ### GitHub API (Free)
  ```python
  # GitHub Actions API (Free)
  GET https://api.github.com/repos/my-org/my-repo/actions/runs
  Response: {
    "workflow_runs": [
      {
        "id": 123456789,
        "status": "completed",
        "conclusion": "success",
        "created_at": "2024-01-01T12:00:00Z"
      }
    ]
  }
  
  # GitHub Issues API (Free)
  GET https://api.github.com/repos/my-org/my-repo/issues
  # Can track latency-related issues and performance reports
  
  # GitHub Commits API (Free)
  GET https://api.github.com/repos/my-org/my-repo/commits
  # Can track performance-related commits
  ```
  
  ### Slack API (Free)
  ```python
  # Slack Chat API (Free)
  POST https://slack.com/api/chat.postMessage
  {
    "channel": "#inference-alerts",
    "text": "High latency detected: 250ms average over 5 minutes"
  }
  
  # Slack Users API (Free)
  GET https://slack.com/api/users.list
  # Can notify relevant teams about latency issues
  
  # Slack Channels API (Free)
  GET https://slack.com/api/conversations.list
  # Can create dedicated latency monitoring channels
  ```
  
  ## 🌐 Geographic & Network APIs (Free)
  
  ### MaxMind GeoIP2 (Free Database)
  ```python
  # Download free GeoLite2 database
  wget https://github.com/P3TERX/GeoLite.mmdb/raw/download/GeoLite2-City.mmdb
  
  # Use with Python
  import geoip2.database
  reader = geoip2.database.Reader('GeoLite2-City.mmdb')
  response = reader.city('8.8.8.8')
  print(response.city.name)  # Mountain View
  print(response.country.name)  # United States
  ```
  
  ### IPinfo.io (Free 1000 requests/month)
  ```python
  GET https://ipinfo.io/8.8.8.8/json?token=FREE_TOKEN
  Response: {
    "ip": "8.8.8.8",
    "city": "Mountain View",
    "region": "California",
    "country": "US",
    "loc": "37.4056,-122.0775",
    "org": "AS15169 Google LLC"
  }
  ```
  
  ### OpenStreetMap Nominatim (Free)
  ```python
  GET https://nominatim.openstreetmap.org/search?q=Mountain%20View&format=json
  Response: [
    {
      "lat": "37.4056",
      "lon": "-122.0775",
      "display_name": "Mountain View, California, United States"
    }
  ]
  ```
  
  ## 📈 Performance Testing APIs (Free)
  
  ### WebPageTest API (Free)
  ```python
  GET https://www.webpagetest.org/runtest.php?url=https://api.example.com/inference&f=json
  Response: {
    "testId": "240101_AB_123",
    "userUrl": "https://www.webpagetest.org/result/240101_AB_123/"
  }
  
  GET https://www.webpagetest.org/jsonResult.php?test=240101_AB_123
  Response: {
    "data": {
      "median": {
        "firstView": {
          "loadTime": 1250,
          "TTFB": 450,
          "fullyLoaded": 1800
        }
      }
    }
  }
  ```
  
  ### GTmetrix API (Free tier)
  ```python
  POST https://gtmetrix.com/api/0.1/test
  {
    "url": "https://api.example.com/inference",
    "location": "1"  # Vancouver
  }
  # Provides performance and latency metrics
  ```
  
  ## 🔧 Integration Strategy
  
  ### 1. Latency Measurement Stack
  ```python
  class LatencyIntegrator:
      def __init__(self):
          self.geolocation = IPGeolocationAPI()
          self.network_metrics = NetworkMetricsAPI()
          self.performance = PerformanceTestingAPI()
      
      def measure_inference_latency(self, provider, endpoint, user_location):
          # 1. Get user geolocation
          user_geo = self.geolocation.get_location(user_location)
          
          # 2. Find nearest edge locations
          edge_locations = self.get_edge_locations(provider, user_geo)
          
          # 3. Measure actual latency
          latency = self.performance.measure_latency(endpoint)
          
          # 4. Get network distance
          network_distance = self.network_metrics.get_distance(user_geo, edge_locations)
          
          return {
              'user_location': user_geo,
              'edge_location': edge_locations,
              'actual_latency': latency,
              'network_distance': network_distance,
              'optimal_route': self.calculate_optimal_route(user_geo, edge_locations)
          }
  ```
  
  ### 2. Arbitrage Decision Engine
  ```python
  class InferenceArbitrageEngine:
      def route_request(self, request):
          # 1. Get all provider latencies
          latencies = {}
          for provider in self.providers:
              latency = self.measure_latency(provider, request.user_location)
              latencies[provider] = latency
          
          # 2. Filter by latency requirements
          viable = {p: l for p, l in latencies.items() if l['actual_latency'] <= request.max_latency}
          
          # 3. Optimize for cost + latency
          optimal = self.optimize_cost_latency(viable, request)
          
          return optimal
  ```
  
  ## 🎯 Free API Limitations & Workarounds
  
  ### Rate Limits:
  - Most free APIs have rate limits (1000-10000 requests/month)
  - Solution: Implement caching and batch requests
  - Solution: Use multiple providers for redundancy
  
  ### Data Freshness:
  - Some APIs update data infrequently
  - Solution: Combine multiple data sources
  - Solution: Implement real-time measurements
  
  ### Coverage Gaps:
  - Some regions have limited API coverage
  - Solution: Use multiple geolocation providers
  - Solution: Implement fallback mechanisms
  
  ## 📊 Summary of Required Free APIs
  
  | Category | APIs Needed | Cost | Use Case |
  |----------|------------|------|---------|
  | **Geolocation** | IPapi.co, IPinfo.io, MaxMind | Free | User location |
  | **Network Metrics** | Cloudflare Radar, Pingdom | Free | Network latency |
  | **Edge Locations** | Cloudflare, Fastly | Free | Edge routing |
  | **Cloud Metrics** | AWS CloudWatch, GCP Monitoring | Free tier | Performance |
  | **Provider APIs** | RunPod, Lambda, CoreWeave | Free tier | Provider data |
  | **DevOps APIs** | Kubernetes, Grafana, Jenkins | Free | Monitoring |
  | **Communication** | GitHub, Slack | Free | Notifications |
  | **Performance** | WebPageTest, GTmetrix | Free tier | Latency testing |
  
  **Total Cost: $0/month** (using free tiers and open-source APIs)
  **Total APIs: ~15-20 different services**
  **Integration Complexity: Medium-High**
  
  This gives you everything needed to build a comprehensive latency arbitrage system without any API costs!
  EOT
}
