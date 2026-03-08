# Terradev API Documentation

## Overview

Terradev provides a comprehensive REST API for GPU arbitrage, cost optimization, and cloud resource management. This API enables developers to integrate real-time GPU price comparison, automated deployment, and advanced financial modeling into their applications.

## Base URL

```
https://api.terradev.io/v1
```

## Authentication

Terradev uses JWT-based authentication with API key support. All requests must include a valid authentication token.

### JWT Authentication

```bash
curl -H "Authorization: Bearer <your-jwt-token>" \
     https://api.terradev.io/v1/gpu/pricing
```

### API Key Authentication

```bash
curl -H "X-API-Key: <your-api-key>" \
     https://api.terradev.io/v1/gpu/pricing
```

## Rate Limiting

- **Free Tier**: 100 requests per minute
- **Paid Tier**: 1000 requests per minute
- **Enterprise**: Custom limits

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Total requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Error Handling

Terradev uses standard HTTP status codes with detailed error responses:

```json
{
  "error": {
    "code": "GPU_NOT_FOUND",
    "message": "GPU type not found",
    "details": {
      "gpu_type": "invalid_gpu",
      "available_types": ["a100", "h100", "a10g"]
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

## Endpoints

### GPU Pricing

#### Get GPU Prices

Retrieve current GPU pricing across all providers.

```http
GET /gpu/pricing
```

**Query Parameters:**
- `gpu_types` (optional): Comma-separated list of GPU types
- `providers` (optional): Comma-separated list of providers
- `regions` (optional): Comma-separated list of regions
- `spot_only` (optional): Return only spot prices (default: true)

**Response:**
```json
{
  "data": {
    "aws": {
      "a100": {
        "on_demand_price": 4.06,
        "spot_price": 1.22,
        "currency": "USD",
        "region": "us-west-2",
        "instance_type": "p4d.24xlarge",
        "availability": "available",
        "last_updated": "2024-01-15T10:30:00Z"
      }
    },
    "runpod": {
      "a100": {
        "on_demand_price": 2.50,
        "spot_price": 0.89,
        "currency": "USD",
        "region": "us-west-2",
        "instance_type": "A100-40GB",
        "availability": "available",
        "last_updated": "2024-01-15T10:30:00Z"
      }
    }
  },
  "metadata": {
    "total_providers": 6,
    "total_gpu_types": 5,
    "last_updated": "2024-01-15T10:30:00Z",
    "data_freshness": "5 minutes"
  }
}
```

#### Get GPU Price History

Get historical pricing data for analysis.

```http
GET /gpu/pricing/history
```

**Query Parameters:**
- `gpu_type` (required): GPU type
- `provider` (optional): Specific provider
- `period` (optional): Time period (1h, 24h, 7d, 30d)
- `granularity` (optional): Data granularity (1m, 5m, 1h, 1d)

**Response:**
```json
{
  "data": {
    "gpu_type": "a100",
    "provider": "aws",
    "period": "24h",
    "granularity": "1h",
    "prices": [
      {
        "timestamp": "2024-01-14T10:00:00Z",
        "on_demand_price": 4.06,
        "spot_price": 1.22,
        "availability": "available"
      }
    ]
  },
  "metadata": {
    "total_points": 24,
    "period_start": "2024-01-14T10:00:00Z",
    "period_end": "2024-01-15T10:00:00Z"
  }
}
```

### Arbitrage Analysis

#### Find Arbitrage Opportunities

Discover the best GPU arbitrage opportunities.

```http
POST /arbitrage/opportunities
```

**Request Body:**
```json
{
  "gpu_type": "a100",
  "max_budget": 5.0,
  "confidence_threshold": 0.8,
  "max_risk_score": 0.3,
  "providers": ["aws", "gcp", "runpod"],
  "regions": ["us-west-2", "us-east-1"],
  "spot_only": true,
  "include_risk_analysis": true,
  "include_volatility_analysis": true
}
```

**Response:**
```json
{
  "data": {
    "opportunities": [
      {
        "rank": 1,
        "provider": "runpod",
        "instance_type": "A100-40GB",
        "gpu_type": "a100",
        "hourly_cost": 0.89,
        "on_demand_price": 2.50,
        "savings_percentage": 64.4,
        "confidence_score": 0.92,
        "success_probability": 0.95,
        "risk_level": "low",
        "region": "us-west-2",
        "availability": "available",
        "features": ["spot_pricing", "high_availability"],
        "estimated_savings": 1.61,
        "deployment_time": "2 minutes"
      }
    ],
    "analysis": {
      "total_opportunities": 15,
      "avg_savings_percentage": 58.3,
      "best_provider": "runpod",
      "market_efficiency": 0.87,
      "volatility_index": 0.15,
      "risk_assessment": {
        "var_95": 0.12,
        "expected_shortfall": 0.18,
        "sharpe_ratio": 2.8
      }
    }
  },
  "metadata": {
    "analysis_timestamp": "2024-01-15T10:30:00Z",
    "data_freshness": "2 minutes",
    "computation_time": "1.2 seconds"
  }
}
```

#### Enhanced Arbitrage Analysis

Advanced analysis with risk modeling and Greeks.

```http
POST /arbitrage/enhanced-analysis
```

**Request Body:**
```json
{
  "gpu_type": "a100",
  "max_budget": 10.0,
  "analysis_depth": "comprehensive",
  "risk_modeling": true,
  "volatility_forecasting": true,
  "greeks_calculation": true,
  "tou_pricing": true,
  "liquidity_analysis": true,
  "correlation_analysis": true
}
```

**Response:**
```json
{
  "data": {
    "opportunities": [
      {
        "provider": "aws",
        "instance_type": "p4d.24xlarge",
        "gpu_type": "a100",
        "base_price": 4.06,
        "risk_adjusted_price": 1.35,
        "volatility_adjusted_price": 1.42,
        "tou_adjusted_price": 1.38,
        "final_adjusted_price": 1.45,
        "success_probability": 0.88,
        "confidence_score": 0.85,
        "total_score": 0.82,
        "greeks": {
          "delta": 0.75,
          "gamma": 0.12,
          "theta": -0.03,
          "vega": 0.18,
          "rho": 0.05
        },
        "risk_metrics": {
          "var_95": 0.15,
          "expected_shortfall": 0.22,
          "max_drawdown": 0.08,
          "correlation_risk": 0.25
        },
        "volatility_metrics": {
          "current_volatility": 0.15,
          "forecast_volatility": 0.12,
          "volatility_trend": "decreasing"
        }
      }
    ],
    "market_analysis": {
      "volatility_regime": "normal",
      "market_efficiency": 0.85,
      "arbitrage_opportunities": 12,
      "risk_distribution": {
        "low": 8,
        "medium": 3,
        "high": 1
      }
    }
  }
}
```

### Risk Management

#### Calculate Risk Metrics

Get comprehensive risk metrics for GPU instances.

```http
POST /risk/metrics
```

**Request Body:**
```json
{
  "provider": "aws",
  "gpu_type": "a100",
  "instance_type": "p4d.24xlarge",
  "time_horizon": 30,
  "confidence_level": 0.95,
  "include_correlation": true,
  "include_liquidity": true
}
```

**Response:**
```json
{
  "data": {
    "provider": "aws",
    "gpu_type": "a100",
    "instance_type": "p4d.24xlarge",
    "risk_metrics": {
      "var_95": 0.15,
      "expected_shortfall": 0.22,
      "max_drawdown": 0.08,
      "sharpe_ratio": 2.5,
      "sortino_ratio": 3.2,
      "beta": 0.85,
      "alpha": 0.12,
      "correlation_risk": 0.25,
      "liquidity_risk": 0.10,
      "concentration_risk": 0.05
    },
    "greeks": {
      "delta": 0.75,
      "gamma": 0.15,
      "theta": -0.05,
      "vega": 0.25,
      "rho": 0.08
    },
    "stress_test_results": {
      "market_crash": -0.35,
      "provider_outage": -0.12,
      "demand_spike": 0.28
    }
  }
}
```

#### Volatility Analysis

Get volatility analysis and forecasting.

```http
POST /risk/volatility
```

**Request Body:**
```json
{
  "gpu_type": "a100",
  "providers": ["aws", "gcp", "runpod"],
  "forecast_horizon": 7,
  "model_type": "realized_volatility",
  "confidence_intervals": true
}
```

**Response:**
```json
{
  "data": {
    "gpu_type": "a100",
    "volatility_analysis": {
      "current_volatility": 0.15,
      "historical_volatility": 0.12,
      "volatility_trend": "increasing",
      "volatility_regime": "normal",
      "mean_reversion_level": 0.11,
      "volatility_forecast": [
        {
          "date": "2024-01-16",
          "forecast_volatility": 0.14,
          "confidence_interval_lower": 0.11,
          "confidence_interval_upper": 0.17
        }
      ]
    },
    "provider_comparison": {
      "aws": {"volatility": 0.15, "trend": "stable"},
      "gcp": {"volatility": 0.12, "trend": "decreasing"},
      "runpod": {"volatility": 0.18, "trend": "increasing"}
    }
  }
}
```

### Cost Management

#### Get Cost Analysis

Analyze costs and savings opportunities.

```http
POST /cost/analysis
```

**Request Body:**
```json
{
  "time_period": "30d",
  "user_id": "user_123",
  "include_projections": true,
  "include_recommendations": true,
  "granularity": "daily"
}
```

**Response:**
```json
{
  "data": {
    "cost_summary": {
      "total_cost": 1250.50,
      "total_savings": 890.25,
      "savings_percentage": 41.6,
      "avg_hourly_cost": 1.75,
      "cost_trend": "increasing"
    },
    "cost_breakdown": {
      "by_provider": {
        "aws": {"cost": 450.25, "percentage": 36.0},
        "gcp": {"cost": 380.15, "percentage": 30.4},
        "runpod": {"cost": 320.10, "percentage": 25.6}
      },
      "by_gpu_type": {
        "a100": {"cost": 780.30, "percentage": 62.4},
        "h100": {"cost": 320.15, "percentage": 25.6},
        "a10g": {"cost": 150.05, "percentage": 12.0}
      }
    },
    "optimization_recommendations": [
      {
        "type": "spot_migration",
        "potential_savings": 125.30,
        "confidence": 0.85,
        "description": "Migrate 3 instances to spot pricing",
        "implementation_effort": "low"
      }
    ],
    "forecast": {
      "next_month_cost": 1320.75,
      "confidence": 0.78,
      "factors": ["market_trends", "seasonal_patterns"]
    }
  }
}
```

#### Set Budget Alerts

Configure budget monitoring and alerts.

```http
POST /cost/budget-alerts
```

**Request Body:**
```json
{
  "budget_amount": 1000.0,
  "alert_thresholds": [50, 75, 90, 100],
  "notification_channels": ["email", "slack"],
  "alert_frequency": "daily",
  "include_forecasts": true
}
```

**Response:**
```json
{
  "data": {
    "budget_id": "budget_123456",
    "budget_amount": 1000.0,
    "current_spend": 450.25,
    "remaining_budget": 549.75,
    "alert_thresholds": [50, 75, 90, 100],
    "status": "active",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### Deployment Management

#### Deploy GPU Instance

Deploy to the optimal GPU instance.

```http
POST /deployment/deploy
```

**Request Body:**
```json
{
  "gpu_type": "a100",
  "provider": "runpod",
  "instance_type": "A100-40GB",
  "region": "us-west-2",
  "spot_pricing": true,
  "duration_hours": 4,
  "configuration": {
    "image": "pytorch/pytorch:latest",
    "command": ["python", "train.py"],
    "environment": {
      "BATCH_SIZE": "32",
      "LEARNING_RATE": "0.001"
    },
    "resources": {
      "cpu": 8,
      "memory": "32Gi",
      "storage": "100Gi"
    }
  }
}
```

**Response:**
```json
{
  "data": {
    "deployment_id": "deploy_123456",
    "provider": "runpod",
    "instance_type": "A100-40GB",
    "gpu_type": "a100",
    "status": "initializing",
    "estimated_cost": 3.56,
    "estimated_duration": "4 hours",
    "region": "us-west-2",
    "created_at": "2024-01-15T10:30:00Z",
    "estimated_ready_time": "2024-01-15T10:32:00Z"
  }
}
```

#### Get Deployment Status

Check deployment status and metrics.

```http
GET /deployment/{deployment_id}/status
```

**Response:**
```json
{
  "data": {
    "deployment_id": "deploy_123456",
    "status": "running",
    "provider": "runpod",
    "instance_type": "A100-40GB",
    "gpu_type": "a100",
    "start_time": "2024-01-15T10:32:00Z",
    "current_cost": 0.89,
    "estimated_total_cost": 3.56,
    "runtime_hours": 1.0,
    "utilization": {
      "gpu_utilization": 0.85,
      "cpu_utilization": 0.45,
      "memory_utilization": 0.62
    },
    "logs_url": "https://runpod.io/logs/deploy_123456",
    "metrics_url": "https://runpod.io/metrics/deploy_123456"
  }
}
```

#### Stop Deployment

Stop a running deployment.

```http
POST /deployment/{deployment_id}/stop
```

**Response:**
```json
{
  "data": {
    "deployment_id": "deploy_123456",
    "status": "stopping",
    "final_cost": 1.78,
    "total_runtime": 2.0,
    "stopped_at": "2024-01-15T12:32:00Z"
  }
}
```

### User Management

#### Get User Profile

Get current user information and usage.

```http
GET /user/profile
```

**Response:**
```json
{
  "data": {
    "user_id": "user_123",
    "email": "user@example.com",
    "username": "user123",
    "tier": "paid",
    "created_at": "2024-01-01T00:00:00Z",
    "usage": {
      "gpu_hours_used": 125.5,
      "monthly_limit": 1000,
      "total_cost": 890.25,
      "total_savings": 680.50
    },
    "permissions": [
      "gpu_arbitrage:read",
      "gpu_arbitrage:write",
      "cost:read",
      "deployment:write"
    ],
    "api_keys": [
      {
        "id": "key_123",
        "name": "Production Key",
        "created_at": "2024-01-01T00:00:00Z",
        "last_used": "2024-01-15T10:30:00Z",
        "usage_count": 1250
      }
    ]
  }
}
```

#### Update User Settings

Update user preferences and settings.

```http
PUT /user/settings
```

**Request Body:**
```json
{
  "notification_preferences": {
    "email_alerts": true,
    "slack_alerts": false,
    "budget_alerts": true,
    "deployment_alerts": true
  },
  "default_preferences": {
    "preferred_providers": ["aws", "runpod"],
    "preferred_regions": ["us-west-2"],
    "spot_pricing": true,
    "max_budget": 10.0
  }
}
```

### System Health

#### Health Check

Check system health and status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "arbitrage_engine": "healthy",
    "risk_engine": "healthy"
  },
  "metrics": {
    "uptime": "72h 15m 30s",
    "requests_per_minute": 125,
    "active_deployments": 45,
    "error_rate": 0.002
  }
}
```

#### System Metrics

Get detailed system metrics.

```http
GET /metrics
```

**Response:**
```json
{
  "data": {
    "api_metrics": {
      "requests_per_minute": 125,
      "average_response_time": 0.15,
      "error_rate": 0.002,
      "active_connections": 45
    },
    "arbitrage_metrics": {
      "scans_per_hour": 180,
      "opportunities_found": 25,
      "avg_savings_percentage": 58.3,
      "success_rate": 0.95
    },
    "resource_metrics": {
      "cpu_utilization": 0.65,
      "memory_utilization": 0.78,
      "disk_utilization": 0.45,
      "network_io": 125.5
    }
  }
}
```

## SDKs and Libraries

### Python SDK

```python
from terradev import TerradevClient

# Initialize client
client = TerradevClient(
    api_key="your-api-key",
    base_url="https://api.terradev.io/v1"
)

# Get GPU prices
prices = client.gpu.get_prices(gpu_types=["a100", "h100"])

# Find arbitrage opportunities
opportunities = client.arbitrage.find_opportunities(
    gpu_type="a100",
    max_budget=5.0
)

# Deploy instance
deployment = client.deployment.deploy(
    gpu_type="a100",
    provider="runpod",
    duration_hours=4
)
```

### JavaScript SDK

```javascript
import { TerradevClient } from 'terradev-js';

// Initialize client
const client = new TerradevClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.terradev.io/v1'
});

// Get GPU prices
const prices = await client.gpu.getPrices({
  gpuTypes: ['a100', 'h100']
});

// Find arbitrage opportunities
const opportunities = await client.arbitrage.findOpportunities({
  gpuType: 'a100',
  maxBudget: 5.0
});
```

### Go SDK

```go
package main

import (
    "github.com/terradev/terradev-go"
)

func main() {
    // Initialize client
    client := terradev.NewClient("your-api-key")
    
    // Get GPU prices
    prices, err := client.GPU.GetPrices(&terradev.GPUPricesRequest{
        GPUTypes: []string{"a100", "h100"},
    })
    
    // Find arbitrage opportunities
    opportunities, err := client.Arbitrage.FindOpportunities(&terradev.OpportunitiesRequest{
        GPUType: "a100",
        MaxBudget: 5.0,
    })
}
```

## Webhooks

Configure webhooks to receive real-time notifications.

### Create Webhook

```http
POST /webhooks
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["deployment.created", "deployment.completed", "budget.exceeded"],
  "secret": "your-webhook-secret",
  "active": true
}
```

### Webhook Events

#### Deployment Created

```json
{
  "event": "deployment.created",
  "data": {
    "deployment_id": "deploy_123456",
    "provider": "runpod",
    "gpu_type": "a100",
    "estimated_cost": 3.56,
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

#### Budget Exceeded

```json
{
  "event": "budget.exceeded",
  "data": {
    "budget_id": "budget_123456",
    "current_spend": 1050.25,
    "budget_amount": 1000.0,
    "exceeded_by": 50.25,
    "alert_threshold": 100
  }
}
```

## Rate Limits and Quotas

| Tier | Requests/Minute | GPU Hours/Month | Features |
|------|------------------|-----------------|----------|
| Free | 100 | 10 | Basic arbitrage, spot pricing |
| Paid | 1000 | Unlimited | Advanced risk analysis, API access |
| Enterprise | Custom | Unlimited | Custom features, dedicated support |

## Support

- **Documentation**: https://docs.terradev.io
- **API Reference**: https://api.terradev.io/docs
- **Support Email**: support@terradev.io
- **Status Page**: https://status.terradev.io

## Changelog

### v1.0.0 (2024-01-15)
- Initial API release
- GPU pricing endpoints
- Arbitrage analysis
- Risk management
- Cost optimization
- Deployment management

### v1.1.0 (Planned)
- Enhanced volatility forecasting
- Multi-cloud deployment orchestration
- Advanced analytics dashboards
- Custom alerting rules
