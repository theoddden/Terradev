# Critical Management Widgets for Terradev
# Target: AWS, Azure, GCP, RunPod, Lambda, CoreWeave APIs + Grafana, Jenkins, GitHub, Slack

## 1. Grafana Dashboard Integration
# GPU arbitrage metrics dashboard

resource "grafana_dashboard" "terradev_gpu_arbitrage" {
  config_json = jsonencode({
    annotations = {
      list = []
    }
    editable = true
    gnetId = null
    graphTooltip = 0
    id = null
    links = []
    liveNow = false
    panels = [
      {
        aliasColors = {}
        bars = false
        dashLength = 10
        dashes = false
        datasource = "Terradev-Prometheus"
        fill = 1
        fillGradient = 0
        gridPos = {
          h = 8
          w = 12
          x = 0
          y = 0
        }
        hiddenSeries = false
        id = 1
        legend = {
          avg = false
          current = false
          max = false
          min = false
          show = true
          total = false
          values = false
        }
        lines = true
        linewidth = 1
        nullPointMode = "null"
        options = {
          DataLinks = []
          alertThreshold = true
        }
        percentage = false
        pluginVersion = "7.5.7"
        pointradius = 2
        points = false
        renderer = "flot"
        seriesOverrides = []
        spaceLength = 10
        stack = false
        steppedLine = false
        targets = [
          {
            expr = "terradev_gpu_savings_percentage"
            interval = ""
            legendFormat = "GPU Savings %"
            refId = "A"
          },
          {
            expr = "terradev_arbitrage_confidence"
            interval = ""
            legendFormat = "Arbitrage Confidence"
            refId = "B"
          }
        ]
        thresholds = []
        timeFrom = null
        timeRegions = []
        timeShift = null
        title = "GPU Arbitrage Performance"
        tooltip = {
          shared = true
          sort = 0
          value_type = "individual"
        }
        type = "graph"
        xaxis = {
          buckets = null
          mode = "time"
          name = null
          show = true
          values = []
        }
        yaxes = [
          {
            format = "percent"
            label = null
            logBase = 1
            max = "100"
            min = "0"
            show = true
          },
          {
            format = "short"
            label = null
            logBase = 1
            max = null
            min = null
            show = true
          }
        ]
        yaxis = {
          align = false
          alignLevel = null
        }
      },
      {
        datasource = "Terradev-Prometheus"
        fieldConfig = {
          defaults = {
            color = {
              mode = "palette-classic"
            }
            custom = {
              align = null
              displayMode = "auto"
            }
            mappings = []
            thresholds = {
              mode = "absolute"
              steps = [
                {
                  color = "green"
                  value = null
                },
                {
                  color = "red"
                  value = 80
                }
              ]
            }
          }
          overrides = []
        }
        gridPos = {
          h = 8
          w = 12
          x = 12
          y = 0
        }
        id = 2
        options = {
          showHeader = true
        }
        pluginVersion = "7.5.7"
        targets = [
          {
            expr = "terradev_provider_prices"
            format = "table"
            instant = true
            refId = "A"
          }
        ]
        title = "Current GPU Prices by Provider"
        type = "table"
      },
      {
        datasource = "Terradev-Prometheus"
        fieldConfig = {
          defaults = {
            color = {
              mode = "thresholds"
            }
            mappings = []
            thresholds = {
              mode = "absolute"
              steps = [
                {
                  color = "green"
                  value = null
                },
                {
                  color = "yellow"
                  value = 50
                },
                {
                  color = "red"
                  value = 80
                }
              ]
            }
          }
          overrides = []
        }
        gridPos = {
          h = 8
          w = 6
          x = 0
          y = 8
        }
        id = 3
        options = {
          colorMode = "value"
          graphMode = "area"
          justifyMode = "auto"
          orientation = "auto"
          reduceOptions = {
            values = false
            calcs = [
              "lastNotNull"
            ]
            fields = ""
          }
          textMode = "auto"
        }
        pluginVersion = "7.5.7"
        targets = [
          {
            expr = "terradev_total_savings_this_month"
            instant = true
            refId = "A"
          }
        ]
        title = "Total Monthly Savings"
        type = "stat"
      },
      {
        datasource = "Terradev-Prometheus"
        fieldConfig = {
          defaults = {
            color = {
              mode = "thresholds"
            }
            mappings = []
            thresholds = {
              mode = "absolute"
              steps = [
                {
                  color = "red"
                  value = null
                },
                {
                  color = "yellow"
                  value = 0.3
                },
                {
                  color = "green"
                  value = 0.7
                }
              ]
            }
          }
          overrides = []
        }
        gridPos = {
          h = 8
          w = 6
          x = 6
          y = 8
        }
        id = 4
        options = {
          colorMode = "value"
          graphMode = "area"
          justifyMode = "auto"
          orientation = "auto"
          reduceOptions = {
            values = false
            calcs = [
              "lastNotNull"
            ]
            fields = ""
          }
          textMode = "auto"
        }
        pluginVersion = "7.5.7"
        targets = [
          {
            expr = "terradev_market_efficiency"
            instant = true
            refId = "A"
          }
        ]
        title = "Market Efficiency"
        type = "stat"
      }
    ]
    refresh = "5s"
    schemaVersion = 27
    style = "dark"
    tags = [
      "terradev"
      "gpu"
      "arbitrage"
    ]
    templating = {
      list = []
    }
    time = {
      from = "now-1h"
      to = "now"
    }
    timepicker = {}
    timezone = ""
    title = "Terradev GPU Arbitrage Dashboard"
    uid = "terradev-gpu-arbitrage"
    version = 1
  })
}

## 2. Jenkins Plugin Configuration
# GPU arbitrage monitoring for Jenkins

resource "local_file" "jenkins_terradev_plugin" {
  filename = "${path.module}/jenkins-plugins/terradev.hpi"
  content = "" # This would be the compiled Jenkins plugin
}

# Jenkins configuration for Terradev
resource "local_file" "jenkins_config" {
  filename = "${path.module}/jenkins-config.xml"
  content = xmlencode({
    hudson = {
      pluginManager = {
        plugins = [
          {
            "terradev-gpu-arbitrage" = {
              version = "1.0.0"
              enabled = true
              pinned = false
            }
          }
        ]
      }
      
      views = [
        {
          name = "Terradev GPU Dashboard"
          type = "hudson.model.AllView"
          filter = "true"
          recurse = "false"
          
          jobNames = ["gpu-arbitrage-*"]
          
          property = [
            {
              "hudson.plugins.view.dashboard.DashboardPortlet" = {
                name = "Terradev GPU Savings"
                plugin = "dashboard@latest"
                properties = {
                  terradev_api_endpoint = "https://api.terradev.io/v1"
                  terradev_api_key = "${var.terradev_api_key}"
                  refresh_interval = "300"
                  show_savings_chart = "true"
                  show_deployment_status = "true"
                  show_provider_comparison = "true"
                }
              }
            }
          ]
        }
      ]
    }
  })
}

## 3. GitHub Actions Integration
# GPU cost optimization in CI/CD

resource "local_file" "github_actions_workflow" {
  filename = "${path.module}/.github/workflows/terradev-gpu-optimization.yml"
  content = yamlencode({
    name = "Terradev GPU Optimization"
    
    on = {
      push = {
        branches = ["main", "develop"]
      }
      pull_request = {
        branches = ["main"]
      }
      workflow_dispatch = {
        inputs = {
          gpu_type = {
            description = "GPU type to use"
            required = false
            default = "a100"
            type = "choice"
            options = ["a100", "h100", "a10g"]
          }
          hours_needed = {
            description = "GPU hours needed"
            required = false
            default = "4"
            type = "string"
          }
        }
      }
    }
    
    env = {
      TERRADEV_API_KEY = "${{ secrets.TERRADEV_API_KEY }}"
      TERRADEV_TIER = "${{ secrets.TERRADEV_TIER }}"
    }
    
    jobs = {
      "gpu-arbitrage-analysis" = {
        runs-on = "ubuntu-latest"
        outputs = {
          should_deploy = "${{ steps.find-gpu.outputs.should_deploy }}"
          provider = "${{ steps.find-gpu.outputs.provider }}"
          gpu_type = "${{ steps.find-gpu.outputs.gpu_type }}"
          price = "${{ steps.find-gpu.outputs.price }}"
          savings = "${{ steps.find-gpu.outputs.savings }}"
          confidence = "${{ steps.find-gpu.outputs.confidence }}"
        }
        
        steps = [
          {
            name = "Checkout"
            uses = "actions/checkout@v3"
          },
          {
            name = "Setup Terradev CLI"
            run = <<-EOT
              curl -fsSL https://api.terradev.io/install.sh | sh
              echo "$HOME/.terradev/bin" >> $GITHUB_PATH
              terradev login --api-key $TERRADEV_API_KEY
            EOT
          },
          {
            name = "Find Optimal GPU"
            id = "find-gpu"
            run = <<-EOT
              GPU_TYPE="${{ github.event.inputs.gpu_type || 'a100' }}"
              HOURS_NEEDED="${{ github.event.inputs.hours_needed || '4' }}"
              
              echo "Finding optimal GPU for $GPU_TYPE for $HOURS_NEEDED hours..."
              
              RESULT=$(terradev find-optimal-gpu \
                --gpu-type "$GPU_TYPE" \
                --hours "$HOURS_NEEDED" \
                --confidence-threshold 0.7 \
                --max-risk-score 0.5 \
                --format json)
              
              echo "Result: $RESULT"
              
              # Parse results
              SHOULD_DEPLOY=$(echo "$RESULT" | jq -r '.should_deploy')
              PROVIDER=$(echo "$RESULT" | jq -r '.provider')
              GPU_TYPE_FOUND=$(echo "$RESULT" | jq -r '.gpu_type')
              PRICE=$(echo "$RESULT" | jq -r '.price')
              SAVINGS=$(echo "$RESULT" | jq -r '.savings_percentage')
              CONFIDENCE=$(echo "$RESULT" | jq -r '.arbitrage_confidence')
              
              echo "should_deploy=$SHOULD_DEPLOY" >> $GITHUB_OUTPUT
              echo "provider=$PROVIDER" >> $GITHUB_OUTPUT
              echo "gpu_type=$GPU_TYPE_FOUND" >> $GITHUB_OUTPUT
              echo "price=$PRICE" >> $GITHUB_OUTPUT
              echo "savings=$SAVINGS" >> $GITHUB_OUTPUT
              echo "confidence=$CONFIDENCE" >> $GITHUB_OUTPUT
              
              echo "## 🚀 Terradev GPU Arbitrage Results" >> $GITHUB_STEP_SUMMARY
              echo "| Metric | Value |" >> $GITHUB_STEP_SUMMARY
              echo "|--------|-------|" >> $GITHUB_STEP_SUMMARY
              echo "| Provider | $PROVIDER |" >> $GITHUB_STEP_SUMMARY
              echo "| GPU Type | $GPU_TYPE_FOUND |" >> $GITHUB_STEP_SUMMARY
              echo "| Price | \$$PRICE/hour |" >> $GITHUB_STEP_SUMMARY
              echo "| Savings | $SAVINGS% |" >> $GITHUB_STEP_SUMMARY
              echo "| Confidence | $CONFIDENCE |" >> $GITHUB_STEP_SUMMARY
              echo "| Should Deploy | $SHOULD_DEPLOY |" >> $GITHUB_STEP_SUMMARY
            EOT
          },
          {
            name = "Deploy Training Job"
            if = "steps.find-gpu.outputs.should_deploy == 'true'"
            run = <<-EOT
              PROVIDER="${{ steps.find-gpu.outputs.provider }}"
              GPU_TYPE="${{ steps.find-gpu.outputs.gpu_type }}"
              PRICE="${{ steps.find-gpu.outputs.price }}"
              
              echo "🚀 Deploying to $PROVIDER $GPU_TYPE at $PRICE/hour"
              
              # Deploy using Terradev
              terradev deploy \
                --provider "$PROVIDER" \
                --gpu-type "$GPU_TYPE" \
                --hours "${{ github.event.inputs.hours_needed || '4' }}" \
                --job-id "github-${{ github.run_id }}"
              
              echo "✅ Deployment successful!"
            EOT
          },
          {
            name = "Skip Deployment"
            if = "steps.find-gpu.outputs.should_deploy != 'true'"
            run = <<-EOT
              echo "⏸️ Skipping deployment - conditions not optimal"
              echo "Confidence: ${{ steps.find-gpu.outputs.confidence }}"
              echo "Risk Score: ${{ steps.find-gpu.outputs.risk_score }}"
              echo "Recommendation: Wait for better conditions"
            EOT
          }
        ]
      }
      
      "training-job" = {
        needs = "gpu-arbitrage-analysis"
        if = "needs.gpu-arbitrage-analysis.outputs.should_deploy == 'true'"
        runs-on = "ubuntu-latest"
        
        steps = [
          {
            name = "Setup Training Environment"
            run = <<-EOT
              echo "Setting up training environment on ${{ needs.gpu-arbitrage-analysis.outputs.provider }}"
              # Setup training script here
            EOT
          },
          {
            name = "Run Training"
            run = <<-EOT
              echo "Running training on ${{ needs.gpu-arbitrage-analysis.outputs.gpu_type }}"
              # Run actual training here
            EOT
          }
        ]
      }
    }
  })
}

## 4. Slack Integration
# Real-time GPU arbitrage notifications

resource "local_file" "slack_bot_config" {
  filename = "${path.module}/slack-bot/config.json"
  content = jsonencode({
    bot_token = "${var.slack_bot_token}"
    app_token = "${var.slack_app_token}"
    
    terradev_api = {
      endpoint = "https://api.terradev.io/v1"
      api_key = "${var.terradev_api_key}"
    }
    
    notifications = {
      gpu_deployment = {
        enabled = true
        channels = ["#gpu-alerts", "#ml-team"]
        template = "🚀 **GPU Deployment**\n• Provider: {provider}\n• GPU: {gpu_type}\n• Price: ${price}/hour\n• Savings: {savings}%\n• Confidence: {confidence}%"
      }
      
      savings_milestone = {
        enabled = true
        channels = ["#finance", "#management"]
        template = "💰 **Savings Milestone**\n• Total Saved: ${total_savings}\n• This Month: ${monthly_savings}\n• Deployments: {deployment_count}"
      }
      
      market_opportunity = {
        enabled = true
        channels = ["#gpu-opportunities"]
        template = "🎯 **Market Opportunity**\n• GPU: {gpu_type}\n• Provider: {provider}\n• Price Drop: {price_drop}%\n• Recommended: {action}"
      }
      
      risk_alert = {
        enabled = true
        channels = ["#ops-alerts"]
        template = "⚠️ **Risk Alert**\n• Provider: {provider}\n• Risk Score: {risk_score}\n• Recommendation: {recommendation}"
      }
    }
    
    commands = {
      "/gpu-status" = {
        description = "Show current GPU deployment status"
        action = "show_gpu_status"
      }
      
      "/gpu-savings" = {
        description = "Show total savings this month"
        action = "show_savings"
      }
      
      "/gpu-find" = {
        description = "Find optimal GPU for training"
        parameters = ["gpu_type", "hours"]
        action = "find_optimal_gpu"
      }
      
      "/gpu-deploy" = {
        description = "Deploy to optimal GPU"
        parameters = ["gpu_type", "hours"]
        action = "deploy_gpu"
      }
    }
  })
}

# Slack bot implementation
resource "local_file" "slack_bot_code" {
  filename = "${path.module}/slack-bot/bot.py"
  content = <<-EOT
  import os
  import json
  import requests
  from slack_bolt import App
  from slack_bolt.adapter.socket_mode import SocketModeHandler
  
  # Initialize Slack app
  app = App(token=os.environ["SLACK_BOT_TOKEN"])
  
  # Terradev API configuration
  TERRADEV_API = os.environ.get("TERRADEV_API", "https://api.terradev.io/v1")
  TERRADEV_KEY = os.environ.get("TERRADEV_API_KEY")
  
  @app.command("/gpu-status")
  def gpu_status_command(ack, respond):
      ack()
      
      try:
          response = requests.get(
              f"{TERRADEV_API}/status",
              headers={"Authorization": f"Bearer {TERRADEV_KEY}"}
          )
          data = response.json()
          
          blocks = [
              {
                  "type": "header",
                  "text": {"type": "plain_text", "text": "🚀 GPU Deployment Status"}
              },
              {
                  "type": "section",
                  "fields": [
                      {"type": "mrkdwn", "text": f"*Active Deployments:* {data.get('active_deployments', 0)}"},
                      {"type": "mrkdwn", "text": f"*Total Savings:* ${data.get('total_savings', 0)}"}
                  ]
              }
          ]
          
          respond(blocks=blocks)
          
      except Exception as e:
          respond(text=f"Error getting GPU status: {str(e)}")
  
  @app.command("/gpu-savings")
  def gpu_savings_command(ack, respond):
      ack()
      
      try:
          response = requests.get(
              f"{TERRADEV_API}/savings",
              headers={"Authorization": f"Bearer {TERRADEV_KEY}"}
          )
          data = response.json()
          
          blocks = [
              {
                  "type": "header",
                  "text": {"type": "plain_text", "text": "💰 GPU Savings Summary"}
              },
              {
                  "type": "section",
                  "fields": [
                      {"type": "mrkdwn", "text": f"*This Month:* ${data.get('monthly_savings', 0)}"},
                      {"type": "mrkdwn", "text": f"*Average Savings:* {data.get('average_savings_pct', 0)}%"}
                  ]
              }
          ]
          
          respond(blocks=blocks)
          
      except Exception as e:
          respond(text=f"Error getting savings data: {str(e)}")
  
  @app.command("/gpu-find")
  def gpu_find_command(ack, respond, command):
      ack()
      
      # Parse parameters
      params = dict(param.split('=') for param in command['text'].split() if '=' in param)
      gpu_type = params.get('gpu_type', 'a100')
      hours = params.get('hours', '4')
      
      try:
          response = requests.post(
              f"{TERRADEV_API}/find-optimal-gpu",
              headers={"Authorization": f"Bearer {TERRADEV_KEY}"},
              json={
                  "gpu_type": gpu_type,
                  "hours": int(hours),
                  "confidence_threshold": 0.7
              }
          )
          data = response.json()
          
          if data.get('should_deploy'):
              blocks = [
                  {
                      "type": "header",
                      "text": {"type": "plain_text", "text": "🎯 Optimal GPU Found"}
                  },
                  {
                      "type": "section",
                      "fields": [
                          {"type": "mrkdwn", "text": f"*Provider:* {data.get('provider')}"},
                          {"type": "mrkdwn", "text": f"*GPU Type:* {data.get('gpu_type')}"},
                          {"type": "mrkdwn", "text": f"*Price:* ${data.get('price')}/hour"},
                          {"type": "mrkdwn", "text": f"*Savings:* {data.get('savings_percentage')}%"}
                      ]
                  },
                  {
                      "type": "actions",
                      "elements": [
                          {
                              "type": "button",
                              "text": {"type": "plain_text", "text": "🚀 Deploy Now"},
                              "action_id": "deploy_gpu",
                              "value": json.dumps(data)
                          }
                      ]
                  }
              ]
          else:
              blocks = [
                  {
                      "type": "header",
                      "text": {"type": "plain_text", "text": "⏸️ Wait for Better Conditions"}
                  },
                  {
                      "type": "section",
                      "text": {"type": "mrkdwn", "text": f"Current conditions not optimal:\n• Confidence: {data.get('arbitrage_confidence', 0)}\n• Risk Score: {data.get('risk_score', 0)}\n• Recommendation: {data.get('optimal_timing', 'unknown')}"}
                  }
              ]
          
          respond(blocks=blocks)
          
      except Exception as e:
          respond(text=f"Error finding optimal GPU: {str(e)}")
  
  @app.action("deploy_gpu")
  def deploy_gpu_action(ack, respond, body, action):
      ack()
      
      try:
          gpu_data = json.loads(action['value'])
          
          response = requests.post(
              f"{TERRADEV_API}/deploy",
              headers={"Authorization": f"Bearer {TERRADEV_KEY}"},
              json=gpu_data
          )
          
          if response.status_code == 200:
              respond(text="✅ GPU deployment started successfully!")
          else:
              respond(text=f"❌ Deployment failed: {response.text}")
              
      except Exception as e:
          respond(text=f"Error deploying GPU: {str(e)}")
  
  # Start the app
  if __name__ == "__main__":
      handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
      handler.start()
  EOT
}

## 5. Cloud Provider API Integrations
# Direct API connections for all supported providers

resource "local_file" "provider_apis_config" {
  filename = "${path.module}/config/provider-apis.json"
  content = jsonencode({
    providers = {
      aws = {
        name = "Amazon Web Services"
        api_endpoint = "https://pricing.us-east-1.amazonaws.com"
        authentication = {
          type = "aws_signature_v4"
          region = "us-east-1"
          service = "pricing"
        }
        pricing_api = {
          base_url = "https://pricing.us-east-1.amazonaws.com"
          endpoint = "offers/v1.0/aws/AmazonEC2/current/index.json"
          rate_limit = "100 requests/second"
        }
        compute_api = {
          base_url = "https://ec2.amazonaws.com"
          regions = ["us-west-2", "us-east-1", "eu-west-1"]
        }
        gpu_types = {
          a100 = "p4d.24xlarge"
          h100 = "p5.48xlarge"
          a10g = "g5.xlarge"
        }
      }
      
      gcp = {
        name = "Google Cloud Platform"
        api_endpoint = "https://cloudbilling.googleapis.com"
        authentication = {
          type = "oauth2"
          scopes = ["https://www.googleapis.com/auth/cloud-billing"]
        }
        pricing_api = {
          base_url = "https://cloudbilling.googleapis.com"
          endpoint = "v1/services/6F81-5844-456A/skus"
          rate_limit = "100 requests/second"
        }
        compute_api = {
          base_url = "https://compute.googleapis.com"
          regions = ["us-west2", "us-central1", "europe-west1"]
        }
        gpu_types = {
          a100 = "a2-highgpu-8g"
          h100 = "a2-ultragpu-8g"
          a10g = "g2-standard-8"
        }
      }
      
      azure = {
        name = "Microsoft Azure"
        api_endpoint = "https://prices.azure.com"
        authentication = {
          type = "oauth2"
          tenant_id = "${var.azure_tenant_id}"
          client_id = "${var.azure_client_id}"
          client_secret = "${var.azure_client_secret}"
        }
        pricing_api = {
          base_url = "https://prices.azure.com"
          endpoint = "api/retail/prices"
          rate_limit = "1000 requests/hour"
        }
        compute_api = {
          base_url = "https://management.azure.com"
          regions = ["westus2", "eastus", "westeurope"]
        }
        gpu_types = {
          a100 = "Standard_ND96asr_v4"
          h100 = "Standard_ND96isr_H100_v5"
          a10g = "Standard_NC48ads_A100_v4"
        }
      }
      
      runpod = {
        name = "RunPod"
        api_endpoint = "https://api.runpod.io"
        authentication = {
          type = "api_key"
          api_key = "${var.runpod_api_key}"
        }
        pricing_api = {
          base_url = "https://api.runpod.io"
          endpoint = "v1/gpu/pricing"
          rate_limit = "60 requests/minute"
        }
        compute_api = {
          base_url = "https://api.runpod.io"
          regions = ["us-west-2", "us-east-1", "eu-west-1"]
        }
        gpu_types = {
          a100 = "A100-80GB"
          h100 = "H100-80GB"
          a10g = "A10G-24GB"
        }
      }
      
      lambda = {
        name = "Lambda Labs"
        api_endpoint = "https://api.labs.lambda.cloud"
        authentication = {
          type = "api_key"
          api_key = "${var.lambda_api_key}"
        }
        pricing_api = {
          base_url = "https://api.labs.lambda.cloud"
          endpoint = "v1/gpu/pricing"
          rate_limit = "60 requests/minute"
        }
        compute_api = {
          base_url = "https://api.labs.lambda.cloud"
          regions = ["us-west", "us-east", "eu-west"]
        }
        gpu_types = {
          a100 = "A100"
          h100 = "H100"
          a10g = "A10G"
        }
      }
      
      coreweave = {
        name = "CoreWeave"
        api_endpoint = "https://api.coreweave.com"
        authentication = {
          type = "api_key"
          api_key = "${var.coreweave_api_key}"
        }
        pricing_api = {
          base_url = "https://api.coreweave.com"
          endpoint = "v1/gpu/pricing"
          rate_limit = "100 requests/minute"
        }
        compute_api = {
          base_url = "https://api.coreweave.com"
          regions = ["us-west-2", "us-east-1", "eu-west-1"]
        }
        gpu_types = {
          a100 = "A100-80GB"
          h100 = "H100-80GB"
          a10g = "A10G-24GB"
        }
      }
    }
    
    rate_limiting = {
      default = "100 requests/second"
      aws = "100 requests/second"
      gcp = "100 requests/second"
      azure = "1000 requests/hour"
      runpod = "60 requests/minute"
      lambda = "60 requests/minute"
      coreweave = "100 requests/minute"
    }
    
    retry_policy = {
      max_retries = 3
      backoff_factor = 2
      retry_on_status = [429, 500, 502, 503, 504]
    }
  })
}
