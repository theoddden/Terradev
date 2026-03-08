# DevOps & CI/CD Platform Integrations
# Widgets for popular DevOps platforms and CI/CD systems

## 1. GitHub Actions Integration
# GitHub App for GPU cost optimization in CI/CD

resource "github_actions_secret" "terradev_api_key" {
  repository = "my-ml-repo"
  secret_name = "TERRADEV_API_KEY"
  plaintext_value = var.terradev_api_key
}

# GitHub Actions workflow with Terradev integration
resource "local_file" "github_workflow" {
  filename = "${path.module}/.github/workflows/gpu-optimized-ci.yml"
  content = yamlencode({
    name = "GPU-Optimized ML Training"
    
    on = {
      push = {
        branches = ["main"]
      }
    }
    
    jobs = {
      "gpu-arbitrage" = {
        runs-on = "ubuntu-latest"
        steps = [
          {
            name = "Setup Terradev"
            uses = "terradev/setup@v1"
            with = {
              api_key = "${{ secrets.TERRADEV_API_KEY }}"
            }
          },
          {
            name = "Find Optimal GPU"
            id = "find-gpu"
            run = "terradev find-gpu --type a100 --hours 8 --confidence 0.8"
          },
          {
            name = "Deploy Training Job"
            if = "steps.find-gpu.outputs.should_deploy == 'true'"
            run = "terradev deploy --provider ${{ steps.find-gpu.outputs.provider }} --gpu ${{ steps.find-gpu.outputs.gpu_type }}"
          }
        ]
      }
    }
  })
}

## 2. GitLab CI/CD Integration
# GitLab CI template with Terradev optimization

resource "local_file" "gitlab_ci_template" {
  filename = "${path.module}/.gitlab-ci.yml"
  content = yamlencode({
    stages = ["gpu-arbitrage", "training", "cleanup"]
    
    variables = {
      TERRADEV_API_KEY = "$TERRADEV_API_KEY"
    }
    
    "gpu-arbitrage" = {
      stage = "gpu-arbitrage"
      script = [
        "curl -s https://api.terradev.io/v1/find-optimal-gpu \\",
        "  -H \"Authorization: Bearer $TERRADEV_API_KEY\" \\",
        "  -H \"Content-Type: application/json\" \\",
        "  -d '{\"gpu_type\": \"a100\", \"hours\": 8, \"confidence_threshold\": 0.8}' \\",
        "  > gpu_config.json",
        "export SHOULD_DEPLOY=$(jq -r '.should_deploy' gpu_config.json)",
        "if [ \"$SHOULD_DEPLOY\" = \"true\" ]; then",
        "  echo \"Deploying to optimal GPU configuration\"",
        "  export GPU_PROVIDER=$(jq -r '.provider' gpu_config.json)",
        "  export GPU_TYPE=$(jq -r '.gpu_type' gpu_config.json)",
        "  export GPU_PRICE=$(jq -r '.price' gpu_config.json)",
        "  echo \"GPU_PROVIDER=$GPU_PROVIDER\" >> gpu_env.txt",
        "  echo \"GPU_TYPE=$GPU_TYPE\" >> gpu_env.txt",
        "  echo \"GPU_PRICE=$GPU_PRICE\" >> gpu_env.txt",
        "fi"
      ]
      artifacts = {
        paths = ["gpu_config.json", "gpu_env.txt"]
        reports = {
          junit = ["gpu_arbitrage_report.xml"]
        }
      }
    }
    
    "training" = {
      stage = "training"
      dependencies = ["gpu-arbitrage"]
      script = [
        "source gpu_env.txt",
        "if [ -n \"$GPU_PROVIDER\" ]; then",
        "  echo \"Starting training on $GPU_PROVIDER $GPU_TYPE at $GPU_PRICE/hour\"",
        "  # Training script here",
        "fi"
      ]
    }
  })
}

## 3. Jenkins Plugin Integration
# Jenkins plugin for Terradev GPU optimization

resource "local_file" "jenkins_plugin_config" {
  filename = "${path.module}/jenkins-plugin-config.xml"
  content = xmlencode({
    hudson = {
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
                  terradev_api_key = "${var.terradev_api_key}"
                  refresh_interval = "300"  # 5 minutes
                  show_savings_chart = "true"
                  show_deployment_status = "true"
                }
              }
            }
          ]
        }
      ]
    }
  })
}

## 4. Azure DevOps Integration
# Azure DevOps extension for Terradev

resource "local_file" "azure_devops_extension" {
  filename = "${path.module}/azure-devops-extension.json"
  content = jsonencode({
    manifestVersion = 1
    id = "terradev-gpu-arbitrage"
    name = "Terradev GPU Arbitrage"
    version = "1.0.0"
    
    publisher = "terradev"
    targets = [
      {
        id = "Microsoft.VisualStudio.Services"
      }
    ]
    
    description = "Optimize GPU costs across cloud providers with real-time arbitrage"
    
    categories = ["Build and Release"]
    
    contributions = [
      {
        id = "terradev-dashboard-widget"
        type = "ms.vss-dashboards-web.widget"
        targets = [
          {
            id = "ms.vss-dashboards-web.widget-catalog"
          }
        ]
        properties = {
          name = "Terradev GPU Arbitrage"
          description = "Shows GPU cost savings and deployment recommendations"
          uri = "dist/terradev-widget.html"
          sizes = [
            { rowSpan = 1, columnSpan = 2 },
            { rowSpan = 2, columnSpan = 3 }
          ]
          scopes = ["project"]
        }
      }
    ]
  })
}

## 5. CircleCI Orb Integration
# CircleCI Orb for Terradev GPU optimization

resource "local_file" "circleci_orb" {
  filename = "${path.module}/circleci-orb.yml"
  content = yamlencode({
    orbs = {
      terradev = "terradev/gpu-arbitrage@1.0.0"
    }
    
    version = 2.1
    
    jobs = {
      "gpu-optimized-training" = {
        docker = [
          { image = "cimg/python:3.9" }
        ]
        
        steps = [
          { checkout = {} },
          {
            terradev/setup = {
              api_key = "TERRADEV_API_KEY"
            }
          },
          {
            terradev/find_optimal_gpu = {
              gpu_type = "a100"
              hours = 8
              confidence_threshold = 0.8
            }
          },
          {
            when = {
              condition = "<< steps.find_optimal_gpu.outputs.should_deploy >>"
              steps = [
                {
                  terradev/deploy = {
                    provider = "<< steps.find_optimal_gpu.outputs.provider >>"
                    gpu_type = "<< steps.find_optimal_gpu.outputs.gpu_type >>"
                  }
                }
              ]
            }
          }
        ]
      }
    }
    
    workflows = {
      version = 2
      "gpu-optimization" = {
        jobs = ["gpu-optimized-training"]
      }
    }
  })
}
