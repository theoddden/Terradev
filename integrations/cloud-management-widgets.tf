# Cloud Management Platform Integrations
# Widgets for major cloud management and monitoring platforms

## 1. AWS Management Console Widget
# Integrates with AWS Management Console as a custom widget

resource "aws_cloudwatch_dashboard" "terradev_widget" {
  dashboard_name = "Terradev-GPU-Arbitrage"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["Terradev", "GPU_Savings", "Percentage", "InstanceId", "i-1234567890abcdef0"],
            [".", "GPU_Cost", "Hourly", ".", "."],
            [".", "Deployment_Time", "Seconds", ".", "."]
          ]
          period = 300
          stat   = "Average"
          region = "us-west-2"
          title  = "Terradev GPU Arbitrage Metrics"
          yAxis = {
            left = {
              min = 0
              max = 100
            }
          }
        }
      },
      {
        type   = "text"
        x      = 0
        y      = 6
        width  = 24
        height = 3

        properties = {
          markdown = "# Terradev GPU Arbitrage Status\n\n**Current Savings**: 79%\n**Active Deployments**: 3\n**Market Efficiency**: 78%\n\n[Open Terradev Dashboard](https://terradev.io/dashboard)"
        }
      }
    ]
  })
}

## 2. Azure Dashboard Widget
# Integrates with Azure Monitor dashboards

resource "azurerm_monitor_dashboard" "terradev_widget" {
  name                = "Terradev-GPU-Arbitrage"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  dashboard_properties = jsonencode({
    lenses = [
      {
        order = 1
        parts = [
          {
            position = {
              x = 0
              y = 0
              width = 6
              height = 3
            }
            metadata = {
              title = "Terradev GPU Savings"
              type = "Extension/HubsExtension"
              settings = {
                content = {
                  options = {
                    chart = {
                      series = [
                        {
                          data = [
                            { x = "AWS", y = 79 },
                            { x = "GCP", y = 72 },
                            { x = "Azure", y = 68 }
                          ]
                        }
                      ]
                      type = "Bar"
                    }
                  }
                }
              }
            }
          }
        ]
      }
    ]
  })
}

## 3. GCP Cloud Monitoring Widget
# Integrates with Google Cloud Monitoring

resource "google_monitoring_dashboard" "terradev_widget" {
  dashboard_json = jsonencode({
    displayName = "Terradev GPU Arbitrage"
    gridLayout = {
      columns = "2"
      widgets = [
        {
          title = "GPU Cost Savings"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  prometheusQuery = {
                    query = "terradev_gpu_savings_percentage"
                  }
                }
                plotType = "LINE"
              }
            ]
            timeshiftDuration = "0s"
            yAxis = {
              scale = "LINEAR"
            }
          }
        }
      ]
    }
  })
}
