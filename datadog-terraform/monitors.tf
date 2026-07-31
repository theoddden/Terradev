resource "datadog_monitor" "terradev_budget_alert" {
  name    = "[Terradev] GPU Budget Alert"
  type    = "metric alert"
  query   = "avg(last_1h):avg:terradev.gpu.budget_utilization{*} > 80"
  message = "GPU budget >80%. Current: {{value}}%\n\n- Switch to spot\n- Downsize idle GPUs\n- Shut idle instances\n\n@slack-terradev-alerts"

  monitor_thresholds {
    critical = 80
    warning  = 60
  }

  notify_no_data = false
  tags           = ["terradev", "finops", "budget"]
  priority       = 2
}

resource "datadog_monitor" "terradev_cost_spike" {
  name    = "[Terradev] GPU Cost Spike"
  type    = "metric alert"
  query   = "pct_change(avg(last_1h),last_4h):avg:terradev.gpu.cost_per_hour{*} > 50"
  message = "GPU cost spiked >50% vs 4h baseline.\n\n- Unintended expensive provision?\n- Spot fallback?\n\n@slack-terradev-alerts"

  monitor_thresholds {
    critical = 50
    warning  = 25
  }

  notify_no_data = false
  tags           = ["terradev", "finops", "cost-spike"]
  priority       = 2
}

resource "datadog_monitor" "terradev_idle_gpu" {
  name    = "[Terradev] Idle GPU Detection"
  type    = "metric alert"
  query   = "avg(last_30m):avg:terradev.training.gpu_util{*} by {instance_id} < 10"
  message = "GPU {{instance_id.name}} <10% util for 30m. Terminate or downsize.\n\n@slack-terradev-alerts"

  monitor_thresholds {
    critical = 10
    warning  = 25
  }

  notify_no_data = false
  tags           = ["terradev", "finops", "idle"]
  priority       = 3
}

resource "datadog_monitor" "terradev_spot_risk" {
  name    = "[Terradev] Spot Volatility"
  type    = "metric alert"
  query   = "avg(last_15m):avg:terradev.price.volatility{spot:true} by {provider,gpu_type} > 100"
  message = "High spot volatility {{provider.name}} {{gpu_type.name}}. Vol: {{value}}%\n\n- Checkpoint auto-save\n- On-demand fallback\n\n@slack-terradev-alerts"

  monitor_thresholds {
    critical = 100
    warning  = 60
  }

  notify_no_data = false
  tags           = ["terradev", "finops", "spot"]
  priority       = 1
}

resource "datadog_monitor" "terradev_provider_degraded" {
  name    = "[Terradev] Provider Degraded"
  type    = "metric alert"
  query   = "avg(last_1h):avg:terradev.provider.reliability{*} by {provider} < 70"
  message = "Provider {{provider.name}} reliability <70. Score: {{value}}\n\n@slack-terradev-alerts"

  monitor_thresholds {
    critical = 70
    warning  = 85
  }

  notify_no_data = false
  tags           = ["terradev", "finops", "provider"]
  priority       = 3
}

resource "datadog_monitor" "terradev_egress_anomaly" {
  name    = "[Terradev] Egress Anomaly"
  type    = "metric alert"
  query   = "avg(last_1h):anomalies(avg:terradev.egress.cost{*}, 'agile', 3) >= 1"
  message = "Anomalous egress cost.\n\n- Cross-cloud transfers?\n- Missing compression?\n\n@slack-terradev-alerts"

  monitor_thresholds {
    critical = 1
    warning  = 0
  }

  notify_no_data = false
  tags           = ["terradev", "finops", "egress"]
  priority       = 3
}