locals {
  base_tags = concat(["terradev", "finops", "gpu"], var.extra_tags)
}

# ── Budget Alert ─────────────────────────────────────────────────────────────

resource "datadog_monitor" "terradev_budget_alert" {
  count = var.create_monitors ? 1 : 0

  name    = "[Terradev] GPU Budget Alert"
  type    = "metric alert"
  query   = "avg(last_1h):avg:terradev.gpu.budget_utilization{*} > ${var.budget_threshold_critical}"
  message = <<-EOT
    GPU budget utilization has exceeded ${var.budget_threshold_critical}%.

    Current: {{value}}%

    Recommended actions:
    - Switch to spot instances
    - Downsize underutilized GPUs
    - Shut down idle instances

    ${var.notification_channel}
  EOT

  monitor_thresholds {
    critical = var.budget_threshold_critical
    warning  = var.budget_threshold_warning
  }

  notify_no_data    = false
  renotify_interval = 60
  tags              = concat(local.base_tags, ["budget"])
  priority          = 2

  lifecycle {
    create_before_destroy = true
  }
}

# ── Cost Spike ───────────────────────────────────────────────────────────────

resource "datadog_monitor" "terradev_cost_spike" {
  count = var.create_monitors ? 1 : 0

  name    = "[Terradev] GPU Cost Spike"
  type    = "metric alert"
  query   = "pct_change(avg(last_1h),last_4h):avg:terradev.gpu.cost_per_hour{*} > ${var.cost_spike_threshold}"
  message = <<-EOT
    GPU hourly cost spiked >${var.cost_spike_threshold}% vs 4h baseline.

    Possible causes:
    - Unintended expensive provision
    - Spot to on-demand fallback
    - Provider price increase

    ${var.notification_channel}
  EOT

  monitor_thresholds {
    critical = var.cost_spike_threshold
    warning  = var.cost_spike_threshold / 2
  }

  notify_no_data    = false
  renotify_interval = 120
  tags              = concat(local.base_tags, ["cost-spike"])
  priority          = 2

  lifecycle {
    create_before_destroy = true
  }
}

# ── Idle GPU Detection ───────────────────────────────────────────────────────

resource "datadog_monitor" "terradev_idle_gpu" {
  count = var.create_monitors ? 1 : 0

  name    = "[Terradev] Idle GPU Detection"
  type    = "metric alert"
  query   = "avg(last_30m):avg:terradev.training.gpu_util{*} by {instance_id} < ${var.idle_gpu_threshold}"
  message = <<-EOT
    GPU instance {{instance_id.name}} has <${var.idle_gpu_threshold}% utilization for 30 minutes.

    Consider terminating or downsizing this instance.

    ${var.notification_channel}
  EOT

  monitor_thresholds {
    critical = var.idle_gpu_threshold
    warning  = var.idle_gpu_threshold * 2.5
  }

  notify_no_data    = false
  renotify_interval = 60
  tags              = concat(local.base_tags, ["idle"])
  priority          = 3

  lifecycle {
    create_before_destroy = true
  }
}

# ── Spot Instance Volatility ─────────────────────────────────────────────────

resource "datadog_monitor" "terradev_spot_risk" {
  count = var.create_monitors ? 1 : 0

  name    = "[Terradev] Spot Instance Volatility"
  type    = "metric alert"
  query   = "avg(last_15m):avg:terradev.price.volatility{spot:true} by {provider,gpu_type} > ${var.spot_volatility_threshold}"
  message = <<-EOT
    High spot price volatility for {{provider.name}} {{gpu_type.name}}.
    Volatility: {{value}}%

    Mitigation:
    - Enable checkpoint auto-save
    - Prepare on-demand fallback
    - Consider provider switch

    ${var.notification_channel}
  EOT

  monitor_thresholds {
    critical = var.spot_volatility_threshold
    warning  = var.spot_volatility_threshold * 0.6
  }

  notify_no_data    = false
  renotify_interval = 30
  tags              = concat(local.base_tags, ["spot"])
  priority          = 1

  lifecycle {
    create_before_destroy = true
  }
}

# ── Provider Degraded ────────────────────────────────────────────────────────

resource "datadog_monitor" "terradev_provider_degraded" {
  count = var.create_monitors ? 1 : 0

  name    = "[Terradev] Provider Degraded"
  type    = "metric alert"
  query   = "avg(last_1h):avg:terradev.provider.reliability{*} by {provider} < ${var.provider_reliability_threshold}"
  message = <<-EOT
    Provider {{provider.name}} reliability dropped below ${var.provider_reliability_threshold}.
    Score: {{value}}

    Check provider status page and consider routing traffic elsewhere.

    ${var.notification_channel}
  EOT

  monitor_thresholds {
    critical = var.provider_reliability_threshold
    warning  = var.provider_reliability_threshold + 15
  }

  notify_no_data    = false
  renotify_interval = 120
  tags              = concat(local.base_tags, ["provider"])
  priority          = 3

  lifecycle {
    create_before_destroy = true
  }
}

# ── Egress Cost Anomaly ──────────────────────────────────────────────────────

resource "datadog_monitor" "terradev_egress_anomaly" {
  count = var.create_monitors ? 1 : 0

  name    = "[Terradev] Egress Cost Anomaly"
  type    = "metric alert"
  query   = "avg(last_1h):anomalies(avg:terradev.egress.cost{*}, 'agile', 3) >= 1"
  message = <<-EOT
    Anomalous egress cost detected.

    Check for:
    - Unintended cross-cloud transfers
    - Missing data compression
    - Redundant data movement

    ${var.notification_channel}
  EOT

  monitor_thresholds {
    critical = 1
  }

  notify_no_data = false
  tags           = concat(local.base_tags, ["egress"])
  priority       = 3

  lifecycle {
    create_before_destroy = true
  }
}
