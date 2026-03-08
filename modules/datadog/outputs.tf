output "monitor_ids" {
  description = "Map of monitor name to Datadog monitor ID"
  value = var.create_monitors ? {
    budget_alert      = datadog_monitor.terradev_budget_alert[0].id
    cost_spike        = datadog_monitor.terradev_cost_spike[0].id
    idle_gpu          = datadog_monitor.terradev_idle_gpu[0].id
    spot_risk         = datadog_monitor.terradev_spot_risk[0].id
    provider_degraded = datadog_monitor.terradev_provider_degraded[0].id
    egress_anomaly    = datadog_monitor.terradev_egress_anomaly[0].id
  } : {}
}

output "dashboard_id" {
  description = "Datadog dashboard ID"
  value       = var.create_dashboard ? datadog_dashboard.terradev_gpu_finops[0].id : null
}

output "dashboard_url" {
  description = "Datadog dashboard URL"
  value       = var.create_dashboard ? datadog_dashboard.terradev_gpu_finops[0].url : null
}
