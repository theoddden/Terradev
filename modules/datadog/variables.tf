variable "datadog_api_key" {
  description = "Datadog API Key"
  type        = string
  sensitive   = true
}

variable "datadog_app_key" {
  description = "Datadog Application Key"
  type        = string
  sensitive   = true
}

variable "datadog_site" {
  description = "Datadog site (datadoghq.com, datadoghq.eu, us3.datadoghq.com, us5.datadoghq.com, ap1.datadoghq.com)"
  type        = string
  default     = "datadoghq.com"
}

variable "notification_channel" {
  description = "Notification target for alerts (e.g. @slack-terradev-alerts, @pagerduty-gpu-oncall)"
  type        = string
  default     = "@slack-terradev-alerts"
}

variable "budget_threshold_critical" {
  description = "Budget utilization critical threshold (%)"
  type        = number
  default     = 80
}

variable "budget_threshold_warning" {
  description = "Budget utilization warning threshold (%)"
  type        = number
  default     = 60
}

variable "cost_spike_threshold" {
  description = "Cost spike detection threshold (% change over 4h)"
  type        = number
  default     = 50
}

variable "idle_gpu_threshold" {
  description = "GPU utilization below this % for 30m triggers idle alert"
  type        = number
  default     = 10
}

variable "spot_volatility_threshold" {
  description = "Spot price volatility threshold (annualized %)"
  type        = number
  default     = 100
}

variable "provider_reliability_threshold" {
  description = "Provider reliability score below this triggers alert (0-100)"
  type        = number
  default     = 70
}

variable "create_dashboard" {
  description = "Whether to create the GPU FinOps dashboard"
  type        = bool
  default     = true
}

variable "create_monitors" {
  description = "Whether to create GPU FinOps monitors"
  type        = bool
  default     = true
}

variable "extra_tags" {
  description = "Additional tags to apply to all resources"
  type        = list(string)
  default     = []
}
