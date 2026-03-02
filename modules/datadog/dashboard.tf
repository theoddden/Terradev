resource "datadog_dashboard" "terradev_gpu_finops" {
  count = var.create_dashboard ? 1 : 0

  title       = "Terradev GPU FinOps"
  description = "Multi-cloud GPU cost intelligence — provisioned by Terradev"
  layout_type = "ordered"

  # ── Row 1: Key Figures ───────────────────────────────────────────────────

  widget {
    query_value_definition {
      title = "Hourly Spend"
      request {
        q          = "sum:terradev.gpu.cost_per_hour{*}"
        aggregator = "avg"
      }
      custom_unit = "$/hr"
      precision   = 2
    }
    widget_layout {
      x      = 0
      y      = 0
      width  = 3
      height = 2
    }
  }

  widget {
    query_value_definition {
      title = "Monthly Projected"
      request {
        q          = "avg:terradev.gpu.monthly_projected{*}"
        aggregator = "avg"
      }
      custom_unit = "$"
      precision   = 0
    }
    widget_layout {
      x      = 3
      y      = 0
      width  = 3
      height = 2
    }
  }

  widget {
    query_value_definition {
      title = "Active GPUs"
      request {
        q          = "sum:terradev.provisions.active{*}"
        aggregator = "avg"
      }
      precision = 0
    }
    widget_layout {
      x      = 6
      y      = 0
      width  = 3
      height = 2
    }
  }

  widget {
    query_value_definition {
      title = "Budget Used"
      request {
        q          = "avg:terradev.gpu.budget_utilization{*}"
        aggregator = "avg"
      }
      custom_unit = "%"
      precision   = 1
      conditional_formats {
        comparator = ">"
        value      = 80
        palette    = "white_on_red"
      }
      conditional_formats {
        comparator = ">"
        value      = 60
        palette    = "white_on_yellow"
      }
      conditional_formats {
        comparator = "<="
        value      = 60
        palette    = "white_on_green"
      }
    }
    widget_layout {
      x      = 9
      y      = 0
      width  = 3
      height = 2
    }
  }

  # ── Row 2: Cost Over Time ────────────────────────────────────────────────

  widget {
    timeseries_definition {
      title = "GPU Cost/hr by Provider"
      request {
        q            = "avg:terradev.gpu.cost_per_hour{*} by {provider}"
        display_type = "bars"
      }
    }
    widget_layout {
      x      = 0
      y      = 2
      width  = 6
      height = 3
    }
  }

  widget {
    timeseries_definition {
      title = "Quote Prices by GPU"
      request {
        q            = "avg:terradev.price.quote{*} by {gpu_type,provider}"
        display_type = "line"
      }
    }
    widget_layout {
      x      = 6
      y      = 2
      width  = 6
      height = 3
    }
  }

  # ── Row 3: Provider Health ───────────────────────────────────────────────

  widget {
    toplist_definition {
      title = "Provider Reliability Ranking"
      request {
        q = "avg:terradev.provider.reliability{*} by {provider}"
      }
    }
    widget_layout {
      x      = 0
      y      = 5
      width  = 4
      height = 3
    }
  }

  widget {
    timeseries_definition {
      title = "Price Volatility"
      request {
        q            = "avg:terradev.price.volatility{*} by {provider,gpu_type}"
        display_type = "line"
      }
    }
    widget_layout {
      x      = 4
      y      = 5
      width  = 4
      height = 3
    }
  }

  widget {
    timeseries_definition {
      title = "Quote API Latency (ms)"
      request {
        q            = "avg:terradev.provider.latency_ms{*} by {provider}"
        display_type = "line"
      }
    }
    widget_layout {
      x      = 8
      y      = 5
      width  = 4
      height = 3
    }
  }

  # ── Row 4: Training & Egress ─────────────────────────────────────────────

  widget {
    timeseries_definition {
      title = "Training GPU Utilization"
      request {
        q            = "avg:terradev.training.gpu_util{*} by {job_id}"
        display_type = "line"
      }
      yaxis {
        max = "100"
      }
    }
    widget_layout {
      x      = 0
      y      = 8
      width  = 6
      height = 3
    }
  }

  widget {
    timeseries_definition {
      title = "Egress Cost"
      request {
        q            = "sum:terradev.egress.cost{*} by {src_provider,dst_provider}.as_count()"
        display_type = "bars"
      }
    }
    widget_layout {
      x      = 6
      y      = 8
      width  = 6
      height = 3
    }
  }

  # ── Row 5: Event Stream ─────────────────────────────────────────────────

  widget {
    event_stream_definition {
      title      = "Terradev Events"
      query      = "source:terradev"
      event_size = "l"
    }
    widget_layout {
      x      = 0
      y      = 11
      width  = 12
      height = 3
    }
  }

  tags = concat(["terradev", "finops", "gpu"], var.extra_tags)

  lifecycle {
    create_before_destroy = true
  }
}
