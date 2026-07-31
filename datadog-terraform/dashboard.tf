resource "datadog_dashboard_json" "terradev_gpu_finops" {
  dashboard = jsonencode({
    title       = "Terradev GPU FinOps"
    description = "Multi-cloud GPU cost intelligence — provisioned by Terradev"
    layout_type = "ordered"
    tags        = ["terradev", "finops", "gpu"]
    widgets     = jsondecode(<<-EOT
[
  {
    "definition": {
      "type": "query_value",
      "title": "Hourly Spend",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "sum:terradev.gpu.cost_per_hour{*}"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "scalar"
        }
      ],
      "custom_unit": "$/hr"
    },
    "layout": {
      "x": 0,
      "y": 0,
      "width": 3,
      "height": 2
    }
  },
  {
    "definition": {
      "type": "query_value",
      "title": "Monthly Projected",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "avg:terradev.gpu.monthly_projected{*}"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "scalar"
        }
      ],
      "custom_unit": "$"
    },
    "layout": {
      "x": 3,
      "y": 0,
      "width": 3,
      "height": 2
    }
  },
  {
    "definition": {
      "type": "query_value",
      "title": "Active GPUs",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "sum:terradev.provisions.active{*}"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "scalar"
        }
      ]
    },
    "layout": {
      "x": 6,
      "y": 0,
      "width": 3,
      "height": 2
    }
  },
  {
    "definition": {
      "type": "query_value",
      "title": "Budget Used",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "avg:terradev.gpu.budget_utilization{*}"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "scalar"
        }
      ],
      "custom_unit": "%",
      "conditional_formats": [
        {
          "comparator": ">",
          "value": 80,
          "palette": "white_on_red"
        },
        {
          "comparator": ">",
          "value": 60,
          "palette": "white_on_yellow"
        },
        {
          "comparator": "<=",
          "value": 60,
          "palette": "white_on_green"
        }
      ]
    },
    "layout": {
      "x": 9,
      "y": 0,
      "width": 3,
      "height": 2
    }
  },
  {
    "definition": {
      "type": "timeseries",
      "title": "GPU Cost/hr by Provider",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "avg:terradev.gpu.cost_per_hour{*} by {provider}"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "timeseries",
          "display_type": "bars"
        }
      ]
    },
    "layout": {
      "x": 0,
      "y": 2,
      "width": 6,
      "height": 3
    }
  },
  {
    "definition": {
      "type": "timeseries",
      "title": "Quote Prices by GPU",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "avg:terradev.price.quote{*} by {gpu_type,provider}"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "timeseries",
          "display_type": "line"
        }
      ]
    },
    "layout": {
      "x": 6,
      "y": 2,
      "width": 6,
      "height": 3
    }
  },
  {
    "definition": {
      "type": "toplist",
      "title": "Provider Reliability",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "avg:terradev.provider.reliability{*} by {provider}"
            }
          ],
          "formulas": [
            {
              "formula": "a",
              "limit": {
                "count": 20,
                "order": "desc"
              }
            }
          ],
          "response_format": "scalar"
        }
      ]
    },
    "layout": {
      "x": 0,
      "y": 5,
      "width": 4,
      "height": 3
    }
  },
  {
    "definition": {
      "type": "timeseries",
      "title": "Price Volatility",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "avg:terradev.price.volatility{*} by {provider,gpu_type}"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "timeseries",
          "display_type": "line"
        }
      ]
    },
    "layout": {
      "x": 4,
      "y": 5,
      "width": 4,
      "height": 3
    }
  },
  {
    "definition": {
      "type": "timeseries",
      "title": "Quote API Latency",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "avg:terradev.provider.latency_ms{*} by {provider}"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "timeseries",
          "display_type": "line"
        }
      ]
    },
    "layout": {
      "x": 8,
      "y": 5,
      "width": 4,
      "height": 3
    }
  },
  {
    "definition": {
      "type": "timeseries",
      "title": "Training GPU Util",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "avg:terradev.training.gpu_util{*} by {job_id}"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "timeseries",
          "display_type": "line"
        }
      ]
    },
    "layout": {
      "x": 0,
      "y": 8,
      "width": 6,
      "height": 3
    }
  },
  {
    "definition": {
      "type": "timeseries",
      "title": "Egress Cost",
      "requests": [
        {
          "queries": [
            {
              "data_source": "metrics",
              "name": "a",
              "query": "sum:terradev.egress.cost{*} by {src_provider,dst_provider}.as_count()"
            }
          ],
          "formulas": [
            {
              "formula": "a"
            }
          ],
          "response_format": "timeseries",
          "display_type": "bars"
        }
      ]
    },
    "layout": {
      "x": 6,
      "y": 8,
      "width": 6,
      "height": 3
    }
  },
  {
    "definition": {
      "type": "event_stream",
      "title": "Terradev Events",
      "query": "source:terradev",
      "tags_execution": "and",
      "event_size": "l"
    },
    "layout": {
      "x": 0,
      "y": 11,
      "width": 12,
      "height": 3
    }
  }
]
EOT
    )
  })
}