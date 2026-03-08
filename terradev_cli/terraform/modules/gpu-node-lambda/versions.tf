terraform {
  required_providers {
    lambda = {
      source  = "lambda-labs/lambda"
      version = "~> 1.0"
    }
    tailscale = {
      source  = "tailscale/tailscale"
      version = "~> 0.13"
    }
  }
}
