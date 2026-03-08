terraform {
  required_providers {
    vastai = {
      source  = "vastai/vastai"
      version = "~> 1.0"
    }
    tailscale = {
      source  = "tailscale/tailscale"
      version = "~> 0.13"
    }
  }
}
