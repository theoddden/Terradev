terraform {
  required_providers {
    hyperstack = {
      source  = "hyperstack/hyperstack"
      version = "~> 1.0"
    }
    tailscale = {
      source  = "tailscale/tailscale"
      version = "~> 0.13"
    }
  }
}
