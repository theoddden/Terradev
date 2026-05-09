#!/bin/bash
# Terradev GPU Check — Quick local GPU status for OpenClaw agents
# Usage: ./gpu-check.sh [quote|status|overflow]

set -e

ACTION="${1:-status}"

case "$ACTION" in
  status)
    echo "=== Local GPU Status ==="
    if command -v nvidia-smi &>/dev/null; then
      nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r name mem_used mem_total util; do
        pct=$((mem_used * 100 / mem_total))
        echo "GPU: $name"
        echo "  VRAM: ${mem_used}MB / ${mem_total}MB (${pct}%)"
        echo "  Utilization: ${util}%"
        if [ "$pct" -gt 90 ]; then
          echo "  ⚠️  VRAM > 90% — consider cloud overflow"
          echo "  Run: terradev quote -g A10G"
        fi
      done
    else
      echo "No local NVIDIA GPU detected."
      echo "Use Terradev to provision cloud GPUs:"
      echo "  terradev quote -g A100"
    fi
    echo ""
    echo "=== Cloud Instances ==="
    if command -v terradev &>/dev/null; then
      terradev --skip-onboarding status 2>/dev/null || echo "No cloud instances running."
    else
      echo "Terradev CLI not installed. Run: pip install terradev-cli"
    fi
    ;;

  quote)
    GPU_TYPE="${2:-A100}"
    echo "=== Real-Time GPU Quotes: $GPU_TYPE ==="
    if command -v terradev &>/dev/null; then
      terradev --skip-onboarding quote -g "$GPU_TYPE" 2>/dev/null
    else
      echo "Terradev CLI not installed. Run: pip install terradev-cli"
    fi
    ;;

  overflow)
    echo "=== GPU Overflow Check ==="
    LOCAL_UTIL=0
    if command -v nvidia-smi &>/dev/null; then
      LOCAL_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    fi

    if [ "$LOCAL_UTIL" -gt 85 ] 2>/dev/null; then
      echo "🔴 Local GPU at ${LOCAL_UTIL}% — overflow recommended"
      echo ""
      echo "Cheapest cloud options right now:"
      terradev --skip-onboarding quote -g A10G 2>/dev/null || echo "Configure providers: terradev setup runpod --quick"
      echo ""
      echo "Quick provision:"
      echo "  terradev provision -g A10G --max-price 1.50"
      echo "  terradev run --gpu A10G --image pytorch/pytorch:latest -c 'python train.py'"
    else
      echo "🟢 Local GPU at ${LOCAL_UTIL}% — no overflow needed"
      echo "Tip: Use 'terradev quote -g H100' to check cloud prices anyway"
    fi
    ;;

  k8s)
    echo "=== K8s GPU Cluster Commands ==="
    echo "Create:  terradev k8s create my-cluster --gpu H100 --count 4 --multi-cloud --prefer-spot"
    echo "List:    terradev k8s list"
    echo "Info:    terradev k8s info <cluster>"
    echo "Destroy: terradev k8s destroy <cluster>"
    ;;

  *)
    echo "Usage: gpu-check.sh [status|quote|overflow|k8s]"
    echo "  status   — Local GPU + cloud instance status"
    echo "  quote    — Real-time GPU pricing (default: A100)"
    echo "  overflow — Check if local GPU needs cloud burst"
    echo "  k8s      — K8s cluster management commands"
    ;;
esac
