# 🚀 Terradev GPU Cloud — OpenClaw Skill

**Cross-cloud GPU provisioning, K8s clusters, and inference overflow for OpenClaw agents.**

Your local GPU maxed out? One command to burst to cloud. Need a K8s cluster with H100s? Done. Want real-time pricing across 11+ providers? Instant.

## What This Skill Does

| Capability | Example |
|-----------|---------|
| **GPU Price Quotes** | "Find me the cheapest H100 right now" |
| **Multi-Cloud Provisioning** | "Spin up 4 A100s across the cheapest clouds" |
| **K8s GPU Clusters** | "Create a Kubernetes cluster with 8 H100 nodes" |
| **Inference Deployment** | "Deploy Llama 2 to a serverless endpoint" |
| **HuggingFace Spaces** | "Share my model on HuggingFace with one click" |
| **GPU Overflow** | "My local GPU is full, burst this job to cloud" |
| **Instance Management** | "Show me all running instances and costs" |
| **Cost Optimization** | "Find cheaper alternatives for my running GPUs" |

## Install

### Via ClawHub
```bash
clawhub install terradev-gpu-cloud
```

### Manual
```bash
# 1. Install Terradev CLI
pip install terradev-cli

# 2. Configure at least one provider
terradev setup runpod --quick
terradev configure --provider runpod

# 3. Copy the skill folder to your OpenClaw skills directory
cp -r terradev-gpu-cloud ~/.openclaw/skills/
```

## Demo

```
You: "Find me the cheapest H100 right now"

🔍 Querying 11 providers in parallel...

┌──────────────┬──────────┬──────────┬────────┐
│ Provider     │ GPU      │ Price/hr │ Region │
├──────────────┼──────────┼──────────┼────────┤
│ RunPod       │ H100 80G │ $1.89    │ US-TX  │
│ Lambda Labs  │ H100 80G │ $1.99    │ US-TX  │
│ Vast.ai      │ H100 80G │ $2.15    │ US-OR  │
│ CoreWeave    │ H100 80G │ $2.49    │ US-NJ  │
│ TensorDock   │ H100 80G │ $2.79    │ US-TX  │
│ AWS (spot)   │ p5.xlg   │ $3.21    │ us-e-1 │
│ GCP (spot)   │ a3-high  │ $3.89    │ us-c-1 │
└──────────────┴──────────┴──────────┴────────┘

💰 Best price: RunPod H100 @ $1.89/hr (47% cheaper than AWS)
```

```
You: "Create a K8s cluster with 4 H100s for training"

🔧 Creating multi-cloud GPU cluster...
   ├── Karpenter NodeClass: spot-first H100 scheduling
   ├── KEDA autoscaling: 90% GPU utilization trigger
   ├── CNI ordering: EKS v21 race condition handled
   └── Node pools: RunPod + Lambda (cheapest spots)

✅ Cluster 'training-cluster' ready in 47 seconds
   4x H100 80GB @ $1.89/hr avg = $7.56/hr total
```

```
You: "My RTX 4090 is maxed out running inference, overflow to cloud"

📊 Local GPU: RTX 4090 24GB — 98% utilized
🌊 Overflow strategy: Burst to cloud A10G (similar VRAM, $0.76/hr)

terradev provision -g A10G -n 2 --parallel 6

✅ 2x A10G provisioned on Vast.ai @ $0.76/hr
   Endpoint: ssh root@<ip> -p 22
   Run: terradev execute -i <id> -c "python serve.py"
```

---

## BYOAPI — Your Keys Stay Local

This skill never sees or stores your API keys. All credentials remain on your machine and are passed directly to cloud providers. This is the secure way to do cloud automation.

## Supported Providers

RunPod · Vast.ai · AWS · GCP · Azure · Lambda Labs · CoreWeave · TensorDock · Oracle Cloud · Crusoe Cloud · DigitalOcean · HyperStack

---

*Built with ❤️ by the Terradev team*: https://github.com/theoddden/Terradev
- **PyPI**: https://pypi.org/project/terradev-cli/
- **Docs**: https://theodden.github.io/Terradev/

## License

MIT License — see [LICENSE](LICENSE) file for details.
