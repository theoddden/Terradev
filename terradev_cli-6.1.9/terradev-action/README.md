# Terradev GPU Provision — GitHub Action

Provision GPU instances across 10 cloud providers with cost-optimized arbitrage, directly from your CI/CD pipeline.

## Quick Start

```yaml
- uses: theoddden/terradev-action@v1
  with:
    gpu-type: A100
    dry-run: true
  env:
    TERRADEV_RUNPOD_KEY: ${{ secrets.RUNPOD_API_KEY }}
```

## Usage Examples

### Quote GPU prices

```yaml
- uses: theoddden/terradev-action@v1
  with:
    command: quote
    gpu-type: A100
```

### Provision cheapest instance

```yaml
- uses: theoddden/terradev-action@v1
  id: gpu
  with:
    command: provision
    gpu-type: H100
    max-price: "2.00"
    providers: runpod,vastai,lambda
  env:
    TERRADEV_RUNPOD_KEY: ${{ secrets.RUNPOD_API_KEY }}
    TERRADEV_VASTAI_KEY: ${{ secrets.VASTAI_API_KEY }}
    TERRADEV_LAMBDA_KEY: ${{ secrets.LAMBDA_API_KEY }}

- name: Use provisioned instance
  run: echo "Instance ${{ steps.gpu.outputs.instance-id }} at ${{ steps.gpu.outputs.price }}"
```

### Provision + run Docker training job

```yaml
- uses: theoddden/terradev-action@v1
  with:
    command: run
    gpu-type: A100
    image: pytorch/pytorch:latest
    run-command: "python train.py --epochs 10"
    max-price: "1.50"
  env:
    TERRADEV_RUNPOD_KEY: ${{ secrets.RUNPOD_API_KEY }}
```

### Multi-GPU parallel provision

```yaml
- uses: theoddden/terradev-action@v1
  with:
    command: provision
    gpu-type: H100
    count: "4"
    parallel: "6"
    providers: runpod,vastai,lambda,coreweave
  env:
    TERRADEV_RUNPOD_KEY: ${{ secrets.RUNPOD_API_KEY }}
    TERRADEV_VASTAI_KEY: ${{ secrets.VASTAI_API_KEY }}
    TERRADEV_LAMBDA_KEY: ${{ secrets.LAMBDA_API_KEY }}
    TERRADEV_COREWEAVE_KEY: ${{ secrets.COREWEAVE_API_KEY }}
```

### Full ML training pipeline

```yaml
name: Train Model
on:
  push:
    branches: [main]
    paths: ['training/**']

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: theoddden/terradev-action@v1
        id: gpu
        with:
          command: run
          gpu-type: A100
          image: my-org/training:latest
          run-command: "python train.py --config config.yaml"
          max-price: "1.50"
        env:
          TERRADEV_RUNPOD_KEY: ${{ secrets.RUNPOD_API_KEY }}
          TERRADEV_VASTAI_KEY: ${{ secrets.VASTAI_API_KEY }}

      - name: Training output
        run: echo "${{ steps.gpu.outputs.output }}"
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `command` | Terradev command (quote, provision, run, status, manage, optimize) | No | `provision` |
| `gpu-type` | GPU type (A100, H100, RTX4090, etc.) | No | — |
| `count` | Number of instances | No | `1` |
| `max-price` | Maximum $/hr | No | — |
| `providers` | Comma-separated provider list | No | all configured |
| `parallel` | Max parallel threads | No | `6` |
| `dry-run` | Plan without launching | No | `false` |
| `image` | Docker image (for `run`) | No | — |
| `run-command` | Container command (for `run`) | No | — |
| `keep-alive` | Keep instance after completion | No | `false` |
| `extra-args` | Additional CLI arguments | No | — |
| `terradev-version` | terradev-cli version | No | `latest` |

## Outputs

| Output | Description |
|--------|-------------|
| `instance-id` | Provisioned instance ID |
| `provider` | Cloud provider used |
| `price` | Hourly price |
| `output` | Full CLI output |

## Environment Variables (Secrets)

Configure provider credentials as GitHub Secrets:

| Secret | Provider |
|--------|----------|
| `TERRADEV_RUNPOD_KEY` | RunPod |
| `TERRADEV_VASTAI_KEY` | Vast.ai |
| `TERRADEV_AWS_ACCESS_KEY` + `TERRADEV_AWS_SECRET_KEY` | AWS |
| `TERRADEV_GCP_PROJECT` + `TERRADEV_GCP_CREDENTIALS` | Google Cloud |
| `TERRADEV_AZURE_SUBSCRIPTION` + `TERRADEV_AZURE_TENANT` + `TERRADEV_AZURE_CLIENT_ID` + `TERRADEV_AZURE_CLIENT_SECRET` | Azure |
| `TERRADEV_LAMBDA_KEY` | Lambda Labs |
| `TERRADEV_COREWEAVE_KEY` | CoreWeave |
| `TERRADEV_TENSORDOCK_KEY` + `TERRADEV_TENSORDOCK_SECRET` | TensorDock |
| `TERRADEV_ORACLE_TENANCY` + `TERRADEV_ORACLE_USER` + `TERRADEV_ORACLE_KEY` | Oracle Cloud |
| `TERRADEV_BASETEN_KEY` | Baseten |

## License

MIT
