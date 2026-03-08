# Terradev CLI - Docker Hub Distribution

## 🐳 Docker Images

Terradev CLI is available as Docker images for easy deployment and integration into containerized workflows.

## 📦 Available Images

### Primary Image
```bash
docker pull theoddden/terradev:2.9.2
```

### Latest Version
```bash
docker pull theoddden/terradev:latest
```

### Development Version
```bash
docker pull theoddden/terradev:dev
```

## 🚀 Quick Start

### Basic Usage
```bash
# Pull the image
docker pull theoddden/terradev:2.9.2

# Run GPU pricing quote
docker run --rm theoddden/terradev:2.9.2 python -m terradev_cli quote -g A100

# Interactive shell
docker run -it --rm theoddden/terradev:2.9.2 bash
```

### With Cloud Credentials
```bash
# Mount your AWS credentials
docker run --rm \
  -v ~/.aws:/home/terradev/.aws:ro \
  theoddden/terradev:2.9.2 \
  python -m terradev_cli quote -g A100

# Mount multiple cloud credentials
docker run --rm \
  -v ~/.aws:/home/terradev/.aws:ro \
  -v ~/.gcp:/home/terradev/.gcp:ro \
  -v ~/.azure:/home/terradev/.azure:ro \
  theoddden/terradev:2.9.2 \
  python -m terradev_cli provision -g A100 --duration 4
```

## 🐙 Docker Compose

### Complete Setup
```yaml
version: '3.8'

services:
  terradev-cli:
    image: theoddden/terradev:2.9.2
    container_name: terradev-cli
    environment:
      - TERRADEV_API_URL=https://api.terradev.cloud
      - TERRADEV_FALLBACK_URL=http://34.207.59.52:8080
    volumes:
      - ~/.aws:/home/terradev/.aws:ro
      - ~/.gcp:/home/terradev/.gcp:ro
      - ~/.azure:/home/terradev/.azure:ro
      - ./config:/home/terradev/.terradev
    command: python -m terradev_cli status
    restart: unless-stopped

  terradev-jupyter:
    image: theoddden/terradev:2.9.2
    container_name: terradev-jupyter
    environment:
      - TERRADEV_API_URL=https://api.terradev.cloud
    volumes:
      - ./notebooks:/home/terradev/notebooks
      - ~/.aws:/home/terradev/.aws:ro
    ports:
      - "8888:8888"
    command: >
      bash -c "
        pip install jupyterlab &&
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
      "
    restart: unless-stopped
```

### Usage
```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f terradev-cli

# Stop services
docker-compose down
```

## 🔧 Advanced Usage

### Custom Configuration
```bash
# Create config directory
mkdir -p ./config

# Run with custom config
docker run --rm \
  -v ./config:/home/terradev/.terradev \
  -v ~/.aws:/home/terradev/.aws:ro \
  theoddden/terradev:2.9.2 \
  python -m terradev_cli configure
```

### API Service
```bash
# Run as API service
docker run -d \
  --name terradev-api \
  -p 8080:8080 \
  -v ~/.aws:/home/terradev/.aws:ro \
  theoddden/terradev:2.9.2 \
  python -c "
from flask import Flask, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/quote/<gpu_type>')
def quote(gpu_type):
    try:
        result = subprocess.run(['python', '-m', 'terradev_cli', 'quote', '-g', gpu_type], 
                              capture_output=True, text=True)
        return jsonify({'gpu_type': gpu_type, 'output': result.stdout})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

app.run(host='0.0.0.0', port=8080)
"
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: terradev-cli
spec:
  replicas: 1
  selector:
    matchLabels:
      app: terradev-cli
  template:
    metadata:
      labels:
        app: terradev-cli
    spec:
      containers:
      - name: terradev-cli
        image: theoddden/terradev:2.9.2
        command: ["python", "-m", "terradev_cli", "status"]
        env:
        - name: TERRADEV_API_URL
          value: "https://api.terradev.cloud"
        volumeMounts:
        - name: aws-credentials
          mountPath: /home/terradev/.aws
          readOnly: true
      volumes:
      - name: aws-credentials
        secret:
          secretName: aws-credentials
```

## 🏷️ Image Tags

| Tag | Description |
|-----|-------------|
| `2.9.2` | Latest stable release |
| `latest` | Always points to latest stable |
| `dev` | Development version |
| `2.9` | Major version 2.9 |
| `2` | Major version 2 |

## 🔒 Security

- **Non-root user**: Container runs as `terradev` user (not root)
- **Minimal base image**: Based on `python:3.9-slim`
- **Health checks**: Built-in health check endpoint
- **Read-only credentials**: Cloud credentials mounted read-only

## 📊 Performance

- **Image size**: ~200MB compressed
- **Startup time**: <5 seconds
- **Memory usage**: ~50MB idle
- **CPU usage**: Minimal for CLI operations

## 🔄 Updates

### Automatic Updates
```bash
# Pull latest version
docker pull theoddden/terradev:latest

# Update running container
docker-compose pull && docker-compose up -d
```

### Version Pinning
```bash
# Pin to specific version
docker pull theoddden/terradev:2.9.2

# Use in production
docker run --rm theoddden/terradev:2.9.2 python -m terradev_cli --version
```

## 🐛 Troubleshooting

### Common Issues

1. **Permission denied**
   ```bash
   # Fix: Ensure proper volume permissions
   docker run --rm \
     -v ~/.aws:/home/terradev/.aws:ro \
     theoddden/terradev:2.9.2 \
     python -m terradev_cli status
   ```

2. **Network issues**
   ```bash
   # Fix: Check network connectivity
   docker run --rm theoddden/terradev:2.9.2 curl -I https://api.terradev.cloud
   ```

3. **Memory issues**
   ```bash
   # Fix: Increase memory limit
   docker run --rm --memory=512m theoddden/terradev:2.9.2 python -m terradev_cli status
   ```

### Debug Mode
```bash
# Run with debug output
docker run --rm \
  -e DEBUG=1 \
  theoddden/terradev:2.9.2 \
  python -m terradev_cli --verbose status
```

## 🤝 Contributing

To build the image locally:
```bash
# Build from source
docker build -f Dockerfile.hub -t theoddden/terradev:local .

# Test locally built image
docker run --rm theoddden/terradev:local python -m terradev_cli --version
```

## 📄 License

This Docker image is licensed under the Business Source License 1.1 (BUSL-1.1).

## 🔗 Links

- **Docker Hub**: https://hub.docker.com/r/theoddden/terradev
- **GitHub Repository**: https://github.com/theoddden/terradev
- **PyPI Package**: https://pypi.org/project/terradev-cli/
- **Documentation**: https://github.com/theoddden/terradev/blob/main/README.md
