# Terradev CLI Docker Image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TERRADEV_CLI_VERSION=2.9.2

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Terradev CLI
COPY . .
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash terradev
USER terradev
WORKDIR /home/terradev

# Copy CLI to user home
COPY --chown=terradev:terradev /app /home/terradev/terradev-cli
RUN pip install -e /home/terradev/terradev-cli

# Add CLI to PATH
ENV PATH="/home/terradev/terradev-cli:$PATH"

# Expose common ports
EXPOSE 8000 8080 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m terradev_cli --version || exit 1

# Default command
CMD ["python", "-m", "terradev_cli", "--help"]

# Labels for Docker Hub
LABEL maintainer="theoddden" \
      version="2.9.2" \
      description="Terradev CLI - Cross-cloud GPU optimization platform" \
      org.opencontainers.image.source="https://github.com/theoddden/terradev" \
      org.opencontainers.image.licenses="BUSL-1.1"
