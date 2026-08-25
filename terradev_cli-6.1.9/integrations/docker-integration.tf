# Docker Integration for Terradev
# Essential for containerized ML workflows

resource "local_file" "docker_integration" {
  filename = "${path.module}/docker/Dockerfile.terradev"
  content = <<-EOT
  # Terradev GPU-Optimized Docker Image
  # Automatically optimizes GPU costs for containerized workloads
  
  FROM python:3.9-slim
  
  # Install system dependencies
  RUN apt-get update && apt-get install -y \
      curl \
      jq \
      git \
      && rm -rf /var/lib/apt/lists/*
  
  # Install Terradev CLI
  RUN curl -fsSL https://api.terradev.io/install.sh | sh
  
  # Install Python ML dependencies
  COPY requirements.txt /tmp/requirements.txt
  RUN pip install --no-cache-dir -r /tmp/requirements.txt
  
  # Install Terradev Python SDK
  RUN pip install terradev-sdk
  
  # Create app directory
  WORKDIR /app
  
  # Copy application code
  COPY . /app/
  
  # Environment variables
  ENV TERRADEV_API_ENDPOINT="https://api.terradev.io/v1"
  ENV PYTHONPATH="/app"
  
  # Health check
  HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
      CMD terradev version || exit 1
  
  # Default command
  CMD ["python", "main.py"]
  EOT
}

# Docker Compose Integration
resource "local_file" "docker_compose" {
  filename = "${path.module}/docker/docker-compose.yml"
  content = yamlencode({
    version = "3.8"
    
    services = {
      terradev_optimizer = {
        build = {
          context = "."
          dockerfile = "Dockerfile.terradev"
        }
        environment = [
          "TERRADEV_API_KEY=${var.terradev_api_key}",
          "GPU_TYPE=a100",
          "HOURS_NEEDED=8",
          "CONFIDENCE_THRESHOLD=0.8"
        ]
        volumes = [
          "./data:/app/data",
          "./models:/app/models",
          "./logs:/app/logs"
        ]
        command = ["python", "optimized_training.py"]
      }
      
      mlflow_server = {
        image = "python:3.9-slim"
        command = [
          "sh", "-c",
          "pip install mlflow && mlflow server --host 0.0.0.0 --port 5000"
        ]
        ports = ["5000:5000"]
        volumes = ["./mlruns:/mlruns"]
      }
      
      jupyter_lab = {
        image = "jupyter/scipy-notebook:latest"
        ports = ["8888:8888"]
        environment = [
          "JUPYTER_ENABLE_LAB=yes"
        ]
        volumes = [
          "./notebooks:/home/jovyan/work",
          "./data:/home/jovyan/data"
        ]
        command = [
          "start-notebook.sh",
          "--NotebookApp.token=''",
          "--NotebookApp.password=''"
        ]
      }
    }
    
    volumes = {
      mlruns = {}
      data = {}
      models = {}
      logs = {}
      notebooks = {}
    }
  })
}

# Docker SDK Integration
resource "local_file" "docker_sdk" {
  filename = "${path.module}/docker/terradev_docker_sdk.py"
  content = <<-EOT
  """
  Terradev Docker SDK
  Integrates GPU optimization with Docker containers
  """
  
  import docker
  import json
  import os
  import requests
  from typing import Dict, List, Optional, Any
  from dataclasses import dataclass
  import logging
  
  logger = logging.getLogger(__name__)
  
  @dataclass
  class GPUOptimizedContainer:
      """Container configuration with GPU optimization"""
      image: str
      gpu_type: str
      hours_needed: int
      confidence_threshold: float = 0.7
      max_risk_score: float = 0.5
      environment: Dict[str, str] = None
      volumes: Dict[str, str] = None
      command: List[str] = None
      auto_deploy: bool = True
  
  class TerradevDockerManager:
      """Manages Docker containers with Terradev GPU optimization"""
      
      def __init__(self, terradev_api_key: str, api_endpoint: str = "https://api.terradev.io/v1"):
          self.api_key = terradev_api_key
          self.api_endpoint = api_endpoint
          self.docker_client = docker.from_env()
          
      def find_optimal_gpu(self, gpu_type: str, hours: int, confidence_threshold: float = 0.7) -> Dict[str, Any]:
          """Find optimal GPU configuration"""
          try:
              response = requests.post(
                  f"{self.api_endpoint}/find-optimal-gpu",
                  headers={"Authorization": f"Bearer {self.api_key}"},
                  json={
                      "gpu_type": gpu_type,
                      "hours": hours,
                      "confidence_threshold": confidence_threshold
                  }
              )
              
              if response.status_code == 200:
                  return response.json()
              else:
                  raise Exception(f"Terradev API error: {response.status_code}")
                  
          except Exception as e:
              logger.error(f"Error finding optimal GPU: {e}")
              return None
      
      def deploy_gpu_container(self, container_config: GPUOptimizedContainer) -> Dict[str, Any]:
          """Deploy container with GPU optimization"""
          
          # Find optimal GPU if auto_deploy is enabled
          gpu_info = None
          if container_config.auto_deploy:
              gpu_info = self.find_optimal_gpu(
                  container_config.gpu_type,
                  container_config.hours_needed,
                  container_config.confidence_threshold
              )
              
              if not gpu_info or not gpu_info.get('should_deploy'):
                  raise Exception("GPU conditions not optimal for deployment")
          
          # Prepare container configuration
          container_kwargs = {
              "image": container_config.image,
              "environment": container_config.environment or {},
              "volumes": container_config.volumes or {},
              "detach": True,
              "remove": False
          }
          
          # Add GPU configuration
          if gpu_info:
              # Add GPU provider info to environment
              container_kwargs["environment"].update({
                  "TERRADEV_PROVIDER": gpu_info['provider'],
                  "TERRADEV_GPU_TYPE": gpu_info['gpu_type'],
                  "TERRADEV_GPU_PRICE": str(gpu_info['price']),
                  "TERRADEV_DEPLOYMENT_ID": gpu_info.get('deployment_id', ''),
                  "CUDA_VISIBLE_DEVICES": "0"
              })
              
              # Configure GPU runtime
              container_kwargs["runtime"] = "nvidia"
              container_kwargs["device_requests"] = [
                  docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])
              ]
          
          # Add command if specified
          if container_config.command:
              container_kwargs["command"] = container_config.command
          
          try:
              # Deploy container
              container = self.docker_client.containers.run(**container_kwargs)
              
              # Log deployment
              logger.info(f"Deployed container {container.id[:12]} with GPU optimization")
              
              if gpu_info:
                  logger.info(f"GPU: {gpu_info['provider']} {gpu_info['gpu_type']} at ${gpu_info['price']}/hour")
              
              return {
                  "container_id": container.id,
                  "status": "running",
                  "gpu_info": gpu_info,
                  "container": container
              }
              
          except Exception as e:
              logger.error(f"Error deploying container: {e}")
              raise
      
      def create_training_container(self, 
                                  image: str,
                                  training_script: str,
                                  gpu_type: str = "a100",
                                  hours: int = 4,
                                  dataset_path: str = "/data",
                                  model_path: str = "/models") -> Dict[str, Any]:
          """Create optimized training container"""
          
          container_config = GPUOptimizedContainer(
              image=image,
              gpu_type=gpu_type,
              hours_needed=hours,
              environment={
                  "TRAINING_SCRIPT": training_script,
                  "DATASET_PATH": dataset_path,
                  "MODEL_PATH": model_path,
                  "PYTHONPATH": "/app"
              },
              volumes={
                  os.path.abspath(dataset_path): {"bind": dataset_path, "mode": "ro"},
                  os.path.abspath(model_path): {"bind": model_path, "mode": "rw"}
              },
              command=["python", training_script],
              auto_deploy=True
          )
          
          return self.deploy_gpu_container(container_config)
      
      def create_jupyter_container(self,
                                notebook_dir: str = "./notebooks",
                                gpu_type: str = "a10g",
                                hours: int = 2) -> Dict[str, Any]:
          """Create GPU-optimized Jupyter container"""
          
          container_config = GPUOptimizedContainer(
              image="jupyter/tensorflow-notebook:latest",
              gpu_type=gpu_type,
              hours_needed=hours,
              environment={
                  "JUPYTER_ENABLE_LAB": "yes",
                  "JUPYTER_TOKEN": ""
              },
              volumes={
                  os.path.abspath(notebook_dir): {"bind": "/home/jovyan/work", "mode": "rw"}
              },
              ports={"8888/tcp": None},
              auto_deploy=True
          )
          
          return self.deploy_gpu_container(container_config)
      
      def list_optimized_containers(self) -> List[Dict[str, Any]]:
          """List all Terradev-optimized containers"""
          
          optimized_containers = []
          
          for container in self.docker_client.containers.list(all=True):
              # Check if container has Terradev environment variables
              env_vars = container.attrs.get('Config', {}).get('Env', [])
              is_terradev = any('TERRADEV_' in env for env in env_vars)
              
              if is_terradev:
                  container_info = {
                      "id": container.id[:12],
                      "name": container.name,
                      "status": container.status,
                      "image": container.image.tags[0] if container.image.tags else "unknown",
                      "gpu_provider": None,
                      "gpu_type": None,
                      "gpu_price": None
                  }
                  
                  # Extract GPU info from environment
                  for env in env_vars:
                      if env.startswith('TERRADEV_PROVIDER='):
                          container_info["gpu_provider"] = env.split('=', 1)[1]
                      elif env.startswith('TERRADEV_GPU_TYPE='):
                          container_info["gpu_type"] = env.split('=', 1)[1]
                      elif env.startswith('TERRADEV_GPU_PRICE='):
                          container_info["gpu_price"] = env.split('=', 1)[1]
                  
                  optimized_containers.append(container_info)
          
          return optimized_containers
      
      def stop_container(self, container_id: str, force: bool = False) -> bool:
          """Stop a Terradev-optimized container"""
          
          try:
              container = self.docker_client.containers.get(container_id)
              
              if force:
                  container.force()
              else:
                  container.stop()
              
              logger.info(f"Stopped container {container_id[:12]}")
              return True
              
          except Exception as e:
              logger.error(f"Error stopping container {container_id[:12]}: {e}")
              return False
      
      def get_container_metrics(self, container_id: str) -> Dict[str, Any]:
          """Get metrics for a running container"""
          
          try:
              container = self.docker_client.containers.get(container_id)
              
              # Get container stats
              stats = container.stats(stream=False)
              
              # Calculate cost metrics
              env_vars = container.attrs.get('Config', {}).get('Env', [])
              gpu_price = 0.0
              for env in env_vars:
                  if env.startswith('TERRADEV_GPU_PRICE='):
                      gpu_price = float(env.split('=', 1)[1])
                      break
              
              # Calculate runtime cost
              runtime_seconds = stats.get('read', 0) / 1000000000  # Convert nanoseconds to seconds
              runtime_hours = runtime_seconds / 3600
              current_cost = runtime_hours * gpu_price
              
              return {
                  "container_id": container_id[:12],
                  "status": container.status,
                  "cpu_usage": stats.get('cpu_stats', {}).get('cpu_usage', {}).get('total_usage', 0),
                  "memory_usage": stats.get('memory_stats', {}).get('usage', 0),
                  "gpu_price_per_hour": gpu_price,
                  "runtime_hours": runtime_hours,
                  "current_cost": current_cost
              }
              
          except Exception as e:
              logger.error(f"Error getting container metrics: {e}")
              return {}
  
  # Example usage
  def main():
      """Example usage of Terradev Docker SDK"""
      
      # Initialize manager
      manager = TerradevDockerManager(api_key="your-api-key")
      
      # Create training container
      print("Creating GPU-optimized training container...")
      training_result = manager.create_training_container(
          image="pytorch/pytorch:latest",
          training_script="train_model.py",
          gpu_type="a100",
          hours=8
      )
      
      print(f"Training container deployed: {training_result['container_id']}")
      
      # Create Jupyter container
      print("Creating GPU-optimized Jupyter container...")
      jupyter_result = manager.create_jupyter_container(
          notebook_dir="./notebooks",
          gpu_type="a10g",
          hours=2
      )
      
      print(f"Jupyter container deployed: {jupyter_result['container_id']}")
      
      # List all optimized containers
      print("\nOptimized containers:")
      for container in manager.list_optimized_containers():
          print(f"  {container['id']}: {container['gpu_type']} on {container['gpu_provider']} (${container['gpu_price']}/hour)")
      
      # Get container metrics
      if training_result['container_id']:
          metrics = manager.get_container_metrics(training_result['container_id'])
          print(f"\nTraining container metrics:")
          print(f"  Runtime: {metrics['runtime_hours']:.2f} hours")
          print(f"  Current cost: ${metrics['current_cost']:.4f}")
  
  if __name__ == "__main__":
      main()
  EOT
}
