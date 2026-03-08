# MLflow Integration for ML Teams
# Essential for experiment tracking and cost optimization

resource "local_file" "mlflow_integration" {
  filename = "${path.module}/mlflow/terradev_plugin.py"
  content = <<-EOT
  """
  Terradev MLflow Plugin
  Tracks GPU costs and optimizations in MLflow experiments
  """
  
  import mlflow
  import mlflow.entities
  from mlflow.tracking.context import Context
  import json
  import requests
  from datetime import datetime
  from typing import Dict, Any, Optional
  
  class TerradevContext(Context):
      """MLflow context for Terradev GPU arbitrage"""
      
      def __init__(self, terradev_api_key: str, api_endpoint: str = "https://api.terradev.io/v1"):
          self.api_key = terradev_api_key
          self.api_endpoint = api_endpoint
          self.gpu_info = None
          
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
                  self.gpu_info = response.json()
                  return self.gpu_info
              else:
                  raise Exception(f"Terradev API error: {response.status_code}")
                  
          except Exception as e:
              print(f"Error finding optimal GPU: {e}")
              return None
          
      def deploy_gpu(self, gpu_type: str, hours: int, experiment_id: str = None) -> Dict[str, Any]:
          """Deploy GPU and track in MLflow"""
          # Find optimal GPU first
          gpu_info = self.find_optimal_gpu(gpu_type, hours)
          
          if not gpu_info or not gpu_info.get('should_deploy'):
              raise Exception("No suitable GPU found or conditions not optimal")
          
          # Deploy GPU
          try:
              response = requests.post(
                  f"{self.api_endpoint}/deploy",
                  headers={"Authorization": f"Bearer {self.api_key}"},
                  json={
                      "provider": gpu_info['provider'],
                      "gpu_type": gpu_info['gpu_type'],
                      "hours": hours,
                      "experiment_id": experiment_id
                  }
              )
              
              if response.status_code == 200:
                  deployment = response.json()
                  
                  # Log to MLflow
                  with mlflow.start_run(nested=True) as run:
                      mlflow.log_params({
                          "gpu_provider": gpu_info['provider'],
                          "gpu_type": gpu_info['gpu_type'],
                          "gpu_price_per_hour": gpu_info['price'],
                          "gpu_hours_requested": hours,
                          "estimated_total_cost": gpu_info['price'] * hours,
                          "arbitrage_confidence": gpu_info.get('arbitrage_confidence', 0),
                          "risk_score": gpu_info.get('risk_score', 0),
                          "market_efficiency": gpu_info.get('market_efficiency', 0),
                          "deployment_id": deployment.get('deployment_id')
                      })
                      
                      mlflow.set_tag("terradev_deployment", "true")
                      mlflow.set_tag("gpu_optimization", "true")
                      
                  return deployment
              else:
                  raise Exception(f"Deployment failed: {response.status_code}")
                  
          except Exception as e:
              print(f"Error deploying GPU: {e}")
              raise
          
      def log_training_metrics(self, metrics: Dict[str, float], cost_metrics: Dict[str, float] = None):
          """Log training metrics with cost tracking"""
          with mlflow.start_run(nested=True) as run:
              # Log training metrics
              for metric_name, value in metrics.items():
                  mlflow.log_metric(metric_name, value)
              
              # Log cost metrics if available
              if cost_metrics:
                  for metric_name, value in cost_metrics.items():
                      mlflow.log_metric(f"cost_{metric_name}", value)
              
              # Calculate and log cost efficiency
              if 'accuracy' in metrics and cost_metrics:
                  cost_per_accuracy = cost_metrics.get('total_cost', 0) / metrics['accuracy']
                  mlflow.log_metric("cost_per_accuracy", cost_per_accuracy)
  
  class TerradevMLflowTracker(mlflow.tracking.TrackingClient):
      """Enhanced MLflow tracker with Terradev cost tracking"""
      
      def __init__(self, tracking_uri: str, terradev_context: TerradevContext):
          super().__init__(tracking_uri)
          self.terradev = terradev_context
      
      def log_run_start(self, run_id, experiment_id, user_id, start_time, tags, run_name):
          """Override to log GPU deployment at run start"""
          super().log_run_start(run_id, experiment_id, user_id, start_time, tags, run_name)
          
          # Check if this is a GPU training run
          if tags and tags.get('gpu_type'):
              gpu_type = tags.get('gpu_type')
              hours = int(tags.get('gpu_hours', 4))
              
              try:
                  deployment = self.terradev.deploy_gpu(gpu_type, hours, experiment_id)
                  
                  # Add deployment info to tags
                  enhanced_tags = tags.copy()
                  enhanced_tags.update({
                      'terradev_deployment_id': deployment.get('deployment_id'),
                      'terradev_provider': deployment.get('provider'),
                      'terradev_gpu_price': str(deployment.get('price_per_hour', 0))
                  })
                  
                  # Update run tags
                  self.set_tags(run_id, enhanced_tags)
                  
              except Exception as e:
                  print(f"Failed to deploy GPU via Terradev: {e}")
  
  def setup_terradev_mlflow(api_key: str, tracking_uri: str = None):
      """Setup MLflow with Terradev integration"""
      terradev_ctx = TerradevContext(api_key)
      
      # Set tracking URI if provided
      if tracking_uri:
          mlflow.set_tracking_uri(tracking_uri)
      
      # Register Terradev context
      mlflow.get_context_registry().register("terradev", terradev_ctx)
      
      return terradev_ctx
  
  # Example usage decorator
  def with_terradev_gpu(gpu_type: str, hours: int = 4, confidence_threshold: float = 0.7):
      """Decorator to automatically optimize GPU for MLflow runs"""
      def decorator(func):
          def wrapper(*args, **kwargs):
              # Get Terradev context
              terradev_ctx = mlflow.get_context_registry().get("terradev")
              
              if not terradev_ctx:
                  raise Exception("Terradev context not found. Call setup_terradev_mlflow() first.")
              
              # Find optimal GPU
              gpu_info = terradev_ctx.find_optimal_gpu(gpu_type, hours, confidence_threshold)
              
              if not gpu_info or not gpu_info.get('should_deploy'):
                  print(f"GPU conditions not optimal for {gpu_type}. Skipping deployment.")
                  return None
              
              # Deploy GPU
              deployment = terradev_ctx.deploy_gpu(gpu_type, hours)
              
              print(f"Deployed {gpu_type} on {gpu_info['provider']} at ${gpu_info['price']}/hour")
              
              # Run the function with GPU context
              with mlflow.start_run() as run:
                  mlflow.log_params({
                      "gpu_provider": gpu_info['provider'],
                      "gpu_type": gpu_info['gpu_type'],
                      "gpu_price_per_hour": gpu_info['price'],
                      "deployment_id": deployment.get('deployment_id')
                  })
                  
                  # Run the original function
                  result = func(*args, **kwargs)
                  
                  # Log completion
                  mlflow.set_tag("status", "completed")
                  
              return result
          
          return wrapper
      return decorator
  EOT
}

# MLflow Example Usage
resource "local_file" "mlflow_example" {
  filename = "${path.module}/mlflow/examples/training_with_terradev.py"
  content = <<-EOT
  """
  Example: ML training with Terradev GPU optimization via MLflow
  """
  
  import mlflow
  import mlflow.pytorch
  import torch
  import torch.nn as nn
  from transformers import LlamaForCausalLM, LlamaTokenizer
  from terradev_plugin import setup_terradev_mlflow, with_terradev_gpu
  
  # Setup MLflow with Terradev
  terradev_ctx = setup_terradev_mlflow(
      api_key="your-terradev-api-key",
      tracking_uri="http://localhost:5000"
  )
  
  # Example 1: Automatic GPU optimization with decorator
  @with_terradev_gpu(gpu_type="a100", hours=8, confidence_threshold=0.8)
  def train_llama_model():
      """Train LLaMA model with automatic GPU optimization"""
      
      # Load model
      model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
      tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
      
      # Training configuration
      training_config = {
          "batch_size": 32,
          "learning_rate": 1e-4,
          "epochs": 10,
          "model_name": "llama-2-7b-finetuned"
      }
      
      mlflow.log_params(training_config)
      
      # Simulate training
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model = model.to(device)
      
      # Training loop
      for epoch in range(training_config['epochs']):
          # Simulate training step
          loss = torch.tensor(2.0 - epoch * 0.1)  # Simulated decreasing loss
          accuracy = 0.5 + epoch * 0.05  # Simulated increasing accuracy
          
          mlflow.log_metrics({
              "loss": loss.item(),
              "accuracy": accuracy,
              "epoch": epoch
          }, step=epoch)
          
          print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")
      
      # Log model
      mlflow.pytorch.log_model(model, "model")
      
      return {
          "final_loss": loss.item(),
          "final_accuracy": accuracy,
          "model_path": "model"
      }
  
  # Example 2: Manual GPU optimization
  def manual_gpu_training():
      """Manual GPU optimization with detailed tracking"""
      
      with mlflow.start_run(run_name="manual-gpu-optimization") as run:
          # Set tags for Terradev
          mlflow.set_tags({
              "gpu_type": "h100",
              "gpu_hours": "6",
              "model": "bert-base",
              "dataset": "squad"
          })
          
          # Find optimal GPU
          gpu_info = terradev_ctx.find_optimal_gpu("h100", 6, confidence_threshold=0.7)
          
          if gpu_info and gpu_info.get('should_deploy'):
              print(f"Found optimal GPU: {gpu_info['provider']} {gpu_info['gpu_type']} at ${gpu_info['price']}/hour")
              
              # Deploy GPU
              deployment = terradev_ctx.deploy_gpu("h100", 6, run.info.experiment_id)
              
              # Log deployment info
              mlflow.log_params({
                  "gpu_provider": gpu_info['provider'],
                  "gpu_price_per_hour": gpu_info['price'],
                  "arbitrage_confidence": gpu_info.get('arbitrage_confidence', 0),
                  "risk_score": gpu_info.get('risk_score', 0),
                  "deployment_id": deployment.get('deployment_id')
              })
              
              # Simulate BERT training
              model = torch.nn.Sequential(
                  torch.nn.Linear(768, 256),
                  torch.nn.ReLU(),
                  torch.nn.Linear(256, 2)
              )
              
              optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
              criterion = torch.nn.CrossEntropyLoss()
              
              # Training metrics
              total_cost = 0
              for epoch in range(5):
                  # Simulate training
                  loss = torch.randn(1).item()
                  accuracy = torch.rand(1).item()
                  
                  mlflow.log_metrics({
                      "loss": abs(loss),
                      "accuracy": accuracy,
                      "epoch": epoch
                  }, step=epoch)
                  
                  # Track cost
                  epoch_cost = gpu_info['price'] * 1.2  # 1.2 hours per epoch
                  total_cost += epoch_cost
                  mlflow.log_metric("cumulative_cost", total_cost, step=epoch)
              
              # Log final cost metrics
              mlflow.log_metrics({
                  "total_training_cost": total_cost,
                  "cost_per_epoch": total_cost / 5,
                  "estimated_savings": gpu_info.get('savings_percentage', 0)
              })
              
              # Log model
              mlflow.pytorch.log_model(model, "bert_model")
              
              return {
                  "total_cost": total_cost,
                  "final_accuracy": accuracy,
                  "deployment_id": deployment.get('deployment_id')
              }
          else:
              print("GPU conditions not optimal for training")
              return None
  
  if __name__ == "__main__":
      print("Starting ML training with Terradev GPU optimization...")
      
      # Run automatic GPU optimization
      print("\n=== Automatic GPU Optimization ===")
      result1 = train_llama_model()
      
      # Run manual GPU optimization
      print("\n=== Manual GPU Optimization ===")
      result2 = manual_gpu_training()
      
      print("\nTraining completed!")
  EOT
}
