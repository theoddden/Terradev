# Terradev CLI - Jupyter Integration

## 📓 Jupyter Notebook Support

Terradev CLI integrates seamlessly with Jupyter notebooks and JupyterLab, making it perfect for interactive GPU cost analysis and ML workflows.

## 🚀 Installation in Jupyter

### PyPI Installation (Recommended)
```bash
pip install terradev-cli[jupyter]==2.9.2
```

### Direct Installation
```bash
pip install terradev-cli==2.9.2
pip install jupyterlab matplotlib pandas seaborn
```

## 📊 Example Notebooks

### 1. GPU Cost Analysis Notebook
```python
# Install in notebook
!pip install terradev-cli==2.9.2 -q

# Import libraries
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to get GPU pricing
def get_gpu_quote(gpu_type):
    result = subprocess.run(
        ['python', '-m', 'terradev_cli', 'quote', '-g', gpu_type],
        capture_output=True, text=True
    )
    return result.stdout

# Compare GPU prices
gpu_types = ['A100', 'H100', 'V100', 'RTX4090']
for gpu in gpu_types:
    print(f"\n=== {gpu} Pricing ===")
    print(get_gpu_quote(gpu))
```

### 2. Cost Optimization Dashboard
```python
# Create interactive dashboard
import ipywidgets as widgets
from IPython.display import display, clear_output

# GPU selector
gpu_selector = widgets.Dropdown(
    options=['A100', 'H100', 'V100', 'RTX4090', 'L40', 'A6000'],
    value='A100',
    description='GPU Type:'
)

# Duration slider
duration_slider = widgets.IntSlider(
    value=4,
    min=1,
    max=24,
    description='Duration (hours):'
)

# Analysis button
analyze_btn = widgets.Button(description='Analyze Costs')

# Output area
output = widgets.Output()

def analyze_costs(b):
    with output:
        clear_output()
        gpu = gpu_selector.value
        duration = duration_slider.value
        
        print(f"🔍 Analyzing {gpu} for {duration} hours...")
        
        # Get pricing data
        quote = get_gpu_quote(gpu)
        print(quote)
        
        # Parse and visualize (implementation depends on output format)
        # Add your parsing logic here

analyze_btn.on_click(analyze_costs)

# Display widgets
display(gpu_selector, duration_slider, analyze_btn, output)
```

### 3. Multi-Cloud Comparison
```python
# Multi-cloud cost comparison
def compare_providers(gpu_types):
    results = {}
    
    for gpu in gpu_types:
        quote = get_gpu_quote(gpu)
        # Parse quote to extract provider prices
        # This depends on the actual output format
        results[gpu] = parse_quote(quote)  # Implement parsing function
    
    # Create DataFrame
    df = pd.DataFrame(results).T
    return df

# Usage
gpu_types = ['A100', 'H100', 'V100']
comparison_df = compare_providers(gpu_types)
print("Multi-Cloud Pricing Comparison:")
display(comparison_df)

# Visualization
plt.figure(figsize=(12, 6))
comparison_df.plot(kind='bar', figsize=(12, 6))
plt.title('GPU Pricing Comparison Across Providers')
plt.ylabel('Price ($/hour)')
plt.xticks(rotation=45)
plt.legend(title='Cloud Provider')
plt.tight_layout()
plt.show()
```

## 🔧 JupyterLab Extension

### Terradev CLI Extension
Create a custom JupyterLab extension for Terradev CLI:

```python
# terradev_jupyterlab/extension.py
from jupyter_server.extension.application import ExtensionApp
from jupyter_server.extension.handler import ExtensionHandler
import subprocess
import json

class TerradevHandler(ExtensionHandler):
    def get(self):
        gpu_type = self.get_query_argument('gpu', 'A100')
        
        try:
            result = subprocess.run(
                ['python', '-m', 'terradev_cli', 'quote', '-g', gpu_type],
                capture_output=True, text=True
            )
            
            response = {
                'gpu_type': gpu_type,
                'quote': result.stdout,
                'success': True
            }
        except Exception as e:
            response = {
                'gpu_type': gpu_type,
                'error': str(e),
                'success': False
            }
        
        self.finish(json.dumps(response))

class TerradevApp(ExtensionApp):
    name = "terradev"
    
    def initialize_settings(self):
        self.handlers.append(
            (r"/quote", TerradevHandler)
        )
```

## 📚 JupyterHub Integration

### JupyterHub Configuration
```python
# jupyterhub_config.py
c.Spawner.cmd = ['jupyter-labhub']
c.Spawner.environment.update({
    'TERRADEV_API_URL': 'https://api.terradev.cloud',
    'TERRADEV_FALLBACK_URL': 'http://34.207.59.52:8080'
})

# Pre-install Terradev CLI in user environments
c.Spawner.pre_spawn_hook = function(spawner):
    subprocess.run(['pip', 'install', 'terradev-cli==2.9.2'])
```

### Docker-based JupyterHub
```dockerfile
# Dockerfile for JupyterHub with Terradev
FROM jupyterhub/jupyterhub:latest

# Install Terradev CLI
RUN pip install terradev-cli[jupyter]==2.9.2

# Copy custom configuration
COPY jupyterhub_config.py /etc/jupyterhub/

# Install additional notebook dependencies
RUN pip install matplotlib pandas seaborn plotly ipywidgets

EXPOSE 8000
CMD ["jupyterhub"]
```

## 🎯 Use Cases

### 1. Research & Development
```python
# Research notebook for GPU cost optimization
import terradev_cli

# Analyze different GPU configurations
configs = [
    {'gpu': 'A100', 'memory': '40GB', 'provider': 'aws'},
    {'gpu': 'H100', 'memory': '80GB', 'provider': 'gcp'},
    {'gpu': 'V100', 'memory': '32GB', 'provider': 'azure'}
]

for config in configs:
    cost = terradev_cli.get_cost_estimate(config)
    print(f"{config['gpu']} on {config['provider']}: ${cost}/hour")
```

### 2. ML Pipeline Integration
```python
# ML pipeline with cost optimization
def train_model_with_cost_optimization(model_config):
    # Get optimal GPU for this workload
    optimal_gpu = terradev_cli.find_optimal_gpu(
        memory_requirement=model_config['memory_gb'],
        compute_requirement=model_config['flops']
    )
    
    print(f"Optimal GPU: {optimal_gpu['type']} at ${optimal_gpu['price']}/hr")
    
    # Provision and train
    deployment = terradev_cli.provision_gpu(
        gpu_type=optimal_gpu['type'],
        duration=model_config['training_hours']
    )
    
    return deployment
```

### 3. Interactive Cost Dashboard
```python
# Interactive dashboard using Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_cost_dashboard():
    # Get pricing data
    gpu_types = ['A100', 'H100', 'V100', 'RTX4090']
    pricing_data = {}
    
    for gpu in gpu_types:
        pricing_data[gpu] = terradev_cli.get_pricing(gpu)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Comparison', 'Cost per TFLOP', 'Memory Efficiency', 'Provider Distribution')
    )
    
    # Add traces
    fig.add_trace(go.Bar(x=gpu_types, y=[pricing_data[g]['min_price'] for g in gpu_types]), row=1, col=1)
    
    fig.update_layout(title_text="GPU Cost Analysis Dashboard")
    fig.show()

create_cost_dashboard()
```

## 📦 Package Structure

### Jupyter Framework Classifiers
The package includes these PyPI classifiers for Jupyter discovery:
- `Framework :: Jupyter`
- `Framework :: Jupyter :: JupyterLab`
- `Topic :: Scientific/Engineering :: Artificial Intelligence`

### Installation Options
```bash
# Basic installation
pip install terradev-cli

# With Jupyter support
pip install terradev-cli[jupyter]

# Development installation
pip install terradev-cli[jupyter,dev]
```

## 🔄 Auto-completion in Jupyter

### IPython Magic Commands
```python
# Custom magic commands for Terradev CLI
from IPython.core.magic import register_line_magic

@register_line_magic
def terradev(line):
    """Terradev CLI magic command"""
    args = line.split()
    cmd = ['python', '-m', 'terradev_cli'] + args
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    return result.stdout

# Usage in notebook
%terradev quote -g A100
%terradev status
%terradev provision -g H100 --duration 4
```

## 📊 Performance Tips

### 1. Caching Results
```python
# Cache pricing data to avoid repeated API calls
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_gpu_quote(gpu_type):
    return get_gpu_quote(gpu_type)
```

### 2. Async Operations
```python
# Async pricing for multiple GPUs
import asyncio
import concurrent.futures

async def get_multiple_quotes(gpu_types):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, get_gpu_quote, gpu)
            for gpu in gpu_types
        ]
        results = await asyncio.gather(*tasks)
    
    return dict(zip(gpu_types, results))
```

## 🤝 Community Integration

### JupyterHub Community Forum
Post your notebooks and integrations in the JupyterHub community forum:

1. **Show and Tell Section**: Share your cost analysis notebooks
2. **Help & Support**: Get help with integration issues
3. **Feature Requests**: Suggest new Jupyter-specific features

### Contributing
```bash
# Contribute to Jupyter integration
git clone https://github.com/theoddden/terradev.git
cd terradev

# Create new notebook
jupyter lab notebooks/

# Test integration
python -m terradev_cli --version
```

## 📄 License

Terradev CLI is licensed under the Business Source License 1.1 (BUSL-1.1), free for evaluation, testing, and internal business use.

## 🔗 Links

- **PyPI**: https://pypi.org/project/terradev-cli/
- **GitHub**: https://github.com/theoddden/terradev
- **JupyterHub Forum**: https://discourse.jupyter.org/
- **Documentation**: https://github.com/theoddden/terradev/blob/main/README.md
