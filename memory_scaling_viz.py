#!/usr/bin/env python3
"""
Visualization 4: Memory Utilization and Scaling Analysis
Focused on GPU memory usage patterns and scaling behavior in agentic AI workloads
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("plasma")

@dataclass
class ThroughputScenario:
    """Represents different agentic AI workload scenarios"""
    name: str
    avg_prompt_tokens: int
    avg_response_tokens: int
    requests_per_second: float
    concurrent_users: int
    latency_sensitivity: float
    memory_pressure: float
    gpu_count: int
    model_size_gb: float

class MemoryScalingViz:
    """Visualization focused on memory utilization and scaling analysis"""
    
    def __init__(self):
        self.scenarios = self._generate_scenarios()
        self.configurations = self._get_configurations()
        self.gpu_configs = self._get_gpu_configs()
        
    def _generate_scenarios(self) -> List[ThroughputScenario]:
        """Generate realistic agentic AI workload scenarios"""
        return [
            ThroughputScenario("Code Generation Agent", 850, 420, 15, 50, 0.7, 0.6, 2, 14.0),
            ThroughputScenario("Research Assistant", 1200, 680, 8, 25, 0.4, 0.3, 4, 28.0),
            ThroughputScenario("Customer Service Bot", 320, 180, 45, 200, 0.9, 0.8, 1, 7.0),
            ThroughputScenario("Document Analysis Agent", 2500, 1200, 3, 10, 0.2, 0.4, 8, 56.0),
            ThroughputScenario("Multi-Modal Agent", 600, 350, 25, 80, 0.6, 0.7, 2, 16.0)
        ]
    
    def _get_configurations(self) -> Dict[str, Dict]:
        """Define different memory configurations"""
        return {
            "Default": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": False
            },
            "Memory Optimized": {
                "max_num_batched_tokens": 8192,
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 512,
                "enable_prefix_caching": True,
                "enable_kv_cache_offloading": True,
                "enable_speculative_decoding": True
            },
            "High Concurrency": {
                "max_num_batched_tokens": 4096,
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 1024,
                "enable_prefix_caching": True,
                "enable_kv_cache_offloading": True,
                "enable_speculative_decoding": False
            }
        }
    
    def _get_gpu_configs(self) -> List[Dict]:
        """Define different GPU configurations"""
        return [
            {"name": "A100 40GB", "memory_gb": 40, "bandwidth_gb_s": 1555},
            {"name": "A100 80GB", "memory_gb": 80, "bandwidth_gb_s": 2039},
            {"name": "H100 80GB", "memory_gb": 80, "bandwidth_gb_s": 3350},
            {"name": "RTX 4090", "memory_gb": 24, "bandwidth_gb_s": 1008},
            {"name": "RTX 3090", "memory_gb": 24, "bandwidth_gb_s": 936}
        ]
    
    def calculate_memory_metrics(self, scenario: ThroughputScenario, config: Dict, gpu_config: Dict) -> Dict:
        """Calculate memory and performance metrics"""
        total_tokens = scenario.avg_prompt_tokens + scenario.avg_response_tokens
        
        # Model memory (weights)
        model_memory = scenario.model_size_gb
        
        # KV cache memory calculation
        kv_cache_per_token = 2.0  # Approximate bytes per token for KV cache
        kv_cache_memory = (scenario.concurrent_users * total_tokens * kv_cache_per_token) / (1024**3)
        
        # Apply KV cache offloading if enabled
        if config["enable_kv_cache_offloading"]:
            kv_cache_memory *= 0.3  # 70% reduction with offloading
        
        # Prefix cache memory
        prefix_cache_memory = 0
        if config["enable_prefix_caching"]:
            prefix_cache_memory = (total_tokens * 0.5) / (1024**3)  # 50% of sequence length
        
        # Activation memory
        activation_memory = (scenario.concurrent_users * total_tokens * 0.5) / (1024**3)
        
        # Total memory usage
        total_memory = model_memory + kv_cache_memory + prefix_cache_memory + activation_memory
        
        # Memory utilization
        memory_utilization = total_memory / gpu_config["memory_gb"]
        
        # Performance scaling based on memory bandwidth
        bandwidth_factor = gpu_config["bandwidth_gb_s"] / 1555  # Normalize to A100 40GB
        
        # Calculate throughput with memory constraints
        if memory_utilization > config["gpu_memory_utilization"]:
            # Memory constrained - reduce throughput
            constraint_factor = config["gpu_memory_utilization"] / memory_utilization
        else:
            constraint_factor = 1.0
        
        # Base throughput calculation
        base_throughput = scenario.requests_per_second * total_tokens
        
        # Apply optimizations
        batch_efficiency = min(1.0, config["max_num_batched_tokens"] / total_tokens)
        concurrency_factor = min(1.0, config["max_num_seqs"] / scenario.concurrent_users)
        
        effective_throughput = (base_throughput * batch_efficiency * 
                              concurrency_factor * constraint_factor * bandwidth_factor)
        
        return {
            'total_memory_gb': total_memory,
            'model_memory_gb': model_memory,
            'kv_cache_memory_gb': kv_cache_memory,
            'prefix_cache_memory_gb': prefix_cache_memory,
            'activation_memory_gb': activation_memory,
            'memory_utilization': memory_utilization,
            'throughput': effective_throughput,
            'memory_constrained': memory_utilization > config["gpu_memory_utilization"]
        }
    
    def create_visualization(self):
        """Create focused memory utilization and scaling visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agentic AI Token Throughput - Memory Utilization & Scaling Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Memory usage breakdown
        ax1 = axes[0, 0]
        self._plot_memory_breakdown(ax1)
        
        # 2. Memory utilization across scenarios
        ax2 = axes[0, 1]
        self._plot_memory_utilization(ax2)
        
        # 3. GPU scaling analysis
        ax3 = axes[1, 0]
        self._plot_gpu_scaling(ax3)
        
        # 4. Memory optimization recommendations
        ax4 = axes[1, 1]
        self._plot_memory_recommendations(ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_memory_breakdown(self, ax):
        """Plot memory usage breakdown by component"""
        # Use A100 40GB as reference
        gpu_config = self.gpu_configs[0]
        config = self.configurations["Memory Optimized"]
        
        data = []
        for scenario in self.scenarios:
            metrics = self.calculate_memory_metrics(scenario, config, gpu_config)
            data.append({
                'Scenario': scenario.name,
                'Model Weights': metrics['model_memory_gb'],
                'KV Cache': metrics['kv_cache_memory_gb'],
                'Prefix Cache': metrics['prefix_cache_memory_gb'],
                'Activations': metrics['activation_memory_gb']
            })
        
        df = pd.DataFrame(data).set_index('Scenario')
        
        # Create stacked bar chart
        df.plot(kind='bar', stacked=True, ax=ax, colormap='plasma')
        
        ax.set_title('Memory Usage Breakdown (A100 40GB, Memory Optimized)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Memory Usage (GB)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Memory Component', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add GPU memory limit line
        ax.axhline(y=gpu_config['memory_gb'], color='red', linestyle='--', alpha=0.7, label='GPU Memory Limit')
    
    def _plot_memory_utilization(self, ax):
        """Plot memory utilization across scenarios and configurations"""
        gpu_config = self.gpu_configs[0]  # A100 40GB
        
        data = []
        for scenario in self.scenarios:
            for config_name, config in self.configurations.items():
                metrics = self.calculate_memory_metrics(scenario, config, gpu_config)
                data.append({
                    'Scenario': scenario.name,
                    'Configuration': config_name,
                    'Memory Utilization': metrics['memory_utilization'],
                    'Constrained': metrics['memory_constrained']
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        sns.barplot(data=df, x='Scenario', y='Memory Utilization', hue='Configuration', ax=ax)
        ax.set_title('Memory Utilization by Scenario and Configuration', fontweight='bold', fontsize=12)
        ax.set_ylabel('Memory Utilization (Fraction of GPU)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add 95% utilization threshold line
        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='{:.2f}', fontsize=8)
    
    def _plot_gpu_scaling(self, ax):
        """Plot GPU scaling analysis"""
        scenario = self.scenarios[2]  # Customer Service Bot (high concurrency)
        config = self.configurations["Memory Optimized"]
        
        data = []
        for gpu_config in self.gpu_configs:
            metrics = self.calculate_memory_metrics(scenario, config, gpu_config)
            data.append({
                'GPU': gpu_config["name"],
                'Memory (GB)': gpu_config["memory_gb"],
                'Bandwidth (GB/s)': gpu_config["bandwidth_gb_s"],
                'Throughput': metrics['throughput'],
                'Memory Utilization': metrics['memory_utilization'],
                'Constrained': metrics['memory_constrained']
            })
        
        df = pd.DataFrame(data)
        
        # Create dual-axis plot
        ax2 = ax.twinx()
        
        # Plot throughput on primary axis
        bars1 = ax.bar(df['GPU'], df['Throughput'], alpha=0.7, color='blue', label='Throughput')
        ax.set_ylabel('Throughput (tokens/sec)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Plot memory utilization on secondary axis
        bars2 = ax2.bar(df['GPU'], df['Memory Utilization'], alpha=0.7, color='red', label='Memory Utilization')
        ax2.set_ylabel('Memory Utilization', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title('GPU Scaling Analysis (Customer Service Bot)', fontweight='bold', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Mark constrained GPUs
        for i, (gpu, constrained) in enumerate(zip(df['GPU'], df['Constrained'])):
            if constrained:
                ax.text(i, df['Throughput'][i] + 1000, '⚠️', ha='center', fontsize=12)
    
    def _plot_memory_recommendations(self, ax):
        """Plot memory optimization recommendations"""
        ax.axis('off')
        
        recommendations = []
        for scenario in self.scenarios:
            # Test with A100 40GB
            gpu_config = self.gpu_configs[0]
            
            best_config = ""
            best_throughput = 0
            memory_issues = []
            
            for config_name, config in self.configurations.items():
                metrics = self.calculate_memory_metrics(scenario, config, gpu_config)
                
                if metrics['throughput'] > best_throughput and not metrics['memory_constrained']:
                    best_throughput = metrics['throughput']
                    best_config = config_name
                
                if metrics['memory_constrained']:
                    memory_issues.append(config_name)
            
            # Get GPU recommendation
            gpu_recommendation = self._get_gpu_recommendation(scenario, gpu_config)
            
            recommendations.append({
                'Scenario': scenario.name,
                'Best Config': best_config,
                'Memory Issues': ', '.join(memory_issues) if memory_issues else 'None',
                'GPU Recommendation': gpu_recommendation,
                'Key Optimization': self._get_key_optimization(scenario)
            })
        
        # Create table
        table_data = []
        for rec in recommendations:
            table_data.append([
                rec['Scenario'],
                rec['Best Config'],
                rec['Memory Issues'],
                rec['GPU Recommendation'],
                rec['Key Optimization']
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Scenario', 'Best Config', 'Memory Issues', 'GPU Recommendation', 'Key Optimization'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.18, 0.18, 0.15, 0.2, 0.29])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(5):
                cell = table[i, j]
                if i == 0:  # Header row
                    cell.set_facecolor('#9B59B6')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    # Color code memory issues
                    if table_data[i-1][2] != 'None':
                        cell.set_facecolor('#FFE5E5')  # Light red for issues
                    else:
                        cell.set_facecolor('#E5FFE5')  # Light green for no issues
        
        ax.set_title('Memory Optimization Recommendations', fontweight='bold', fontsize=12, pad=20)
    
    def _get_gpu_recommendation(self, scenario: ThroughputScenario, current_gpu: Dict) -> str:
        """Get GPU recommendation based on scenario requirements"""
        if scenario.model_size_gb > 40:
            return "A100 80GB or H100 80GB"
        elif scenario.concurrent_users > 100:
            return "A100 80GB (for larger KV cache)"
        elif scenario.memory_pressure > 0.7:
            return "H100 80GB (higher bandwidth)"
        else:
            return "A100 40GB sufficient"
    
    def _get_key_optimization(self, scenario: ThroughputScenario) -> str:
        """Get key memory optimization for scenario"""
        if scenario.concurrent_users > 100:
            return "Enable KV cache offloading"
        elif scenario.memory_pressure > 0.7:
            return "Reduce max_num_seqs, enable offloading"
        elif scenario.latency_sensitivity > 0.7:
            return "Enable prefix caching"
        else:
            return "Increase GPU utilization to 95%"

def main():
    """Main execution function"""
    print("💾 Generating Memory Utilization & Scaling Visualization...")
    
    # Create visualization
    viz = MemoryScalingViz()
    fig = viz.create_visualization()
    
    # Save visualization
    output_file = "/Users/theowolfenden/CascadeProjects/Terradev/memory_scaling_analysis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Memory scaling analysis saved: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
