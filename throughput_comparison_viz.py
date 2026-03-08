#!/usr/bin/env python3
"""
Visualization 1: Throughput Comparison Analysis
Focused on token throughput across different agentic AI scenarios and configurations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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

class ThroughputComparisonViz:
    """Visualization focused on throughput comparison analysis"""
    
    def __init__(self):
        self.scenarios = self._generate_scenarios()
        self.optimization_configs = self._get_optimization_configs()
        
    def _generate_scenarios(self) -> List[ThroughputScenario]:
        """Generate realistic agentic AI workload scenarios"""
        return [
            ThroughputScenario("Code Generation Agent", 850, 420, 15, 50, 0.7, 0.6, 2, 14.0),
            ThroughputScenario("Research Assistant", 1200, 680, 8, 25, 0.4, 0.3, 4, 28.0),
            ThroughputScenario("Customer Service Bot", 320, 180, 45, 200, 0.9, 0.8, 1, 7.0),
            ThroughputScenario("Document Analysis Agent", 2500, 1200, 3, 10, 0.2, 0.4, 8, 56.0),
            ThroughputScenario("Multi-Modal Agent", 600, 350, 25, 80, 0.6, 0.7, 2, 16.0)
        ]
    
    def _get_optimization_configs(self) -> Dict[str, Dict]:
        """Define vLLM optimization configurations"""
        return {
            "Default": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": False
            },
            "Throughput Optimized": {
                "max_num_batched_tokens": 16384,
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 1024,
                "enable_prefix_caching": True,
                "enable_kv_cache_offloading": True,
                "enable_speculative_decoding": True
            },
            "Latency Optimized": {
                "max_num_batched_tokens": 4096,
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 512,
                "enable_prefix_caching": True,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": True
            },
            "Balanced": {
                "max_num_batched_tokens": 8192,
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 768,
                "enable_prefix_caching": True,
                "enable_kv_cache_offloading": True,
                "enable_speculative_decoding": True
            }
        }
    
    def calculate_throughput_metrics(self, scenario: ThroughputScenario, config: Dict) -> Dict:
        """Calculate throughput metrics based on scenario and configuration"""
        total_tokens = scenario.avg_prompt_tokens + scenario.avg_response_tokens
        
        # Efficiency factors
        batch_efficiency = min(1.0, config["max_num_batched_tokens"] / total_tokens)
        concurrency_factor = min(1.0, config["max_num_seqs"] / scenario.concurrent_users)
        memory_efficiency = 1.0 - (scenario.memory_pressure * (1 - config["gpu_memory_utilization"]))
        
        # Optimization bonuses
        prefix_cache_bonus = 1.15 if config["enable_prefix_caching"] else 1.0
        kv_offload_bonus = 1.25 if config["enable_kv_cache_offloading"] else 1.0
        speculative_bonus = 1.18 if config["enable_speculative_decoding"] else 1.0
        
        # Latency penalty
        latency_penalty = 1.0 - (scenario.latency_sensitivity * 0.2)
        
        # Calculate effective throughput
        base_throughput = scenario.requests_per_second * total_tokens
        effective_throughput = (base_throughput * batch_efficiency * 
                              concurrency_factor * memory_efficiency * 
                              prefix_cache_bonus * kv_offload_bonus * 
                              speculative_bonus * latency_penalty)
        
        return {
            "tokens_per_second": effective_throughput,
            "throughput_efficiency": effective_throughput / base_throughput
        }
    
    def create_visualization(self):
        """Create focused throughput comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agentic AI Token Throughput Analysis - Configuration Comparison', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Main throughput comparison
        ax1 = axes[0, 0]
        self._plot_main_throughput_comparison(ax1)
        
        # 2. Throughput improvement percentages
        ax2 = axes[0, 1]
        self._plot_throughput_improvements(ax2)
        
        # 3. Efficiency analysis
        ax3 = axes[1, 0]
        self._plot_efficiency_analysis(ax3)
        
        # 4. Configuration recommendations
        ax4 = axes[1, 1]
        self._plot_recommendations(ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_main_throughput_comparison(self, ax):
        """Plot main throughput comparison across scenarios and configurations"""
        data = []
        for scenario in self.scenarios:
            for config_name, config in self.optimization_configs.items():
                metrics = self.calculate_throughput_metrics(scenario, config)
                data.append({
                    'Scenario': scenario.name,
                    'Configuration': config_name,
                    'Tokens/Second': metrics['tokens_per_second']
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        sns.barplot(data=df, x='Scenario', y='Tokens/Second', hue='Configuration', ax=ax)
        ax.set_title('Token Throughput by Scenario and Configuration', fontweight='bold', fontsize=12)
        ax.set_ylabel('Tokens per Second')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='{:,.0f}', fontsize=8)
    
    def _plot_throughput_improvements(self, ax):
        """Plot throughput improvement percentages compared to default"""
        improvements = []
        for scenario in self.scenarios:
            default_metrics = self.calculate_throughput_metrics(scenario, self.optimization_configs["Default"])
            
            for config_name in ["Throughput Optimized", "Latency Optimized", "Balanced"]:
                optimized_metrics = self.calculate_throughput_metrics(scenario, self.optimization_configs[config_name])
                improvement = (optimized_metrics['tokens_per_second'] / default_metrics['tokens_per_second'] - 1) * 100
                
                improvements.append({
                    'Scenario': scenario.name,
                    'Configuration': config_name,
                    'Improvement (%)': improvement
                })
        
        df = pd.DataFrame(improvements)
        
        # Create grouped bar chart
        sns.barplot(data=df, x='Scenario', y='Improvement (%)', hue='Configuration', ax=ax)
        ax.set_title('Throughput Improvement vs Default (%)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Improvement (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='{:.1f}%', fontsize=8)
        
        # Add reference line at 0%
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    def _plot_efficiency_analysis(self, ax):
        """Plot throughput efficiency analysis"""
        data = []
        for scenario in self.scenarios:
            for config_name, config in self.optimization_configs.items():
                metrics = self.calculate_throughput_metrics(scenario, config)
                data.append({
                    'Scenario': scenario.name,
                    'Configuration': config_name,
                    'Efficiency': metrics['throughput_efficiency']
                })
        
        df = pd.DataFrame(data)
        
        # Create heatmap
        pivot_df = df.pivot(index='Scenario', columns='Configuration', values='Efficiency')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                   cbar_kws={'label': 'Throughput Efficiency'})
        ax.set_title('Throughput Efficiency Heatmap', fontweight='bold', fontsize=12)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Scenario')
    
    def _plot_recommendations(self, ax):
        """Plot configuration recommendations"""
        ax.axis('off')
        
        recommendations = []
        for scenario in self.scenarios:
            best_throughput = 0
            best_config = ""
            
            for config_name, config in self.optimization_configs.items():
                metrics = self.calculate_throughput_metrics(scenario, config)
                if metrics['tokens_per_second'] > best_throughput:
                    best_throughput = metrics['tokens_per_second']
                    best_config = config_name
            
            recommendations.append({
                'Scenario': scenario.name,
                'Best Config': best_config,
                'Throughput': best_throughput,
                'Key Factor': self._identify_key_factor(scenario)
            })
        
        # Create table
        table_data = []
        for rec in recommendations:
            table_data.append([
                rec['Scenario'],
                rec['Best Config'],
                f"{rec['Throughput']:,.0f} tokens/sec",
                rec['Key Factor']
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Scenario', 'Best Configuration', 'Max Throughput', 'Key Factor'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[i, j]
                if i == 0:  # Header row
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0')
        
        ax.set_title('Configuration Recommendations', fontweight='bold', fontsize=12, pad=20)
    
    def _identify_key_factor(self, scenario: ThroughputScenario) -> str:
        """Identify key optimization factor for scenario"""
        if scenario.latency_sensitivity > 0.8:
            return "Latency Critical"
        elif scenario.memory_pressure > 0.7:
            return "Memory Constrained"
        elif scenario.concurrent_users > 100:
            return "High Concurrency"
        elif scenario.requests_per_second > 30:
            return "High QPS"
        else:
            return "Balanced"

def main():
    """Main execution function"""
    print("📊 Generating Throughput Comparison Visualization...")
    
    # Create visualization
    viz = ThroughputComparisonViz()
    fig = viz.create_visualization()
    
    # Save visualization
    output_file = "/Users/theowolfenden/CascadeProjects/Terradev/throughput_comparison_analysis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Throughput comparison saved: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
