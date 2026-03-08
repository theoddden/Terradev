#!/usr/bin/env python3
"""
Visualization 3: Latency vs Throughput Tradeoff Analysis
Focused on the fundamental tradeoff between latency and throughput in agentic AI workloads
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
sns.set_palette("coolwarm")

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

class LatencyThroughputTradeoffViz:
    """Visualization focused on latency vs throughput tradeoff analysis"""
    
    def __init__(self):
        self.scenarios = self._generate_scenarios()
        self.configurations = self._get_configurations()
        
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
        """Define different configuration points for tradeoff analysis"""
        return {
            "Default": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": False
            },
            "Throughput Focus": {
                "max_num_batched_tokens": 16384,
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 1024,
                "enable_prefix_caching": True,
                "enable_kv_cache_offloading": True,
                "enable_speculative_decoding": True
            },
            "Latency Focus": {
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
    
    def calculate_metrics(self, scenario: ThroughputScenario, config: Dict) -> Dict:
        """Calculate latency and throughput metrics"""
        total_tokens = scenario.avg_prompt_tokens + scenario.avg_response_tokens
        
        # Efficiency factors
        batch_efficiency = min(1.0, config["max_num_batched_tokens"] / total_tokens)
        concurrency_factor = min(1.0, config["max_num_seqs"] / scenario.concurrent_users)
        memory_efficiency = 1.0 - (scenario.memory_pressure * (1 - config["gpu_memory_utilization"]))
        
        # Optimization bonuses
        prefix_cache_bonus = 1.15 if config["enable_prefix_caching"] else 1.0
        kv_offload_bonus = 1.25 if config["enable_kv_cache_offloading"] else 1.0
        speculative_bonus = 1.18 if config["enable_speculative_decoding"] else 1.0
        
        # Calculate throughput
        base_throughput = scenario.requests_per_second * total_tokens
        effective_throughput = (base_throughput * batch_efficiency * 
                              concurrency_factor * memory_efficiency * 
                              prefix_cache_bonus * kv_offload_bonus * 
                              speculative_bonus)
        
        # Apply latency penalty
        latency_penalty = 1.0 - (scenario.latency_sensitivity * 0.2)
        final_throughput = effective_throughput * latency_penalty
        
        # Calculate latency
        base_latency = total_tokens / (scenario.gpu_count * 100)  # ms
        optimized_latency = base_latency / (batch_efficiency * speculative_bonus * 1.2)  # Additional optimization
        
        return {
            'throughput': final_throughput,
            'latency_ms': optimized_latency,
            'batch_efficiency': batch_efficiency,
            'concurrency_factor': concurrency_factor
        }
    
    def create_visualization(self):
        """Create focused latency vs throughput tradeoff visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agentic AI Token Throughput - Latency vs Throughput Tradeoff Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Main tradeoff scatter plot
        ax1 = axes[0, 0]
        self._plot_tradeoff_scatter(ax1)
        
        # 2. Pareto frontier analysis
        ax2 = axes[0, 1]
        self._plot_pareto_frontier(ax2)
        
        # 3. Configuration impact analysis
        ax3 = axes[1, 0]
        self._plot_configuration_impact(ax3)
        
        # 4. Optimal configuration recommendations
        ax4 = axes[1, 1]
        self._plot_optimal_recommendations(ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_tradeoff_scatter(self, ax):
        """Plot main latency vs throughput scatter plot"""
        data = []
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.scenarios)))
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, scenario in enumerate(self.scenarios):
            for j, (config_name, config) in enumerate(self.configurations.items()):
                metrics = self.calculate_metrics(scenario, config)
                data.append({
                    'Scenario': scenario.name,
                    'Configuration': config_name,
                    'Latency (ms)': metrics['latency_ms'],
                    'Throughput': metrics['throughput'],
                    'Color': colors[i],
                    'Marker': markers[j]
                })
        
        df = pd.DataFrame(data)
        
        # Create scatter plot
        for i, scenario in enumerate(self.scenarios):
            scenario_data = df[df['Scenario'] == scenario.name]
            for j, config_name in enumerate(self.configurations.keys()):
                config_data = scenario_data[scenario_data['Configuration'] == config_name]
                if not config_data.empty:
                    ax.scatter(config_data['Latency (ms)'], config_data['Throughput'],
                             c=[colors[i]], marker=markers[j], s=120, alpha=0.8,
                             label=f"{scenario.name}" if j == 0 else "")
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_title('Latency vs Throughput Tradeoff', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add ideal regions
        ax.axhspan(30000, 40000, xmin=0, xmax=0.3, alpha=0.1, color='green', label='Ideal: Low Latency, High Throughput')
        ax.text(2, 35000, 'Ideal Zone', fontsize=10, alpha=0.7)
    
    def _plot_pareto_frontier(self, ax):
        """Plot Pareto frontier analysis"""
        # Calculate Pareto frontier for each scenario
        for scenario in self.scenarios:
            points = []
            for config_name, config in self.configurations.items():
                metrics = self.calculate_metrics(scenario, config)
                points.append((metrics['latency_ms'], metrics['throughput']))
            
            # Sort by latency
            points.sort(key=lambda x: x[0])
            
            # Calculate Pareto frontier
            frontier = []
            max_throughput = 0
            for latency, throughput in points:
                if throughput > max_throughput:
                    frontier.append((latency, throughput))
                    max_throughput = throughput
            
            if frontier:
                latencies, throughputs = zip(*frontier)
                ax.plot(latencies, throughputs, marker='o', linewidth=2, 
                       markersize=8, label=scenario.name, alpha=0.8)
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_title('Pareto Frontier - Optimal Tradeoffs', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _plot_configuration_impact(self, ax):
        """Plot configuration impact on latency and throughput"""
        data = []
        for scenario in self.scenarios:
            default_metrics = self.calculate_metrics(scenario, self.configurations["Default"])
            
            for config_name in ["Throughput Focus", "Latency Focus", "Balanced"]:
                metrics = self.calculate_metrics(scenario, self.configurations[config_name])
                latency_improvement = (1 - metrics['latency_ms'] / default_metrics['latency_ms']) * 100
                throughput_improvement = (metrics['throughput'] / default_metrics['throughput'] - 1) * 100
                
                data.append({
                    'Scenario': scenario.name,
                    'Configuration': config_name,
                    'Latency Improvement (%)': latency_improvement,
                    'Throughput Improvement (%)': throughput_improvement
                })
        
        df = pd.DataFrame(data)
        
        # Create scatter plot
        sns.scatterplot(data=df, x='Latency Improvement (%)', y='Throughput Improvement (%)', 
                       hue='Configuration', style='Scenario', s=100, ax=ax)
        
        ax.set_xlabel('Latency Improvement (%)', fontsize=12)
        ax.set_ylabel('Throughput Improvement (%)', fontsize=12)
        ax.set_title('Configuration Impact Analysis', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(15, 15, 'Win-Win', fontsize=10, alpha=0.7, ha='center')
        ax.text(-15, 15, 'Latency Win', fontsize=10, alpha=0.7, ha='center')
        ax.text(15, -15, 'Throughput Win', fontsize=10, alpha=0.7, ha='center')
        ax.text(-15, -15, 'Lose-Lose', fontsize=10, alpha=0.7, ha='center')
    
    def _plot_optimal_recommendations(self, ax):
        """Plot optimal configuration recommendations"""
        ax.axis('off')
        
        recommendations = []
        for scenario in self.scenarios:
            best_config = ""
            best_score = -float('inf')
            
            for config_name, config in self.configurations.items():
                metrics = self.calculate_metrics(scenario, config)
                
                # Calculate composite score based on scenario's latency sensitivity
                if scenario.latency_sensitivity > 0.7:
                    # Latency-focused scenario
                    score = metrics['throughput'] / (metrics['latency_ms'] ** 2)
                else:
                    # Throughput-focused scenario
                    score = metrics['throughput'] / metrics['latency_ms']
                
                if score > best_score:
                    best_score = score
                    best_config = config_name
            
            recommendations.append({
                'Scenario': scenario.name,
                'Latency Sensitivity': scenario.latency_sensitivity,
                'Optimal Config': best_config,
                'Reasoning': self._get_reasoning(scenario.latency_sensitivity, best_config)
            })
        
        # Create table
        table_data = []
        for rec in recommendations:
            table_data.append([
                rec['Scenario'],
                f"{rec['Latency Sensitivity']:.2f}",
                rec['Optimal Config'],
                rec['Reasoning']
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Scenario', 'Latency Sensitivity', 'Optimal Config', 'Reasoning'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.15, 0.2, 0.45])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[i, j]
                if i == 0:  # Header row
                    cell.set_facecolor('#4169E1')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    # Color code by latency sensitivity
                    sensitivity = float(table_data[i-1][1])
                    if sensitivity > 0.7:
                        cell.set_facecolor('#E5E5FF')  # Light blue for latency-sensitive
                    else:
                        cell.set_facecolor('#FFE5E5')  # Light red for throughput-focused
        
        ax.set_title('Optimal Configuration Recommendations', fontweight='bold', fontsize=12, pad=20)
    
    def _get_reasoning(self, latency_sensitivity: float, config: str) -> str:
        """Get reasoning for configuration recommendation"""
        if latency_sensitivity > 0.7:
            return "High latency sensitivity - prioritize low latency configurations"
        elif latency_sensitivity > 0.4:
            return "Balanced requirements - choose balanced configuration"
        else:
            return "Low latency sensitivity - prioritize throughput optimization"

def main():
    """Main execution function"""
    print("⚖️ Generating Latency vs Throughput Tradeoff Visualization...")
    
    # Create visualization
    viz = LatencyThroughputTradeoffViz()
    fig = viz.create_visualization()
    
    # Save visualization
    output_file = "/Users/theowolfenden/CascadeProjects/Terradev/latency_throughput_tradeoff.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Latency vs throughput tradeoff saved: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
