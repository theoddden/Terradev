#!/usr/bin/env python3
"""
Visualization 2: Bottleneck Analysis
Focused on identifying and visualizing throughput bottlenecks in agentic AI workloads
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
sns.set_palette("viridis")

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

class BottleneckAnalysisViz:
    """Visualization focused on bottleneck analysis"""
    
    def __init__(self):
        self.scenarios = self._generate_scenarios()
        self.default_config = self._get_default_config()
        
    def _generate_scenarios(self) -> List[ThroughputScenario]:
        """Generate realistic agentic AI workload scenarios"""
        return [
            ThroughputScenario("Code Generation Agent", 850, 420, 15, 50, 0.7, 0.6, 2, 14.0),
            ThroughputScenario("Research Assistant", 1200, 680, 8, 25, 0.4, 0.3, 4, 28.0),
            ThroughputScenario("Customer Service Bot", 320, 180, 45, 200, 0.9, 0.8, 1, 7.0),
            ThroughputScenario("Document Analysis Agent", 2500, 1200, 3, 10, 0.2, 0.4, 8, 56.0),
            ThroughputScenario("Multi-Modal Agent", 600, 350, 25, 80, 0.6, 0.7, 2, 16.0)
        ]
    
    def _get_default_config(self) -> Dict:
        """Get default vLLM configuration"""
        return {
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 256,
            "enable_prefix_caching": False,
            "enable_kv_cache_offloading": False,
            "enable_speculative_decoding": False
        }
    
    def calculate_bottleneck_metrics(self, scenario: ThroughputScenario) -> Dict:
        """Calculate bottleneck metrics for a scenario"""
        total_tokens = scenario.avg_prompt_tokens + scenario.avg_response_tokens
        
        # Calculate individual bottleneck factors
        batch_efficiency = min(1.0, self.default_config["max_num_batched_tokens"] / total_tokens)
        concurrency_factor = min(1.0, self.default_config["max_num_seqs"] / scenario.concurrent_users)
        memory_efficiency = 1.0 - (scenario.memory_pressure * (1 - self.default_config["gpu_memory_utilization"]))
        
        # Calculate bottleneck severity (0 = no bottleneck, 1 = severe bottleneck)
        batch_bottleneck = 1.0 - batch_efficiency
        concurrency_bottleneck = 1.0 - concurrency_factor
        memory_bottleneck = 1.0 - memory_efficiency
        latency_bottleneck = scenario.latency_sensitivity
        
        # Calculate overall bottleneck score
        overall_bottleneck = max(batch_bottleneck, concurrency_bottleneck, memory_bottleneck, latency_bottleneck)
        
        return {
            'batch_efficiency': batch_efficiency,
            'concurrency_factor': concurrency_factor,
            'memory_efficiency': memory_efficiency,
            'batch_bottleneck': batch_bottleneck,
            'concurrency_bottleneck': concurrency_bottleneck,
            'memory_bottleneck': memory_bottleneck,
            'latency_bottleneck': latency_bottleneck,
            'overall_bottleneck': overall_bottleneck,
            'primary_bottleneck': self._identify_primary_bottleneck(batch_bottleneck, concurrency_bottleneck, 
                                                                  memory_bottleneck, latency_bottleneck)
        }
    
    def _identify_primary_bottleneck(self, batch: float, concurrency: float, memory: float, latency: float) -> str:
        """Identify the primary bottleneck"""
        bottlenecks = {
            'Batch Processing': batch,
            'Concurrency Limits': concurrency,
            'Memory Constraints': memory,
            'Latency Sensitivity': latency
        }
        return max(bottlenecks, key=bottlenecks.get)
    
    def create_visualization(self):
        """Create focused bottleneck analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agentic AI Token Throughput - Bottleneck Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Bottleneck severity comparison
        ax1 = axes[0, 0]
        self._plot_bottleneck_severity(ax1)
        
        # 2. Efficiency factors breakdown
        ax2 = axes[0, 1]
        self._plot_efficiency_breakdown(ax2)
        
        # 3. Bottleneck heatmap
        ax3 = axes[1, 0]
        self._plot_bottleneck_heatmap(ax3)
        
        # 4. Bottleneck mitigation strategies
        ax4 = axes[1, 1]
        self._plot_mitigation_strategies(ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_bottleneck_severity(self, ax):
        """Plot bottleneck severity across scenarios"""
        data = []
        for scenario in self.scenarios:
            metrics = self.calculate_bottleneck_metrics(scenario)
            data.append({
                'Scenario': scenario.name,
                'Batch Processing': metrics['batch_bottleneck'],
                'Concurrency Limits': metrics['concurrency_bottleneck'],
                'Memory Constraints': metrics['memory_bottleneck'],
                'Latency Sensitivity': metrics['latency_bottleneck'],
                'Overall': metrics['overall_bottleneck']
            })
        
        df = pd.DataFrame(data).set_index('Scenario')
        
        # Create horizontal bar chart
        df[['Batch Processing', 'Concurrency Limits', 'Memory Constraints', 'Latency Sensitivity']].plot(
            kind='barh', ax=ax, width=0.8)
        
        ax.set_title('Bottleneck Severity by Scenario', fontweight='bold', fontsize=12)
        ax.set_xlabel('Bottleneck Severity (0 = None, 1 = Severe)')
        ax.legend(title='Bottleneck Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    
    def _plot_efficiency_breakdown(self, ax):
        """Plot efficiency factors breakdown"""
        data = []
        for scenario in self.scenarios:
            metrics = self.calculate_bottleneck_metrics(scenario)
            data.append({
                'Scenario': scenario.name,
                'Batch Efficiency': metrics['batch_efficiency'],
                'Concurrency Efficiency': metrics['concurrency_factor'],
                'Memory Efficiency': metrics['memory_efficiency']
            })
        
        df = pd.DataFrame(data).set_index('Scenario')
        
        # Create stacked bar chart
        df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        
        ax.set_title('Efficiency Factors Breakdown', fontweight='bold', fontsize=12)
        ax.set_ylabel('Efficiency Factor')
        ax.set_xlabel('Scenario')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Efficiency Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _plot_bottleneck_heatmap(self, ax):
        """Plot bottleneck heatmap"""
        data = []
        for scenario in self.scenarios:
            metrics = self.calculate_bottleneck_metrics(scenario)
            data.append({
                'Scenario': scenario.name,
                'Batch Processing': metrics['batch_bottleneck'],
                'Concurrency Limits': metrics['concurrency_bottleneck'],
                'Memory Constraints': metrics['memory_bottleneck'],
                'Latency Sensitivity': metrics['latency_bottleneck']
            })
        
        df = pd.DataFrame(data).set_index('Scenario')
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.2f', cmap='Reds', ax=ax, 
                   cbar_kws={'label': 'Bottleneck Severity'})
        ax.set_title('Bottleneck Severity Heatmap', fontweight='bold', fontsize=12)
        ax.set_xlabel('Bottleneck Type')
        ax.set_ylabel('Scenario')
    
    def _plot_mitigation_strategies(self, ax):
        """Plot bottleneck mitigation strategies"""
        ax.axis('off')
        
        strategies = []
        for scenario in self.scenarios:
            metrics = self.calculate_bottleneck_metrics(scenario)
            primary = metrics['primary_bottleneck']
            strategy = self._get_mitigation_strategy(primary)
            
            strategies.append({
                'Scenario': scenario.name,
                'Primary Bottleneck': primary,
                'Severity': metrics['overall_bottleneck'],
                'Mitigation Strategy': strategy,
                'Priority': self._get_priority(metrics['overall_bottleneck'])
            })
        
        # Create table
        table_data = []
        for s in strategies:
            table_data.append([
                s['Scenario'],
                s['Primary Bottleneck'],
                f"{s['Severity']:.2f}",
                s['Mitigation Strategy'],
                s['Priority']
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Scenario', 'Primary Bottleneck', 'Severity', 'Mitigation Strategy', 'Priority'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.2, 0.1, 0.35, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(5):
                cell = table[i, j]
                if i == 0:  # Header row
                    cell.set_facecolor('#FF6B6B')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    # Color code by priority
                    priority = table_data[i-1][4]
                    if priority == 'High':
                        cell.set_facecolor('#FFE5E5')
                    elif priority == 'Medium':
                        cell.set_facecolor('#FFF5E5')
                    else:
                        cell.set_facecolor('#E5F5E5')
        
        ax.set_title('Bottleneck Mitigation Strategies', fontweight='bold', fontsize=12, pad=20)
    
    def _get_mitigation_strategy(self, bottleneck: str) -> str:
        """Get mitigation strategy for bottleneck type"""
        strategies = {
            'Batch Processing': 'Increase max-num-batched-tokens (2048→16384)',
            'Concurrency Limits': 'Increase max-num-seqs (256→1024)',
            'Memory Constraints': 'Enable KV cache offloading, increase GPU utilization',
            'Latency Sensitivity': 'Enable speculative decoding, optimize batch size'
        }
        return strategies.get(bottleneck, 'Optimize configuration parameters')
    
    def _get_priority(self, severity: float) -> str:
        """Get priority based on severity"""
        if severity > 0.7:
            return 'High'
        elif severity > 0.4:
            return 'Medium'
        else:
            return 'Low'

def main():
    """Main execution function"""
    print("🔍 Generating Bottleneck Analysis Visualization...")
    
    # Create visualization
    viz = BottleneckAnalysisViz()
    fig = viz.create_visualization()
    
    # Save visualization
    output_file = "/Users/theowolfenden/CascadeProjects/Terradev/bottleneck_analysis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Bottleneck analysis saved: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
