#!/usr/bin/env python3
"""
Visualization 5: Optimization Impact Analysis
Focused on quantifying the impact of individual and combined optimizations on agentic AI throughput
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
sns.set_palette("magma")

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

class OptimizationImpactViz:
    """Visualization focused on optimization impact analysis"""
    
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
        """Define individual optimization configurations"""
        return {
            "Baseline": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": False
            },
            "Batch Size Only": {
                "max_num_batched_tokens": 16384,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": False
            },
            "Memory Util Only": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": False
            },
            "Concurrency Only": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 1024,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": False
            },
            "Prefix Cache Only": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 256,
                "enable_prefix_caching": True,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": False
            },
            "KV Offload Only": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": True,
                "enable_speculative_decoding": False
            },
            "Speculative Only": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": True
            },
            "All Optimizations": {
                "max_num_batched_tokens": 16384,
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 1024,
                "enable_prefix_caching": True,
                "enable_kv_cache_offloading": True,
                "enable_speculative_decoding": True
            }
        }
    
    def calculate_metrics(self, scenario: ThroughputScenario, config: Dict) -> Dict:
        """Calculate throughput and latency metrics"""
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
        
        # Calculate throughput
        base_throughput = scenario.requests_per_second * total_tokens
        effective_throughput = (base_throughput * batch_efficiency * 
                              concurrency_factor * memory_efficiency * 
                              prefix_cache_bonus * kv_offload_bonus * 
                              speculative_bonus * latency_penalty)
        
        # Calculate latency
        base_latency = total_tokens / (scenario.gpu_count * 100)  # ms
        optimized_latency = base_latency / (batch_efficiency * speculative_bonus)
        
        return {
            'throughput': effective_throughput,
            'latency_ms': optimized_latency,
            'batch_efficiency': batch_efficiency,
            'concurrency_factor': concurrency_factor,
            'memory_efficiency': memory_efficiency
        }
    
    def create_visualization(self):
        """Create focused optimization impact visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agentic AI Token Throughput - Optimization Impact Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Individual optimization impact
        ax1 = axes[0, 0]
        self._plot_individual_optimization_impact(ax1)
        
        # 2. Cumulative optimization effect
        ax2 = axes[0, 1]
        self._plot_cumulative_effect(ax2)
        
        # 3. ROI analysis
        ax3 = axes[1, 0]
        self._plot_roi_analysis(ax3)
        
        # 4. Implementation roadmap
        ax4 = axes[1, 1]
        self._plot_implementation_roadmap(ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_individual_optimization_impact(self, ax):
        """Plot impact of individual optimizations"""
        # Use Customer Service Bot as representative scenario
        scenario = self.scenarios[2]  # Customer Service Bot
        baseline_metrics = self.calculate_metrics(scenario, self.optimization_configs["Baseline"])
        
        data = []
        for config_name, config in self.optimization_configs.items():
            if config_name != "Baseline" and config_name != "All Optimizations":
                metrics = self.calculate_metrics(scenario, config)
                improvement = (metrics['throughput'] / baseline_metrics['throughput'] - 1) * 100
                latency_improvement = (1 - metrics['latency_ms'] / baseline_metrics['latency_ms']) * 100
                
                data.append({
                    'Optimization': config_name.replace(' Only', ''),
                    'Throughput Improvement (%)': improvement,
                    'Latency Improvement (%)': latency_improvement
                })
        
        df = pd.DataFrame(data)
        
        # Create horizontal bar chart
        df_sorted = df.sort_values('Throughput Improvement (%)', ascending=True)
        
        ax.barh(df_sorted['Optimization'], df_sorted['Throughput Improvement (%)'], 
               alpha=0.7, color='blue', label='Throughput')
        
        # Add latency improvements as markers
        ax2 = ax.twiny()
        ax2.scatter(df_sorted['Latency Improvement (%)'], df_sorted['Optimization'], 
                   color='red', s=100, alpha=0.8, label='Latency', marker='s')
        
        ax.set_xlabel('Throughput Improvement (%)', color='blue')
        ax2.set_xlabel('Latency Improvement (%)', color='red')
        ax.tick_params(axis='x', labelcolor='blue')
        ax2.tick_params(axis='x', labelcolor='red')
        
        ax.set_title('Individual Optimization Impact (Customer Service Bot)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Optimization')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for i, (throughput, latency) in enumerate(zip(df_sorted['Throughput Improvement (%)'], 
                                                     df_sorted['Latency Improvement (%)'])):
            ax.text(throughput + 1, i, f'{throughput:.1f}%', va='center', fontsize=8)
            ax2.text(latency + 1, i, f'{latency:.1f}%', va='center', fontsize=8, color='red')
    
    def _plot_cumulative_effect(self, ax):
        """Plot cumulative effect of optimizations"""
        scenario = self.scenarios[1]  # Research Assistant
        
        # Define optimization order
        optimization_order = [
            ("Baseline", "Baseline"),
            ("+ Memory Util", "Memory Util Only"),
            ("+ Batch Size", "Batch Size Only"),
            ("+ Concurrency", "Concurrency Only"),
            ("+ Prefix Cache", "Prefix Cache Only"),
            ("+ KV Offload", "KV Offload Only"),
            ("+ Speculative", "Speculative Only"),
            ("All Combined", "All Optimizations")
        ]
        
        # Build cumulative configurations
        cumulative_configs = {}
        current_config = self.optimization_configs["Baseline"].copy()
        
        data = []
        for step_name, config_name in optimization_order:
            if config_name != "Baseline":
                # Add the optimization to current config
                new_config = self.optimization_configs[config_name]
                for key, value in new_config.items():
                    if value != current_config[key]:
                        current_config[key] = value
            
            metrics = self.calculate_metrics(scenario, current_config)
            baseline_metrics = self.calculate_metrics(scenario, self.optimization_configs["Baseline"])
            improvement = (metrics['throughput'] / baseline_metrics['throughput'] - 1) * 100
            
            data.append({
                'Step': step_name,
                'Throughput': metrics['throughput'],
                'Improvement (%)': improvement,
                'Latency (ms)': metrics['latency_ms']
            })
        
        df = pd.DataFrame(data)
        
        # Create line chart
        ax2 = ax.twinx()
        
        line1 = ax.plot(df['Step'], df['Throughput'], marker='o', linewidth=2, 
                       markersize=8, color='blue', label='Throughput')
        line2 = ax2.plot(df['Step'], df['Latency (ms)'], marker='s', linewidth=2, 
                        markersize=8, color='red', label='Latency')
        
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Throughput (tokens/sec)', color='blue')
        ax2.set_ylabel('Latency (ms)', color='red')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title('Cumulative Optimization Effect (Research Assistant)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotations
        for i, improvement in enumerate(df['Improvement (%)']):
            if i > 0:  # Skip baseline
                ax.text(i, df['Throughput'][i] + 500, f'+{improvement:.1f}%', 
                       ha='center', fontsize=8, alpha=0.7)
    
    def _plot_roi_analysis(self, ax):
        """Plot ROI analysis for optimizations"""
        # Define implementation costs (relative units)
        implementation_costs = {
            "Memory Util": 1,      # Easy - just change parameter
            "Batch Size": 2,       # Medium - requires testing
            "Concurrency": 2,      # Medium - requires testing
            "Prefix Cache": 3,     # Hard - requires infrastructure
            "KV Offload": 4,       # Very Hard - requires hardware support
            "Speculative": 3       # Hard - requires model compatibility
        }
        
        # Calculate average improvements across all scenarios
        avg_improvements = {}
        for opt_name, cost in implementation_costs.items():
            config_key = f"{opt_name} Only"
            improvements = []
            
            for scenario in self.scenarios:
                baseline_metrics = self.calculate_metrics(scenario, self.optimization_configs["Baseline"])
                opt_metrics = self.calculate_metrics(scenario, self.optimization_configs[config_key])
                improvement = (opt_metrics['throughput'] / baseline_metrics['throughput'] - 1) * 100
                improvements.append(improvement)
            
            avg_improvements[opt_name] = np.mean(improvements)
        
        # Calculate ROI
        roi_data = []
        for opt_name, cost in implementation_costs.items():
            improvement = avg_improvements[opt_name]
            roi = improvement / cost  # Improvement per cost unit
            
            roi_data.append({
                'Optimization': opt_name,
                'Cost': cost,
                'Avg Improvement (%)': improvement,
                'ROI': roi
            })
        
        df = pd.DataFrame(roi_data)
        
        # Create scatter plot
        scatter = ax.scatter(df['Cost'], df['Avg Improvement (%)'], 
                           s=df['ROI']*100, alpha=0.6, c=df['ROI'], 
                           cmap='viridis', edgecolors='black')
        
        # Add labels for each point
        for i, row in df.iterrows():
            ax.annotate(row['Optimization'], (row['Cost'], row['Avg Improvement (%)']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Implementation Cost (Relative Units)')
        ax.set_ylabel('Average Throughput Improvement (%)')
        ax.set_title('Optimization ROI Analysis', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for ROI
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('ROI (Improvement/Cost)')
    
    def _plot_implementation_roadmap(self, ax):
        """Plot implementation roadmap"""
        ax.axis('off')
        
        # Define roadmap phases
        roadmap = [
            {
                'Phase': 'Phase 1: Quick Wins',
                'Timeline': 'Week 1-2',
                'Optimizations': ['Memory Utilization', 'Batch Size'],
                'Effort': 'Low',
                'Impact': 'High',
                'Priority': 'Critical'
            },
            {
                'Phase': 'Phase 2: Core Features',
                'Timeline': 'Week 3-4',
                'Optimizations': ['Concurrency Limits', 'Prefix Caching'],
                'Effort': 'Medium',
                'Impact': 'Medium-High',
                'Priority': 'High'
            },
            {
                'Phase': 'Phase 3: Advanced',
                'Timeline': 'Week 5-6',
                'Optimizations': ['Speculative Decoding'],
                'Effort': 'High',
                'Impact': 'Medium',
                'Priority': 'Medium'
            },
            {
                'Phase': 'Phase 4: Infrastructure',
                'Timeline': 'Week 7-8',
                'Optimizations': ['KV Cache Offloading'],
                'Effort': 'Very High',
                'Impact': 'High',
                'Priority': 'Context-dependent'
            }
        ]
        
        # Create table
        table_data = []
        for phase in roadmap:
            table_data.append([
                phase['Phase'],
                phase['Timeline'],
                ', '.join(phase['Optimizations']),
                phase['Effort'],
                phase['Impact'],
                phase['Priority']
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Phase', 'Timeline', 'Optimizations', 'Effort', 'Impact', 'Priority'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.1, 0.3, 0.1, 0.1, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.5)
        
        # Style the table
        colors = ['#E8F5E8', '#FFF5E5', '#FFE5E5', '#E5E5FF']
        for i in range(len(table_data) + 1):
            for j in range(6):
                cell = table[i, j]
                if i == 0:  # Header row
                    cell.set_facecolor('#2E4057')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor(colors[i-1])
        
        ax.set_title('Optimization Implementation Roadmap', fontweight='bold', fontsize=12, pad=20)

def main():
    """Main execution function"""
    print("📈 Generating Optimization Impact Analysis Visualization...")
    
    # Create visualization
    viz = OptimizationImpactViz()
    fig = viz.create_visualization()
    
    # Save visualization
    output_file = "/Users/theowolfenden/CascadeProjects/Terradev/optimization_impact_analysis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Optimization impact analysis saved: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
