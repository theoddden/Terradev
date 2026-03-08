#!/usr/bin/env python3
"""
Detailed Breakdown: How the Figures Were Calculated
Step-by-step mathematical analysis of the throughput calculations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

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

class FigureCalculationBreakdown:
    """Detailed breakdown of how all figures were calculated"""
    
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
        """Define different optimization configurations"""
        return {
            "Baseline": {
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_kv_cache_offloading": False,
                "enable_speculative_decoding": False
            },
            "Optimized": {
                "max_num_batched_tokens": 16384,
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 1024,
                "enable_prefix_caching": True,
                "enable_kv_cache_offloading": True,
                "enable_speculative_decoding": True
            }
        }
    
    def detailed_calculation_breakdown(self, scenario_name: str):
        """Show detailed step-by-step calculation for a specific scenario"""
        scenario = next(s for s in self.scenarios if s.name == scenario_name)
        
        print(f"\n{'='*80}")
        print(f"DETAILED CALCULATION BREAKDOWN: {scenario_name}")
        print(f"{'='*80}")
        
        print(f"\n📊 INPUT PARAMETERS:")
        print(f"  Prompt Tokens: {scenario.avg_prompt_tokens}")
        print(f"  Response Tokens: {scenario.avg_response_tokens}")
        print(f"  QPS: {scenario.requests_per_second}")
        print(f"  Concurrent Users: {scenario.concurrent_users}")
        print(f"  Latency Sensitivity: {scenario.latency_sensitivity}")
        print(f"  Memory Pressure: {scenario.memory_pressure}")
        print(f"  GPU Count: {scenario.gpu_count}")
        print(f"  Model Size: {scenario.model_size_gb} GB")
        
        total_tokens = scenario.avg_prompt_tokens + scenario.avg_response_tokens
        print(f"\n🔢 DERIVED VALUES:")
        print(f"  Total Tokens per Request: {total_tokens}")
        
        for config_name, config in self.configurations.items():
            print(f"\n{'='*60}")
            print(f"CONFIGURATION: {config_name}")
            print(f"{'='*60}")
            
            print(f"\n⚙️  CONFIGURATION PARAMETERS:")
            print(f"  Max Batched Tokens: {config['max_num_batched_tokens']}")
            print(f"  GPU Memory Utilization: {config['gpu_memory_utilization']}")
            print(f"  Max Sequences: {config['max_num_seqs']}")
            print(f"  Prefix Caching: {config['enable_prefix_caching']}")
            print(f"  KV Cache Offloading: {config['enable_kv_cache_offloading']}")
            print(f"  Speculative Decoding: {config['enable_speculative_decoding']}")
            
            # Step 1: Calculate Base Throughput
            base_throughput = scenario.requests_per_second * total_tokens
            print(f"\n📈 STEP 1: BASE THROUGHPUT")
            print(f"  Formula: QPS × Total Tokens")
            print(f"  Calculation: {scenario.requests_per_second} × {total_tokens}")
            print(f"  Result: {base_throughput:,} tokens/second")
            
            # Step 2: Calculate Efficiency Factors
            batch_efficiency = min(1.0, config["max_num_batched_tokens"] / total_tokens)
            concurrency_factor = min(1.0, config["max_num_seqs"] / scenario.concurrent_users)
            memory_efficiency = 1.0 - (scenario.memory_pressure * (1 - config["gpu_memory_utilization"]))
            
            print(f"\n⚡ STEP 2: EFFICIENCY FACTORS")
            print(f"  Batch Efficiency: min(1.0, {config['max_num_batched_tokens']} / {total_tokens}) = {batch_efficiency:.3f}")
            print(f"  Concurrency Factor: min(1.0, {config['max_num_seqs']} / {scenario.concurrent_users}) = {concurrency_factor:.3f}")
            print(f"  Memory Efficiency: 1.0 - ({scenario.memory_pressure} × (1 - {config['gpu_memory_utilization']})) = {memory_efficiency:.3f}")
            
            # Step 3: Calculate Optimization Bonuses
            prefix_cache_bonus = 1.15 if config["enable_prefix_caching"] else 1.0
            kv_offload_bonus = 1.25 if config["enable_kv_cache_offloading"] else 1.0
            speculative_bonus = 1.18 if config["enable_speculative_decoding"] else 1.0
            
            print(f"\n🚀 STEP 3: OPTIMIZATION BONUSES")
            print(f"  Prefix Cache Bonus: {prefix_cache_bonus:.2f}x" + (" (enabled)" if config['enable_prefix_caching'] else " (disabled)"))
            print(f"  KV Offload Bonus: {kv_offload_bonus:.2f}x" + (" (enabled)" if config['enable_kv_cache_offloading'] else " (disabled)"))
            print(f"  Speculative Bonus: {speculative_bonus:.2f}x" + (" (enabled)" if config['enable_speculative_decoding'] else " (disabled)"))
            
            # Step 4: Calculate Latency Penalty
            latency_penalty = 1.0 - (scenario.latency_sensitivity * 0.2)
            print(f"\n⏱️  STEP 4: LATENCY PENALTY")
            print(f"  Formula: 1.0 - (Latency Sensitivity × 0.2)")
            print(f"  Calculation: 1.0 - ({scenario.latency_sensitivity} × 0.2)")
            print(f"  Result: {latency_penalty:.3f}")
            
            # Step 5: Calculate Effective Throughput
            effective_throughput = (base_throughput * batch_efficiency * 
                                  concurrency_factor * memory_efficiency * 
                                  prefix_cache_bonus * kv_offload_bonus * 
                                  speculative_bonus * latency_penalty)
            
            print(f"\n🎯 STEP 5: EFFECTIVE THROUGHPUT")
            print(f"  Formula: Base Throughput × All Factors × All Bonuses × Latency Penalty")
            print(f"  Calculation: {base_throughput:,.0f} × {batch_efficiency:.3f} × {concurrency_factor:.3f} × {memory_efficiency:.3f} × {prefix_cache_bonus:.2f} × {kv_offload_bonus:.2f} × {speculative_bonus:.2f} × {latency_penalty:.3f}")
            print(f"  Result: {effective_throughput:,.0f} tokens/second")
            
            # Step 6: Calculate Latency
            base_latency = total_tokens / (scenario.gpu_count * 100)  # ms
            optimized_latency = base_latency / (batch_efficiency * speculative_bonus * 1.2)
            
            print(f"\n⏱️  STEP 6: LATENCY CALCULATION")
            print(f"  Base Latency: {total_tokens} / ({scenario.gpu_count} × 100) = {base_latency:.2f} ms")
            print(f"  Optimized Latency: {base_latency:.2f} / ({batch_efficiency:.3f} × {speculative_bonus:.2f} × 1.2) = {optimized_latency:.2f} ms")
            
            # Step 7: Calculate Memory Usage
            model_memory = scenario.model_size_gb
            kv_cache_memory = (scenario.concurrent_users * total_tokens * 2) / (1024**3)
            if config["enable_kv_cache_offloading"]:
                kv_cache_memory *= 0.3
            prefix_cache_memory = (total_tokens * 0.5) / (1024**3) if config["enable_prefix_caching"] else 0
            activation_memory = (scenario.concurrent_users * total_tokens * 0.5) / (1024**3)
            
            total_memory = model_memory + kv_cache_memory + prefix_cache_memory + activation_memory
            memory_utilization = total_memory / 40  # Assume A100 40GB
            
            print(f"\n💾 STEP 7: MEMORY CALCULATION")
            print(f"  Model Memory: {model_memory:.1f} GB")
            print(f"  KV Cache Memory: {kv_cache_memory:.2f} GB")
            print(f"  Prefix Cache Memory: {prefix_cache_memory:.2f} GB")
            print(f"  Activation Memory: {activation_memory:.2f} GB")
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Memory Utilization: {total_memory:.2f} / 40 = {memory_utilization:.3f} ({memory_utilization*100:.1f}%)")
            
            # Store results for comparison
            if config_name == "Baseline":
                baseline_results = {
                    'throughput': effective_throughput,
                    'latency': optimized_latency,
                    'memory_utilization': memory_utilization
                }
            else:
                optimized_results = {
                    'throughput': effective_throughput,
                    'latency': optimized_latency,
                    'memory_utilization': memory_utilization
                }
        
        # Step 8: Calculate Improvements
        print(f"\n{'='*60}")
        print(f"IMPROVEMENT CALCULATIONS")
        print(f"{'='*60}")
        
        throughput_improvement = (optimized_results['throughput'] / baseline_results['throughput'] - 1) * 100
        latency_improvement = (1 - optimized_results['latency'] / baseline_results['latency']) * 100
        memory_change = (optimized_results['memory_utilization'] - baseline_results['memory_utilization']) * 100
        
        print(f"\n📊 PERFORMANCE IMPROVEMENTS:")
        print(f"  Throughput Improvement: ({optimized_results['throughput']:,.0f} / {baseline_results['throughput']:,.0f} - 1) × 100 = {throughput_improvement:.1f}%")
        print(f"  Latency Improvement: (1 - {optimized_results['latency']:.2f} / {baseline_results['latency']:.2f}) × 100 = {latency_improvement:.1f}%")
        print(f"  Memory Utilization Change: ({optimized_results['memory_utilization']:.3f} - {baseline_results['memory_utilization']:.3f}) × 100 = {memory_change:.1f}%")
        
        return {
            'baseline': baseline_results,
            'optimized': optimized_results,
            'improvements': {
                'throughput': throughput_improvement,
                'latency': latency_improvement,
                'memory': memory_change
            }
        }
    
    def show_all_calculations(self):
        """Show calculations for all scenarios"""
        print("\n" + "="*100)
        print("COMPREHENSIVE FIGURE CALCULATION BREAKDOWN")
        print("="*100)
        print("This shows exactly how every number in the visualizations was calculated.")
        print("All calculations use synthetic data based on realistic agentic AI patterns.")
        
        all_results = {}
        
        for scenario in self.scenarios:
            results = self.detailed_calculation_breakdown(scenario.name)
            all_results[scenario.name] = results
        
        # Summary table
        print(f"\n{'='*100}")
        print("SUMMARY OF ALL CALCULATIONS")
        print(f"{'='*100}")
        
        print(f"\n{'Scenario':<25} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12}")
        print("-" * 65)
        
        total_baseline = 0
        total_optimized = 0
        
        for scenario_name, results in all_results.items():
            baseline = results['baseline']['throughput']
            optimized = results['optimized']['throughput']
            improvement = results['improvements']['throughput']
            
            print(f"{scenario_name:<25} {baseline:>10,.0f} {optimized:>10,.0f} {improvement:>10.1f}%")
            
            total_baseline += baseline
            total_optimized += optimized
        
        print("-" * 65)
        overall_improvement = (total_optimized / total_baseline - 1) * 100
        print(f"{'TOTAL':<25} {total_baseline:>10,.0f} {total_optimized:>10,.0f} {overall_improvement:>10.1f}%")
        
        print(f"\n🔍 DATA SOURCES:")
        print(f"  • Scenario parameters: Based on typical agentic AI workloads")
        print(f"  • vLLM configurations: From Terradev optimization guide")
        print(f"  • Optimization bonuses: Industry benchmark estimates")
        print(f"  • Memory calculations: Standard GPU memory models")
        
        print(f"\n⚠️  IMPORTANT NOTES:")
        print(f"  • This is SYNTHETIC data for demonstration purposes")
        print(f"  • Real-world results will vary based on actual workloads")
        print(f"  • Calculations are simplified models of complex systems")
        print(f"  • Network effects and other factors are not included")

def main():
    """Main execution function"""
    calculator = FigureCalculationBreakdown()
    calculator.show_all_calculations()

if __name__ == "__main__":
    main()
