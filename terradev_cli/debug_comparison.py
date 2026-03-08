#!/usr/bin/env python3
from core.mla_vram_estimator import MLA_VRAMEstimator

estimator = MLA_VRAMEstimator()

print('=== Standard vs MLA Comparison ===')
comparison = estimator.compare_standard_vs_mla('deepseek-v3', 16384, 80.0)

print(f'Standard MHA KV: {comparison["standard_mha_estimate"]["kv_cache_gb"]:.2f}GB')
print(f'MLA KV: {comparison["mla_estimate"]["kv_cache_gb"]:.2f}GB')
print(f'Compression Ratio: {comparison["savings"]["kv_cache_compression_ratio"]:.2f}x')
print(f'Cost Savings: {comparison["savings"]["cost_savings_percent"]:.1f}%')

print('\n=== GPU Count Test ===')
estimate = estimator.estimate_vram('llama-3-70b', 8192, 1, 80.0)
print(f'Llama-3-70B 8K context: {estimate.gpu_count} GPUs')
