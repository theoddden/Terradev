#!/usr/bin/env python3
from core.mla_vram_estimator import MLA_VRAMEstimator

estimator = MLA_VRAMEstimator()

print('=== DeepSeek V3 Analysis ===')
estimate = estimator.estimate_vram('deepseek-v3', 4096, 1, 80.0)
print(f'4K context - KV: {estimate.kv_cache_gb:.2f}GB, Total: {estimate.total_gb:.2f}GB, GPUs: {estimate.gpu_count}')

estimate = estimator.estimate_vram('deepseek-v3', 8192, 1, 80.0)
print(f'8K context - KV: {estimate.kv_cache_gb:.2f}GB, Total: {estimate.total_gb:.2f}GB, GPUs: {estimate.gpu_count}')

print('\n=== Test Expectations vs Reality ===')
print('Test expects 4K: 0.6GB KV, 82GB Total, 18 GPUs')
print('Test expects 8K: 1.2GB KV, 82.6GB Total, 18 GPUs')
