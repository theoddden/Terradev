#!/usr/bin/env python3
import asyncio
from core.weight_streaming_benchmarks import WeightStreamingBenchmarks

async def debug_streaming():
    benchmarks = WeightStreamingBenchmarks()
    
    print('=== Debug Weight Streaming Simulations ===')
    
    # Test basic streaming simulation
    streaming_time = await benchmarks._simulate_weight_streaming(
        total_layers=32,
        enable_streaming=True
    )
    print(f'32 layers streaming: {streaming_time:.1f}s')
    
    traditional_time = await benchmarks._simulate_weight_streaming(
        total_layers=32,
        enable_streaming=False
    )
    print(f'32 layers traditional: {traditional_time:.1f}s')
    
    improvement = traditional_time / streaming_time
    print(f'Improvement factor: {improvement:.1f}x')
    
    # Test parallel downloads
    parallel_1 = await benchmarks._simulate_weight_streaming(
        total_layers=80,
        enable_streaming=True,
        parallel_downloads=1
    )
    parallel_8 = await benchmarks._simulate_weight_streaming(
        total_layers=80,
        enable_streaming=True,
        parallel_downloads=8
    )
    print(f'Parallel 1x: {parallel_1:.1f}s')
    print(f'Parallel 8x: {parallel_8:.1f}s')
    print(f'Parallel improvement: {parallel_1/parallel_8:.1f}x')

asyncio.run(debug_streaming())
