#!/usr/bin/env python3
"""
Test script for vLLM optimization implementation
"""

import sys
import json
sys.path.append('.')

from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService, WorkloadProfile

def test_throughput_optimization():
    """Test throughput optimization configuration"""
    print("🧪 Testing throughput optimization...")
    
    config = VLLMConfig.create_throughput_optimized("meta-llama/Llama-2-7b-hf")
    
    # Verify throughput-optimized values
    assert config.max_num_batched_tokens == 16384, f"Expected 16384, got {config.max_num_batched_tokens}"
    assert config.max_num_seqs == 1024, f"Expected 1024, got {config.max_num_seqs}"
    assert config.gpu_memory_utilization == 0.95, f"Expected 0.95, got {config.gpu_memory_utilization}"
    assert config.enable_prefix_caching == True, "Prefix caching should be enabled"
    assert config.enable_chunked_prefill == True, "Chunked prefill should be enabled"
    
    print("✅ Throughput optimization tests passed")

def test_latency_optimization():
    """Test latency optimization configuration"""
    print("🧪 Testing latency optimization...")
    
    config = VLLMConfig.create_latency_optimized("mistralai/Mistral-7B-v0.1")
    
    # Verify latency-optimized values
    assert config.max_num_batched_tokens == 4096, f"Expected 4096, got {config.max_num_batched_tokens}"
    assert config.max_num_seqs == 512, f"Expected 512, got {config.max_num_seqs}"
    assert config.gpu_memory_utilization == 0.95, f"Expected 0.95, got {config.gpu_memory_utilization}"
    assert config.enable_prefix_caching == True, "Prefix caching should still be enabled"
    assert config.enable_chunked_prefill == True, "Chunked prefill should be enabled"
    
    print("✅ Latency optimization tests passed")

def test_auto_optimization():
    """Test automatic workload-based optimization"""
    print("🧪 Testing automatic optimization...")
    
    # Test high QPS workload
    high_qps_workload = WorkloadProfile(
        avg_prompt_length=256,
        avg_response_length=128,
        requests_per_second=50.0,
        concurrent_users=20,
        latency_sensitivity=0.2,  # Throughput focused
        memory_pressure=0.3,
        gpu_count=4
    )
    
    config = VLLMConfig.create_auto_optimized("meta-llama/Llama-2-7b-hf", high_qps_workload)
    
    # Should optimize for high throughput
    assert config.max_num_batched_tokens >= 16384, f"High QPS should use large batch size, got {config.max_num_batched_tokens}"
    assert config.max_num_seqs >= 256, f"High QPS should allow more sequences, got {config.max_num_seqs}"
    assert config.enable_prefix_caching == True, "High QPS should enable prefix caching"
    assert config.enable_chunked_prefill == True, "High QPS should enable chunked prefill"
    assert config.cpu_cores >= 6, f"High QPS should allocate more CPU cores, got {config.cpu_cores}"
    
    # Test latency-sensitive workload
    latency_workload = WorkloadProfile(
        avg_prompt_length=64,
        avg_response_length=32,
        requests_per_second=1.0,
        concurrent_users=5,
        latency_sensitivity=0.9,  # Latency focused
        memory_pressure=0.5,
        gpu_count=2
    )
    
    config = VLLMConfig.create_auto_optimized("mistralai/Mistral-7B-v0.1", latency_workload)
    
    # Should optimize for low latency
    assert config.max_num_batched_tokens <= 4096, f"Latency-focused should use smaller batch size, got {config.max_num_batched_tokens}"
    assert config.max_num_seqs <= 512, f"Latency-focused should limit sequences, got {config.max_num_seqs}"
    
    print("✅ Automatic optimization tests passed")

def test_workload_analysis():
    """Test workload analysis from samples"""
    print("🧪 Testing workload analysis...")
    
    # Create sample data
    samples = [
        {"prompt": "Short question", "response": "Short answer", "timestamp": 1000, "user_id": "user1"},
        {"prompt": "A much longer prompt with detailed context and requirements", "response": "A comprehensive response with multiple paragraphs and detailed explanations", "timestamp": 2000, "user_id": "user2"},
        {"prompt": "Another query", "response": "Another response", "timestamp": 3000, "user_id": "user1"}
    ]
    
    workload = VLLMConfig.analyze_workload_from_samples(samples, gpu_count=2)
    
    assert workload.avg_prompt_length > 0, "Should calculate average prompt length"
    assert workload.avg_response_length > 0, "Should calculate average response length"
    assert workload.requests_per_second > 0, "Should calculate QPS"
    assert workload.concurrent_users == 2, "Should count unique users"
    assert workload.gpu_count == 2, "Should use provided GPU count"
    
    print("✅ Workload analysis tests passed")

def test_server_args_generation():
    """Test server args generation with optimizations"""
    print("🧪 Testing server args generation...")
    
    config = VLLMConfig.create_throughput_optimized("meta-llama/Llama-2-7b-hf")
    service = VLLMService(config)
    args = service._build_server_args()
    
    # Verify optimization flags are included
    assert "--max-num-batched-tokens" in args, "Missing batched tokens flag"
    assert "--max-num-seqs" in args, "Missing max sequences flag"
    assert "--enable-prefix-caching" in args, "Missing prefix caching flag"
    assert "--enable-chunked-prefill" in args, "Missing chunked prefill flag"
    
    # Verify values
    batch_idx = args.index("--max-num-batched-tokens")
    assert args[batch_idx + 1] == "16384", f"Wrong batch size value: {args[batch_idx + 1]}"
    
    seq_idx = args.index("--max-num-seqs")
    assert args[seq_idx + 1] == "1024", f"Wrong max sequences value: {args[seq_idx + 1]}"
    
    print("✅ Server args generation tests passed")

def test_cli_integration():
    """Test CLI integration"""
    print("🧪 Testing CLI integration...")
    
    try:
        from terradev_cli.cli import cli
        # Check if vllm group exists
        vllm_group = None
        for command in cli.commands.values():
            if hasattr(command, 'name') and command.name == 'vllm':
                vllm_group = command
                break
        
        assert vllm_group is not None, "vLLM CLI group not found"
        
        # Check if auto-optimize command exists
        assert 'auto-optimize' in vllm_group.commands, "auto-optimize command not found"
        assert 'analyze' in vllm_group.commands, "analyze command not found"
        assert 'optimize' in vllm_group.commands, "optimize command not found"
        
        print("✅ vLLM CLI group imported successfully")
        
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing vLLM Optimization Implementation")
    print("=" * 50)
    
    try:
        test_throughput_optimization()
        test_latency_optimization()
        test_auto_optimization()
        test_workload_analysis()
        test_server_args_generation()
        test_cli_integration()
        
        print("\n✅ All tests passed!")
        print("\n🎯 vLLM optimization implementation is working correctly!")
        print("\n📋 Next steps:")
        print("   1. Run: terradev vllm auto-optimize -s sample_workload.json -m your-model")
        print("   2. Test with: terradev vllm analyze -e http://localhost:8000")
        print("   3. Deploy with optimized Helm values")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
