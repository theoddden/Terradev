#!/usr/bin/env python3
"""
Comprehensive test suite for SGLang optimization stack

Tests all workload types, hardware profiles, and optimization configurations
to ensure correct auto-detection and parameter application.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from ml_services.sglang_service import (
    SGLangService, SGLangConfig, WorkloadType, SchedulePolicy, 
    AttentionBackend, SpeculativeAlgorithm, DeepEPMode, HardwareProfile
)


class TestSGLangOptimizer:
    """Test SGLang optimization engine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SGLangService()
        self.optimizer = self.service.optimizer
    
    def test_hardware_detection_h100(self):
        """Test H100 hardware detection"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="NVIDIA H100 80GB, 81920\n"
            )
            
            profile = self.optimizer.detect_hardware()
            
            assert profile.gpu_type == "H100"
            assert profile.memory_gb == 80
            assert profile.supports_fp8 == True
            assert profile.default_attention_backend == AttentionBackend.FLASHINFER
    
    def test_hardware_detection_h20(self):
        """Test H20 hardware detection"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="NVIDIA H20 96GB, 98304\n"
            )
            
            profile = self.optimizer.detect_hardware()
            
            assert profile.gpu_type == "H20"
            assert profile.memory_gb == 96
            assert profile.supports_fa3 == True
            assert "swapab" in profile.special_optimizations
    
    def test_model_type_detection_deepseek(self):
        """Test DeepSeek model detection"""
        model_type, config = self.optimizer.detect_model_type("deepseek-ai/DeepSeek-V3")
        
        assert model_type == "deepseek"
        assert config["type"] == "moe"
        assert config["experts"] == 64
        assert config["workload_type"] == WorkloadType.MOE_MODEL
    
    def test_model_type_detection_llama(self):
        """Test Llama model detection"""
        model_type, config = self.optimizer.detect_model_type("meta-llama/Llama-2-7b-hf")
        
        assert model_type == "llama"
        assert config["type"] == "dense"
        assert config["workload_type"] == WorkloadType.AGENTIC_CHAT
    
    def test_workload_type_detection_user_description(self):
        """Test workload type detection from user description"""
        # Test agentic detection
        workload = self.service.detect_workload_type(
            "llama", 
            "Multi-turn conversational AI agent with tool usage"
        )
        assert workload == WorkloadType.AGENTIC_CHAT
        
        # Test batch detection
        workload = self.service.detect_workload_type(
            "llama",
            "Batch processing dataset evaluation"
        )
        assert workload == WorkloadType.BATCH_INFERENCE
        
        # Test low latency detection
        workload = self.service.detect_workload_type(
            "llama",
            "Real-time API with low TTFT requirements"
        )
        assert workload == WorkloadType.LOW_LATENCY


class TestAgenticChatOptimizations:
    """Test agentic/multi-turn chat optimizations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SGLangService()
        self.hardware = HardwareProfile(
            gpu_type="H100",
            memory_gb=80,
            bandwidth_gbps=2000,
            architecture="hopper"
        )
    
    def test_agentic_chat_optimization(self):
        """Test agentic chat workload optimization"""
        config = SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.AGENTIC_CHAT
        )
        
        optimized = self.service._optimize_agentic_chat(config, self.hardware)
        
        # Verify LPM schedule policy
        assert optimized.schedule_policy == SchedulePolicy.LPM
        assert optimized.disable_radix_cache == False
        
        # Verify memory settings
        assert optimized.mem_fraction_static == 0.82
        assert optimized.chunked_prefill_size == 8192
        assert optimized.max_running_requests == 256
        
        # Verify cache-aware routing
        assert optimized.env_vars["SGLANG_CACHE_AWARE_ROUTING"] == "1"
        
        # Verify hardware-specific tuning
        assert optimized.attention_backend == AttentionBackend.FLASHINFER
        assert optimized.kv_cache_dtype == "fp8_e4m3"
    
    def test_agentic_chat_launch_command(self):
        """Test agentic chat launch command generation"""
        config = SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.AGENTIC_CHAT,
            schedule_policy=SchedulePolicy.LPM,
            mem_fraction_static=0.82,
            chunked_prefill_size=8192,
            max_running_requests=256,
            kv_cache_dtype="fp8_e4m3",
            env_vars={"SGLANG_CACHE_AWARE_ROUTING": "1"}
        )
        
        command = self.service.generate_launch_command(config)
        
        assert "--schedule-policy lpm" in command
        assert "--mem-fraction-static 0.82" in command
        assert "--chunked-prefill-size 8192" in command
        assert "--kv-cache-dtype fp8_e4m3" in command
        assert "SGLANG_CACHE_AWARE_ROUTING=1" in command
    
    def test_agentic_chat_performance_summary(self):
        """Test agentic chat performance summary"""
        config = SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.AGENTIC_CHAT,
            schedule_policy=SchedulePolicy.LPM
        )
        
        summary = self.service.get_optimization_summary(config)
        
        assert summary["workload_type"] == "agentic_chat"
        assert "LPM schedule policy for prefix sharing" in summary["optimizations_applied"]
        assert "RadixAttention enabled for cache hits" in summary["optimizations_applied"]
        assert summary["performance_expectations"]["cache_hit_rate"] == "75-90%"
        assert summary["performance_expectations"]["gpu_utilization"] == "95-98%"


class TestBatchInferenceOptimizations:
    """Test high-throughput batch inference optimizations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SGLangService()
        self.hardware = HardwareProfile(
            gpu_type="H100",
            memory_gb=80,
            bandwidth_gbps=2000,
            architecture="hopper"
        )
    
    def test_batch_inference_optimization(self):
        """Test batch inference workload optimization"""
        config = SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.BATCH_INFERENCE
        )
        
        optimized = self.service._optimize_batch_inference(config, self.hardware)
        
        # Verify FCFS schedule policy
        assert optimized.schedule_policy == SchedulePolicy.FCFS
        assert optimized.disable_radix_cache == True
        
        # Verify memory settings
        assert optimized.mem_fraction_static == 0.85
        assert optimized.chunked_prefill_size == 16384
        assert optimized.max_running_requests == 512
        
        # Verify CUDA graphs
        assert optimized.cuda_graph_bs == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        
        # Verify FP8 quantization
        assert optimized.quantization == "fp8"
        assert optimized.kv_cache_dtype == "fp8_e4m3"
        
        # Verify torch compile
        assert optimized.enable_torch_compile == True
    
    def test_batch_inference_launch_command(self):
        """Test batch inference launch command generation"""
        config = SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.BATCH_INFERENCE,
            schedule_policy=SchedulePolicy.FCFS,
            disable_radix_cache=True,
            quantization="fp8",
            enable_torch_compile=True
        )
        
        command = self.service.generate_launch_command(config)
        
        assert "--schedule-policy fcfs" in command
        assert "--disable-radix-cache" in command
        assert "--quantization fp8" in command
        assert "--enable-torch-compile" in command


class TestLowLatencyOptimizations:
    """Test low-latency/real-time optimizations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SGLangService()
        self.hardware = HardwareProfile(
            gpu_type="H100",
            memory_gb=80,
            bandwidth_gbps=2000,
            architecture="hopper"
        )
    
    def test_low_latency_optimization(self):
        """Test low latency workload optimization"""
        config = SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.LOW_LATENCY
        )
        
        optimized = self.service._optimize_low_latency(config, self.hardware)
        
        # Verify LPM schedule policy
        assert optimized.schedule_policy == SchedulePolicy.LPM
        
        # Verify conservative memory settings
        assert optimized.mem_fraction_static == 0.75
        assert optimized.chunked_prefill_size == 4096
        assert optimized.max_running_requests == 64
        
        # Verify small CUDA graphs
        assert optimized.cuda_graph_bs == [1, 2, 4, 8, 16, 32]
        
        # Verify EAGLE3 speculative decoding
        assert optimized.speculative_algorithm == SpeculativeAlgorithm.EAGLE
        assert optimized.speculative_num_steps == 3
        assert optimized.enable_spec_v2 == True
        
        # Verify Spec V2 environment
        assert optimized.env_vars["SGLANG_ENABLE_SPEC_V2"] == "1"
    
    def test_low_latency_launch_command(self):
        """Test low latency launch command generation"""
        config = SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.LOW_LATENCY,
            speculative_algorithm=SpeculativeAlgorithm.EAGLE,
            speculative_num_steps=3,
            enable_spec_v2=True,
            env_vars={"SGLANG_ENABLE_SPEC_V2": "1"}
        )
        
        command = self.service.generate_launch_command(config)
        
        assert "--speculative-algorithm EAGLE" in command
        assert "--speculative-num-steps 3" in command
        assert "SGLANG_ENABLE_SPEC_V2=1" in command


class TestMoEOptimizations:
    """Test MoE model optimizations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SGLangService()
        self.hardware = HardwareProfile(
            gpu_type="H100",
            memory_gb=80,
            bandwidth_gbps=2000,
            architecture="hopper"
        )
    
    def test_deepseek_moe_optimization(self):
        """Test DeepSeek MoE optimization"""
        config = SGLangConfig(
            model_path="deepseek-ai/DeepSeek-V3",
            workload_type=WorkloadType.MOE_MODEL
        )
        
        optimized = self.service._optimize_moe_model(config, self.hardware)
        
        # Verify DeepSeek-specific TP/EP
        assert optimized.tp == 8
        assert optimized.ep == 8
        assert optimized.ep_num_redundant_experts == 32
        
        # Verify MoE settings
        assert optimized.enable_dp_attention == True
        assert optimized.moe_a2a_backend == "deepep"
        assert optimized.moe_runner_backend == "deep_gemm"
        assert optimized.deepep_mode == DeepEPMode.AUTO
        assert optimized.enable_eplb == True
        assert optimized.enable_two_batch_overlap == True
        assert optimized.enable_single_batch_overlap == True
    
    def test_h20_moe_optimization(self):
        """Test H20 MoE optimization"""
        hardware = HardwareProfile(
            gpu_type="H20",
            memory_gb=96,
            bandwidth_gbps=900,
            architecture="hopper",
            supports_fa3=True
        )
        
        config = SGLangConfig(
            model_path="deepseek-ai/DeepSeek-V3",
            workload_type=WorkloadType.MOE_MODEL
        )
        
        optimized = self.service._optimize_moe_model(config, hardware)
        
        # Verify H20-specific optimizations
        assert optimized.moe_runner_backend == "swapab"
        assert optimized.attention_backend == AttentionBackend.FA3
        assert optimized.quantization == "fp8"
        assert optimized.kv_cache_dtype == "fp8_e4m3"
    
    def test_gb200_moe_optimization(self):
        """Test GB200 MoE optimization"""
        hardware = HardwareProfile(
            gpu_type="GB200",
            memory_gb=192,
            bandwidth_gbps=1800,
            architecture="blackwell"
        )
        
        config = SGLangConfig(
            model_path="deepseek-ai/DeepSeek-V3",
            workload_type=WorkloadType.MOE_MODEL
        )
        
        optimized = self.service._optimize_moe_model(config, hardware)
        
        # Verify GB200-specific optimizations
        assert optimized.moe_dense_tp_size == 1
        assert optimized.enable_dp_lm_head == True
        assert optimized.chunked_prefill_size == 524288


class TestPDDisaggregatedOptimizations:
    """Test PD disaggregated serving optimizations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SGLangService()
        self.hardware = HardwareProfile(
            gpu_type="H100",
            memory_gb=80,
            bandwidth_gbps=2000,
            architecture="hopper"
        )
    
    def test_prefill_node_optimization(self):
        """Test prefill node optimization"""
        config = SGLangConfig(
            model_path="deepseek-ai/DeepSeek-V3",
            workload_type=WorkloadType.PD_DISAGGREGATED,
            disaggregation_mode="prefill"
        )
        
        optimized = self.service._optimize_pd_disaggregated(config, self.hardware)
        
        # Verify prefill-specific settings
        assert optimized.mem_fraction_static == 0.85
        assert optimized.chunked_prefill_size == 524288
        assert optimized.max_running_requests == 8192
        assert optimized.disable_radix_cache == True
        assert optimized.deepep_mode == DeepEPMode.NORMAL
        assert optimized.page_size == 1
        
        # Verify prefill environment
        assert optimized.env_vars["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] == "4"
        assert optimized.env_vars["SGLANG_TBO_DEBUG"] == "1"
    
    def test_decode_node_optimization(self):
        """Test decode node optimization"""
        config = SGLangConfig(
            model_path="deepseek-ai/DeepSeek-V3",
            workload_type=WorkloadType.PD_DISAGGREGATED,
            disaggregation_mode="decode"
        )
        
        optimized = self.service._optimize_pd_disaggregated(config, self.hardware)
        
        # Verify decode-specific settings
        assert optimized.mem_fraction_static == 0.82
        assert optimized.max_running_requests == 4096
        assert optimized.deepep_mode == DeepEPMode.LOW_LATENCY
    
    def test_pd_common_settings(self):
        """Test PD common settings"""
        config = SGLangConfig(
            model_path="deepseek-ai/DeepSeek-V3",
            workload_type=WorkloadType.PD_DISAGGREGATED,
            disaggregation_mode="prefill"
        )
        
        optimized = self.service._optimize_pd_disaggregated(config, self.hardware)
        
        # Verify common PD settings
        assert optimized.enable_dp_attention == True
        assert optimized.enable_dp_lm_head == True
        assert optimized.enable_deepep_moe == True


class TestStructuredOutputOptimizations:
    """Test structured output/RAG/JSON decoding optimizations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SGLangService()
        self.hardware = HardwareProfile(
            gpu_type="H100",
            memory_gb=80,
            bandwidth_gbps=2000,
            architecture="hopper"
        )
    
    def test_structured_output_optimization(self):
        """Test structured output optimization"""
        config = SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.STRUCTURED_OUTPUT
        )
        
        optimized = self.service._optimize_structured_output(config, self.hardware)
        
        # Verify structured output settings
        assert optimized.schedule_policy == SchedulePolicy.LPM
        assert optimized.mem_fraction_static == 0.80
        assert optimized.chunked_prefill_size == 8192
        assert optimized.enable_xgrammar == True
        
        # Verify xGrammar environment
        assert optimized.env_vars["SGLANG_XGRAMMAR_ENABLED"] == "1"
    
    def test_rag_workload_optimization(self):
        """Test RAG workload optimization"""
        config = SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.RAG_WORKLOAD
        )
        
        optimized = self.service._optimize_rag_workload(config, self.hardware)
        
        # Verify RAG settings
        assert optimized.schedule_policy == SchedulePolicy.LPM
        assert optimized.mem_fraction_static == 0.80
        assert optimized.chunked_prefill_size == 8192
        assert optimized.disable_radix_cache == False
        
        # Verify RadixAttention environment
        assert optimized.env_vars["SGLANG_RADIX_ATTENTION"] == "1"


class TestConfigurationValidation:
    """Test configuration validation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SGLangService()
    
    def test_fp8_compatibility_warning(self):
        """Test FP8 compatibility warning"""
        config = SGLangConfig(
            model_path="test/model",
            workload_type=WorkloadType.AGENTIC_CHAT,
            quantization="fp8"
        )
        
        with patch.object(self.service.optimizer, 'detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareProfile(
                gpu_type="AMD_MI300X",
                memory_gb=192,
                bandwidth_gbps=3500,
                architecture="cdna",
                supports_fp8=False
            )
            
            warnings = self.service.validate_config(config)
            
            assert any("FP8 quantization not supported" in w for w in warnings)
    
    def test_latency_concurrency_warning(self):
        """Test latency concurrency warning"""
        config = SGLangConfig(
            model_path="test/model",
            workload_type=WorkloadType.LOW_LATENCY,
            max_running_requests=256
        )
        
        warnings = self.service.validate_config(config)
        
        assert any("High max-running-requests may impact latency" in w for w in warnings)
    
    def test_moe_memory_warning(self):
        """Test MoE memory warning"""
        config = SGLangConfig(
            model_path="test/model",
            workload_type=WorkloadType.MOE_MODEL,
            ep_num_redundant_experts=64
        )
        
        warnings = self.service.validate_config(config)
        
        assert any("High redundant expert count may exceed memory" in w for w in warnings)


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SGLangService()
    
    def test_deepseek_agentic_workflow(self):
        """Test DeepSeek agentic workflow optimization"""
        config = self.service.create_optimized_config(
            model_path="deepseek-ai/DeepSeek-V3",
            user_description="Agentic AI assistant with multi-turn conversations"
        )
        
        # Should detect as MoE model but optimize for agentic workload
        assert config.workload_type == WorkloadType.AGENTIC_CHAT
        assert config.tp == 8  # DeepSeek default
        assert config.ep == 8  # DeepSeek default
        assert config.schedule_policy == SchedulePolicy.LPM
        
        # Should have both MoE and agentic optimizations
        assert config.enable_eplb == True  # MoE optimization
        assert config.env_vars.get("SGLANG_CACHE_AWARE_ROUTING") == "1"  # Agentic optimization
    
    def test_llama_batch_processing(self):
        """Test Llama batch processing optimization"""
        config = self.service.create_optimized_config(
            model_path="meta-llama/Llama-2-70b-hf",
            workload_type=WorkloadType.BATCH_INFERENCE
        )
        
        # Verify batch optimizations
        assert config.schedule_policy == SchedulePolicy.FCFS
        assert config.disable_radix_cache == True
        assert config.enable_torch_compile == True
        assert config.quantization == "fp8"
    
    def test_real_time_api_deployment(self):
        """Test real-time API deployment optimization"""
        config = self.service.create_optimized_config(
            model_path="meta-llama/Llama-2-7b-hf",
            user_description="Real-time API with strict latency requirements"
        )
        
        # Verify low latency optimizations
        assert config.workload_type == WorkloadType.LOW_LATENCY
        assert config.speculative_algorithm == SpeculativeAlgorithm.EAGLE
        assert config.enable_spec_v2 == True
        assert config.max_running_requests == 64


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
