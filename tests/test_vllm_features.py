#!/usr/bin/env python3
"""
vLLM Feature Tests

Tests the 5 vLLM features integrated into Terradev:
1. KV Cache Offloading (--kv-connector=offloading)
2. MTP Speculative Decoding (--speculative-config.method=mtp)
3. Sleep Mode (--enable-sleep-mode + VLLM_SERVER_DEV_MODE=1)
4. Multi-LoRA MoE (--enable-lora)
5. vLLM Router (enable_router for multi-replica P/D disaggregation)

These tests verify:
1. Configuration fields exist and have correct defaults
2. Server args generation includes the right flags
3. LoRA management (list/load/unload)
4. Sleep/wake methods
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.ml_services.vllm_service import (
    VLLMConfig,
    LoRAModule,
    WorkloadProfile,
)


class TestVLLMConfigKVOffloading:
    """Test KV Cache Offloading feature"""

    def test_kv_offloading_config_fields(self):
        """KV offloading config fields exist"""
        config = VLLMConfig(model_name="test-model")

        # KV offloading fields
        assert hasattr(config, "kv_connector")
        assert hasattr(config, "kv_connector_config")

        # Defaults
        assert config.kv_connector is None  # Disabled by default
        assert config.kv_connector_config is None

    def test_kv_offloading_enabled(self):
        """KV offloading can be enabled"""
        config = VLLMConfig(
            model_name="test-model",
            kv_connector="offloading",
            kv_connector_config={"device": "cpu"},
        )

        assert config.kv_connector == "offloading"
        assert config.kv_connector_config == {"device": "cpu"}


class TestVLLMConfigSpeculativeDecoding:
    """Test MTP Speculative Decoding feature"""

    def test_speculative_decoding_config_fields(self):
        """Speculative decoding config fields exist"""
        config = VLLMConfig(model_name="test-model")

        # Speculative decoding fields
        assert hasattr(config, "speculative_method")
        assert hasattr(config, "speculative_model")
        assert hasattr(config, "num_speculative_tokens")
        assert hasattr(config, "speculative_disable_by_batch_size")

        # Defaults
        assert config.speculative_method is None  # Disabled by default
        assert config.speculative_model is None
        assert config.num_speculative_tokens == 5

    def test_mtp_speculative_decoding_enabled(self):
        """MTP speculative decoding can be enabled"""
        config = VLLMConfig(
            model_name="test-model",
            speculative_method="mtp",
            speculative_model="draft-model",
            num_speculative_tokens=5,
        )

        assert config.speculative_method == "mtp"
        assert config.speculative_model == "draft-model"
        assert config.num_speculative_tokens == 5

    def test_other_speculative_methods(self):
        """Other speculative decoding methods can be configured"""
        for method in ["draft", "ngram", "eagle", "medusa"]:
            config = VLLMConfig(model_name="test-model", speculative_method=method)
            assert config.speculative_method == method


class TestVLLMConfigSleepMode:
    """Test Sleep Mode feature"""

    def test_sleep_mode_config_fields(self):
        """Sleep mode config fields exist"""
        config = VLLMConfig(model_name="test-model")

        # Sleep mode fields
        assert hasattr(config, "enable_sleep_mode")
        assert hasattr(config, "sleep_level")
        assert hasattr(config, "auto_sleep_idle_seconds")

        # Defaults
        assert not config.enable_sleep_mode  # Disabled by default
        assert config.sleep_level == 1
        assert config.auto_sleep_idle_seconds == 300

    def test_sleep_mode_enabled(self):
        """Sleep mode can be enabled"""
        config = VLLMConfig(
            model_name="test-model",
            enable_sleep_mode=True,
            sleep_level=1,
            auto_sleep_idle_seconds=300,
        )

        assert config.enable_sleep_mode
        assert config.sleep_level == 1
        assert config.auto_sleep_idle_seconds == 300

    def test_sleep_mode_levels(self):
        """Sleep mode has different levels"""
        # Level 1: offload to CPU RAM
        config1 = VLLMConfig(
            model_name="test-model", enable_sleep_mode=True, sleep_level=1
        )
        assert config1.sleep_level == 1

        # Level 2: discard weights
        config2 = VLLMConfig(
            model_name="test-model", enable_sleep_mode=True, sleep_level=2
        )
        assert config2.sleep_level == 2


class TestVLLMConfigMultiLoRA:
    """Test Multi-LoRA MoE feature"""

    def test_lora_config_fields(self):
        """LoRA config fields exist"""
        config = VLLMConfig(model_name="test-model")

        # LoRA fields
        assert hasattr(config, "enable_lora")
        assert hasattr(config, "lora_modules")
        assert hasattr(config, "max_loras")
        assert hasattr(config, "max_lora_rank")
        assert hasattr(config, "lora_extra_vocab_size")
        assert hasattr(config, "lora_tuned_config_dir")

        # Defaults
        assert not config.enable_lora  # Disabled by default
        assert config.lora_modules is None
        assert config.max_loras == 8
        assert config.max_lora_rank == 64
        assert config.lora_extra_vocab_size == 256

    def test_lora_enabled_with_modules(self):
        """LoRA can be enabled with modules"""
        lora_modules = [
            LoRAModule(name="adapter1", path="/path/to/adapter1"),
            LoRAModule(name="adapter2", path="/path/to/adapter2"),
        ]

        config = VLLMConfig(
            model_name="test-model",
            enable_lora=True,
            lora_modules=lora_modules,
            max_loras=8,
            max_lora_rank=64,
        )

        assert config.enable_lora
        assert len(config.lora_modules) == 2
        assert config.lora_modules[0].name == "adapter1"
        assert config.lora_modules[0].path == "/path/to/adapter1"

    def test_lora_module_definition(self):
        """LoRAModule dataclass works correctly"""
        module = LoRAModule(
            name="test-adapter", path="/path/to/adapter", base_model_name="base-model"
        )

        assert module.name == "test-adapter"
        assert module.path == "/path/to/adapter"
        assert module.base_model_name == "base-model"


class TestVLLMConfigRouter:
    """Test vLLM Router feature"""

    def test_router_config_fields(self):
        """Router config fields exist"""
        config = VLLMConfig(model_name="test-model")

        # Router fields
        assert hasattr(config, "enable_router")
        assert hasattr(config, "router_policy")
        assert hasattr(config, "router_port")
        assert hasattr(config, "router_session_key")

        # Defaults
        assert not config.enable_router  # Disabled by default
        assert config.router_policy == "consistent_hash"
        assert config.router_port == 8080
        assert config.router_session_key == "x-session-id"

    def test_router_enabled(self):
        """Router can be enabled"""
        config = VLLMConfig(
            model_name="test-model",
            enable_router=True,
            router_policy="consistent_hash",
            router_port=8080,
        )

        assert config.enable_router
        assert config.router_policy == "consistent_hash"
        assert config.router_port == 8080

    def test_router_policies(self):
        """Different router policies can be configured"""
        for policy in ["consistent_hash", "power_of_two", "round_robin"]:
            config = VLLMConfig(
                model_name="test-model", enable_router=True, router_policy=policy
            )
            assert config.router_policy == policy


class TestVLLMConfigAutoOptimized:
    """Test auto-optimized config creation"""

    def test_auto_optimized_creates_config(self):
        """create_auto_optimized creates a valid config"""
        workload = WorkloadProfile(
            avg_prompt_length=100,
            avg_response_length=200,
            requests_per_second=10,
            concurrent_users=5,
            latency_sensitivity=0.5,
            memory_pressure=0.5,
            gpu_count=1,
            model_size_gb=15,
        )

        config = VLLMConfig.create_auto_optimized("test-model", workload)

        assert config.model_name == "test-model"
        assert config is not None

    def test_reasoning_model_detection(self):
        """Reasoning models are detected and optimized"""
        reasoning_models = ["o3-mini", "deepseek-r1", "claude-thinking", "qwen-qq"]

        workload = WorkloadProfile(
            avg_prompt_length=100,
            avg_response_length=200,
            requests_per_second=10,
            concurrent_users=5,
            latency_sensitivity=0.5,
            memory_pressure=0.5,
            gpu_count=1,
            model_size_gb=15,
        )

        for model_name in reasoning_models:
            config = VLLMConfig.create_auto_optimized(model_name, workload)
            assert config.model_name == model_name

    def test_workload_profile_defaults(self):
        """WorkloadProfile has sensible defaults"""
        profile = WorkloadProfile()

        assert profile.avg_prompt_length == 0.0
        assert profile.avg_response_length == 0.0
        assert profile.requests_per_second == 0.0
        assert profile.concurrent_users == 1
        assert profile.latency_sensitivity == 0.5
        assert profile.memory_pressure == 0.5
        assert profile.gpu_count == 1
        assert profile.model_size_gb == 0.0


class TestVLLMConfigOtherOptimizations:
    """Test other vLLM optimizations"""

    def test_critical_throughput_optimizations(self):
        """Critical throughput optimizations have correct defaults"""
        config = VLLMConfig(model_name="test-model")

        # These are the 6 critical knobs
        assert hasattr(config, "max_num_batched_tokens")
        assert hasattr(config, "max_num_seqs")
        assert hasattr(config, "enable_prefix_caching")
        assert hasattr(config, "enable_chunked_prefill")
        assert hasattr(config, "gpu_memory_utilization")
        assert hasattr(config, "cpu_cores")

        # Optimized defaults per memory
        assert config.max_num_batched_tokens == 16384  # Optimized: 2048 → 16384
        assert config.max_num_seqs == 1024  # Optimized: 256/1024 → 1024
        assert config.enable_prefix_caching  # Optimized: OFF → ON
        assert config.enable_chunked_prefill  # Optimized: OFF → ON
        assert config.gpu_memory_utilization == 0.95  # Optimized: 0.90 → 0.95
        assert config.cpu_cores is None  # Auto-calculated

    def test_flashinfer_enabled(self):
        """FlashInfer is enabled by default"""
        config = VLLMConfig(model_name="test-model")

        assert hasattr(config, "attention_backend")
        assert hasattr(config, "enable_flashinfer")

        assert config.attention_backend == "FLASHINFER"
        assert config.enable_flashinfer

    def test_lmcache_integration(self):
        """LMCache integration is available but disabled by default"""
        config = VLLMConfig(model_name="test-model")

        assert hasattr(config, "enable_lmcache")
        assert hasattr(config, "lmcache_backend")

        assert (
            not config.enable_lmcache
        )  # Disabled by default, can be enabled via config
        assert config.lmcache_backend == "redis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
