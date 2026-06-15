#!/usr/bin/env python3
"""Tests for providers/real_pricing.py"""

import pytest
import asyncio
from terradev_cli.providers.real_pricing import RealGPUPricing


def run_async(coro):
    """Helper to run async functions in sync tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestRealGPUPricing:
    """Test RealGPUPricing"""

    def test_initialization(self):
        """RealGPUPricing initialization"""
        pricing = RealGPUPricing()
        assert pricing.session is None

    def test_get_session(self):
        """Get or create aiohttp session"""
        pricing = RealGPUPricing()
        session = run_async(pricing.get_session())
        assert session is not None
        assert pricing.session is not None

    def test_close_session(self):
        """Close aiohttp session"""
        pricing = RealGPUPricing()
        run_async(pricing.get_session())
        run_async(pricing.close())
        assert pricing.session is None

    def test_get_aws_pricing_a100(self):
        """Get AWS pricing for A100"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_aws_pricing("A100"))
        
        assert len(quotes) > 0
        assert all(q["provider"] == "aws" for q in quotes)
        assert all(q["gpu_type"] == "A100" for q in quotes)
        
        # Should have both spot and on-demand quotes
        spot_quotes = [q for q in quotes if q["spot"]]
        ondemand_quotes = [q for q in quotes if not q["spot"]]
        assert len(spot_quotes) > 0
        assert len(ondemand_quotes) > 0

    def test_get_aws_pricing_h100(self):
        """Get AWS pricing for H100"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_aws_pricing("H100"))
        
        assert len(quotes) > 0
        assert all(q["gpu_type"] == "H100" for q in quotes)

    def test_get_aws_pricing_invalid_gpu(self):
        """Get AWS pricing for invalid GPU returns empty"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_aws_pricing("INVALID_GPU"))
        assert quotes == []

    def test_get_azure_pricing_a100(self):
        """Get Azure pricing for A100"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_azure_pricing("A100"))
        
        assert len(quotes) > 0
        assert all(q["provider"] == "azure" for q in quotes)
        assert all(q["gpu_type"] == "A100" for q in quotes)

    def test_get_azure_pricing_h100(self):
        """Get Azure pricing for H100"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_azure_pricing("H100"))
        
        assert len(quotes) > 0
        assert all(q["gpu_type"] == "H100" for q in quotes)

    def test_get_azure_pricing_invalid_gpu(self):
        """Get Azure pricing for invalid GPU returns empty"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_azure_pricing("INVALID_GPU"))
        assert quotes == []

    def test_get_gcp_pricing_a100(self):
        """Get GCP pricing for A100"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_gcp_pricing("A100"))
        
        assert len(quotes) > 0
        assert all(q["provider"] == "gcp" for q in quotes)
        assert all(q["gpu_type"] == "A100" for q in quotes)

    def test_get_gcp_pricing_h100(self):
        """Get GCP pricing for H100"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_gcp_pricing("H100"))
        
        assert len(quotes) > 0
        assert all(q["gpu_type"] == "H100" for q in quotes)

    def test_get_gcp_pricing_invalid_gpu(self):
        """Get GCP pricing for invalid GPU returns empty"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_gcp_pricing("INVALID_GPU"))
        assert quotes == []

    def test_get_runpod_pricing_a100(self):
        """Get RunPod pricing for A100"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_runpod_pricing("A100"))
        
        assert len(quotes) > 0
        assert all(q["provider"] == "runpod" for q in quotes)
        assert all(q["gpu_type"] == "A100" for q in quotes)
        assert all("tier" in q for q in quotes)

    def test_get_runpod_pricing_rtx4090(self):
        """Get RunPod pricing for RTX4090"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_runpod_pricing("RTX4090"))
        
        assert len(quotes) > 0
        assert all(q["gpu_type"] == "RTX4090" for q in quotes)

    def test_get_runpod_pricing_invalid_gpu(self):
        """Get RunPod pricing for invalid GPU returns empty"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_runpod_pricing("INVALID_GPU"))
        assert quotes == []

    def test_get_vastai_pricing_a100(self):
        """Get Vast.ai pricing for A100"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_vastai_pricing("A100"))
        
        assert len(quotes) > 0
        assert all(q["provider"] == "vastai" for q in quotes)
        assert all(q["gpu_type"] == "A100" for q in quotes)
        assert all("tier" in q for q in quotes)

    def test_get_vastai_pricing_rtx4090(self):
        """Get Vast.ai pricing for RTX4090"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_vastai_pricing("RTX4090"))
        
        assert len(quotes) > 0
        assert all(q["gpu_type"] == "RTX4090" for q in quotes)

    def test_get_vastai_pricing_invalid_gpu(self):
        """Get Vast.ai pricing for invalid GPU returns empty"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_vastai_pricing("INVALID_GPU"))
        assert quotes == []

    def test_get_coreweave_pricing_a100(self):
        """Get CoreWeave pricing for A100"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_coreweave_pricing("A100"))
        
        assert len(quotes) > 0
        assert all(q["provider"] == "coreweave" for q in quotes)
        assert all(q["gpu_type"] == "A100" for q in quotes)
        assert all("tier" in q for q in quotes)

    def test_get_coreweave_pricing_rtx4090(self):
        """Get CoreWeave pricing for RTX4090"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_coreweave_pricing("RTX4090"))
        
        assert len(quotes) > 0
        assert all(q["gpu_type"] == "RTX4090" for q in quotes)

    def test_get_coreweave_pricing_invalid_gpu(self):
        """Get CoreWeave pricing for invalid GPU returns empty"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_coreweave_pricing("INVALID_GPU"))
        assert quotes == []

    def test_get_all_provider_quotes(self):
        """Get quotes from all providers"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_all_provider_quotes("A100"))
        
        assert len(quotes) > 0
        providers = set(q["provider"] for q in quotes)
        assert "aws" in providers
        assert "azure" in providers
        assert "gcp" in providers
        assert "runpod" in providers
        assert "vastai" in providers
        assert "coreweave" in providers

    def test_get_all_provider_quotes_invalid_gpu(self):
        """Get all provider quotes for invalid GPU returns empty"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_all_provider_quotes("INVALID_GPU"))
        assert quotes == []

    def test_aws_pricing_structure(self):
        """AWS pricing has required fields"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_aws_pricing("A100"))
        
        for quote in quotes:
            assert "instance_type" in quote
            assert "gpu_type" in quote
            assert "price_per_hour" in quote
            assert "region" in quote
            assert "available" in quote
            assert "provider" in quote
            assert "spot" in quote
            assert "gpu_count" in quote
            assert "price_per_gpu" in quote

    def test_azure_pricing_structure(self):
        """Azure pricing has required fields"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_azure_pricing("A100"))
        
        for quote in quotes:
            assert "instance_type" in quote
            assert "gpu_type" in quote
            assert "price_per_hour" in quote
            assert "region" in quote
            assert "available" in quote
            assert "provider" in quote
            assert "spot" in quote
            assert "gpu_count" in quote
            assert "price_per_gpu" in quote

    def test_gcp_pricing_structure(self):
        """GCP pricing has required fields"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_gcp_pricing("A100"))
        
        for quote in quotes:
            assert "instance_type" in quote
            assert "gpu_type" in quote
            assert "price_per_hour" in quote
            assert "region" in quote
            assert "available" in quote
            assert "provider" in quote
            assert "spot" in quote
            assert "gpu_count" in quote
            assert "price_per_gpu" in quote

    def test_custom_region_aws(self):
        """AWS pricing with custom region"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_aws_pricing("A100", "us-west-2"))
        
        assert len(quotes) > 0
        assert all(q["region"] == "us-west-2" for q in quotes)

    def test_custom_region_azure(self):
        """Azure pricing with custom region"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_azure_pricing("A100", "westus"))
        
        assert len(quotes) > 0
        assert all(q["region"] == "westus" for q in quotes)

    def test_custom_region_gcp(self):
        """GCP pricing with custom region"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_gcp_pricing("A100", "us-east1"))
        
        assert len(quotes) > 0
        assert all(q["region"] == "us-east1" for q in quotes)

    def test_price_per_gpu_calculation(self):
        """Price per GPU is calculated correctly"""
        pricing = RealGPUPricing()
        quotes = run_async(pricing.get_aws_pricing("A100"))
        
        for quote in quotes:
            expected = quote["price_per_hour"] / quote["gpu_count"]
            assert abs(quote["price_per_gpu"] - expected) < 0.01

    def test_session_cleanup(self):
        """Session is properly cleaned up"""
        pricing = RealGPUPricing()
        run_async(pricing.get_session())
        assert pricing.session is not None
        
        run_async(pricing.close())
        assert pricing.session is None
        
        # Can get new session after close
        run_async(pricing.get_session())
        assert pricing.session is not None
