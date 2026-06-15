#!/usr/bin/env python3
"""Tests for providers/demo_mode.py"""

import pytest
import asyncio
from terradev_cli.providers.demo_mode import DemoModeProvider, DemoModeManager


def run_async(coro):
    """Helper to run async functions in sync tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestDemoModeProvider:
    """Test DemoModeProvider"""

    def test_initialization(self):
        """Provider initialization with name"""
        provider = DemoModeProvider("runpod")
        assert provider.name == "runpod"
        assert provider.demo_data is not None

    def test_demo_pricing_structure(self):
        """Demo pricing data has expected structure"""
        provider = DemoModeProvider("runpod")
        assert "runpod" in provider.demo_data
        assert "A100" in provider.demo_data["runpod"]
        assert "H100" in provider.demo_data["runpod"]
        assert "price" in provider.demo_data["runpod"]["A100"]
        assert "note" in provider.demo_data["runpod"]["A100"]

    def test_get_demo_quotes_valid_gpu(self):
        """Get demo quotes for valid GPU type"""
        provider = DemoModeProvider("runpod")
        result = run_async(provider.get_demo_quotes("A100"))
        assert len(result) == 1
        assert result[0]["provider"] == "Runpod"
        assert result[0]["gpu_type"] == "A100"
        assert result[0]["demo_mode"] is True
        assert "DEMO DATA" in result[0]["note"]

    def test_get_demo_quotes_invalid_gpu(self):
        """Get demo quotes for invalid GPU type returns empty"""
        provider = DemoModeProvider("runpod")
        result = run_async(provider.get_demo_quotes("INVALID_GPU"))
        assert result == []

    def test_get_demo_quotes_h100(self):
        """Get demo quotes for H100"""
        provider = DemoModeProvider("aws")
        result = run_async(provider.get_demo_quotes("H100"))
        assert len(result) == 1
        assert result[0]["gpu_type"] == "H100"
        assert result[0]["provider"] == "Aws"

    def test_multiple_providers(self):
        """Different providers have different pricing"""
        runpod = DemoModeProvider("runpod")
        aws = DemoModeProvider("aws")
        
        runpod_quotes = run_async(runpod.get_demo_quotes("A100"))
        aws_quotes = run_async(aws.get_demo_quotes("A100"))
        
        assert runpod_quotes[0]["price"] != aws_quotes[0]["price"]
        assert runpod_quotes[0]["region"] != aws_quotes[0]["region"]

    def test_demo_data_has_all_providers(self):
        """Demo data includes all expected providers"""
        provider = DemoModeProvider("runpod")
        expected_providers = [
            "runpod", "vastai", "aws", "azure", "gcp",
            "coreweave", "lambda_labs", "tensordock", "oracle", "crusoe"
        ]
        for provider_name in expected_providers:
            assert provider_name in provider.demo_data


class TestDemoModeManager:
    """Test DemoModeManager"""

    def test_initialization(self):
        """Manager initializes with all providers"""
        manager = DemoModeManager()
        assert len(manager.providers) == 10
        assert "runpod" in manager.providers
        assert "aws" in manager.providers
        assert "gcp" in manager.providers

    def test_get_all_demo_quotes(self):
        """Get demo quotes from all providers"""
        manager = DemoModeManager()
        result = run_async(manager.get_all_demo_quotes("A100"))
        
        # Should get quotes from all providers that have A100 data
        assert len(result) > 0
        assert all(q["gpu_type"] == "A100" for q in result)
        assert all(q["demo_mode"] is True for q in result)

    def test_get_all_demo_quotes_h100(self):
        """Get demo quotes for H100 from all providers"""
        manager = DemoModeManager()
        result = run_async(manager.get_all_demo_quotes("H100"))
        
        assert len(result) > 0
        assert all(q["gpu_type"] == "H100" for q in result)

    def test_get_all_demo_quotes_invalid_gpu(self):
        """Get demo quotes for invalid GPU returns empty"""
        manager = DemoModeManager()
        result = run_async(manager.get_all_demo_quotes("INVALID_GPU"))
        assert result == []

    def test_print_demo_disclaimer(self, capsys):
        """Print demo mode disclaimer"""
        manager = DemoModeManager()
        manager.print_demo_disclaimer()
        
        captured = capsys.readouterr()
        assert "DEMO MODE" in captured.out
        assert "STATIC PRICING DATA" in captured.out
        assert "NOT REAL-TIME PRICING" in captured.out

    def test_provider_instances(self):
        """All providers are DemoModeProvider instances"""
        manager = DemoModeManager()
        for provider in manager.providers.values():
            assert isinstance(provider, DemoModeProvider)

    def test_quotes_have_timestamp(self):
        """Demo quotes include timestamp"""
        manager = DemoModeManager()
        result = run_async(manager.get_all_demo_quotes("A100"))
        
        for quote in result:
            assert "timestamp" in quote
            assert quote["timestamp"] is not None

    def test_quotes_have_availability(self):
        """Demo quotes have availability field"""
        manager = DemoModeManager()
        result = run_async(manager.get_all_demo_quotes("A100"))
        
        for quote in result:
            assert quote["availability"] == "demo"
