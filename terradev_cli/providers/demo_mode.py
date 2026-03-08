#!/usr/bin/env python3
"""
Demo Mode Provider - Static demonstration data for testing and showcase
CLEARLY SEPARATED from real provider APIs - only used with explicit --demo flag
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

class DemoModeProvider:
    """Demo mode provider with clearly marked static data for demonstration purposes only"""
    
    def __init__(self, provider_name: str):
        self.name = provider_name
        self.demo_data = self._get_demo_pricing()
    
    def _get_demo_pricing(self) -> Dict[str, Dict[str, Any]]:
        """Demo pricing data - CLEARLY MARKED AS DEMONSTRATION ONLY"""
        return {
            "runpod": {
                "A100": {"price": 1.64, "region": "us-east", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 3.49, "region": "us-east", "note": "DEMO DATA - NOT REAL PRICING"},
            },
            "vastai": {
                "A100": {"price": 1.10, "region": "us-east", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 2.80, "region": "us-east", "note": "DEMO DATA - NOT REAL PRICING"},
            },
            "aws": {
                "A100": {"price": 4.80, "region": "us-east-1", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 15.50, "region": "us-east-1", "note": "DEMO DATA - NOT REAL PRICING"},
            },
            "azure": {
                "A100": {"price": 9.52, "region": "eastus", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 32.77, "region": "eastus", "note": "DEMO DATA - NOT REAL PRICING"},
            },
            "gcp": {
                "A100": {"price": 3.67, "region": "us-central1", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 45.00, "region": "us-central1", "note": "DEMO DATA - NOT REAL PRICING"},
            },
            "coreweave": {
                "A100": {"price": 2.21, "region": "us-east-04e", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 4.76, "region": "us-east-04e", "note": "DEMO DATA - NOT REAL PRICING"},
            },
            "lambda_labs": {
                "A100": {"price": 1.29, "region": "us-east-1", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 2.49, "region": "us-east-1", "note": "DEMO DATA - NOT REAL PRICING"},
            },
            "tensordock": {
                "A100": {"price": 1.50, "region": "us-east", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 3.20, "region": "us-east", "note": "DEMO DATA - NOT REAL PRICING"},
            },
            "oracle": {
                "A100": {"price": 3.50, "region": "us-ashburn-1", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 5.00, "region": "us-ashburn-1", "note": "DEMO DATA - NOT REAL PRICING"},
            },
            "crusoe": {
                "A100": {"price": 2.20, "region": "us-east", "note": "DEMO DATA - NOT REAL PRICING"},
                "H100": {"price": 4.50, "region": "us-east", "note": "DEMO DATA - NOT REAL PRICING"},
            },
        }
    
    async def get_demo_quotes(self, gpu_type: str) -> List[Dict[str, Any]]:
        """Get demo quotes - CLEARLY MARKED AS DEMONSTRATION DATA"""
        provider_data = self.demo_data.get(self.name, {})
        gpu_data = provider_data.get(gpu_type)
        
        if not gpu_data:
            return []
        
        return [{
            "provider": self.name.title(),
            "price": gpu_data["price"],
            "gpu_type": gpu_type,
            "region": gpu_data["region"],
            "availability": "demo",
            "demo_mode": True,
            "note": gpu_data["note"],
            "timestamp": datetime.now().isoformat(),
        }]

class DemoModeManager:
    """Manages demo mode across all providers"""
    
    def __init__(self):
        self.providers = {
            "runpod": DemoModeProvider("runpod"),
            "vastai": DemoModeProvider("vastai"),
            "aws": DemoModeProvider("aws"),
            "azure": DemoModeProvider("azure"),
            "gcp": DemoModeProvider("gcp"),
            "coreweave": DemoModeProvider("coreweave"),
            "lambda_labs": DemoModeProvider("lambda_labs"),
            "tensordock": DemoModeProvider("tensordock"),
            "oracle": DemoModeProvider("oracle"),
            "crusoe": DemoModeProvider("crusoe"),
        }
    
    async def get_all_demo_quotes(self, gpu_type: str) -> List[Dict[str, Any]]:
        """Get demo quotes from all providers"""
        tasks = []
        for provider_name, provider in self.providers.items():
            tasks.append(provider.get_demo_quotes(gpu_type))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_quotes = []
        
        for result in results:
            if isinstance(result, list):
                all_quotes.extend(result)
        
        return all_quotes
    
    def print_demo_disclaimer(self):
        """Print clear demo mode disclaimer"""
        print("⚠️  DEMO MODE - STATIC PRICING DATA")
        print("=" * 50)
        print("📊 The following quotes are DEMONSTRATION DATA ONLY")
        print("🚫 NOT REAL-TIME PRICING FROM CLOUD PROVIDERS")
        print("🔧 For real pricing, configure your API credentials:")
        print("   terradev configure --provider <provider>")
        print("   terradev quote --gpu-type <gpu>")
        print("=" * 50)
