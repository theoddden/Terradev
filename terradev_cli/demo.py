#!/usr/bin/env python3
"""
Terradev CLI Demo - Demonstration of parallel provisioning capabilities
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any


class MockTerradevEngine:
    """Mock Terradev engine for demonstration"""

    def __init__(self):
        self.providers = [
            "aws",
            "gcp",
            "azure",
            "runpod",
            "vastai",
            "lambda_labs",
            "coreweave",
            "tensordock",
        ]
        self.gpu_types = ["A100", "V100", "RTX4090", "H100", "RTX3090"]
        self.regions = ["us-east-1", "us-west-2", "eu-west-1", "asia-east-1"]

    async def get_parallel_quotes(
        self, gpu_type: str, region: str = None
    ) -> List[Dict[str, Any]]:
        """Get parallel quotes from all providers"""
        print(
            f"🔍 Getting quotes for {gpu_type} from {len(self.providers)} providers..."
        )

        # Simulate parallel API calls with different response times
        tasks = []
        for provider in self.providers:
            task = self._mock_provider_quote(provider, gpu_type, region)
            tasks.append(task)

        # Execute in parallel
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Filter and format results
        quotes = []
        for result in results:
            if result:
                quotes.append(result)

        print(f"✅ Retrieved {len(quotes)} quotes in {total_time:.2f}s")
        return quotes

    async def _mock_provider_quote(
        self, provider: str, gpu_type: str, region: str
    ) -> Dict[str, Any]:
        """Mock provider quote with realistic latency"""
        # Simulate API latency (1-3 seconds)
        await asyncio.sleep(1 + (hash(provider) % 2))

        # Generate realistic pricing
        base_prices = {
            "A100": {
                "aws": 2.5,
                "gcp": 2.3,
                "azure": 2.4,
                "runpod": 2.0,
                "vastai": 1.8,
                "lambda_labs": 2.1,
                "coreweave": 1.9,
                "tensordock": 1.7,
            },
            "V100": {
                "aws": 2.0,
                "gcp": 1.8,
                "azure": 1.9,
                "runpod": 1.6,
                "vastai": 1.4,
                "lambda_labs": 1.7,
                "coreweave": 1.5,
                "tensordock": 1.3,
            },
            "RTX4090": {
                "aws": 1.5,
                "gcp": 1.4,
                "azure": 1.5,
                "runpod": 1.2,
                "vastai": 1.0,
                "lambda_labs": 1.3,
                "coreweave": 1.1,
                "tensordock": 0.9,
            },
            "H100": {
                "aws": 4.0,
                "gcp": 3.8,
                "azure": 3.9,
                "runpod": 3.5,
                "vastai": 3.2,
                "lambda_labs": 3.6,
                "coreweave": 3.3,
                "tensordock": 3.0,
            },
            "RTX3090": {
                "aws": 1.2,
                "gcp": 1.1,
                "azure": 1.2,
                "runpod": 1.0,
                "vastai": 0.8,
                "lambda_labs": 1.1,
                "coreweave": 0.9,
                "tensordock": 0.7,
            },
        }

        base_price = base_prices.get(gpu_type, {}).get(provider, 1.0)

        # Add some randomness
        import random

        price = base_price * (0.9 + random.random() * 0.2)

        # Calculate optimization score
        score = self._calculate_optimization_score(provider, price, gpu_type)

        return {
            "provider": provider,
            "instance_type": f"{provider}-{gpu_type.lower()}-xlarge",
            "gpu_type": gpu_type,
            "price_per_hour": round(price, 4),
            "region": region or "us-east-1",
            "available": random.random() > 0.1,  # 90% availability
            "latency_ms": 10 + (hash(provider) % 100),
            "optimization_score": score,
            "spot_available": random.random() > 0.3,  # 70% spot availability
            "spot_price": round(price * 0.6, 4) if random.random() > 0.3 else None,
        }

    def _calculate_optimization_score(
        self, provider: str, price: float, gpu_type: str
    ) -> float:
        """Calculate optimization score"""
        # Price factor (lower is better)
        price_score = max(0, 1 - (price / 5.0))

        # Provider reliability factor
        reliability_scores = {
            "aws": 0.95,
            "gcp": 0.93,
            "azure": 0.92,
            "runpod": 0.85,
            "vastai": 0.80,
            "lambda_labs": 0.82,
            "coreweave": 0.88,
            "tensordock": 0.78,
        }
        reliability_score = reliability_scores.get(provider, 0.8)

        # Availability factor
        availability_score = 0.9  # Assumed 90% availability

        # Combined score
        score = price_score * 0.4 + reliability_score * 0.3 + availability_score * 0.3

        return round(score, 3)

    def display_quotes_table(self, quotes: List[Dict[str, Any]]):
        """Display quotes in a formatted table"""
        if not quotes:
            print("No quotes available")
            return

        # Sort by optimization score
        quotes.sort(key=lambda x: x["optimization_score"], reverse=True)

        # Table headers
        headers = [
            "Provider",
            "Instance",
            "GPU",
            "Price/Hr",
            "Spot",
            "Region",
            "Score",
            "Latency",
        ]

        # Table rows
        rows = []
        for quote in quotes[:10]:  # Show top 10
            provider_icons = {
                "aws": "🟧",
                "gcp": "🟦",
                "azure": "🟦",
                "runpod": "🚀",
                "vastai": "🌐",
                "lambda_labs": "⚡",
                "coreweave": "🔷",
                "tensordock": "🔶",
            }

            icon = provider_icons.get(quote["provider"], "☁️")
            spot_price = quote.get("spot_price")
            spot_str = f"${spot_price:.4f}" if spot_price else "N/A"

            rows.append(
                [
                    f"{icon} {quote['provider'].upper()}",
                    quote["instance_type"],
                    quote["gpu_type"],
                    f"${quote['price_per_hour']:.4f}",
                    spot_str,
                    quote["region"],
                    f"{quote['optimization_score']:.3f}",
                    f"{quote['latency_ms']}ms",
                ]
            )

        # Print table
        print("\n📊 QUOTE RESULTS:")
        print("=" * 120)

        # Header
        header_row = " | ".join(h.ljust(12) for h in headers)
        print(header_row)
        print("-" * len(header_row))

        # Data rows
        for row in rows:
            data_row = " | ".join(cell.ljust(12) for cell in row)
            print(data_row)

        print("=" * 120)

    def analyze_savings(self, quotes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential savings"""
        if not quotes:
            return {}

        # Find best price
        best_price = min(q["price_per_hour"] for q in quotes)
        worst_price = max(q["price_per_hour"] for q in quotes)
        avg_price = sum(q["price_per_hour"] for q in quotes) / len(quotes)

        # Calculate savings
        savings_vs_worst = ((worst_price - best_price) / worst_price) * 100
        savings_vs_avg = ((avg_price - best_price) / avg_price) * 100

        # Spot savings
        spot_quotes = [q for q in quotes if q.get("spot_price")]
        spot_savings = 0
        if spot_quotes:
            best_spot = min(q["spot_price"] for q in spot_quotes)
            spot_savings = ((best_price - best_spot) / best_price) * 100

        return {
            "best_price": best_price,
            "worst_price": worst_price,
            "avg_price": avg_price,
            "savings_vs_worst": savings_vs_worst,
            "savings_vs_avg": savings_vs_avg,
            "spot_savings": spot_savings,
            "spot_available": len(spot_quotes) > 0,
        }


async def main():
    """Main demonstration function"""
    print("🚀 TERRADEV CLI - PARALLEL PROVISIONING DEMO")
    print("=" * 60)

    engine = MockTerradevEngine()

    # Demo 1: Parallel quoting for A100
    print("\n🎯 DEMO 1: Parallel A100 Quoting")
    print("-" * 40)

    quotes = await engine.get_parallel_quotes("A100", "us-east-1")
    engine.display_quotes_table(quotes)

    # Analyze savings
    analysis = engine.analyze_savings(quotes)
    if analysis:
        print(f"\n💰 COST ANALYSIS:")
        print(f"   Best Price: ${analysis['best_price']:.4f}/hour")
        print(f"   Worst Price: ${analysis['worst_price']:.4f}/hour")
        print(f"   Average Price: ${analysis['avg_price']:.4f}/hour")
        print(f"   Savings vs Worst: {analysis['savings_vs_worst']:.1f}%")
        print(f"   Savings vs Average: {analysis['savings_vs_avg']:.1f}%")
        if analysis["spot_available"]:
            print(f"   Spot Savings: {analysis['spot_savings']:.1f}%")

    # Demo 2: Sequential vs Parallel comparison
    print(f"\n🎯 DEMO 2: Sequential vs Parallel Performance")
    print("-" * 40)

    # Simulate sequential (slow) approach
    print("🐌 Sequential approach (traditional):")
    sequential_start = time.time()
    for provider in engine.providers[:4]:  # Just 4 for demo
        await asyncio.sleep(2)  # Simulate 2s API call
        print(f"   {provider.upper()}: 2.0s")
    sequential_time = time.time() - sequential_start
    print(f"   Total time: {sequential_time:.1f}s")

    # Simulate parallel (fast) approach
    print(f"\n⚡ Parallel approach (Terradev):")
    parallel_start = time.time()
    tasks = [asyncio.sleep(2) for _ in engine.providers[:4]]
    await asyncio.gather(*tasks)
    parallel_time = time.time() - parallel_start
    print(f"   All providers: {parallel_time:.1f}s")

    speedup = sequential_time / parallel_time
    print(f"\n🚀 Speedup: {speedup:.1f}x faster")

    # Demo 3: Multi-GPU comparison
    print(f"\n🎯 DEMO 3: Multi-GPU Type Comparison")
    print("-" * 40)

    gpu_comparison = {}
    for gpu_type in ["A100", "V100", "RTX4090"]:
        quotes = await engine.get_parallel_quotes(gpu_type)
        if quotes:
            best_price = min(q["price_per_hour"] for q in quotes)
            gpu_comparison[gpu_type] = best_price

    print("📊 GPU Price Comparison:")
    for gpu_type, price in gpu_comparison.items():
        print(f"   {gpu_type}: ${price:.4f}/hour")

    # Demo 4: Regional optimization
    print(f"\n🎯 DEMO 4: Regional Price Optimization")
    print("-" * 40)

    regional_analysis = {}
    for region in ["us-east-1", "us-west-2", "eu-west-1"]:
        quotes = await engine.get_parallel_quotes("A100", region)
        if quotes:
            best_price = min(q["price_per_hour"] for q in quotes)
            regional_analysis[region] = best_price

    print("🌍 Regional Price Comparison:")
    for region, price in regional_analysis.items():
        print(f"   {region}: ${price:.4f}/hour")

    # Summary
    print(f"\n🎉 TERRADEV CLI DEMO COMPLETE")
    print("=" * 60)
    print("✅ Parallel provisioning demonstrated")
    print("✅ Multi-cloud optimization shown")
    print("✅ Cost savings calculated")
    print("✅ Performance improvement measured")
    print("✅ Regional optimization analyzed")

    print(f"\n🚀 KEY BENEFITS:")
    print("   ⚡ 4-6x faster provisioning")
    print("   💰 20%+ cost savings")
    print("   🌐 8+ cloud providers")
    print("   🔒 Secure credential management")
    print("   📊 Real-time analytics")
    print("   🐳 Container orchestration")

    print(f"\n🎯 READY FOR PRODUCTION:")
    print("   pip install terradev-cli")
    print("   terradev configure --provider aws")
    print("   terradev quote --gpu-type A100")
    print("   terradev provision --gpu-type A100 --count 2")


if __name__ == "__main__":
    asyncio.run(main())
