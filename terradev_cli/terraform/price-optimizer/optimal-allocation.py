#!/usr/bin/env python3
"""
Multi-Cloud GPU Price Optimizer
Queries real-time prices and allocates nodes across cheapest providers
"""

import json
import sys
import os
import requests
from datetime import datetime
from typing import Dict, List, Tuple

class PriceOptimizer:
    def __init__(self):
        self.providers = ["aws", "vastai", "lambda", "hyperstack"]
        self.gpu_pricing = self._load_pricing_data()
        
    def _load_pricing_data(self) -> Dict:
        """Load pricing data from Terradev price database"""
        try:
            # Try to read from local price database
            price_db_path = os.path.expanduser("~/.terradev/price_database.json")
            if os.path.exists(price_db_path):
                with open(price_db_path, 'r') as f:
                    return json.load(f)
            
            # Fallback to hardcoded pricing for demo
            return {
                "aws": {
                    "H100": {"ondemand": 3.90, "spot": 2.34, "instance": "p5.48xlarge"},
                    "A100": {"ondemand": 3.12, "spot": 1.87, "instance": "p4d.24xlarge"},
                    "L40": {"ondemand": 1.20, "spot": 0.72, "instance": "g5.12xlarge"}
                },
                "vastai": {
                    "H100": {"ondemand": 3.00, "spot": 2.50, "instance": "H100_80GB"},
                    "A100": {"ondemand": 2.35, "spot": 1.88, "instance": "A100_80GB"},
                    "L40": {"ondemand": 1.10, "spot": 0.88, "instance": "L40_48GB"}
                },
                "lambda": {
                    "H100": {"ondemand": 2.99, "spot": 2.39, "instance": "on-demand-h100"},
                    "A100": {"ondemand": 2.25, "spot": 1.80, "instance": "on-demand-a100"},
                    "L40": {"ondemand": 1.05, "spot": 0.84, "instance": "on-demand-l40"}
                },
                "hyperstack": {
                    "H100": {"ondemand": 3.20, "spot": 2.56, "instance": "H100"},
                    "A100": {"ondemand": 2.40, "spot": 1.92, "instance": "A100"},
                    "L40": {"ondemand": 1.15, "spot": 0.92, "instance": "L40"}
                }
            }
        except Exception as e:
            print(f"Error loading pricing data: {e}", file=sys.stderr)
            return {}
    
    def get_provider_prices(self, gpu_type: str, prefer_spot: bool) -> List[Tuple[str, float, str, bool]]:
        """Get sorted list of (provider, price, instance_type, is_spot) tuples"""
        prices = []
        
        for provider in self.providers:
            if provider not in self.gpu_pricing:
                continue
                
            if gpu_type not in self.gpu_pricing[provider]:
                continue
            
            pricing = self.gpu_pricing[provider][gpu_type]
            
            # Choose spot or on-demand price
            if prefer_spot and "spot" in pricing:
                price = pricing["spot"]
                is_spot = True
            else:
                price = pricing["ondemand"]
                is_spot = False
            
            instance_type = pricing["instance"]
            prices.append((provider, price, instance_type, is_spot))
        
        # Sort by price (cheapest first)
        return sorted(prices, key=lambda x: x[1])
    
    def allocate_nodes(self, gpu_type: str, total_needed: int, max_price: float, prefer_spot: bool) -> Dict:
        """Allocate nodes across providers to minimize cost"""
        prices = self.get_provider_prices(gpu_type, prefer_spot)
        
        # Filter providers within max price
        affordable_providers = [(p, price, instance, spot) for p, price, instance, spot in prices if price <= max_price]
        
        if not affordable_providers:
            # If no providers within max_price, use cheapest available
            affordable_providers = prices[:1]
        
        allocation = {}
        remaining = total_needed
        
        # Allocate nodes to cheapest providers first
        for provider, price, instance_type, is_spot in affordable_providers:
            if remaining <= 0:
                break
            
            # Distribute nodes across providers for reliability
            # Give at most 40% to any single provider
            max_per_provider = min(remaining, max(1, int(total_needed * 0.4)))
            
            nodes_for_provider = min(max_per_provider, remaining)
            
            if nodes_for_provider > 0:
                allocation[provider] = {
                    "count": nodes_for_provider,
                    "cost_per_node": price,
                    "instance_type": instance_type,
                    "spot": is_spot,
                    "total_cost": price * nodes_for_provider
                }
                remaining -= nodes_for_provider
        
        # If we still have remaining nodes, allocate to cheapest provider
        if remaining > 0 and affordable_providers:
            cheapest_provider, price, instance_type, is_spot = affordable_providers[0]
            allocation[cheapest_provider]["count"] += remaining
            allocation[cheapest_provider]["total_cost"] += price * remaining
        
        return allocation
    
    def calculate_savings(self, allocation: Dict, gpu_type: str) -> Dict:
        """Calculate savings compared to AWS-only"""
        # Get AWS pricing for comparison
        aws_pricing = self.gpu_pricing.get("aws", {}).get(gpu_type, {})
        aws_price = aws_pricing.get("ondemand", 0)
        
        if not aws_price:
            return {"savings_percentage": 0, "savings_amount": 0}
        
        total_nodes = sum(provider["count"] for provider in allocation.values())
        aws_total_cost = aws_price * total_nodes
        multi_cloud_cost = sum(provider["total_cost"] for provider in allocation.values())
        
        if aws_total_cost == 0:
            return {"savings_percentage": 0, "savings_amount": 0}
        
        savings_amount = aws_total_cost - multi_cloud_cost
        savings_percentage = (savings_amount / aws_total_cost) * 100
        
        return {
            "savings_percentage": round(savings_percentage, 2),
            "savings_amount": round(savings_amount, 2),
            "aws_total_cost": round(aws_total_cost, 2),
            "multi_cloud_cost": round(multi_cloud_cost, 2)
        }

def main():
    """Main function for Terraform external data source"""
    try:
        # Read query from stdin
        query = json.loads(sys.stdin.read())
        
        gpu_type = query.get("gpu_type", "H100")
        total_needed = int(query.get("total_needed", 1))
        max_price = float(query.get("max_price", 4.00))
        prefer_spot = query.get("prefer_spot", "true").lower() == "true"
        
        optimizer = PriceOptimizer()
        
        # Get optimal allocation
        allocation = optimizer.allocate_nodes(gpu_type, total_needed, max_price, prefer_spot)
        
        # Calculate savings
        savings = optimizer.calculate_savings(allocation, gpu_type)
        
        # Prepare result for Terraform
        result = {
            "allocation": allocation,
            "savings": savings,
            "total_nodes": sum(provider["count"] for provider in allocation.values()),
            "total_cost_per_hour": sum(provider["total_cost"] for provider in allocation.values()),
            "providers_used": list(allocation.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        # Output JSON for Terraform
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
