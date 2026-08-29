"""MCP tool schema definitions."""

from typing import Any, List

try:
    from mcp.types import Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Tool = None

TOOLS = []

if Tool is not None:
    TOOLS = [

        Tool(name='price_intel', description='GPU price intelligence with quantitative analytics. Computes delta (rate of change), gamma (acceleration), and annualized realized volatility on GPU spot/on-demand prices across 21+ providers. Identifies cheapest time windows and provider arbitrage opportunities.', inputSchema={'type': 'object', 'properties': {'gpu_type': {'type': 'string', 'description': 'GPU type to analyze', 'enum': ['H100', 'H200', 'H800', 'A100', 'A10G', 'L40S', 'L4', 'T4', 'RTX4090', 'RTX3090', 'V100', 'V100S', 'A6000', 'MI300X']}, 'days': {'type': 'integer', 'description': 'Number of days of history to analyze', 'minimum': 1, 'default': 7}, 'provider': {'type': 'string', 'description': 'Filter to specific provider (optional)'}}, 'required': ['gpu_type']}),
        Tool(name='cost_analyze', description='Deep cost analysis of current GPU infrastructure: per-provider breakdown, utilization efficiency, waste identification, and optimization potential.', inputSchema={'type': 'object', 'properties': {'days': {'type': 'integer', 'description': 'Lookback period in days', 'default': 30}}}),
        Tool(name='cost_optimize_recommend', description='Generate actionable cost optimization recommendations: spot migration, GPU right-sizing, provider arbitrage, idle shutdown, and density packing.', inputSchema={'type': 'object', 'properties': {'target_savings': {'type': 'number', 'description': 'Target savings percentage (e.g. 0.3 for 30%)'}, 'constraints': {'type': 'object', 'description': 'Constraints (min_gpus, max_latency_ms, required_providers)'}}}),
        Tool(name='cost_simulate', description='Simulate cost optimization scenarios with ROI projections. Compare current vs optimized infrastructure costs.', inputSchema={'type': 'object', 'properties': {'scenario': {'type': 'object', 'description': 'Scenario config (gpu_type, provider, count, spot, hours)'}, 'compare_with': {'type': 'object', 'description': 'Current config to compare against'}}, 'required': ['scenario']}),
        Tool(name='price_trends', description='Get GPU price trend analysis with delta (rate of change), gamma (acceleration), and annualized volatility. Identifies cheapest time windows.', inputSchema={'type': 'object', 'properties': {'gpu_type': {'type': 'string', 'description': 'GPU type', 'enum': ['H100', 'H200', 'H800', 'A100', 'A10G', 'L40S', 'L4', 'T4', 'RTX4090', 'V100S', 'A6000', 'MI300X']}, 'hours': {'type': 'integer', 'description': 'Hours of history', 'default': 24}}, 'required': ['gpu_type']}),
        Tool(name='price_spot_risk', description='Spot instance risk assessment per provider. Returns interruption probability, mean time to interruption, and recommended mitigation.', inputSchema={'type': 'object', 'properties': {'gpu_type': {'type': 'string', 'description': 'GPU type'}, 'provider': {'type': 'string', 'description': "Provider to assess (or 'all')"}}, 'required': ['gpu_type']}),
    ]
