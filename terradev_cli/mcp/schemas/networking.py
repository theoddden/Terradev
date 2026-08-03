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

        Tool(name='egress_cheapest_route', description='Find the cheapest egress route between cloud providers/regions for model weights or dataset transfer. Supports multi-hop routing.', inputSchema={'type': 'object', 'properties': {'source_provider': {'type': 'string', 'description': 'Source cloud provider (e.g. aws, gcp, azure)'}, 'source_region': {'type': 'string', 'description': 'Source region (e.g. us-east-1)'}, 'dest_provider': {'type': 'string', 'description': 'Destination cloud provider'}, 'dest_region': {'type': 'string', 'description': 'Destination region'}, 'size_gb': {'type': 'number', 'description': 'Transfer size in GB'}}, 'required': ['source_provider', 'source_region', 'dest_provider', 'dest_region', 'size_gb']}),
        Tool(name='egress_optimize_staging', description='Optimize dataset or model staging across regions by finding the cheapest transfer plan. Integrates with the dataset stager for parallel uploads.', inputSchema={'type': 'object', 'properties': {'source_uri': {'type': 'string', 'description': 'Source data URI (s3://, gs://, local path, or HF dataset ID)'}, 'target_regions': {'type': 'array', 'description': 'Target regions as provider:region strings', 'items': {'type': 'string'}}, 'size_gb': {'type': 'number', 'description': 'Approximate data size in GB'}}, 'required': ['source_uri', 'target_regions', 'size_gb']}),
    ]
