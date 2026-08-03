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

        Tool(name='governance_request_consent', description='Request user consent for data movement across cloud regions. GDPR/SOC2 compliant consent tracking with audit trail.', inputSchema={'type': 'object', 'properties': {'user_id': {'type': 'string', 'description': 'User ID requesting consent'}, 'consent_type': {'type': 'string', 'description': 'Consent type', 'enum': ['data_staging', 'cross_region', 'third_party', 'model_training']}, 'dataset_name': {'type': 'string', 'description': 'Dataset being moved'}, 'source_location': {'type': 'string', 'description': 'Source region/provider'}, 'target_location': {'type': 'string', 'description': 'Target region/provider'}, 'purpose': {'type': 'string', 'description': 'Purpose of data movement'}}, 'required': ['user_id', 'consent_type', 'dataset_name', 'purpose']}),
        Tool(name='governance_record_consent', description='Record a consent response (granted or denied) for a pending consent request.', inputSchema={'type': 'object', 'properties': {'request_id': {'type': 'string', 'description': 'Consent request ID'}, 'user_id': {'type': 'string', 'description': 'User ID'}, 'granted': {'type': 'boolean', 'description': 'Whether consent was granted'}, 'conditions': {'type': 'array', 'description': 'Conditions attached to consent', 'items': {'type': 'string'}}}, 'required': ['request_id', 'user_id', 'granted']}),
        Tool(name='governance_evaluate_opa', description='Evaluate OPA (Open Policy Agent) policies for data access. Checks region restrictions, classification rules, and compliance requirements.', inputSchema={'type': 'object', 'properties': {'user_id': {'type': 'string', 'description': 'User ID to evaluate'}, 'dataset_name': {'type': 'string', 'description': 'Dataset name'}, 'action': {'type': 'string', 'description': 'Action to evaluate', 'enum': ['read', 'write', 'move', 'delete', 'train']}, 'target_location': {'type': 'string', 'description': 'Target location for the action'}}, 'required': ['user_id', 'dataset_name', 'action']}),
        Tool(name='governance_move_data', description='Move data with full governance audit trail. Requires prior consent and OPA policy approval. Tracks integrity, encryption, and compliance.', inputSchema={'type': 'object', 'properties': {'user_id': {'type': 'string', 'description': 'User ID'}, 'consent_request_id': {'type': 'string', 'description': 'Approved consent request ID'}, 'dataset_name': {'type': 'string', 'description': 'Dataset to move'}, 'source_location': {'type': 'string', 'description': 'Source location'}, 'target_location': {'type': 'string', 'description': 'Target location'}}, 'required': ['user_id', 'consent_request_id', 'dataset_name', 'source_location', 'target_location']}),
        Tool(name='governance_movement_history', description='Get data movement audit log. Filter by user, dataset, or time range.', inputSchema={'type': 'object', 'properties': {'user_id': {'type': 'string', 'description': 'Filter by user ID'}, 'dataset_name': {'type': 'string', 'description': 'Filter by dataset'}, 'limit': {'type': 'integer', 'description': 'Max records', 'default': 50}}}),
        Tool(name='governance_compliance_report', description='Generate comprehensive compliance report: consent stats, policy evaluations, data movements, violations. For GDPR/SOC2/HIPAA audits.', inputSchema={'type': 'object', 'properties': {'start_date': {'type': 'string', 'description': 'Start date (ISO format, e.g. 2025-01-01)'}, 'end_date': {'type': 'string', 'description': 'End date (ISO format, e.g. 2025-12-31)'}}, 'required': ['start_date', 'end_date']}),
    ]
