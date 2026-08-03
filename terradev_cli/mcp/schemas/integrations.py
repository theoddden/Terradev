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

        Tool(name='langfuse_configure', description='Configure Langfuse credentials (public key, secret key, host URL).', inputSchema={'type': 'object', 'properties': {'public_key': {'type': 'string', 'description': 'Langfuse public key (pk-lf-...)'}, 'secret_key': {'type': 'string', 'description': 'Langfuse secret key (sk-lf-...)'}, 'host': {'type': 'string', 'description': 'Langfuse host URL', 'default': 'https://cloud.langfuse.com'}}, 'required': ['public_key', 'secret_key']}),
        Tool(name='langfuse_test', description='Test Langfuse connectivity and list accessible projects.', inputSchema={'type': 'object', 'properties': {}}),
        Tool(name='langfuse_traces', description='List recent LLM traces from Langfuse.', inputSchema={'type': 'object', 'properties': {'limit': {'type': 'integer', 'description': 'Max traces to return', 'default': 20}, 'name': {'type': 'string', 'description': 'Filter by trace name'}}}),
        Tool(name='langfuse_trace', description='Get a single Langfuse trace with all observations/spans.', inputSchema={'type': 'object', 'properties': {'trace_id': {'type': 'string', 'description': 'Trace ID'}}, 'required': ['trace_id']}),
        Tool(name='langfuse_scores', description='List evaluation scores from Langfuse, optionally filtered by trace or score name.', inputSchema={'type': 'object', 'properties': {'trace_id': {'type': 'string', 'description': 'Filter by trace ID'}, 'name': {'type': 'string', 'description': 'Filter by score name'}, 'limit': {'type': 'integer', 'description': 'Max scores to return', 'default': 50}}}),
        Tool(name='langfuse_score', description='Create an evaluation score for a Langfuse trace (e.g. quality, accuracy, relevance).', inputSchema={'type': 'object', 'properties': {'trace_id': {'type': 'string', 'description': 'Trace ID to score'}, 'name': {'type': 'string', 'description': 'Score name (e.g. quality, accuracy)'}, 'value': {'type': 'number', 'description': 'Numeric score value'}, 'observation_id': {'type': 'string', 'description': 'Specific observation to score'}, 'comment': {'type': 'string', 'description': 'Optional comment'}}, 'required': ['trace_id', 'name', 'value']}),
        Tool(name='langfuse_datasets', description='List Langfuse datasets for evaluation and fine-tuning.', inputSchema={'type': 'object', 'properties': {'limit': {'type': 'integer', 'description': 'Max datasets to return', 'default': 20}}}),
        Tool(name='langfuse_export_training_data', description='Export Langfuse traces as instruction/response pairs for LoRA fine-tuning. Filters by quality score.', inputSchema={'type': 'object', 'properties': {'limit': {'type': 'integer', 'description': 'Max pairs to export', 'default': 500}, 'name': {'type': 'string', 'description': 'Filter traces by name'}, 'min_score': {'type': 'number', 'description': 'Minimum quality score (0.0-1.0)'}, 'score_name': {'type': 'string', 'description': 'Score name to filter on', 'default': 'quality'}}}),
        Tool(name='langfuse_quality', description='Get aggregated quality metrics from Langfuse scores for drift detection.', inputSchema={'type': 'object', 'properties': {'score_name': {'type': 'string', 'description': 'Score name to aggregate', 'default': 'quality'}, 'limit': {'type': 'integer', 'description': 'Sample size', 'default': 200}}}),
        Tool(name='langfuse_otel_env', description='Print OTEL environment variables for instrumenting LLM apps to send traces to Langfuse.', inputSchema={'type': 'object', 'properties': {'project': {'type': 'string', 'description': 'Langfuse project name', 'default': 'default'}}}),
        Tool(name='langfuse_k8s', description='Generate Kubernetes deployment manifest for self-hosted Langfuse.', inputSchema={'type': 'object', 'properties': {'namespace': {'type': 'string', 'description': 'K8s namespace', 'default': 'observability'}}}),
        Tool(name='databricks_configure', description='Configure Databricks credentials (workspace URL and personal access token).', inputSchema={'type': 'object', 'properties': {'host': {'type': 'string', 'description': 'Databricks workspace URL (e.g. https://adb-xxx.azuredatabricks.net)'}, 'token': {'type': 'string', 'description': 'Databricks personal access token (dapi...)'}}, 'required': ['host', 'token']}),
        Tool(name='databricks_test', description='Test Databricks connectivity and show cluster count.', inputSchema={'type': 'object', 'properties': {}}),
        Tool(name='databricks_jobs', description='List Databricks jobs in the workspace.', inputSchema={'type': 'object', 'properties': {'limit': {'type': 'integer', 'description': 'Max jobs to return', 'default': 25}}}),
        Tool(name='databricks_run', description='Trigger a Databricks job run by job ID.', inputSchema={'type': 'object', 'properties': {'job_id': {'type': 'integer', 'description': 'Databricks job ID'}}, 'required': ['job_id']}),
        Tool(name='databricks_run_status', description='Get status of a Databricks job run by run ID.', inputSchema={'type': 'object', 'properties': {'run_id': {'type': 'integer', 'description': 'Databricks run ID'}}, 'required': ['run_id']}),
        Tool(name='databricks_clusters', description='List all Databricks clusters with state and node type.', inputSchema={'type': 'object', 'properties': {}}),
        Tool(name='databricks_serving_endpoints', description='List Databricks model serving endpoints.', inputSchema={'type': 'object', 'properties': {}}),
        Tool(name='databricks_deploy_model', description='Deploy a registered MLflow model to a Databricks serving endpoint.', inputSchema={'type': 'object', 'properties': {'endpoint_name': {'type': 'string', 'description': 'Serving endpoint name'}, 'model_name': {'type': 'string', 'description': 'Registered model name'}, 'model_version': {'type': 'string', 'description': 'Model version', 'default': '1'}, 'workload_size': {'type': 'string', 'enum': ['Small', 'Medium', 'Large'], 'default': 'Small'}, 'scale_to_zero': {'type': 'boolean', 'default': True}}, 'required': ['endpoint_name', 'model_name']}),
        Tool(name='databricks_query', description='Query a Databricks model serving endpoint with a prompt.', inputSchema={'type': 'object', 'properties': {'endpoint': {'type': 'string', 'description': 'Serving endpoint name'}, 'prompt': {'type': 'string', 'description': 'Prompt text'}}, 'required': ['endpoint', 'prompt']}),
        Tool(name='databricks_mlflow_experiments', description='List MLflow experiments in Databricks workspace.', inputSchema={'type': 'object', 'properties': {'limit': {'type': 'integer', 'description': 'Max experiments to return', 'default': 50}}}),
        Tool(name='databricks_mlflow_models', description='List registered models in the Databricks MLflow Model Registry.', inputSchema={'type': 'object', 'properties': {'limit': {'type': 'integer', 'description': 'Max models to return', 'default': 50}}}),
    ]
