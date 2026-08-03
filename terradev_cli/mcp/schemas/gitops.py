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

        Tool(name='gitops_init', description='Initialize GitOps repository with ArgoCD or Flux CD. Creates cluster manifests, app definitions, policy-as-code templates, and multi-environment structure. Supports GitHub, GitLab, Bitbucket, Azure DevOps.', inputSchema={'type': 'object', 'properties': {'repo': {'type': 'string', 'description': 'Git repository (e.g., my-org/infra)'}, 'tool': {'type': 'string', 'description': 'GitOps tool to use', 'enum': ['argocd', 'flux'], 'default': 'argocd'}, 'provider': {'type': 'string', 'description': 'Git provider', 'enum': ['github', 'gitlab', 'bitbucket', 'azure-devops'], 'default': 'github'}, 'cluster': {'type': 'string', 'description': 'Target cluster name'}}, 'required': ['repo']}),
        Tool(name='gitops_bootstrap', description='Bootstrap ArgoCD or Flux on the cluster.', inputSchema={'type': 'object', 'properties': {'tool': {'type': 'string', 'description': 'GitOps tool', 'enum': ['argocd', 'flux']}, 'cluster': {'type': 'string', 'description': 'Cluster name'}, 'namespace': {'type': 'string', 'description': 'Namespace', 'default': 'gitops-system'}}, 'required': ['tool', 'cluster']}),
        Tool(name='gitops_sync', description='Sync cluster with Git repository.', inputSchema={'type': 'object', 'properties': {'cluster': {'type': 'string', 'description': 'Cluster name'}, 'environment': {'type': 'string', 'description': 'Environment to sync', 'default': 'prod'}, 'tool': {'type': 'string', 'description': 'GitOps tool', 'enum': ['argocd', 'flux'], 'default': 'argocd'}}, 'required': ['cluster']}),
        Tool(name='gitops_validate', description='Validate GitOps configuration.', inputSchema={'type': 'object', 'properties': {'cluster': {'type': 'string', 'description': 'Cluster name'}, 'dry_run': {'type': 'boolean', 'description': 'Dry run validation', 'default': True}}}),
    ]
