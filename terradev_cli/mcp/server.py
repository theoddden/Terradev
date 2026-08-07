#!/usr/bin/env python3
"""
Terradev MCP Server v6.0.6 - Complete Agentic GPU Infrastructure for Claude Code

260+ MCP tools: GPU provisioning, Kubernetes clusters, Karpenter auto-provisioning,
GitOps/ArgoCD automation, event-driven triggers, environment promotion, lineage tracking,
cross-provider migration, vLLM/SGLang/Ollama inference, DeepEval LLM evaluation, Arize Phoenix trace observability,
NeMo Guardrails output safety, Qdrant vector DB, Ray cluster management, W&B/MLflow/DVC,
HuggingFace Hub, Datadog monitoring, data governance, cost optimization, FlashOptim training,
Langfuse LLM observability, vLLM auto-optimization/analysis/benchmarking,
Enterprise SSO, Latitude.sh provider, and Terraform-powered parallel provisioning across 21+ cloud providers.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import time

# Import new feature tools from the sibling module (relative import).
# If the module is absent a WARNING is emitted so operators know which tools
# are missing, rather than silently dropping them.
try:
    from .new_feature_tools import ALL_NEW_TOOLS, COMMAND_MAP as NEW_COMMAND_MAP
except ImportError as _nft_err:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "new_feature_tools module not found (%s) — extended MCP tools will be unavailable.",
        _nft_err,
    )
    ALL_NEW_TOOLS = []
    NEW_COMMAND_MAP = {}
try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from .executor import (
    _UNSAFE_execute_shell_command,
    _get_tf_workspace,
    _list_tf_workspaces,
    _load_datadog_creds,
    _terradev_command,
    _validate_config_dir,
    check_terradev_installation,
    discover_local_gpus,
    enhance_error_message,
    estimate_model_memory,
    execute_safe_command,
    execute_terradev_command,
    execute_terraform_command,
    execute_terraform_parallel,
    generate_inference_terraform_config,
    generate_k8s_terraform_config,
    generate_terraform_config,
    generate_variables_file,
)

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.server.sse import SseServerTransport
    from mcp.server import NotificationOptions
    from mcp.server.sse import TransportSecuritySettings
    from mcp.types import (
        CallToolRequest,
        CallToolResult,
        GetPromptRequest,
        GetPromptResult,
        ListPromptsRequest,
        ListPromptsResult,
        ListResourcesRequest,
        ListResourcesResult,
        ListToolsRequest,
        ListToolsResult,
        ReadResourceRequest,
        ReadResourceResult,
        Resource,
        TextContent,
        TextResourceContents,
        Tool,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    InitializationOptions = None
    stdio_server = None
    SseServerTransport = None
    NotificationOptions = None
    TransportSecuritySettings = None
    CallToolRequest = None
    CallToolResult = None
    GetPromptRequest = None
    GetPromptResult = None
    ListPromptsRequest = None
    ListPromptsResult = None
    ListResourcesRequest = None
    ListResourcesResult = None
    ListToolsRequest = None
    ListToolsResult = None
    ReadResourceRequest = None
    ReadResourceResult = None
    Resource = None
    TextContent = None
    TextResourceContents = None
    Tool = None

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Mount, Route
    import uvicorn
except ImportError:
    # SSE deps are optional — only needed for remote mode
    Starlette = None
    uvicorn = None

logger = logging.getLogger("terradev-mcp")


# Check if terradev CLI is available


# Local GPU Discovery




# ── Datadog credential loader ────────────────────────────────────────────────


# ── Persistent Terraform workspaces ──────────────────────────────────────────
# Critical fix: Terraform state must survive beyond a single tool call.
# Previously used tempfile.TemporaryDirectory which destroyed terraform.tfstate
# immediately after apply, making it impossible to destroy/manage resources later.






# ── Path validation ──────────────────────────────────────────────────────────





# Execute terradev command safely with bug fixes




_SHELL_CMD_ALLOWLIST: set = set()  # Populated by internal callers with hardcoded strings only








# Execute generic Terraform command


# Execute Terraform for parallel provisioning










# Create MCP server with enhanced scaling (only if MCP is available)
if MCP_AVAILABLE:
    server = Server("terradev-mcp")
else:
    # Dummy server for when MCP is not installed - allows decorators to be defined
    class _DummyServer:
        def list_tools(self):
            return lambda f: f
        def call_tool(self):
            return lambda f: f
        def list_resources(self):
            return lambda f: f
        def read_resource(self):
            return lambda f: f
        def list_prompts(self):
            return lambda f: f
        def get_prompt(self):
            return lambda f: f
    server = _DummyServer()

# Global state for lazy loading and scaling
_tools_loaded = False
_tool_registry: Dict[str, callable] = {}
_tool_schemas: Dict[str, Dict] = {}
_load_lock: Optional[asyncio.Lock] = None
_concurrent_requests = 0
_max_concurrent = 100
_request_semaphore: Optional[asyncio.Semaphore] = None


async def _ensure_tools_loaded():
    """Ensure tools are loaded (lazy loading)"""
    global _tools_loaded, _tool_registry, _tool_schemas, _load_lock

    # Lazy-create lock to avoid Python 3.9 event loop binding bug
    if _load_lock is None:
        _load_lock = asyncio.Lock()

    if _tools_loaded:
        return

    async with _load_lock:
        if _tools_loaded:
            return

        # Load tools on first request
        logger.info("Loading MCP tools (lazy loading)...")
        _build_all_tools()
        _tools_loaded = True
        logger.info(f"Loaded {len(_ALL_TOOLS)} tools successfully")


# Import Rust-based optimizer for tool compression and parallel dispatch
try:
    from terradev_mcp_optimizer import MCPOptimizer as RustMCPOptimizer

    optimizer = RustMCPOptimizer(
        enable_compression=True, strip_optional=True, enable_parallel=True
    )
    logger.info("Using Rust-based MCPOptimizer for 10-50x faster tool compression")
except ImportError:
    # Python fallback implementation
    optimizer = None
    logger.warning(
        "Rust MCPOptimizer not available, using Python fallback (install Rust and build for 10-50x speedup)"
    )

# ── Pre-compiled Tool Schemas (built once at module load) ────────────────────
_ALL_TOOLS = None

def _build_all_tools():
    global _ALL_TOOLS
    _ALL_TOOLS = []

    if MCP_AVAILABLE:
        from . import schemas
        _ALL_TOOLS = list(schemas.TOOLS)

    if ALL_NEW_TOOLS:
        _ALL_TOOLS.extend(ALL_NEW_TOOLS)

    return _ALL_TOOLS
# Pre-compress at module load — cached for all subsequent list_tools calls
_COMPRESSED_TOOLS = None

def _get_compressed_tools():
    global _COMPRESSED_TOOLS
    if _COMPRESSED_TOOLS is not None:
        return _COMPRESSED_TOOLS
    if _ALL_TOOLS is None:
        _build_all_tools()
    if optimizer and MCP_AVAILABLE:
        _COMPRESSED_TOOLS = optimizer.compress_tools(_ALL_TOOLS)
        return _COMPRESSED_TOOLS
    # Python fallback: strip optional fields
    def strip_optional_fields(tool):
        if isinstance(tool.inputSchema, dict):
            props = tool.inputSchema.get("properties", {})
            required = tool.inputSchema.get("required", [])
            if isinstance(props, dict) and isinstance(required, list):
                required_set = set(required)
                tool.inputSchema["properties"] = {
                    k: v for k, v in props.items() if k in required_set
                }
        return tool

    _COMPRESSED_TOOLS = [strip_optional_fields(tool) for tool in _ALL_TOOLS]
    return _COMPRESSED_TOOLS


@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available Terradev tools (lazy loading with compression)"""
    global _concurrent_requests, _request_semaphore

    # Lazy-create semaphore to avoid Python 3.9 event loop binding bug
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(_max_concurrent)

    # Adaptive concurrency control
    async with _request_semaphore:
        _concurrent_requests += 1

        try:
            # Ensure tools are loaded
            await _ensure_tools_loaded()

            # Return compressed tools
            return ListToolsResult(tools=_get_compressed_tools())
        finally:
            _concurrent_requests -= 1


from . import router

@server.call_tool()
async def handle_call_tool(name_or_request, arguments=None, **kwargs) -> CallToolResult:
    """Handle tool calls with adaptive concurrency control."""
    global _concurrent_requests, _request_semaphore

    if isinstance(name_or_request, CallToolRequest):
        request = name_or_request
        tool_name = request.params.name
        arguments = request.params.arguments or {}
    else:
        tool_name = name_or_request
        arguments = arguments or {}

    # Lazy-create semaphore to avoid Python 3.9 event loop binding bug
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(_max_concurrent)

    # Adaptive concurrency control
    async with _request_semaphore:
        _concurrent_requests += 1

        try:
            # Ensure tools are loaded
            await _ensure_tools_loaded()

            # Expand compressed namespace tools back to original tool names
            if optimizer:
                original_tool_name, original_arguments = optimizer.expand_call(
                    tool_name, arguments
                )
                tool_name = original_tool_name
                arguments = original_arguments
            else:
                # Python fallback: handle namespace expansion
                if "." in tool_name:
                    parts = tool_name.split(".", 1)
                    tool_name = parts[1]

            # O(1) dispatch through the smart router
            return await router.dispatch(tool_name, arguments, execute_terradev_command)

        except Exception as e:  # noqa: BLE001
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Error: {str(e)}")],
                isError=True,
            )
        finally:
            _concurrent_requests -= 1



@server.list_resources()
async def handle_list_resources() -> ListResourcesResult:
    """List available MCP resources for session-start context and polling."""
    return ListResourcesResult(
        resources=[
            Resource(
                uri="terradev://active_context",
                name="Active Context",
                description="Current Terradev state: running jobs, active instances, spend-to-date, alerts. Read on session start.",
                mimeType="application/json",
            ),
            Resource(
                uri="terradev://instances",
                name="Active Instances",
                description="Currently provisioned GPU instances across all providers.",
                mimeType="application/json",
            ),
            Resource(
                uri="terradev://jobs",
                name="Training Jobs",
                description="All training jobs with status, progress, and ETA.",
                mimeType="application/json",
            ),
            Resource(
                uri="terradev://spend",
                name="Spend Summary",
                description="Cost analytics and spend-to-date across all providers.",
                mimeType="application/json",
            ),
            Resource(
                uri="terradev://alerts",
                name="Alerts",
                description="Active alerts: straggler nodes, budget warnings, drift detected, failed health checks.",
                mimeType="application/json",
            ),
        ]
    )


@server.read_resource()
async def handle_read_resource(request: ReadResourceRequest) -> ReadResourceResult:
    """Read a Terradev resource."""
    uri = str(request.params.uri)

    if uri == "terradev://active_context":
        # Composite: jobs + instances + spend
        jobs = await execute_terradev_command(["train-status", "-f", "json"])
        instances = await execute_terradev_command(["status", "-f", "json"])
        spend = await execute_terradev_command(
            ["analytics", "--days", "7", "-f", "json"]
        )
        context = {
            "jobs": jobs["stdout"] if jobs["success"] else None,
            "instances": instances["stdout"] if instances["success"] else None,
            "spend_7d": spend["stdout"] if spend["success"] else None,
            "suggest_action": "Call active_context tool for formatted recommendations.",
        }
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=json.dumps(context, indent=2),
                )
            ]
        )

    elif uri == "terradev://instances":
        result = await execute_terradev_command(["status", "-f", "json"])
        text = (
            result["stdout"]
            if result["success"]
            else json.dumps({"error": result["stderr"]})
        )
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="application/json", text=text)
            ]
        )

    elif uri == "terradev://jobs":
        result = await execute_terradev_command(["train-status", "-f", "json"])
        text = (
            result["stdout"]
            if result["success"]
            else json.dumps({"error": result["stderr"]})
        )
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="application/json", text=text)
            ]
        )

    elif uri == "terradev://spend":
        result = await execute_terradev_command(
            ["analytics", "--days", "30", "-f", "json"]
        )
        text = (
            result["stdout"]
            if result["success"]
            else json.dumps({"error": result["stderr"]})
        )
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="application/json", text=text)
            ]
        )

    elif uri == "terradev://alerts":
        # Aggregate alerts from multiple sources
        alerts = []
        # Check for straggler nodes via monitor
        monitor = await execute_terradev_command(
            ["monitor", "--check-stragglers", "-f", "json"]
        )
        if monitor["success"] and monitor["stdout"].strip():
            alerts.append({"type": "straggler", "data": monitor["stdout"]})
        # Check cost budget
        cost = await execute_terradev_command(["cost-scaler-status", "-f", "json"])
        if cost["success"] and "exceed" in cost["stdout"].lower():
            alerts.append({"type": "budget_warning", "data": cost["stdout"]})
        # Check drift
        drift = await execute_terradev_command(
            ["manifests", "--check-drift", "-f", "json"]
        )
        if drift["success"] and "drift" in drift["stdout"].lower():
            alerts.append({"type": "drift_detected", "data": drift["stdout"]})
        text = json.dumps({"alerts": alerts, "count": len(alerts)}, indent=2)
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="application/json", text=text)
            ]
        )

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                uri=uri,
                mimeType="application/json",
                text=json.dumps({"error": f"Unknown resource: {uri}"}),
            )
        ]
    )


@server.list_prompts()
async def handle_list_prompts() -> ListPromptsResult:
    """List available prompts"""
    return ListPromptsResult(prompts=[])


@server.get_prompt()
async def handle_get_prompt(request: GetPromptRequest) -> GetPromptResult:
    """Get a prompt"""
    return GetPromptResult(description="", messages=[])


# ---------------------------------------------------------------------------
# OAuth 2.0 PKCE auth for Claude.ai Connectors
# ---------------------------------------------------------------------------

TERRADEV_MCP_BEARER_TOKEN = os.getenv("TERRADEV_MCP_BEARER_TOKEN", "")

# Comma-separated list of exact allowed redirect URIs for OAuth.
# Example: "https://claude.ai/oauth/callback,http://localhost:3000/callback"
# If unset, only localhost/127.0.0.1 URIs are permitted (safe local-dev default).
_ALLOWED_REDIRECT_URIS_RAW = os.getenv("TERRADEV_MCP_ALLOWED_REDIRECT_URIS", "")
_ALLOWED_REDIRECT_URIS: set = (
    {u.strip() for u in _ALLOWED_REDIRECT_URIS_RAW.split(",") if u.strip()}
    if _ALLOWED_REDIRECT_URIS_RAW
    else set()
)


def _is_redirect_uri_allowed(uri: str) -> bool:
    """Return True only if uri is on the explicit allowlist, or is a localhost
    URI when no explicit allowlist is configured."""
    if not uri:
        return False
    if _ALLOWED_REDIRECT_URIS:
        return uri in _ALLOWED_REDIRECT_URIS
    # No allowlist configured — permit localhost / 127.0.0.1 only
    from urllib.parse import urlparse
    try:
        parsed = urlparse(uri)
        return parsed.hostname in ("localhost", "127.0.0.1", "::1")
    except Exception:  # noqa: BLE001
        return False

# In-memory stores (single-instance server)
_auth_codes: Dict[str, Dict[str, Any]] = (
    {}
)  # code -> {client_id, code_challenge, redirect_uri, expires}
_access_tokens: Dict[str, Dict[str, Any]] = {}  # token -> {client_id, expires}


def _cleanup_expired():
    """Remove expired auth codes and tokens."""
    now = time.time()
    for store in (_auth_codes, _access_tokens):
        expired = [k for k, v in store.items() if v.get("expires", 0) < now]
        for k in expired:
            del store[k]


# ---------------------------------------------------------------------------
# OAuth endpoint handlers (added as Starlette routes)
# ---------------------------------------------------------------------------


async def oauth_authorization_server_metadata(request: Request) -> JSONResponse:
    """RFC 8414 — OAuth Authorization Server Metadata."""
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "issuer": base,
            "authorization_endpoint": base + "/authorize",
            "token_endpoint": base + "/token",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
        }
    )


async def oauth_protected_resource(request: Request) -> JSONResponse:
    """RFC 9728 — OAuth Protected Resource Metadata."""
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "resource": base,
            "authorization_servers": [base],
            "bearer_methods_supported": ["header"],
        }
    )


async def oauth_authorize(request: Request) -> Response:
    """OAuth 2.0 Authorization Endpoint — auto-approves if client_id matches our token."""
    from starlette.responses import RedirectResponse

    params = dict(request.query_params)
    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    code_challenge = params.get("code_challenge", "")
    code_challenge_method = params.get("code_challenge_method", "")
    state = params.get("state", "")

    logger.info(
        "OAuth authorize: client_id=%s... redirect=%s", client_id[:16], redirect_uri
    )

    # Validate redirect_uri against allowlist before doing anything else (open-redirect prevention)
    if not _is_redirect_uri_allowed(redirect_uri):
        logger.warning("OAuth authorize rejected: redirect_uri not in allowlist: %s", redirect_uri)
        return JSONResponse(
            {"error": "invalid_request", "error_description": "redirect_uri not allowed"},
            status_code=400,
        )

    # Validate client_id matches our configured token
    if TERRADEV_MCP_BEARER_TOKEN and client_id != TERRADEV_MCP_BEARER_TOKEN:
        logger.warning("OAuth authorize rejected: bad client_id")
        return JSONResponse({"error": "invalid_client"}, status_code=401)

    if code_challenge_method and code_challenge_method != "S256":
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Only S256 supported"},
            status_code=400,
        )

    # Generate authorization code
    _cleanup_expired()
    code = secrets.token_urlsafe(48)
    _auth_codes[code] = {
        "client_id": client_id,
        "code_challenge": code_challenge,
        "redirect_uri": redirect_uri,
        "expires": time.time() + 300,  # 5 min
    }

    # Redirect back to Claude.ai with the code
    sep = "&" if "?" in redirect_uri else "?"
    redirect = redirect_uri + sep + urlencode({"code": code, "state": state})
    logger.info("OAuth authorize: issuing code, redirecting to %s", redirect_uri)
    return RedirectResponse(url=redirect, status_code=302)


async def oauth_token(request: Request) -> JSONResponse:
    """OAuth 2.0 Token Endpoint — exchanges auth code for access token (PKCE)."""
    if request.method == "GET":
        return JSONResponse({"error": "method_not_allowed"}, status_code=405)

    try:
        body = await request.form()
    except Exception:  # noqa: BLE001
        body = {}
    body = dict(body)

    grant_type = body.get("grant_type", "")
    code = body.get("code", "")
    code_verifier = body.get("code_verifier", "")
    client_id = body.get("client_id", "")
    body.get("redirect_uri", "")

    logger.info(
        "OAuth token: grant_type=%s client_id=%s...", grant_type, (client_id or "")[:16]
    )

    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    _cleanup_expired()
    auth_data = _auth_codes.pop(code, None)
    if not auth_data:
        logger.warning("OAuth token: invalid or expired code")
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    # Verify PKCE code_challenge
    if auth_data.get("code_challenge") and code_verifier:
        digest = hashlib.sha256(code_verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if expected != auth_data["code_challenge"]:
            logger.warning("OAuth token: PKCE verification failed")
            return JSONResponse(
                {
                    "error": "invalid_grant",
                    "error_description": "PKCE verification failed",
                },
                status_code=400,
            )

    # Issue access token
    access_token = secrets.token_urlsafe(48)
    _access_tokens[access_token] = {
        "client_id": auth_data["client_id"],
        "expires": time.time() + 86400,  # 24 hours
    }

    logger.info(
        "OAuth token: issued access token for client_id=%s...",
        auth_data["client_id"][:16],
    )
    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 86400,
        }
    )


# ---------------------------------------------------------------------------
# ASGI auth middleware (validates Bearer tokens from OAuth flow)
# ---------------------------------------------------------------------------

# Paths that don't require auth
_PUBLIC_PATHS = frozenset(
    [
        "/health",
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/sse",
        "/authorize",
        "/token",
    ]
)


class OAuthBearerMiddleware:
    """Pure ASGI middleware — validates OAuth Bearer tokens on protected routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "?")

        # Public routes pass through
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Extract auth header
        headers_raw = scope.get("headers", [])
        header_dict = {k.decode(): v.decode() for k, v in headers_raw}
        auth = header_dict.get("authorization", "")

        logger.info(
            "Request: %s %s host=%s auth=%s",
            method,
            path,
            header_dict.get("host", "-"),
            auth[:30] + "..." if auth else "-",
        )

        if not auth.startswith("Bearer "):
            logger.warning("Auth rejected for %s %s (no Bearer token)", method, path)
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        token = auth[7:]  # strip "Bearer "

        # Accept the raw configured token OR any valid OAuth-issued token
        _cleanup_expired()
        if token == TERRADEV_MCP_BEARER_TOKEN:
            await self.app(scope, receive, send)
            return

        if token in _access_tokens:
            await self.app(scope, receive, send)
            return

        logger.warning("Auth rejected for %s %s (invalid token)", method, path)
        response = JSONResponse({"error": "unauthorized"}, status_code=401)
        await response(scope, receive, send)


# ---------------------------------------------------------------------------
# SSE app factory
# ---------------------------------------------------------------------------


def create_sse_app() -> "Starlette":
    """Build the Starlette app that exposes the MCP server over SSE."""
    if Starlette is None:
        print(
            "Error: starlette/uvicorn not installed. "
            "Install with: pip install 'mcp[cli]' starlette uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    security_settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "terradev-mcp.terradev.cloud",
            "localhost:8090",
            "127.0.0.1:8090",
        ],
        allowed_origins=[
            "https://claude.ai",
            "https://www.claude.ai",
            "https://terradev-mcp.terradev.cloud",
        ],
    )
    sse_transport = SseServerTransport("/messages", security_settings=security_settings)

    async def handle_messages(request: Request) -> None:
        await sse_transport.handle_post_message(
            request.scope, request.receive, request._send
        )

    async def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {"status": "ok", "server": "terradev-mcp", "version": "2.0.1"}
        )

    # SSE handler wraps the MCP server
    class SseHandler:
        def __init__(self):
            self._server = server

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return

            # SSE endpoint only accepts GET requests
            if scope["method"] != "GET":
                from starlette.responses import Response

                response = Response(
                    "Method Not Allowed - SSE endpoint only accepts GET",
                    status_code=405,
                )
                await response(scope, receive, send)
                return

            async with sse_transport.connect_sse(scope, receive, send) as streams:
                await self._server.run(
                    streams[0],
                    streams[1],
                    InitializationOptions(
                        server_name="terradev-mcp",
                        server_version="2.0.1",
                        capabilities=self._server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )

    sse_handler = SseHandler()

    inner_app = Starlette(
        debug=False,
        routes=[
            # OAuth 2.0 endpoints (public — handled before auth middleware)
            Route(
                "/.well-known/oauth-authorization-server",
                endpoint=oauth_authorization_server_metadata,
            ),
            Route(
                "/.well-known/oauth-protected-resource",
                endpoint=oauth_protected_resource,
            ),
            Route(
                "/.well-known/oauth-protected-resource/sse",
                endpoint=oauth_protected_resource,
            ),
            Route("/authorize", endpoint=oauth_authorize),
            Route("/token", endpoint=oauth_token, methods=["POST"]),
            # MCP endpoints
            Route("/health", endpoint=health),
            Route("/sse", endpoint=sse_handler),
            Route("/messages/{path:path}", endpoint=handle_messages, methods=["POST"]),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
        ],
    )

    app = OAuthBearerMiddleware(inner_app)
    return app


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def run_stdio():
    """Run in stdio mode (Claude Code / local)."""
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="terradev-mcp",
                    server_version="2.0.1",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception:
        import traceback

        traceback.print_exc()
        raise


def main():
    """Main entry point — supports both stdio and SSE transports."""
    parser = argparse.ArgumentParser(description="Terradev MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (default, for Claude Code) or sse (remote, for Claude.ai Connectors)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="SSE host (default: 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=8080, help="SSE port (default: 8080)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Check if terradev is installed
    if not check_terradev_installation():
        logger.warning(
            "terradev CLI not found. Tools will fail until installed: pip install terradev-cli"
        )

    if not os.getenv("TERRADEV_RUNPOD_KEY"):
        logger.warning(
            "TERRADEV_RUNPOD_KEY not set. Some functionality may be limited."
        )

    if args.transport == "stdio":
        asyncio.run(run_stdio())
    else:
        if not TERRADEV_MCP_BEARER_TOKEN:
            logger.warning(
                "TERRADEV_MCP_BEARER_TOKEN is not set — SSE endpoint is UNAUTHENTICATED. "
                "Set this env var in production."
            )
        app = create_sse_app()
        logger.info("Starting Terradev MCP SSE server on %s:%s", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)


def run_server(transport: str = "stdio", host: Optional[str] = None, port: Optional[int] = None):
    """Start the Terradev MCP server (used by `terradev mcp serve`)."""
    if not MCP_AVAILABLE:
        print(
            "Error: mcp package not installed. Install with: pip install 'mcp[cli]'",
            file=sys.stderr,
        )
        sys.exit(1)

    argv = [sys.argv[0] if sys.argv else "terradev_mcp"]
    if transport != "stdio":
        argv.extend(["--transport", transport])
    if host is not None:
        argv.extend(["--host", host])
    if port is not None:
        argv.extend(["--port", str(port)])

    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv


def list_tools():
    """Print all registered Terradev MCP tools (used by `terradev mcp list-tools`)."""
    if not MCP_AVAILABLE:
        print(
            "Error: mcp package not installed. Install with: pip install 'mcp[cli]'",
            file=sys.stderr,
        )
        return

    _build_all_tools()
    print(f"Terradev MCP tools ({len(_ALL_TOOLS)}):\n")
    for tool in _ALL_TOOLS:
        print(f"  - {tool.name}: {tool.description}")


def install_config(client: str):
    """Install the MCP client configuration for the requested client."""
    import platform

    home = os.path.expanduser("~")
    configs = {
        "claude-desktop": (
            os.path.join(home, "Library", "Application Support", "Claude", "claude_desktop_config.json")
            if platform.system() == "Darwin"
            else os.path.join(home, ".config", "claude", "claude_desktop_config.json")
        ),
        "cursor": os.path.join(home, ".cursor", "mcp.json"),
        "windsurf": os.path.join(home, ".config", "windsurf", "mcp_config.json"),
        "continue": os.path.join(home, ".continue", "mcp_config.json"),
        "cline": os.path.join(home, ".config", "cline", "mcp_config.json"),
    }

    if client not in configs:
        print(f"Error: unsupported MCP client '{client}'", file=sys.stderr)
        print(f"Supported clients: {', '.join(configs.keys())}", file=sys.stderr)
        sys.exit(1)

    path = configs[client]
    terradev_entry = {
        "command": sys.executable,
        "args": ["-m", "terradev_cli", "mcp", "serve"],
    }

    existing = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)
    if "mcpServers" not in existing:
        existing["mcpServers"] = {}
    existing["mcpServers"]["terradev"] = terradev_entry

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Installed Terradev MCP config for {client} at {path}")


# ── Exposed command map used by tests and external callers ───────────────────
COMMAND_MAP: Dict[str, Any] = {}

try:
    _build_all_tools()
    if _ALL_TOOLS:
        COMMAND_MAP = {tool.name: None for tool in _ALL_TOOLS}
    if NEW_COMMAND_MAP:
        COMMAND_MAP.update(NEW_COMMAND_MAP)
except Exception:  # noqa: BLE001
    # Tool build may fail during partial imports; callers should still be able
    # to import COMMAND_MAP without raising.
    pass


if __name__ == "__main__":
    main()
