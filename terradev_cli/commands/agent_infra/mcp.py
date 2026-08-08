#!/usr/bin/env python3
"""``terradev agent mcp`` — high-throughput, dynamic MCP protocol bridge.

The bridge is intentionally decoupled:

- ``McpTransport`` speaks JSON-RPC over stdio or HTTP.
- ``McpServerConnection`` handles the protocol handshake and tool cache.
- ``McpBridge`` aggregates multiple servers and exposes a unified MCP surface.
- ``SchemaCompiler`` turns JSON Schema tool definitions into Pydantic models on
  the fly for request validation and error reporting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import aiohttp
import click
from aiohttp import web
from pydantic import BaseModel, ValidationError, create_model

from .core import (
    JsonRpcMessage,
    McpError,
    McpPrompt,
    McpResource,
    McpServerDefinition,
    McpTool,
    McpTransport,
)
from .otel import Tracer, get_tracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema compiler
# ---------------------------------------------------------------------------


class SchemaCompiler:
    """Compile a JSON Schema into a runtime Pydantic model."""

    _cache: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def _map_type(cls, spec: Dict[str, Any]) -> Any:
        t = spec.get("type")
        if t == "string":
            return str
        if t == "integer":
            return int
        if t == "number":
            return float
        if t == "boolean":
            return bool
        if t == "array":
            item_type = cls._map_type(spec.get("items", {})) if "items" in spec else Any
            return List[item_type]
        if t == "object":
            return Dict[str, Any]
        return Any

    @classmethod
    def _default_for(cls, spec: Dict[str, Any]) -> Any:
        t = spec.get("type")
        if t in ("string", "array", "object"):
            return ...
        return ...

    @classmethod
    def compile(cls, name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Compile a JSON Schema into a Pydantic model class."""
        if name in cls._cache:
            return cls._cache[name]

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        fields: Dict[str, Any] = {}
        for prop_name, prop_spec in properties.items():
            default = ...
            if prop_name not in required:
                if "default" in prop_spec:
                    default = prop_spec["default"]
                else:
                    default = None
            ann = cls._map_type(prop_spec)
            if default is ...:
                fields[prop_name] = (ann, ...)
            else:
                fields[prop_name] = (Optional[ann], default)

        model = create_model(f"{name}_input", __base__=BaseModel, **fields)
        cls._cache[name] = model
        return model

    @classmethod
    def validate(cls, name: str, schema: Dict[str, Any], arguments: Any) -> BaseModel:
        model = cls.compile(name, schema)
        if not isinstance(arguments, dict):
            raise McpError(f"Tool {name} arguments must be an object")
        return model.model_validate(arguments)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class McpRegistry(BaseModel):
    """A registry of MCP server definitions."""

    servers: List[McpServerDefinition] = []

    @classmethod
    def load(cls, path: Path) -> "McpRegistry":
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise McpError(f"Invalid MCP registry JSON at {path}: {exc}") from exc
        if isinstance(data, list):
            return cls(servers=[McpServerDefinition.model_validate(s) for s in data])
        return cls(servers=[McpServerDefinition.model_validate(s) for s in data.get("servers", [])])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

    def add(self, server: McpServerDefinition) -> None:
        self.servers = [s for s in self.servers if s.name != server.name]
        self.servers.append(server)

    def get(self, name: str) -> Optional[McpServerDefinition]:
        for s in self.servers:
            if s.name == name:
                return s
        return None


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class JsonRpcCodec:
    """Encode/decode newline-delimited JSON-RPC messages."""

    @classmethod
    def encode(cls, msg: JsonRpcMessage) -> bytes:
        return (json.dumps(msg.model_dump(exclude_none=True), default=str) + "\n").encode()

    @classmethod
    def decode(cls, line: bytes) -> JsonRpcMessage:
        return JsonRpcMessage.model_validate_json(line.decode(errors="replace"))


class StdioMcpTransport(McpTransport):
    """JSON-RPC over a spawned subprocess's stdin/stdout."""

    name = "stdio"

    def __init__(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args
        self.env = env or {}
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()
        self._request_id = 0

    async def start(self) -> None:
        env = {**dict(os.environ), **self.env}
        try:
            self.proc = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (OSError, ValueError) as exc:
            raise McpError(f"Failed to start stdio MCP server '{self.command}': {exc}") from exc
        if not self.proc.stdin or not self.proc.stdout:
            raise McpError("Failed to start stdio MCP server")

    async def initialize(self) -> JsonRpcMessage:
        init = JsonRpcMessage(
            id=1,
            method="initialize",
            params={
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "terradev-agent-mcp", "version": "1.0.0"},
            },
        )
        return await self.send(init)

    async def _write(self, message: JsonRpcMessage) -> None:
        if not self.proc or self.proc.stdin is None:
            raise McpError("stdio transport not started")
        self.proc.stdin.write(JsonRpcCodec.encode(message))
        await self.proc.stdin.drain()

    async def send(self, message: JsonRpcMessage) -> JsonRpcMessage:
        if not self.proc or self.proc.stdin is None or self.proc.stdout is None:
            raise McpError("stdio transport not started")

        async with self._lock:
            await self._write(message)

            # Read until a line with matching id appears
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    raise McpError("stdio MCP server closed stream")
                try:
                    response = JsonRpcCodec.decode(line)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"Discarding non-JSON line: {line!r} ({exc})")
                    continue

                if message.id is not None and response.id == message.id:
                    return response

    async def send_notification(self, message: JsonRpcMessage) -> None:
        async with self._lock:
            await self._write(message)

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()

    async def read_stderr(self) -> str:
        if not self.proc or self.proc.stderr is None:
            return ""
        data = await self.proc.stderr.read()
        return data.decode(errors="replace")


class HttpMcpTransport(McpTransport):
    """JSON-RPC over Streamable HTTP."""

    name = "http"

    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0

    async def start(self) -> None:
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )

    async def initialize(self) -> JsonRpcMessage:
        return await self.send(JsonRpcMessage(
            id=1,
            method="initialize",
            params={
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "terradev-agent-mcp", "version": "1.0.0"},
            },
        ))

    async def _post(self, message: JsonRpcMessage) -> Dict[str, Any]:
        if not self.session:
            raise McpError("HTTP transport not started")
        async with self.session.post(
            f"{self.base_url}/mcp",
            json=message.model_dump(exclude_none=True),
        ) as resp:
            return await resp.json()

    async def send(self, message: JsonRpcMessage) -> JsonRpcMessage:
        data = await self._post(message)
        return JsonRpcMessage.model_validate(data)

    async def send_notification(self, message: JsonRpcMessage) -> None:
        try:
            await self._post(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"HTTP notification failed: {exc}")

    async def close(self) -> None:
        if self.session:
            await self.session.close()


def _create_transport(server: McpServerDefinition) -> McpTransport:
    if server.transport == "stdio":
        return StdioMcpTransport(
            command=server.command,
            args=server.args,
            env=server.env,
        )
    if server.transport == "http":
        return HttpMcpTransport(
            base_url=server.url or "http://127.0.0.1:8080",
            timeout=server.timeout or 30.0,
        )
    raise McpError(f"Unknown transport: {server.transport}")


# ---------------------------------------------------------------------------
# Server connection
# ---------------------------------------------------------------------------


class McpServerConnection:
    """A single backend MCP server with dynamic schema caching."""

    def __init__(
        self,
        name: str,
        server: McpServerDefinition,
        transport: McpTransport,
        tracer: Optional[Tracer] = None,
    ):
        self.name = name
        self.definition = server
        self.transport = transport
        self.tracer = tracer or get_tracer("terradev.agent.mcp.server")
        self.tools: Dict[str, McpTool] = {}
        self.prompts: Dict[str, McpPrompt] = {}
        self.resources: Dict[str, McpResource] = {}
        self._next_id = 1

    async def start(self) -> None:
        if hasattr(self.transport, "start"):
            await self.transport.start()
        init_req = self._request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "terradev-agent-mcp", "version": "1.0.0"},
        })
        init_resp = await self.transport.send(init_req)
        if init_resp.error:
            raise McpError(f"MCP initialize failed: {init_resp.error}")
        # Confirm initialization
        await self.transport.send_notification(JsonRpcMessage(method="notifications/initialized"))
        await self.refresh_tools()

    def _notification(self, method: str, params: Any = None) -> JsonRpcMessage:
        return JsonRpcMessage(method=method, params=params)

    def _request(self, method: str, params: Any = None) -> JsonRpcMessage:
        self._next_id += 1
        return JsonRpcMessage(id=self._next_id, method=method, params=params)

    async def refresh_tools(self) -> None:
        resp = await self.transport.send(self._request("tools/list"))
        if resp.error:
            raise McpError(f"tools/list failed: {resp.error}")
        tools = resp.result or {}
        for tool in tools.get("tools", []):
            m = McpTool.model_validate(tool)
            self.tools[m.name] = m

    async def list_prompts(self) -> List[McpPrompt]:
        resp = await self.transport.send(self._request("prompts/list"))
        if resp.error:
            return []
        return [McpPrompt.model_validate(p) for p in (resp.result or {}).get("prompts", [])]

    async def list_resources(self) -> List[McpResource]:
        resp = await self.transport.send(self._request("resources/list"))
        if resp.error:
            return []
        return [McpResource.model_validate(r) for r in (resp.result or {}).get("resources", [])]

    async def call_tool(
        self,
        tool_name: str,
        arguments: Any,
        *,
        validate: bool = True,
    ) -> Any:
        tool = self.tools.get(tool_name)
        if not tool:
            raise McpError(f"Tool {tool_name} not found on server {self.name}")

        if validate and tool.inputSchema:
            try:
                SchemaCompiler.validate(f"{self.name}.{tool_name}", tool.inputSchema, arguments)
            except ValidationError as exc:
                raise McpError(f"Invalid arguments for {tool_name}: {exc}") from exc

        with self.tracer.trace(
            "agent.mcp.call_tool",
            {"server": self.name, "tool": tool_name},
        ):
            resp = await self.transport.send(self._request("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            }))
            if resp.error:
                raise McpError(f"tools/call failed: {resp.error}")
            return resp.result

    async def close(self) -> None:
        await self.transport.close()


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class McpBridge:
    """Aggregate multiple MCP server connections behind a single MCP surface."""

    def __init__(self, tracer: Optional[Tracer] = None):
        self.servers: Dict[str, McpServerConnection] = {}
        self.tracer = tracer or get_tracer("terradev.agent.mcp.bridge")

    async def add_server(self, name: str, server: McpServerDefinition) -> McpServerConnection:
        if name in self.servers:
            old = self.servers[name]
            try:
                await old.close()
            except Exception:  # noqa: BLE001
                pass
        transport = _create_transport(server)
        conn = McpServerConnection(name, server, transport, tracer=self.tracer)
        await conn.start()
        self.servers[name] = conn
        return conn

    async def load_registry(self, path: Path) -> None:
        registry = McpRegistry.load(path)
        for server in registry.servers:
            await self.add_server(server.name, server)

    async def list_all_tools(self) -> List[Dict[str, Any]]:
        out = []
        for conn in self.servers.values():
            for tool in conn.tools.values():
                out.append({
                    **tool.model_dump(),
                    "server": conn.name,
                })
        return out

    async def call_tool(self, full_name: str, arguments: Any) -> Any:
        """Call a tool by ``server.tool`` or by tool name if unique."""
        if "." in full_name:
            server_name, tool_name = full_name.split(".", 1)
            conn = self.servers.get(server_name)
            if not conn:
                raise McpError(f"Unknown MCP server: {server_name}")
            return await conn.call_tool(tool_name, arguments)

        # Try unique tool name across all servers
        candidates = []
        for conn in self.servers.values():
            if full_name in conn.tools:
                candidates.append(conn)
        if len(candidates) != 1:
            raise McpError(
                f"Tool {full_name} is ambiguous or missing; use server.{full_name}"
            )
        return await candidates[0].call_tool(full_name, arguments)

    async def handle_message(self, msg: JsonRpcMessage) -> JsonRpcMessage:
        """Dispatch a single JSON-RPC message to the right backend."""
        if msg.id is None:
            return JsonRpcMessage()

        method = msg.method
        params = msg.params or {}

        if method == "initialize":
            return JsonRpcMessage(id=msg.id, result={
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "serverInfo": {"name": "terradev-agent-mcp", "version": "1.0.0"},
            })

        if method == "tools/list":
            return JsonRpcMessage(id=msg.id, result={"tools": await self.list_all_tools()})

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                result = await self.call_tool(name, arguments)
            except McpError as exc:
                return JsonRpcMessage(
                    id=msg.id,
                    error={"code": -32603, "message": str(exc)},
                )
            return JsonRpcMessage(id=msg.id, result=result)

        if method == "prompts/list":
            prompts = []
            for conn in self.servers.values():
                prompts.extend([p.model_dump() for p in await conn.list_prompts()])
            return JsonRpcMessage(id=msg.id, result={"prompts": prompts})

        if method == "resources/list":
            resources = []
            for conn in self.servers.values():
                resources.extend([r.model_dump() for r in await conn.list_resources()])
            return JsonRpcMessage(id=msg.id, result={"resources": resources})

        return JsonRpcMessage(
            id=msg.id,
            error={"code": -32601, "message": f"Method {method} not implemented"},
        )

    async def close(self) -> None:
        for conn in self.servers.values():
            await conn.close()


# ---------------------------------------------------------------------------
# Server transport
# ---------------------------------------------------------------------------


class McpServer:
    """Host an MCP bridge over stdio or HTTP."""

    def __init__(self, bridge: McpBridge, tracer: Optional[Tracer] = None):
        self.bridge = bridge
        self.tracer = tracer or get_tracer("terradev.agent.mcp.server")

    async def serve_stdio(self) -> None:
        """Serve JSON-RPC MCP over stdin/stdout line-by-line."""
        loop = asyncio.get_event_loop()

        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line_bytes = line.encode() if isinstance(line, str) else line
            try:
                msg = JsonRpcCodec.decode(line_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Discarding non-JSON line: {line!r} ({exc})")
                continue

            with self.tracer.trace("agent.mcp.stdio_message", {"method": msg.method}):
                if msg.is_notification():
                    continue
                response = await self.bridge.handle_message(msg)
                sys.stdout.write(JsonRpcCodec.encode(response).decode())
                sys.stdout.flush()

    async def serve_http(
        self,
        port: int = 8080,
        host: str = "127.0.0.1",
        ready_event: Optional[asyncio.Event] = None,
    ) -> None:
        app = web.Application()

        async def mcp_post(request: web.Request) -> web.Response:
            data = await request.json()
            msg = JsonRpcMessage.model_validate(data)
            with self.tracer.trace("agent.mcp.http_message", {"method": msg.method}):
                if msg.is_notification():
                    return web.Response(status=202)
                response = await self.bridge.handle_message(msg)
                return web.json_response(
                    response.model_dump(exclude_none=True),
                    dumps=lambda x: json.dumps(x, default=str),
                )

        async def mcp_get(request: web.Request) -> web.Response:
            return web.json_response({
                "status": "ok",
                "bridge": "terradev-agent-mcp",
                "servers": list(self.bridge.servers.keys()),
            })

        app.router.add_get("/mcp", mcp_get)
        app.router.add_post("/mcp", mcp_post)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        try:
            await site.start()
        except OSError as exc:
            await runner.cleanup()
            raise McpError(f"Failed to start MCP HTTP server on {host}:{port}: {exc}") from exc

        # Resolve the actual bound port when port=0 was requested.
        bound_port = port
        if port == 0 and site._server is not None:
            for sock in site._server.sockets:
                bound_port = sock.getsockname()[1]
                break

        self._http_host = host
        self._http_port = bound_port
        logger.info(f"MCP bridge listening on http://{host}:{bound_port}/mcp")

        if ready_event is not None:
            ready_event.set()

        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await site.stop()
            await runner.cleanup()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("mcp")
def mcp():
    """Universal, high-throughput MCP protocol bridge."""
    pass


@mcp.group("registry")
def registry_group():
    """Manage the MCP server registry."""
    pass


@registry_group.command("add")
@click.option("--name", "-n", required=True, help="Server short name")
@click.option("--transport", default="stdio", type=click.Choice(["stdio", "http"]), help="Transport")
@click.option("--command", help="Command for stdio transport")
@click.option("--args", default="", help="Arguments for stdio transport (comma-separated)")
@click.option("--url", help="URL for http transport")
@click.option("--env", multiple=True, help="Environment KEY=VALUE")
@click.option("--network-allow", multiple=True, help="Allowed network hosts")
@click.option("--isolation", type=click.Choice(["firecracker", "gvisor", "bwrap", "none"]))
@click.option("--config", "config_path", default=lambda: str(Path.home() / ".terradev" / "mcp_registry.json"), help="Registry file path")
def mcp_registry_add(
    name,
    transport,
    command,
    args,
    url,
    env,
    network_allow,
    isolation,
    config_path,
):
    """Add an MCP server to the registry."""
    env_dict: Dict[str, str] = {}
    for e in env:
        if "=" not in e:
            raise click.BadOptionUsage("--env", f"Invalid env entry {e}")
        k, v = e.split("=", 1)
        env_dict[k] = v

    try:
        server = McpServerDefinition(
            name=name,
            transport=transport,
            command=command,
            args=[a.strip() for a in args.split(",") if a.strip()],
            url=url,
            env=env_dict,
            network_allow=list(network_allow),
            isolation=isolation,
        )
    except ValidationError as exc:
        raise click.BadOptionUsage("--name", f"Invalid MCP server definition: {exc}") from exc

    try:
        reg = McpRegistry.load(Path(config_path))
    except McpError as exc:
        raise click.BadOptionUsage("--config", f"{exc}") from exc
    reg.add(server)
    reg.save(Path(config_path))
    click.echo(f"OK: Added MCP server {name} to {config_path}")


@registry_group.command("list")
@click.option("--config", "config_path", default=lambda: str(Path.home() / ".terradev" / "mcp_registry.json"), help="Registry file path")
@click.option("--format", type=click.Choice(["text", "json"]), default="text")
def mcp_registry_list(config_path, format):
    """List MCP servers in the registry."""
    try:
        reg = McpRegistry.load(Path(config_path))
    except McpError as exc:
        raise click.BadOptionUsage("--config", f"{exc}") from exc
    if format == "json":
        click.echo(json.dumps([s.model_dump() for s in reg.servers], indent=2, default=str))
    else:
        click.echo(f"MCP REGISTRY ({len(reg.servers)})")
        for s in reg.servers:
            click.echo(f"  {s.name:<20} {s.transport:<8} {s.command or s.url}")


@mcp.command("serve")
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio", "http"]),
    help="Transport for the bridge",
)
@click.option(
    "--config",
    "-c",
    default=lambda: str(Path.home() / ".terradev" / "mcp_registry.json"),
    type=click.Path(exists=False, dir_okay=False),
    help="Path to the MCP server registry",
)
@click.option("--port", default=8080, help="HTTP port")
@click.option("--host", default="127.0.0.1", help="HTTP host")
def mcp_serve(transport, config, port, host):
    """Start the MCP bridge and expose stdio or HTTP endpoints."""
    import asyncio

    async def _main():
        with get_tracer("terradev.agent.mcp").trace("agent.mcp.serve"):
            bridge = McpBridge()
            config_path = Path(config)
            if config_path.exists():
                await bridge.load_registry(config_path)

            server = McpServer(bridge)
            try:
                if transport == "stdio":
                    await server.serve_stdio()
                else:
                    await server.serve_http(port=port, host=host)
            finally:
                await bridge.close()

    try:
        asyncio.run(_main())
    except (click.ClickException, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1)


@mcp.command("call")
@click.option("--server", "-s", required=True, help="Server name or full server.tool")
@click.option("--tool", "-t", help="Tool name if --server is a server name")
@click.option("--input", "-i", default="{}", help="JSON arguments for the tool")
@click.option("--config", "-c", default=lambda: str(Path.home() / ".terradev" / "mcp_registry.json"), help="Registry file path")
@click.option("--timeout", default=30.0, help="Call timeout")
@click.option("--format", type=click.Choice(["text", "json"]), default="text")
def mcp_call(server, tool, input, config, timeout, format):
    """Call a tool on a single MCP server."""
    import asyncio

    async def _main():
        with get_tracer("terradev.agent.mcp").trace("agent.mcp.call"):
            reg = McpRegistry.load(Path(config))
            definition = reg.get(server)
            if not definition:
                raise click.BadOptionUsage("--server", f"Server {server} not found in registry")

            if not tool:
                raise click.BadOptionUsage("--tool", "--tool is required")
            try:
                arguments = json.loads(input) if input else {}
            except json.JSONDecodeError as exc:
                raise click.BadOptionUsage("--input", f"Invalid JSON: {exc}") from exc

            transport = _create_transport(definition)
            conn = McpServerConnection(server, definition, transport)
            await conn.start()
            try:
                result = await conn.call_tool(tool, arguments)

                if format == "json":
                    click.echo(json.dumps(result, indent=2, default=str))
                else:
                    click.echo("OK:")
                    click.echo(json.dumps(result, indent=2, default=str))

                return result
            finally:
                await conn.close()

    try:
        asyncio.run(_main())
    except (click.ClickException, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1)
