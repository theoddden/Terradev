"""
MCP new-feature tool registry.

Add new experimental MCP tools here.  The module is imported by server.py
and its exports are merged into the main tool list at startup.

Exports:
    ALL_NEW_TOOLS  – list of mcp.types.Tool definitions
    COMMAND_MAP    – {tool_name: handler_callable} dict
"""

from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

# Import database connection module
try:
    from terradev_cli.core.database_connection import (
        create_sqlite_connection,
        create_postgresql_connection,
        query_database,
        upsert_database,
        get_database_connection,
    )
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("Database connection module not available")

# Import vault adapter
try:
    from terradev_cli.core.vault_adapter import VaultAdapter
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    logger.warning("Vault adapter not available")

# Import MCP types (optional)
try:
    from mcp.types import Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Tool = None
    logger.warning("mcp.types not available - MCP tools will be disabled")

ALL_NEW_TOOLS: List[Any] = []
COMMAND_MAP: Dict[str, Any] = {}

# Database MCP Tools
if DATABASE_AVAILABLE and MCP_AVAILABLE:

    # Tool: create_sqlite_connection
    ALL_NEW_TOOLS.append(
        Tool(
            name="create_sqlite_connection",
            description="Create a SQLite database connection with auto-table creation. Returns a connection ID for subsequent operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to SQLite database file (optional, defaults to ~/.terradev/databases/{connection_id}.db)"
                    },
                    "table_prefix": {
                        "type": "string",
                        "description": "Prefix for table names (default: terradev_)",
                        "default": "terradev_"
                    },
                    "connection_id": {
                        "type": "string",
                        "description": "Custom connection ID (optional, auto-generated if not provided)"
                    }
                }
            }
        )
    )

    # Tool: create_postgresql_connection
    ALL_NEW_TOOLS.append(
        Tool(
            name="create_postgresql_connection",
            description="Create a PostgreSQL database connection with auto-table creation. Returns a connection ID for subsequent operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "PostgreSQL host address"
                    },
                    "port": {
                        "type": "integer",
                        "description": "PostgreSQL port (default: 5432)",
                        "default": 5432
                    },
                    "database": {
                        "type": "string",
                        "description": "Database name"
                    },
                    "user": {
                        "type": "string",
                        "description": "Database user"
                    },
                    "password": {
                        "type": "string",
                        "description": "Database password"
                    },
                    "table_prefix": {
                        "type": "string",
                        "description": "Prefix for table names (default: terradev_)",
                        "default": "terradev_"
                    },
                    "connection_id": {
                        "type": "string",
                        "description": "Custom connection ID (optional, auto-generated if not provided)"
                    }
                },
                "required": ["host", "database", "user", "password"]
            }
        )
    )

    # Tool: query_database
    ALL_NEW_TOOLS.append(
        Tool(
            name="query_database",
            description="Execute a SELECT query on a database connection. Returns query results as a list of dictionaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "connection_id": {
                        "type": "string",
                        "description": "Database connection ID"
                    },
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters (optional, for parameterized queries)"
                    }
                },
                "required": ["connection_id", "sql"]
            }
        )
    )

    # Tool: upsert_database
    ALL_NEW_TOOLS.append(
        Tool(
            name="upsert_database",
            description="Insert or update data in a database table. Performs upsert operation (insert or update on conflict).",
            inputSchema={
                "type": "object",
                "properties": {
                    "connection_id": {
                        "type": "string",
                        "description": "Database connection ID"
                    },
                    "table": {
                        "type": "string",
                        "description": "Table name (without prefix)"
                    },
                    "data": {
                        "type": "object",
                        "description": "Data to insert/update as key-value pairs"
                    },
                    "conflict_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to check for conflicts (optional)"
                    }
                },
                "required": ["connection_id", "table", "data"]
            }
        )
    )

    # Tool: get_database_connection
    ALL_NEW_TOOLS.append(
        Tool(
            name="get_database_connection",
            description="Get information about a database connection including type, status, and configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "connection_id": {
                        "type": "string",
                        "description": "Database connection ID"
                    }
                },
                "required": ["connection_id"]
            }
        )
    )

    # Tool handlers
    async def handle_create_sqlite_connection(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle create_sqlite_connection tool call"""
        try:
            connection_id = await create_sqlite_connection(
                path=arguments.get("path"),
                table_prefix=arguments.get("table_prefix", "terradev_"),
                connection_id=arguments.get("connection_id")
            )
            return [{
                "type": "text",
                "text": f"Created SQLite connection with ID: {connection_id}"
            }]
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to create SQLite connection: {e}")
            return [{
                "type": "text",
                "text": f"Error creating SQLite connection: {str(e)}"
            }]

    async def handle_create_postgresql_connection(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle create_postgresql_connection tool call"""
        try:
            connection_id = await create_postgresql_connection(
                host=arguments["host"],
                port=arguments.get("port", 5432),
                database=arguments["database"],
                user=arguments["user"],
                password=arguments["password"],
                table_prefix=arguments.get("table_prefix", "terradev_"),
                connection_id=arguments.get("connection_id")
            )
            return [{
                "type": "text",
                "text": f"Created PostgreSQL connection with ID: {connection_id}"
            }]
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to create PostgreSQL connection: {e}")
            return [{
                "type": "text",
                "text": f"Error creating PostgreSQL connection: {str(e)}"
            }]

    async def handle_query_database(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle query_database tool call"""
        try:
            results = await query_database(
                connection_id=arguments["connection_id"],
                sql=arguments["sql"],
                params=arguments.get("params")
            )
            return [{
                "type": "text",
                "text": f"Query results: {results}"
            }]
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to query database: {e}")
            return [{
                "type": "text",
                "text": f"Error querying database: {str(e)}"
            }]

    async def handle_upsert_database(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle upsert_database tool call"""
        try:
            success = await upsert_database(
                connection_id=arguments["connection_id"],
                table=arguments["table"],
                data=arguments["data"],
                conflict_columns=arguments.get("conflict_columns")
            )
            return [{
                "type": "text",
                "text": f"Upsert {'succeeded' if success else 'failed'}"
            }]
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to upsert database: {e}")
            return [{
                "type": "text",
                "text": f"Error upserting database: {str(e)}"
            }]

    async def handle_get_database_connection(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle get_database_connection tool call"""
        try:
            connection_info = await get_database_connection(arguments["connection_id"])
            if connection_info:
                return [{
                    "type": "text",
                    "text": f"Connection info: {connection_info}"
                }]
            else:
                return [{
                    "type": "text",
                    "text": f"Connection not found: {arguments['connection_id']}"
                }]
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to get database connection: {e}")
            return [{
                "type": "text",
                "text": f"Error getting database connection: {str(e)}"
            }]

    # Register command handlers
    COMMAND_MAP["create_sqlite_connection"] = handle_create_sqlite_connection
    COMMAND_MAP["create_postgresql_connection"] = handle_create_postgresql_connection
    COMMAND_MAP["query_database"] = handle_query_database
    COMMAND_MAP["upsert_database"] = handle_upsert_database
    COMMAND_MAP["get_database_connection"] = handle_get_database_connection

# Vault MCP Tools
if VAULT_AVAILABLE and MCP_AVAILABLE:

    _VAULT_TOOLS = [
        Tool(
            name="vault_set",
            description="Store a secret in the Terradev vault. Values are encrypted at rest.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Cloud provider name (e.g. runpod, aws, vastai)",
                    },
                    "key": {
                        "type": "string",
                        "description": "Credential key name (e.g. api_key, secret_key)",
                    },
                    "value": {
                        "type": "string",
                        "description": "Secret value to store",
                    },
                    "no_persist": {
                        "type": "boolean",
                        "description": "Keep the secret in memory only; do not write the vault file",
                        "default": False,
                    },
                },
                "required": ["provider", "key", "value"],
            },
        ),
        Tool(
            name="vault_get",
            description="Retrieve a stored secret. By default the value is masked.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name",
                    },
                    "key": {
                        "type": "string",
                        "description": "Credential key name",
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "Return the raw secret value instead of a masked preview",
                        "default": False,
                    },
                },
                "required": ["provider", "key"],
            },
        ),
        Tool(
            name="vault_list",
            description="List stored provider and key names. Values are never shown.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="vault_remove",
            description="Remove a provider or a single key from the vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name",
                    },
                    "key": {
                        "type": "string",
                        "description": "Optional key name. If omitted, the whole provider is removed.",
                    },
                },
                "required": ["provider"],
            },
        ),
        Tool(
            name="vault_sync",
            description="Import TERRADEV_* environment variables into the vault for supported cloud providers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show what would be imported without persisting",
                        "default": False,
                    },
                    "no_persist": {
                        "type": "boolean",
                        "description": "Keep imported secrets in env only; do not write the vault file",
                        "default": False,
                    },
                    "all": {
                        "type": "boolean",
                        "description": "Import every TERRADEV_* variable, not just supported cloud providers",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="vault_verify",
            description="Check which providers are fully configured and which keys are missing.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="vault_env",
            description="Print environment-style export lines for a provider. By default values are masked.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name",
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "Return raw export values",
                        "default": False,
                    },
                },
                "required": ["provider"],
            },
        ),
        Tool(
            name="vault_run",
            description="Run a shell command with vault secrets injected into the environment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (will be split into arguments)",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Only inject secrets for this provider (optional)",
                    },
                    "no_exec": {
                        "type": "boolean",
                        "description": "Build the env and print export lines without running the command",
                        "default": False,
                    },
                },
                "required": ["command"],
            },
        ),
    ]

    ALL_NEW_TOOLS.extend(_VAULT_TOOLS)

    def _mask_secret(value: str) -> str:
        """Return a masked preview of a secret."""
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}***{value[-4:]}"

    async def handle_vault_set(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            vault = VaultAdapter()
            if arguments.get("no_persist"):
                vault._no_persist = True
            vault.set(arguments["provider"], arguments["key"], arguments["value"])
            return [{
                "type": "text",
                "text": f"Stored {arguments['provider']}.{arguments['key']}"
            }]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to store vault secret: {e}")
            return [{
                "type": "text",
                "text": f"Error storing vault secret: {str(e)}"
            }]

    async def handle_vault_get(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            vault = VaultAdapter()
            value = vault.get(arguments["provider"], arguments["key"])
            if value is None:
                return [{
                    "type": "text",
                    "text": f"No secret found for {arguments['provider']}.{arguments['key']}"
                }]
            if arguments.get("raw"):
                return [{"type": "text", "text": value}]
            return [{"type": "text", "text": _mask_secret(value)}]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get vault secret: {e}")
            return [{"type": "text", "text": f"Error getting vault secret: {str(e)}"}]

    async def handle_vault_list(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            vault = VaultAdapter()
            creds = vault.all_credentials()
            lines = [f"{provider}: {', '.join(sorted(keys))}" for provider, keys in sorted(creds.items())]
            if not lines:
                return [{"type": "text", "text": "No credentials in the vault."}]
            return [{"type": "text", "text": "\n".join(lines)}]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to list vault: {e}")
            return [{"type": "text", "text": f"Error listing vault: {str(e)}"}]

    async def handle_vault_remove(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            vault = VaultAdapter()
            removed = vault.remove(arguments["provider"], arguments.get("key"))
            if removed:
                return [{"type": "text", "text": f"Removed {arguments['provider']}" + (f".{arguments['key']}" if arguments.get('key') else "")}]
            return [{"type": "text", "text": f"No matching secret found for {arguments['provider']}"}]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to remove vault secret: {e}")
            return [{"type": "text", "text": f"Error removing vault secret: {str(e)}"}]

    async def handle_vault_sync(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            vault = VaultAdapter()
            if arguments.get("no_persist"):
                vault._no_persist = True
            base = vault.load_credentials()
            merged = vault.load_env_credentials(base, known_only=not arguments.get("all", False))
            imported = []
            for provider, provider_creds in merged.items():
                for key in provider_creds:
                    if provider not in base or key not in base.get(provider, {}):
                        imported.append(f"{provider}.{key}")
            if arguments.get("dry_run"):
                return [{"type": "text", "text": f"Would import {len(imported)} secret(s): " + ", ".join(imported)}]
            if not imported:
                return [{"type": "text", "text": "No new TERRADEV_* secrets to import."}]
            if not arguments.get("no_persist"):
                vault.save_credentials(merged)
            return [{"type": "text", "text": f"Imported {len(imported)} secret(s): " + ", ".join(imported)}]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to sync vault: {e}")
            return [{"type": "text", "text": f"Error syncing vault: {str(e)}"}]

    async def handle_vault_verify(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            vault = VaultAdapter()
            status = vault.verify()
            lines = ["Configured providers:"] + [f"  - {p}" for p in status["configured"]]
            if status["missing"]:
                lines.append("Missing keys:")
                for provider, keys in status["missing"].items():
                    lines.append(f"  - {provider}: {', '.join(keys)}")
            return [{"type": "text", "text": "\n".join(lines)}]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to verify vault: {e}")
            return [{"type": "text", "text": f"Error verifying vault: {str(e)}"}]

    async def handle_vault_env(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            import shlex
            vault = VaultAdapter()
            env = vault.to_env(arguments["provider"])
            if not env:
                return [{"type": "text", "text": f"No credentials for {arguments['provider']}"}]
            raw = arguments.get("raw", False)
            lines = []
            for name, value in env.items():
                if raw:
                    lines.append(f"export {name}={shlex.quote(value)}")
                else:
                    lines.append(f"export {name}={_mask_secret(value)}")
            return [{"type": "text", "text": "\n".join(lines)}]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to build vault env: {e}")
            return [{"type": "text", "text": f"Error building vault env: {str(e)}"}]

    async def handle_vault_run(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            import shlex
            import subprocess
            vault = VaultAdapter()
            run_env = vault.build_run_env(arguments.get("provider"))
            if arguments.get("no_exec"):
                lines = [
                    f"export {name}={shlex.quote(value)}"
                    for name, value in run_env.items()
                    if name.startswith(vault.ENV_PREFIX)
                ]
                return [{"type": "text", "text": "\n".join(lines) if lines else "No vault secrets to export."}]
            cmd = shlex.split(arguments["command"])
            proc = subprocess.run(cmd, env=run_env, capture_output=True, text=True)
            output = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0:
                return [{"type": "text", "text": f"Command failed (exit {proc.returncode}):\n{output}"}]
            return [{"type": "text", "text": output or "Command completed with no output."}]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to run command with vault: {e}")
            return [{"type": "text", "text": f"Error running command with vault: {str(e)}"}]

    # Register vault command handlers
    COMMAND_MAP["vault_set"] = handle_vault_set
    COMMAND_MAP["vault_get"] = handle_vault_get
    COMMAND_MAP["vault_list"] = handle_vault_list
    COMMAND_MAP["vault_remove"] = handle_vault_remove
    COMMAND_MAP["vault_sync"] = handle_vault_sync
    COMMAND_MAP["vault_verify"] = handle_vault_verify
    COMMAND_MAP["vault_env"] = handle_vault_env
    COMMAND_MAP["vault_run"] = handle_vault_run
