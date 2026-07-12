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
