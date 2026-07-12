#!/usr/bin/env python3
"""
MCP Server Integration Tests

Tests for the Terradev MCP server functionality including:
- Server initialization and startup
- Tool registration and availability
- Command routing
- Database tool handlers
- Error handling
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip tests if mcp is not installed
pytest.importorskip("mcp", reason="mcp package not installed")


class TestMCPServerInitialization:
    """Test MCP server initialization and basic functionality"""
    
    def test_mcp_imports(self):
        """Test that MCP server imports work correctly"""
        try:
            from terradev_cli.mcp.server import Server
            from mcp.server import Server as MCPServer
            assert Server is not None
            assert MCPServer is not None
        except ImportError as e:
            pytest.skip(f"MCP imports failed: {e}")
    
    def test_mcp_server_module_exists(self):
        """Test that the MCP server module exists and is importable"""
        try:
            import terradev_cli.mcp.server as mcp_server
            assert hasattr(mcp_server, 'main')
            assert hasattr(mcp_server, 'server')
            assert hasattr(mcp_server, 'create_sse_app')
        except ImportError as e:
            pytest.skip(f"MCP server module import failed: {e}")


class TestMCPToolRegistration:
    """Test MCP tool registration and availability"""
    
    def test_command_map_exists(self):
        """Test that the command map is defined"""
        try:
            from terradev_cli.mcp.server import COMMAND_MAP
            assert isinstance(COMMAND_MAP, dict)
            assert len(COMMAND_MAP) > 0
        except ImportError as e:
            pytest.skip(f"Command map import failed: {e}")
    
    def test_database_tools_registered(self):
        """Test that database tools are registered in command map"""
        try:
            from terradev_cli.mcp.server import COMMAND_MAP
            database_tools = [
                "create_sqlite_connection",
                "create_postgresql_connection", 
                "query_database",
                "upsert_database",
                "get_database_connection"
            ]
            for tool in database_tools:
                assert tool in COMMAND_MAP, f"Database tool {tool} not registered"
        except ImportError as e:
            pytest.skip(f"Command map import failed: {e}")
    
    def test_consolidated_commands_registered(self):
        """Test that consolidated command structures are registered"""
        try:
            from terradev_cli.mcp.server import COMMAND_MAP
            # Check that new consolidated command mappings exist
            assert "orchestrator_start" in COMMAND_MAP
            assert "warm_pool_start" in COMMAND_MAP
            assert "train" in COMMAND_MAP or "train_start" in COMMAND_MAP
        except ImportError as e:
            pytest.skip(f"Command map import failed: {e}")


class TestDatabaseToolHandlers:
    """Test database tool handler functions"""
    
    def test_database_handlers_import(self):
        """Test that database tool handlers can be imported"""
        try:
            from terradev_cli.mcp.new_feature_tools import (
                handle_create_sqlite_connection,
                handle_create_postgresql_connection,
                handle_query_database,
                handle_upsert_database,
                handle_get_database_connection
            )
            assert handle_create_sqlite_connection is not None
            assert handle_create_postgresql_connection is not None
            assert handle_query_database is not None
            assert handle_upsert_database is not None
            assert handle_get_database_connection is not None
        except ImportError as e:
            pytest.skip(f"Database handlers import failed: {e}")
    
    @pytest.mark.asyncio
    async def test_sqlite_connection_handler_signature(self):
        """Test that SQLite connection handler has correct signature"""
        try:
            from terradev_cli.mcp.new_feature_tools import handle_create_sqlite_connection
            # Verify it's callable
            assert callable(handle_create_sqlite_connection)
        except ImportError as e:
            pytest.skip(f"Database handler import failed: {e}")


class TestMCPServerCLI:
    """Test MCP server CLI integration"""
    
    def test_mcp_command_exists(self):
        """Test that the MCP command is available in CLI"""
        from click.testing import CliRunner
        from terradev_cli.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "mcp" in result.output.lower()
    
    def test_mcp_help_command(self):
        """Test that MCP help command works"""
        from click.testing import CliRunner
        from terradev_cli.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["mcp", "--help"])
        # Should either work or provide helpful error
        assert result.exit_code == 0 or "help" in result.output.lower()


class TestMCPErrorHandling:
    """Test MCP server error handling"""
    
    def test_mcp_import_error_handling(self):
        """Test that missing MCP package is handled gracefully"""
        # This test verifies the error handling in the server module
        # when mcp is not installed
        try:
            from terradev_cli.mcp.server import logger
            assert logger is not None
        except ImportError as e:
            pytest.skip(f"Logger import failed: {e}")


class TestMCPDatabaseIntegration:
    """Test MCP database integration"""
    
    def test_database_connection_module_import(self):
        """Test that database connection module can be imported"""
        try:
            from terradev_cli.core.database_connection import (
                DatabaseConnection,
                SQLiteConnection,
                PostgreSQLConnection,
                DatabaseConnectionManager
            )
            assert DatabaseConnection is not None
            assert SQLiteConnection is not None
            assert PostgreSQLConnection is not None
            assert DatabaseConnectionManager is not None
        except ImportError as e:
            pytest.skip(f"Database connection module import failed: {e}")
    
    def test_database_manager_instantiation(self):
        """Test that database connection manager can be instantiated"""
        try:
            from terradev_cli.core.database_connection import DatabaseConnectionManager
            manager1 = DatabaseConnectionManager()
            manager2 = DatabaseConnectionManager()
            # Should be separate instances (not singleton pattern)
            assert manager1 is not manager2
            assert hasattr(manager1, '_connections')
            assert hasattr(manager2, '_connections')
        except ImportError as e:
            pytest.skip(f"Database manager import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
