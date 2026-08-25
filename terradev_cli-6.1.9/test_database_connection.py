#!/usr/bin/env python3
"""
Test script for database connection system
"""

import asyncio
import sys
import os

# Add the terradev_cli module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'terradev_cli'))

from terradev_cli.core.database_connection import (
    create_sqlite_connection,
    create_postgresql_connection,
    query_database,
    upsert_database,
    get_database_connection,
    DatabaseConnectionManager,
    DatabaseType,
    DatabaseConnectionConfig
)


async def test_sqlite_connection():
    """Test SQLite connection creation and operations"""
    print("Testing SQLite Connection...")
    
    try:
        # Create SQLite connection
        connection_id = await create_sqlite_connection(
            path="/tmp/test_terradev.db",
            table_prefix="test_"
        )
        print(f"✓ Created SQLite connection: {connection_id}")
        
        # Get connection info
        conn_info = await get_database_connection(connection_id)
        print(f"✓ Connection info: {conn_info}")
        
        # Test query
        results = await query_database(
            connection_id,
            "SELECT * FROM test_dataset_versions LIMIT 10"
        )
        print(f"✓ Query executed: {results}")
        
        # Test upsert
        success = await upsert_database(
            connection_id,
            "dataset_versions",
            {"version_hash": "test123", "dataset_id": "test_dataset"},
            conflict_columns=["version_hash"]
        )
        print(f"✓ Upsert executed: {success}")
        
        print("✓ SQLite connection test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ SQLite connection test failed: {e}\n")
        return False


async def test_postgresql_connection():
    """Test PostgreSQL connection creation and operations"""
    print("Testing PostgreSQL Connection...")
    
    try:
        # Create PostgreSQL connection (with test credentials)
        connection_id = await create_postgresql_connection(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_password",
            table_prefix="test_"
        )
        print(f"✓ Created PostgreSQL connection: {connection_id}")
        
        # Get connection info
        conn_info = await get_database_connection(connection_id)
        print(f"✓ Connection info: {conn_info}")
        
        # Test query
        results = await query_database(
            connection_id,
            "SELECT * FROM test_workflow_runs LIMIT 10"
        )
        print(f"✓ Query executed: {results}")
        
        # Test upsert
        success = await upsert_database(
            connection_id,
            "workflow_runs",
            {"run_id": "test_run_123", "status": "completed"},
            conflict_columns=["run_id"]
        )
        print(f"✓ Upsert executed: {success}")
        
        print("✓ PostgreSQL connection test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ PostgreSQL connection test failed: {e}\n")
        return False


async def test_connection_manager():
    """Test connection manager functionality"""
    print("Testing Connection Manager...")
    
    try:
        manager = DatabaseConnectionManager()
        
        # Test connection ID generation
        conn_id_1 = manager.generate_connection_id()
        conn_id_2 = manager.generate_connection_id()
        print(f"✓ Generated connection IDs: {conn_id_1}, {conn_id_2}")
        assert conn_id_1 != conn_id_2, "Connection IDs should be unique"
        
        # Test connection config
        config = DatabaseConnectionConfig(
            database_type=DatabaseType.SQLITE,
            connection_id=conn_id_1,
            table_prefix="test_",
            sqlite_path="/tmp/test_manager.db"
        )
        print(f"✓ Created connection config: {config.database_type}")
        
        print("✓ Connection manager test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Connection manager test failed: {e}\n")
        return False


async def main():
    """Run all tests"""
    print("=" * 50)
    print("Database Connection System Tests")
    print("=" * 50 + "\n")
    
    results = []
    
    # Test connection manager
    results.append(await test_connection_manager())
    
    # Test SQLite connection
    results.append(await test_sqlite_connection())
    
    # Test PostgreSQL connection
    results.append(await test_postgresql_connection())
    
    # Summary
    print("=" * 50)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 50)
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
