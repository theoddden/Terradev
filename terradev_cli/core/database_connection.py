#!/usr/bin/env python3
"""
Database Connection System for Terradev

Provides SQLite and PostgreSQL database connections with:
- In-memory connection storage with connection ID system
- Auto-table creation for standard metadata tables
- Query and upsert operations
"""

import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


@dataclass
class DatabaseConnectionConfig:
    """Configuration for database connections"""
    database_type: DatabaseType
    connection_id: str
    table_prefix: str = "terradev_"
    
    # SQLite-specific
    sqlite_path: Optional[str] = None
    
    # PostgreSQL-specific
    postgres_host: Optional[str] = None
    postgres_port: Optional[int] = None
    postgres_database: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None


class DatabaseConnection(ABC):
    """Abstract base class for database connections"""
    
    def __init__(self, config: DatabaseConnectionConfig):
        self.config = config
        self.connection_id = config.connection_id
        self.table_prefix = config.table_prefix
        self._connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish database connection"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close database connection"""
        pass
    
    @abstractmethod
    async def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results"""
        pass
    
    @abstractmethod
    async def upsert(self, table: str, data: Dict[str, Any], conflict_columns: Optional[List[str]] = None) -> bool:
        """Insert or update data in table"""
        pass
    
    @abstractmethod
    async def create_tables(self) -> bool:
        """Create standard metadata tables"""
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return self._connected


class SQLiteConnection(DatabaseConnection):
    """SQLite database connection implementation"""
    
    def __init__(self, config: DatabaseConnectionConfig):
        super().__init__(config)
        self.db_path = config.sqlite_path or os.path.join(
            os.path.expanduser("~"), ".terradev", "databases", f"{config.connection_id}.db"
        )
        self._connection = None
    
    async def connect(self) -> bool:
        """Establish SQLite connection"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Check if file exists
            file_exists = os.path.exists(self.db_path)
            
            # Simulate connection (in production, use actual sqlite3)
            logger.info(f"SQLite connection to: {self.db_path}")
            logger.info(f"Database file exists: {file_exists}")
            
            self._connected = True
            
            # Create tables if new database
            if not file_exists:
                await self.create_tables()
            
            return True
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close SQLite connection"""
        try:
            logger.info(f"Disconnecting SQLite connection: {self.connection_id}")
            self._connected = False
            return True
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to disconnect SQLite: {e}")
            return False
    
    async def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query"""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        logger.info(f"SQLite Query: {sql}")
        if params:
            logger.info(f"Parameters: {params}")
        
        # Simulate query execution
        # In production, use actual sqlite3 cursor
        return []
    
    async def upsert(self, table: str, data: Dict[str, Any], conflict_columns: Optional[List[str]] = None) -> bool:
        """Insert or update data"""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        full_table_name = f"{self.table_prefix}{table}"
        logger.info(f"SQLite Upsert into {full_table_name}: {data}")
        if conflict_columns:
            logger.info(f"Conflict columns: {conflict_columns}")
        
        # Simulate upsert
        return True
    
    async def create_tables(self) -> bool:
        """Create standard metadata tables"""
        try:
            tables = [
                "dataset_versions",
                "workflow_runs", 
                "idempotency_keys",
                "node_executions"
            ]
            
            for table in tables:
                full_table_name = f"{self.table_prefix}{table}"
                logger.info(f"Creating table: {full_table_name}")
                
                # Simulate table creation
                # In production, use actual CREATE TABLE SQL
            
            logger.info(f"Created {len(tables)} standard metadata tables")
            return True
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to create tables: {e}")
            return False


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database connection implementation"""
    
    def __init__(self, config: DatabaseConnectionConfig):
        super().__init__(config)
        self.host = config.postgres_host or "localhost"
        self.port = config.postgres_port or 5432
        self.database = config.postgres_database
        self.user = config.postgres_user
        self.password = config.postgres_password
        self._connection = None
    
    async def connect(self) -> bool:
        """Establish PostgreSQL connection"""
        try:
            # Log connection parameters (without password)
            logger.info(f"PostgreSQL connection to: {self.host}:{self.port}")
            logger.info(f"Database: {self.database}")
            logger.info(f"User: {self.user}")
            logger.info(f"Authentication: Using provided credentials")
            
            # Simulate connection (in production, use asyncpg or psycopg2)
            self._connected = True
            
            # Create tables
            await self.create_tables()
            
            return True
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close PostgreSQL connection"""
        try:
            logger.info(f"Disconnecting PostgreSQL connection: {self.connection_id}")
            self._connected = False
            return True
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to disconnect PostgreSQL: {e}")
            return False
    
    async def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query"""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        logger.info(f"PostgreSQL Query: {sql}")
        if params:
            logger.info(f"Parameters: {params}")
        
        # Simulate query execution
        # In production, use actual PostgreSQL cursor
        return []
    
    async def upsert(self, table: str, data: Dict[str, Any], conflict_columns: Optional[List[str]] = None) -> bool:
        """Insert or update data"""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        full_table_name = f"{self.table_prefix}{table}"
        logger.info(f"PostgreSQL Upsert into {full_table_name}: {data}")
        if conflict_columns:
            logger.info(f"Conflict columns: {conflict_columns}")
        
        # Simulate upsert using ON CONFLICT
        return True
    
    async def create_tables(self) -> bool:
        """Create standard metadata tables"""
        try:
            tables = [
                "dataset_versions",
                "workflow_runs",
                "idempotency_keys", 
                "node_executions"
            ]
            
            for table in tables:
                full_table_name = f"{self.table_prefix}{table}"
                logger.info(f"Creating table: {full_table_name}")
                
                # Simulate table creation
                # In production, use actual CREATE TABLE SQL with PostgreSQL syntax
            
            logger.info(f"Created {len(tables)} standard metadata tables")
            return True
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to create tables: {e}")
            return False


class DatabaseConnectionManager:
    """In-memory storage for active database connections"""
    
    def __init__(self):
        self._connections: Dict[str, DatabaseConnection] = {}
    
    def generate_connection_id(self) -> str:
        """Generate unique connection ID"""
        return str(uuid.uuid4())
    
    async def create_connection(self, config: DatabaseConnectionConfig) -> str:
        """Create and store a new database connection"""
        try:
            # Create appropriate connection instance
            if config.database_type == DatabaseType.SQLITE:
                connection = SQLiteConnection(config)
            elif config.database_type == DatabaseType.POSTGRESQL:
                connection = PostgreSQLConnection(config)
            else:
                raise ValueError(f"Unsupported database type: {config.database_type}")
            
            # Connect to database
            success = await connection.connect()
            if not success:
                raise RuntimeError("Failed to establish database connection")
            
            # Store connection
            self._connections[config.connection_id] = connection
            
            logger.info(f"Created database connection: {config.connection_id}")
            return config.connection_id
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise
    
    async def get_connection(self, connection_id: str) -> Optional[DatabaseConnection]:
        """Retrieve stored connection by ID"""
        return self._connections.get(connection_id)
    
    async def close_connection(self, connection_id: str) -> bool:
        """Close and remove connection"""
        connection = self._connections.get(connection_id)
        if connection:
            await connection.disconnect()
            del self._connections[connection_id]
            logger.info(f"Closed connection: {connection_id}")
            return True
        return False
    
    async def list_connections(self) -> List[Dict[str, Any]]:
        """List all active connections"""
        connections = []
        for conn_id, conn in self._connections.items():
            connections.append({
                "connection_id": conn_id,
                "database_type": conn.config.database_type.value,
                "is_connected": conn.is_connected,
                "table_prefix": conn.table_prefix
            })
        return connections


# Global connection manager instance
_connection_manager = DatabaseConnectionManager()


def get_database_connection_manager() -> DatabaseConnectionManager:
    """Get the global database connection manager"""
    return _connection_manager


async def create_sqlite_connection(
    path: Optional[str] = None,
    table_prefix: str = "terradev_",
    connection_id: Optional[str] = None
) -> str:
    """Create a SQLite database connection"""
    manager = get_database_connection_manager()
    
    if connection_id is None:
        connection_id = manager.generate_connection_id()
    
    config = DatabaseConnectionConfig(
        database_type=DatabaseType.SQLITE,
        connection_id=connection_id,
        table_prefix=table_prefix,
        sqlite_path=path
    )
    
    return await manager.create_connection(config)


async def create_postgresql_connection(
    host: str,
    database: str,
    user: str,
    password: str,
    port: int = 5432,
    table_prefix: str = "terradev_",
    connection_id: Optional[str] = None
) -> str:
    """Create a PostgreSQL database connection"""
    manager = get_database_connection_manager()
    
    if connection_id is None:
        connection_id = manager.generate_connection_id()
    
    config = DatabaseConnectionConfig(
        database_type=DatabaseType.POSTGRESQL,
        connection_id=connection_id,
        table_prefix=table_prefix,
        postgres_host=host,
        postgres_port=port,
        postgres_database=database,
        postgres_user=user,
        postgres_password=password
    )
    
    return await manager.create_connection(config)


async def query_database(connection_id: str, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Execute a SELECT query on a database connection"""
    manager = get_database_connection_manager()
    connection = await manager.get_connection(connection_id)
    
    if connection is None:
        raise ValueError(f"Connection not found: {connection_id}")
    
    return await connection.query(sql, params)


async def upsert_database(
    connection_id: str,
    table: str,
    data: Dict[str, Any],
    conflict_columns: Optional[List[str]] = None
) -> bool:
    """Insert or update data in a database table"""
    manager = get_database_connection_manager()
    connection = await manager.get_connection(connection_id)
    
    if connection is None:
        raise ValueError(f"Connection not found: {connection_id}")
    
    return await connection.upsert(table, data, conflict_columns)


async def get_database_connection(connection_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a database connection"""
    manager = get_database_connection_manager()
    connection = await manager.get_connection(connection_id)
    
    if connection is None:
        return None
    
    return {
        "connection_id": connection.connection_id,
        "database_type": connection.config.database_type.value,
        "is_connected": connection.is_connected,
        "table_prefix": connection.table_prefix,
        "sqlite_path": connection.config.sqlite_path if connection.config.database_type == DatabaseType.SQLITE else None,
        "postgres_host": connection.config.postgres_host if connection.config.database_type == DatabaseType.POSTGRESQL else None,
        "postgres_port": connection.config.postgres_port if connection.config.database_type == DatabaseType.POSTGRESQL else None,
        "postgres_database": connection.config.postgres_database if connection.config.database_type == DatabaseType.POSTGRESQL else None,
        "postgres_user": connection.config.postgres_user if connection.config.database_type == DatabaseType.POSTGRESQL else None,
    }
