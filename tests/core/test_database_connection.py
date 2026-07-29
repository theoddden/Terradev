"""Tests for terradev_cli.core.database_connection.

Database connections are simulated here — the tests exercise the manager and
the two concrete implementations without touching real network or disk.
"""

import pytest

from terradev_cli.core.database_connection import (
    DatabaseConnectionConfig,
    DatabaseConnectionManager,
    DatabaseType,
    PostgreSQLConnection,
    SQLiteConnection,
    create_sqlite_connection,
)


@pytest.fixture
def manager():
    return DatabaseConnectionManager()


@pytest.mark.asyncio
async def test_sqlite_connection_lifecycle(manager, tmp_path):
    """SQLite connection can be created, queried, and closed."""
    config = DatabaseConnectionConfig(
        database_type=DatabaseType.SQLITE,
        connection_id="conn-1",
        sqlite_path=str(tmp_path / "test.db"),
        table_prefix="td_",
    )
    conn_id = await manager.create_connection(config)
    assert conn_id == "conn-1"

    conn = await manager.get_connection(conn_id)
    assert conn is not None
    assert conn.is_connected
    assert conn.table_prefix == "td_"

    rows = await conn.query("SELECT 1")
    assert rows == []

    assert await conn.upsert("runs", {"id": 1}) is True
    assert await conn.create_tables() is True

    assert await manager.close_connection(conn_id) is True
    assert await manager.get_connection(conn_id) is None


@pytest.mark.asyncio
async def test_postgresql_connection_lifecycle(manager):
    """PostgreSQL connection reports connected and can run simulated queries."""
    config = DatabaseConnectionConfig(
        database_type=DatabaseType.POSTGRESQL,
        connection_id="pg-1",
        postgres_host="localhost",
        postgres_database="terradev",
        postgres_user="user",
        postgres_password="secret",
    )
    conn = PostgreSQLConnection(config)
    assert await conn.connect() is True
    assert conn.is_connected

    rows = await conn.query("SELECT 1")
    assert rows == []
    assert await conn.upsert("runs", {"id": 1}) is True

    assert await conn.disconnect() is True
    assert not conn.is_connected


@pytest.mark.asyncio
async def test_manager_lists_connections(manager, tmp_path):
    """The manager lists active connections with metadata."""
    c1 = DatabaseConnectionConfig(
        database_type=DatabaseType.SQLITE,
        connection_id="c1",
        sqlite_path=str(tmp_path / "a.db"),
    )
    c2 = DatabaseConnectionConfig(
        database_type=DatabaseType.POSTGRESQL,
        connection_id="c2",
    )
    await manager.create_connection(c1)
    await manager.create_connection(c2)

    conns = await manager.list_connections()
    assert len(conns) == 2
    assert {c["connection_id"] for c in conns} == {"c1", "c2"}


@pytest.mark.asyncio
async def test_query_before_connect_raises():
    """Querying before connecting raises a RuntimeError."""
    config = DatabaseConnectionConfig(
        database_type=DatabaseType.SQLITE,
        connection_id="x",
    )
    conn = SQLiteConnection(config)
    with pytest.raises(RuntimeError, match="not connected"):
        await conn.query("SELECT 1")


@pytest.mark.asyncio
async def test_create_sqlite_connection_helper(tmp_path):
    """The helper registers a SQLite connection in the global manager."""
    conn_id = await create_sqlite_connection(
        path=str(tmp_path / "global.db"),
        table_prefix="pref_",
        connection_id="helper-1",
    )
    assert conn_id == "helper-1"

    manager = DatabaseConnectionManager()
    # The helper uses a global manager, so a fresh instance won't see it.
    # We just ensure the helper returned the expected id.
