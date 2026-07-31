#!/usr/bin/env python3
"""
LoRA Adapter Registry - Centralized metadata tracking for production-grade LoRA management

Tracks adapter versions, replica distribution, tenant mappings, and provides
persistence layer for cross-replica coordination.
"""

from contextlib import closing
import sqlite3
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class AdapterStatus(str, Enum):
    """Adapter lifecycle status"""
    REGISTERED = "registered"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class AdapterVersion:
    """A single version of a LoRA adapter"""
    version_id: str  # UUID
    adapter_name: str
    base_model: str
    path: str
    rank: int
    created_at: datetime
    training_data_hash: Optional[str] = None  # For drift detection
    performance_metrics: Dict[str, float] = field(default_factory=dict)  # eval scores
    status: AdapterStatus = AdapterStatus.REGISTERED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterVersion":
        """Create from dictionary"""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('status'), str):
            data['status'] = AdapterStatus(data['status'])
        return cls(**data)


@dataclass
class AdapterReplicaState:
    """State of an adapter on a specific replica"""
    replica_id: str  # K8s pod ID or instance IP
    adapter_name: str
    version_id: str
    loaded_at: datetime
    last_used: datetime
    memory_footprint_gb: float = 0.0
    is_healthy: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['loaded_at'] = self.loaded_at.isoformat()
        data['last_used'] = self.last_used.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterReplicaState":
        """Create from dictionary"""
        if isinstance(data.get('loaded_at'), str):
            data['loaded_at'] = datetime.fromisoformat(data['loaded_at'])
        if isinstance(data.get('last_used'), str):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclass
class TenantMapping:
    """Mapping from tenant to adapter"""
    tenant_id: str
    adapter_name: str
    assigned_at: datetime
    priority: int = 0  # Higher priority = preferred adapter

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['assigned_at'] = self.assigned_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenantMapping":
        """Create from dictionary"""
        if isinstance(data.get('assigned_at'), str):
            data['assigned_at'] = datetime.fromisoformat(data['assigned_at'])
        return cls(**data)


class AdapterRegistry:
    """
    Centralized registry for LoRA adapter metadata and replica state.

    Provides:
    - Version tracking with rollback support
    - Replica distribution tracking
    - Tenant-to-adapter mapping
    - SQLite persistence
    """

    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else Path.home() / ".terradev" / "lora_registry.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"AdapterRegistry initialized with db at {self.db_path}")

    def _init_db(self):
        """Initialize SQLite database schema"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adapter_versions (
                    version_id TEXT PRIMARY KEY,
                    adapter_name TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    path TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    training_data_hash TEXT,
                    performance_metrics TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS replica_states (
                    replica_id TEXT NOT NULL,
                    adapter_name TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    loaded_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    memory_footprint_gb REAL DEFAULT 0.0,
                    is_healthy BOOLEAN DEFAULT 1,
                    PRIMARY KEY (replica_id, adapter_name)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS tenant_mappings (
                    tenant_id TEXT PRIMARY KEY,
                    adapter_name TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    priority INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_adapter_name 
                ON adapter_versions(adapter_name)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_replica_adapter 
                ON replica_states(adapter_name)
            """)

            conn.commit()

    # ── Adapter Version Management ──

    def register_adapter(
        self,
        adapter_name: str,
        base_model: str,
        path: str,
        rank: int = 64,
        training_data_hash: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version_id: Optional[str] = None,
        status: Optional[AdapterStatus] = None,
    ) -> AdapterVersion:
        """Register a new adapter version"""
        version_id = version_id or str(uuid.uuid4())
        version = AdapterVersion(
            version_id=version_id,
            adapter_name=adapter_name,
            base_model=base_model,
            path=path,
            rank=rank,
            created_at=datetime.now(),
            training_data_hash=training_data_hash,
            performance_metrics=performance_metrics or {},
            metadata=metadata or {},
        )

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO adapter_versions 
                (version_id, adapter_name, base_model, path, rank, created_at, 
                 training_data_hash, performance_metrics, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.version_id,
                    version.adapter_name,
                    version.base_model,
                    version.path,
                    version.rank,
                    version.created_at.isoformat(),
                    version.training_data_hash,
                    json.dumps(version.performance_metrics),
                    version.status.value,
                    json.dumps(version.metadata),
                ),
            )
            conn.commit()

        logger.info(f"Registered adapter version {version_id} for {adapter_name}")
        return version

    def get_adapter_versions(self, adapter_name: str) -> List[AdapterVersion]:
        """Get all versions of an adapter"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT version_id, adapter_name, base_model, path, rank, created_at,
                       training_data_hash, performance_metrics, status, metadata
                FROM adapter_versions
                WHERE adapter_name = ?
                ORDER BY created_at DESC
                """,
                (adapter_name,),
            )
            rows = cursor.fetchall()

        versions = []
        for row in rows:
            version = AdapterVersion(
                version_id=row[0],
                adapter_name=row[1],
                base_model=row[2],
                path=row[3],
                rank=row[4],
                created_at=datetime.fromisoformat(row[5]),
                training_data_hash=row[6],
                performance_metrics=json.loads(row[7]) if row[7] else {},
                status=AdapterStatus(row[8]),
                metadata=json.loads(row[9]) if row[9] else {},
            )
            versions.append(version)

        return versions

    def get_active_version(self, adapter_name: str) -> Optional[AdapterVersion]:
        """Get the currently active version of an adapter"""
        versions = self.get_adapter_versions(adapter_name)
        for version in versions:
            if version.status == AdapterStatus.ACTIVE:
                return version
        return None

    def get_version(self, version_id: str) -> Optional[AdapterVersion]:
        """Get a specific version by ID"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT version_id, adapter_name, base_model, path, rank, created_at,
                       training_data_hash, performance_metrics, status, metadata
                FROM adapter_versions
                WHERE version_id = ?
                """,
                (version_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return AdapterVersion(
            version_id=row[0],
            adapter_name=row[1],
            base_model=row[2],
            path=row[3],
            rank=row[4],
            created_at=datetime.fromisoformat(row[5]),
            training_data_hash=row[6],
            performance_metrics=json.loads(row[7]) if row[7] else {},
            status=AdapterStatus(row[8]),
            metadata=json.loads(row[9]) if row[9] else {},
        )

    def mark_version_active(self, adapter_name: str, version_id: str) -> bool:
        """Mark a specific version as active (deactivates others)"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            # Deactivate all versions of this adapter
            conn.execute(
                """
                UPDATE adapter_versions
                SET status = ?
                WHERE adapter_name = ?
                """,
                (AdapterStatus.REGISTERED.value, adapter_name),
            )

            # Activate the target version
            cursor = conn.execute(
                """
                UPDATE adapter_versions
                SET status = ?
                WHERE version_id = ? AND adapter_name = ?
                """,
                (AdapterStatus.ACTIVE.value, version_id, adapter_name),
            )
            conn.commit()

            return cursor.rowcount > 0

    def update_performance_metrics(
        self, version_id: str, metrics: Dict[str, float]
    ) -> bool:
        """Update performance metrics for a version"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                UPDATE adapter_versions
                SET performance_metrics = ?
                WHERE version_id = ?
                """,
                (json.dumps(metrics), version_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    # ── Replica State Management ──

    def record_replica_load(
        self,
        replica_id: str,
        adapter_name: str,
        version_id: str,
        memory_footprint_gb: float = 0.0,
    ) -> bool:
        """Record that an adapter was loaded on a replica"""
        now = datetime.now()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO replica_states
                (replica_id, adapter_name, version_id, loaded_at, last_used, 
                 memory_footprint_gb, is_healthy)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    replica_id,
                    adapter_name,
                    version_id,
                    now.isoformat(),
                    now.isoformat(),
                    memory_footprint_gb,
                ),
            )
            conn.commit()
        return True

    def record_replica_unload(self, replica_id: str, adapter_name: str) -> bool:
        """Record that an adapter was unloaded from a replica"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                DELETE FROM replica_states
                WHERE replica_id = ? AND adapter_name = ?
                """,
                (replica_id, adapter_name),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_replica_last_used(self, replica_id: str, adapter_name: str) -> bool:
        """Update last used timestamp for a replica-adapter pair"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                UPDATE replica_states
                SET last_used = ?
                WHERE replica_id = ? AND adapter_name = ?
                """,
                (datetime.now().isoformat(), replica_id, adapter_name),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_replicas_with_adapter(self, adapter_name: str) -> List[AdapterReplicaState]:
        """List all replicas that have a specific adapter loaded"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT replica_id, adapter_name, version_id, loaded_at, last_used,
                       memory_footprint_gb, is_healthy
                FROM replica_states
                WHERE adapter_name = ?
                ORDER BY last_used DESC
                """,
                (adapter_name,),
            )
            rows = cursor.fetchall()

        states = []
        for row in rows:
            state = AdapterReplicaState(
                replica_id=row[0],
                adapter_name=row[1],
                version_id=row[2],
                loaded_at=datetime.fromisoformat(row[3]),
                last_used=datetime.fromisoformat(row[4]),
                memory_footprint_gb=row[5],
                is_healthy=bool(row[6]),
            )
            states.append(state)

        return states

    def get_replica_adapters(self, replica_id: str) -> List[AdapterReplicaState]:
        """Get all adapters loaded on a specific replica"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT replica_id, adapter_name, version_id, loaded_at, last_used,
                       memory_footprint_gb, is_healthy
                FROM replica_states
                WHERE replica_id = ?
                ORDER BY last_used DESC
                """,
                (replica_id,),
            )
            rows = cursor.fetchall()

        states = []
        for row in rows:
            state = AdapterReplicaState(
                replica_id=row[0],
                adapter_name=row[1],
                version_id=row[2],
                loaded_at=datetime.fromisoformat(row[3]),
                last_used=datetime.fromisoformat(row[4]),
                memory_footprint_gb=row[5],
                is_healthy=bool(row[6]),
            )
            states.append(state)

        return states

    # ── Tenant Mapping ──

    def map_tenant_to_adapter(
        self, tenant_id: str, adapter_name: str, priority: int = 0
    ) -> bool:
        """Map a tenant to an adapter"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tenant_mappings
                (tenant_id, adapter_name, assigned_at, priority)
                VALUES (?, ?, ?, ?)
                """,
                (tenant_id, adapter_name, datetime.now().isoformat(), priority),
            )
            conn.commit()
        return True

    def get_tenant_adapter(self, tenant_id: str) -> Optional[str]:
        """Get the adapter name for a tenant"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT adapter_name
                FROM tenant_mappings
                WHERE tenant_id = ?
                ORDER BY priority DESC
                LIMIT 1
                """,
                (tenant_id,),
            )
            row = cursor.fetchone()

        return row[0] if row else None

    def list_tenant_mappings(self) -> List[TenantMapping]:
        """List all tenant-to-adapter mappings"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT tenant_id, adapter_name, assigned_at, priority
                FROM tenant_mappings
                ORDER BY priority DESC
                """
            )
            rows = cursor.fetchall()

        mappings = []
        for row in rows:
            mapping = TenantMapping(
                tenant_id=row[0],
                adapter_name=row[1],
                assigned_at=datetime.fromisoformat(row[2]),
                priority=row[3],
            )
            mappings.append(mapping)

        return mappings

    # ── Query Helpers ──

    def list_all_adapters(self) -> List[str]:
        """List all adapter names in the registry"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT adapter_name
                FROM adapter_versions
                ORDER BY adapter_name
                """
            )
            rows = cursor.fetchall()

        return [row[0] for row in rows]

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            total_versions = conn.execute(
                "SELECT COUNT(*) FROM adapter_versions"
            ).fetchone()[0]
            total_replicas = conn.execute(
                "SELECT COUNT(DISTINCT replica_id) FROM replica_states"
            ).fetchone()[0]
            total_states = conn.execute(
                "SELECT COUNT(*) FROM replica_states"
            ).fetchone()[0]
            total_tenants = conn.execute(
                "SELECT COUNT(*) FROM tenant_mappings"
            ).fetchone()[0]
            active_versions = conn.execute(
                "SELECT COUNT(*) FROM adapter_versions WHERE status = ?",
                (AdapterStatus.ACTIVE.value,),
            ).fetchone()[0]

        return {
            "total_adapter_names": len(self.list_all_adapters()),
            "total_versions": total_versions,
            "active_versions": active_versions,
            "total_replicas": total_replicas,
            "total_replica_states": total_states,
            "total_tenants": total_tenants,
        }


def get_lora_registry(db_path: Optional[Path] = None) -> AdapterRegistry:
    """Get the singleton adapter registry instance"""
    return AdapterRegistry(db_path)
