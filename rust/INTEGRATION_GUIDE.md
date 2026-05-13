# Rust Modules Integration Guide

This guide explains how to integrate the new Rust modules into the existing Terradev Python codebase.

## Overview

The Rust modules are designed to be drop-in replacements for existing Python implementations. Each module exposes a Python-friendly API via PyO3 bindings.

## Integration Strategy

### Phase 1: Build System Integration

Add to `pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "terradev_cli"
module-name = "terradev_rust"
```

### Phase 2: Gradual Migration

#### 1. State Machine Engine

**File**: `terradev_cli/core/job_state_manager.py`

```python
# Add at top
try:
    from terradev_state_machine import JobStateMachine as RustJobStateMachine
    USE_RUST_STATE_MACHINE = True
except ImportError:
    USE_RUST_STATE_MACHINE = False
    print("Rust state machine not available, using Python fallback")

# Modify JobStateManager class
class JobStateManager:
    def __init__(self, db_path: Optional[str] = None):
        # ... existing init ...
        if USE_RUST_STATE_MACHINE:
            self._rust_engines: Dict[str, RustJobStateMachine] = {}
    
    def create_job(self, config: Dict[str, Any]) -> JobRecord:
        job_id = str(uuid.uuid4())
        
        if USE_RUST_STATE_MACHINE:
            # Create Rust state machine for this job
            self._rust_engines[job_id] = RustJobStateMachine(job_id)
        
        # ... existing Python logic ...
        return record
    
    def update_status(self, job_id: str, status: JobStatus, **kwargs):
        if USE_RUST_STATE_MACHINE and job_id in self._rust_engines:
            engine = self._rust_engines[job_id]
            
            # Use Rust state transitions
            if status == JobStatus.PREFLIGHT:
                engine.to_preflight()
            elif status == JobStatus.LAUNCHING:
                engine.to_launching(kwargs.get('nodes', []))
            elif status == JobStatus.RUNNING:
                engine.to_running(kwargs.get('total_steps', 0))
            elif status == JobStatus.COMPLETED:
                engine.to_completed(kwargs.get('final_step', 0))
            elif status == JobStatus.FAILED:
                engine.to_failed(kwargs.get('error', ''), kwargs.get('step', 0))
        
        # ... existing Python logic as fallback ...
```

#### 2. Resource Pool Manager

**File**: `terradev_cli/core/warm_pool_manager.py`

```python
# Add at top
try:
    from terradev_resource_pool import PyResourcePool, PyPooledResource, PyEvictionPolicy
    USE_RUST_RESOURCE_POOL = True
except ImportError:
    USE_RUST_RESOURCE_POOL = False

# Modify WarmPoolManager class
class WarmPoolManager:
    def __init__(self, config: WarmPoolConfig, config_dir: Optional[Path] = None):
        # ... existing init ...
        if USE_RUST_RESOURCE_POOL:
            self._rust_pool = PyResourcePool(
                pool_name="warm-pool",
                max_size=config.max_warm_models,
                policy=PyEvictionPolicy(
                    policy_type="idle_timeout",
                    timeout_seconds=config.idle_eviction_minutes * 60
                )
            )
    
    def add_to_pool(self, model_id: str, endpoint: str):
        if USE_RUST_RESOURCE_POOL:
            resource = PyPooledResource(
                id=model_id,
                resource_type="model",
                endpoint=endpoint,
                created_at=datetime.now().isoformat(),
                last_used=datetime.now().isoformat(),
                priority=self.model_priorities.get(model_id, 0)
            )
            self._rust_pool.add(resource)
        
        # ... existing Python logic ...
```

#### 3. Telemetry Pipeline

**File**: Create new file `terradev_cli/core/rust_telemetry.py`

```python
try:
    from terradev_telemetry import PyTelemetryPipeline
    USE_RUST_TELEMETRY = True
except ImportError:
    USE_RUST_TELEMETRY = False

class RustTelemetryBackend:
    def __init__(self):
        if not USE_RUST_TELEMETRY:
            raise ImportError("Rust telemetry not available")
        self.pipeline = PyTelemetryPipeline()
    
    def record(self, name: str, value: float, tags: List[Tuple[str, str]]):
        self.pipeline.record_value(name, value, tags)
    
    def get_histogram(self, name: str) -> Optional[Dict[str, Any]]:
        hist = self.pipeline.get_histogram(name)
        if hist:
            return {
                "min": hist.min,
                "max": hist.max,
                "mean": hist.mean,
                "p50": hist.p50,
                "p95": hist.p95,
                "p99": hist.p99,
                "count": hist.count,
                "sum": hist.sum,
            }
        return None
```

**Integrate in** `terradev_cli/core/monitoring/telemetry.py`:

```python
from .rust_telemetry import RustTelemetryBackend

class TelemetryManager:
    def __init__(self):
        try:
            self.rust_backend = RustTelemetryBackend()
        except ImportError:
            self.rust_backend = None
            logger.info("Rust telemetry not available, using Python backend")
    
    def record_metric(self, name: str, value: float, tags: List[Tuple[str, str]]):
        if self.rust_backend:
            self.rust_backend.record(name, value, tags)
        else:
            # Python fallback
            self._record_python(name, value, tags)
```

#### 4. Snapshot Manager

**File**: `terradev_cli/core/checkpoint_manager.py`

```python
try:
    from terradev_snapshot_manager import PySnapshotManager, PyModelState
    USE_RUST_SNAPSHOT = True
except ImportError:
    USE_RUST_SNAPSHOT = False

class CheckpointManager:
    def __init__(self):
        if USE_RUST_SNAPSHOT:
            self._rust_manager = PySnapshotManager(compression_level=3)
    
    def save_checkpoint(self, state: ModelState) -> str:
        if USE_RUST_SNAPSHOT:
            # Convert to Rust-compatible format
            rust_state = PyModelState(
                job_id=state.job_id,
                step=state.step,
                model_weights=state.model_weights,
                optimizer_state=state.optimizer_state,
                metadata=json.dumps(state.metadata),
                created_at=state.created_at.isoformat()
            )
            
            compressed = self._rust_manager.save_snapshot(rust_state)
            path = f"{self.checkpoint_dir}/{state.job_id}_step{state.step}.bin"
            
            with open(path, 'wb') as f:
                f.write(compressed)
            
            return path
        
        # ... existing Python logic ...
```

#### 5. Distributed Lock Manager

**File**: Create new file `terradev_cli/core/distributed_lock.py`

```python
try:
    from terradev_distributed_lock import PyDistributedLock
    USE_RUST_LOCK = True
except ImportError:
    USE_RUST_LOCK = False

class DistributedLockManager:
    def __init__(self):
        if USE_RUST_LOCK:
            self._rust_lock = PyDistributedLock()
        else:
            self._locks: Dict[str, Tuple[str, datetime]] = {}
    
    async def acquire(self, key: str, holder: str, ttl_seconds: int = 3600) -> Optional[str]:
        if USE_RUST_LOCK:
            grant = await self._rust_lock.acquire(key, holder, ttl_seconds)
            return grant.lease_id
        else:
            # Python fallback with in-memory dict
            if key in self._locks:
                holder, expiry = self._locks[key]
                if datetime.now() < expiry:
                    return None
            lease_id = str(uuid.uuid4())
            self._locks[key] = (lease_id, datetime.now() + timedelta(seconds=ttl_seconds))
            return lease_id
```

#### 6. Connection Pool

**File**: `terradev_cli/providers/base_provider.py`

```python
try:
    from terradev_connection_pool import PyConnectionPool, PyConnectionConfig
    USE_RUST_POOL = True
except ImportError:
    USE_RUST_POOL = False

class BaseProvider:
    def __init__(self, config: ProviderConfig):
        # ... existing init ...
        
        if USE_RUST_POOL:
            self._rust_pool = PyConnectionPool(
                PyConnectionConfig(
                    base_url=self.api_base,
                    max_connections=config.max_connections or 100,
                    timeout_seconds=config.timeout or 30,
                    keep_alive=True
                )
            )
    
    async def _make_request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        if USE_RUST_POOL:
            # Use Rust pool for connection management
            # Would need to expose the reqwest::Client from Rust
            pass
        
        # ... existing Python logic ...
```

#### 7. Event Bus

**File**: `terradev_cli/core/event_system.py` (or create new)

```python
try:
    from terradev_event_bus import PyEventBus, PyEvent
    USE_RUST_EVENT_BUS = True
except ImportError:
    USE_RUST_EVENT_BUS = False

class EventBus:
    def __init__(self):
        if USE_RUST_EVENT_BUS:
            self._rust_bus = PyEventBus()
        else:
            self._subscribers: Dict[str, List[Callable]] = {}
    
    def publish(self, event_type: str, data: Dict[str, Any]):
        if USE_RUST_EVENT_BUS:
            event = PyEvent(
                event_type=event_type,
                data=data
            )
            self._rust_bus.publish(event)
        else:
            # Python fallback
            for callback in self._subscribers.get(event_type, []):
                callback(data)
    
    def subscribe(self, event_type: str, callback: Callable) -> str:
        if USE_RUST_EVENT_BUS:
            return self._rust_bus.subscribe()
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            return str(id(callback))
```

#### 8. Cache Eviction Engine

**File**: `terradev_cli/core/cache_manager.py` (or create new)

```python
try:
    from terradev_cache_eviction import PyCacheEngine, PyCacheEntry, PyEvictionPolicy
    USE_RUST_CACHE = True
except ImportError:
    USE_RUST_CACHE = False

class CacheManager:
    def __init__(self, max_capacity: int = 1000, policy: str = "tinylfu"):
        if USE_RUST_CACHE:
            self._rust_cache = PyCacheEngine(
                max_capacity=max_capacity,
                policy=PyEvictionPolicy(policy_type=policy)
            )
        else:
            self._cache: Dict[str, Any] = {}
    
    def put(self, key: str, value: Any, size_bytes: int = 0):
        if USE_RUST_CACHE:
            entry = PyCacheEntry(
                key=key,
                value=json.dumps(value),
                size_bytes=size_bytes,
                created_at=datetime.now().isoformat(),
                last_accessed=datetime.now().isoformat(),
                access_count=0
            )
            self._rust_cache.put(entry)
        else:
            self._cache[key] = value
    
    def get(self, key: str) -> Optional[Any]:
        if USE_RUST_CACHE:
            entry = self._rust_cache.get(key)
            if entry:
                return json.loads(entry.value)
            return None
        else:
            return self._cache.get(key)
```

## Build and Installation

### Development

```bash
# Build Rust modules in development mode
cd rust
maturin develop

# Or build all modules
cargo build
```

### Production

```bash
# Build release versions
cd rust
maturin build --release

# Install from wheel
pip install target/wheels/terradev_*.whl
```

### CI/CD Integration

Add to `.github/workflows/build.yml`:

```yaml
- name: Install Rust toolchain
  uses: actions-rs/toolchain@v1
  with:
    toolchain: stable

- name: Build Rust modules
  run: |
    cd rust
    cargo build --release
    maturin build --release

- name: Install Terradev
  run: |
    pip install target/wheels/terradev_*.whl
```

## Testing

### Unit Tests

```python
# tests/test_rust_integration.py
import pytest

def test_state_machine_rust():
    from terradev_state_machine import JobStateMachine
    
    job = JobStateMachine("test-job")
    assert job.status == "created"
    
    job.to_preflight()
    assert job.status == "preflight"
    
    job.to_launching(["node-1"])
    assert job.status == "launching"
    
    with pytest.raises(ValueError):
        job.to_preflight()  # Invalid transition

def test_resource_pool_rust():
    from terradev_resource_pool import PyResourcePool, PyPooledResource, PyEvictionPolicy
    from datetime import datetime, timezone
    
    pool = PyResourcePool(
        pool_name="test-pool",
        max_size=5,
        policy=PyEvictionPolicy(policy_type="lru")
    )
    
    resource = PyPooledResource(
        id="test-resource",
        resource_type="gpu",
        endpoint="http://test",
        created_at=datetime.now(timezone.utc).isoformat(),
        last_used=datetime.now(timezone.utc).isoformat(),
        priority=1
    )
    
    pool.add(resource)
    assert pool.size() == 1
    
    retrieved = pool.get("test-resource")
    assert retrieved is not None
    assert retrieved.endpoint == "http://test"
```

### Integration Tests

```python
# tests/test_job_manager_integration.py
def test_job_manager_with_rust_state_machine():
    from terradev_cli.core.job_state_manager import JobStateManager
    
    manager = JobStateManager()
    
    config = {
        "name": "test-job",
        "framework": "pytorch",
        "nodes": ["node-1"],
        "total_steps": 100
    }
    
    job = manager.create_job(config)
    
    # Transition through states
    manager.update_status(job.id, JobStatus.PREFLIGHT)
    assert manager.get_job(job.id).status == JobStatus.PREFLIGHT
    
    manager.update_status(job.id, JobStatus.LAUNCHING, nodes=["node-1"])
    assert manager.get_job(job.id).status == JobStatus.LAUNCHING
```

## Performance Validation

Run benchmarks to validate performance improvements:

```python
# benchmarks/rust_vs_python.py
import time
from terradev_state_machine import JobStateMachine as RustJobStateMachine

def benchmark_state_transitions():
    # Python baseline
    start = time.time()
    for i in range(10000):
        job = JobStateMachinePython(f"job-{i}")
        job.to_preflight()
        job.to_launching(["node-1"])
        job.to_running(1000)
    python_time = time.time() - start
    
    # Rust implementation
    start = time.time()
    for i in range(10000):
        job = RustJobStateMachine(f"job-{i}")
        job.to_preflight()
        job.to_launching(["node-1"])
        job.to_running(1000)
    rust_time = time.time() - start
    
    print(f"Python: {python_time:.3f}s")
    print(f"Rust: {rust_time:.3f}s")
    print(f"Speedup: {python_time/rust_time:.2f}x")
```

## Rollback Strategy

If issues arise, each module has a Python fallback:

```python
try:
    from terradev_state_machine import JobStateMachine
except ImportError:
    # Use Python implementation
    JobStateMachine = PythonJobStateMachine
```

This ensures zero downtime during gradual migration.
