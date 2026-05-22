#![allow(non_local_definitions)]

mod lock;
mod types;

use chrono::Utc;
use lock::DistributedLock;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;
use types::{LockGrant, LockRequest};

#[pymodule]
fn terradev_distributed_lock(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDistributedLock>()?;
    m.add_class::<PyLockRequest>()?;
    m.add_class::<PyLockGrant>()?;
    Ok(())
}

#[pyclass]
pub struct PyDistributedLock {
    inner: DistributedLock,
    runtime: Arc<Runtime>,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyDistributedLock {
    #[new]
    fn new() -> Self {
        let runtime = Arc::new(Runtime::new().unwrap());
        Self {
            inner: DistributedLock::new(),
            runtime,
        }
    }

    fn acquire(&self, key: String, holder: String, ttl_seconds: u64) -> PyResult<PyLockGrant> {
        let request = LockRequest {
            key: key.clone(),
            holder: holder.clone(),
            ttl_seconds,
            requested_at: Utc::now(),
        };

        let grant = self
            .runtime
            .block_on(self.inner.acquire(request))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(grant.into())
    }

    fn release(&self, key: String, holder: String, lease_id: String) -> PyResult<()> {
        self.runtime
            .block_on(self.inner.release(&key, &holder, &lease_id))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    fn renew(
        &self,
        key: String,
        holder: String,
        lease_id: String,
        ttl_seconds: u64,
    ) -> PyResult<PyLockGrant> {
        let grant = self
            .runtime
            .block_on(self.inner.renew(&key, &holder, &lease_id, ttl_seconds))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(grant.into())
    }

    fn is_held(&self, key: String) -> PyResult<bool> {
        Ok(self.runtime.block_on(self.inner.is_held(&key)))
    }

    fn get_holder(&self, key: String) -> PyResult<Option<String>> {
        Ok(self.runtime.block_on(self.inner.get_holder(&key)))
    }

    fn cleanup_expired(&self) -> PyResult<usize> {
        Ok(self.runtime.block_on(self.inner.cleanup_expired()))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyLockRequest {
    #[pyo3(get, set)]
    pub key: String,
    #[pyo3(get, set)]
    pub holder: String,
    #[pyo3(get, set)]
    pub ttl_seconds: u64,
}

#[pyclass]
#[derive(Clone)]
pub struct PyLockGrant {
    #[pyo3(get)]
    pub key: String,
    #[pyo3(get)]
    pub holder: String,
    #[pyo3(get)]
    pub acquired_at: String,
    #[pyo3(get)]
    pub expires_at: String,
    #[pyo3(get)]
    pub lease_id: String,
}

impl From<LockGrant> for PyLockGrant {
    fn from(g: LockGrant) -> Self {
        Self {
            key: g.key,
            holder: g.holder,
            acquired_at: g.acquired_at.to_rfc3339(),
            expires_at: g.expires_at.to_rfc3339(),
            lease_id: g.lease_id,
        }
    }
}
