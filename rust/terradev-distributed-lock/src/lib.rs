mod lock;
mod types;

use lock::DistributedLock;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use types::{LockError, LockGrant, LockRequest};
use chrono::Utc;

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
}

#[pymethods]
impl PyDistributedLock {
    #[new]
    fn new() -> Self {
        Self {
            inner: DistributedLock::new(),
        }
    }
    
    fn acquire(&self, key: String, holder: String, ttl_seconds: u64) -> PyResult<PyLockGrant> {
        Python::with_gil(|py| {
            let request = LockRequest {
                key: key.clone(),
                holder: holder.clone(),
                ttl_seconds,
                requested_at: Utc::now(),
            };
            
            pyo3_asyncio::tokio::future_into_py(py, async move {
                self.inner.acquire(request)
                    .map(|grant| grant.into())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        })
    }
    
    fn release(&self, key: String, holder: String, lease_id: String) -> PyResult<()> {
        Python::with_gil(|py| {
            pyo3_asyncio::tokio::future_into_py(py, async move {
                self.inner.release(&key, &holder, &lease_id)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        })
    }
    
    fn renew(&self, key: String, holder: String, lease_id: String, ttl_seconds: u64) -> PyResult<PyLockGrant> {
        Python::with_gil(|py| {
            pyo3_asyncio::tokio::future_into_py(py, async move {
                self.inner.renew(&key, &holder, &lease_id, ttl_seconds)
                    .map(|grant| grant.into())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        })
    }
    
    fn is_held(&self, key: String) -> PyResult<bool> {
        Python::with_gil(|py| {
            pyo3_asyncio::tokio::future_into_py(py, async move {
                Ok(self.inner.is_held(&key).await)
            })
        })
    }
    
    fn get_holder(&self, key: String) -> PyResult<Option<String>> {
        Python::with_gil(|py| {
            pyo3_asyncio::tokio::future_into_py(py, async move {
                Ok(self.inner.get_holder(&key).await)
            })
        })
    }
    
    fn cleanup_expired(&self) -> PyResult<usize> {
        Python::with_gil(|py| {
            pyo3_asyncio::tokio::future_into_py(py, async move {
                Ok(self.inner.cleanup_expired().await)
            })
        })
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
