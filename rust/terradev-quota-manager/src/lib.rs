#![allow(non_local_definitions)]

mod manager;
mod types;

use manager::QuotaManager;
use pyo3::prelude::*;
use types::{Quota, QuotaRequest};

#[pymodule]
fn terradev_quota_manager(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyQuotaManager>()?;
    m.add_class::<PyQuota>()?;
    m.add_class::<PyQuotaRequest>()?;
    Ok(())
}

#[pyclass]
#[allow(non_local_definitions)]
pub struct PyQuotaManager {
    inner: QuotaManager,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyQuotaManager {
    #[new]
    fn new() -> Self {
        Self {
            inner: QuotaManager::new(),
        }
    }
    
    fn set_quota(&self, resource: String, limit: u64) {
        self.inner.set_quota(resource, limit);
    }
    
    fn check_quota(&self, resource: String, amount: u64) -> PyResult<()> {
        let request = QuotaRequest { resource, amount };
        self.inner.check_quota(&request)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    fn consume_quota(&self, resource: String, amount: u64) -> PyResult<()> {
        let request = QuotaRequest { resource, amount };
        self.inner.consume_quota(&request)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    fn release_quota(&self, resource: String, amount: u64) {
        self.inner.release_quota(&resource, amount);
    }
    
    fn get_quota(&self, resource: String) -> Option<PyQuota> {
        self.inner.get_quota(&resource).map(|q| q.into())
    }
    
    fn list_quotas(&self) -> Vec<PyQuota> {
        self.inner.list_quotas().into_iter().map(|q| q.into()).collect()
    }
    
    fn reset_quota(&self, resource: String) {
        self.inner.reset_quota(&resource);
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyQuota {
    #[pyo3(get)]
    pub resource: String,
    #[pyo3(get)]
    pub limit: u64,
    #[pyo3(get)]
    pub used: u64,
    #[pyo3(get)]
    pub remaining: u64,
}

impl From<Quota> for PyQuota {
    fn from(q: Quota) -> Self {
        Self {
            resource: q.resource,
            limit: q.limit,
            used: q.used,
            remaining: q.remaining,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyQuotaRequest {
    #[pyo3(get, set)]
    pub resource: String,
    #[pyo3(get, set)]
    pub amount: u64,
}
