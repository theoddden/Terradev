#![allow(non_local_definitions)]

mod pool;
mod types;

use chrono::{DateTime, Utc};
use pool::ResourcePool;
use pyo3::prelude::*;
use types::{EvictionPolicy, PooledResource};

#[pymodule]
fn terradev_resource_pool(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyResourcePool>()?;
    m.add_class::<PyPooledResource>()?;
    m.add_class::<PyEvictionPolicy>()?;
    Ok(())
}

#[pyclass]
#[allow(non_local_definitions)]
pub struct PyResourcePool {
    inner: ResourcePool,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyResourcePool {
    #[new]
    fn new(pool_name: String, max_size: usize, policy: PyEvictionPolicy) -> Self {
        Self {
            inner: ResourcePool::new(pool_name, max_size, policy.into()),
        }
    }

    fn add(&mut self, resource: PyPooledResource) -> PyResult<()> {
        self.inner
            .add(resource.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn get(&self, id: String) -> Option<PyPooledResource> {
        self.inner.get(&id).map(|r| r.into())
    }

    fn remove(&self, id: String) -> Option<PyPooledResource> {
        self.inner.remove(&id).map(|r| r.into())
    }

    fn contains(&self, id: String) -> bool {
        self.inner.contains(&id)
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn list(&self) -> Vec<PyPooledResource> {
        self.inner.list().into_iter().map(|r| r.into()).collect()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPooledResource {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub resource_type: String,
    #[pyo3(get, set)]
    pub endpoint: String,
    #[pyo3(get, set)]
    pub created_at: String,
    #[pyo3(get, set)]
    pub last_used: String,
    #[pyo3(get, set)]
    pub priority: u32,
}

impl From<PooledResource> for PyPooledResource {
    fn from(r: PooledResource) -> Self {
        Self {
            id: r.id,
            resource_type: r.resource_type,
            endpoint: r.endpoint,
            created_at: r.created_at.to_rfc3339(),
            last_used: r.last_used.to_rfc3339(),
            priority: r.priority,
        }
    }
}

impl From<PyPooledResource> for PooledResource {
    fn from(p: PyPooledResource) -> Self {
        Self {
            id: p.id,
            resource_type: p.resource_type,
            endpoint: p.endpoint,
            created_at: DateTime::parse_from_rfc3339(&p.created_at)
                .unwrap()
                .with_timezone(&Utc),
            last_used: DateTime::parse_from_rfc3339(&p.last_used)
                .unwrap()
                .with_timezone(&Utc),
            priority: p.priority,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyEvictionPolicy {
    pub policy_type: String,
    pub timeout_seconds: Option<u64>,
}

impl From<PyEvictionPolicy> for EvictionPolicy {
    fn from(p: PyEvictionPolicy) -> Self {
        match p.policy_type.as_str() {
            "lru" => EvictionPolicy::Lru,
            "lfu" => EvictionPolicy::Lfu,
            "priority" => EvictionPolicy::Priority,
            "idle_timeout" => EvictionPolicy::IdleTimeout {
                seconds: p.timeout_seconds.unwrap_or(300),
            },
            _ => EvictionPolicy::Lru,
        }
    }
}
