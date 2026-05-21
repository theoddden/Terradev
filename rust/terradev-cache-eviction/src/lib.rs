#![allow(non_local_definitions)]

mod cache;
mod types;

use cache::CacheEngine;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use types::{CacheEntry, EvictionPolicy};

#[pymodule]
fn terradev_cache_eviction(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCacheEngine>()?;
    m.add_class::<PyCacheEntry>()?;
    m.add_class::<PyEvictionPolicy>()?;
    Ok(())
}

#[pyclass]
pub struct PyCacheEngine {
    inner: CacheEngine,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyCacheEngine {
    #[new]
    fn new(max_capacity: u64, policy: PyEvictionPolicy) -> Self {
        Self {
            inner: CacheEngine::new(max_capacity, policy.into()),
        }
    }
    
    fn put(&self, entry: PyCacheEntry) {
        self.inner.put(entry.into());
    }
    
    fn get(&self, key: String) -> Option<PyCacheEntry> {
        self.inner.get(&key).map(|e| {
            let entry = (*e).clone();
            PyCacheEntry {
                key: entry.key,
                value: entry.value.to_string(),
                size_bytes: entry.size_bytes,
                created_at: entry.created_at.to_rfc3339(),
                last_accessed: entry.last_accessed.to_rfc3339(),
                access_count: entry.access_count,
            }
        })
    }
    
    fn remove(&self, key: String) {
        self.inner.remove(&key);
    }
    
    fn contains(&self, key: String) -> bool {
        self.inner.contains(&key)
    }
    
    fn size(&self) -> u64 {
        self.inner.size()
    }
    
    fn clear(&mut self) {
        self.inner.clear();
    }
    
    fn access_count(&self, key: String) -> u64 {
        self.inner.access_count(&key)
    }
    
    fn policy(&self) -> String {
        match self.inner.policy() {
            EvictionPolicy::LRU => "lru".to_string(),
            EvictionPolicy::ARC => "arc".to_string(),
            EvictionPolicy::TinyLFU => "tinylfu".to_string(),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyCacheEntry {
    #[pyo3(get, set)]
    pub key: String,
    #[pyo3(get, set)]
    pub value: String,
    #[pyo3(get, set)]
    pub size_bytes: u64,
    #[pyo3(get, set)]
    pub created_at: String,
    #[pyo3(get, set)]
    pub last_accessed: String,
    #[pyo3(get, set)]
    pub access_count: u64,
}

impl From<PyCacheEntry> for CacheEntry {
    fn from(p: PyCacheEntry) -> Self {
        Self {
            key: p.key,
            value: serde_json::from_str(&p.value).unwrap_or(serde_json::Value::Null),
            size_bytes: p.size_bytes,
            created_at: chrono::DateTime::parse_from_rfc3339(&p.created_at).unwrap().with_timezone(&chrono::Utc).into(),
            last_accessed: chrono::DateTime::parse_from_rfc3339(&p.last_accessed).unwrap().with_timezone(&chrono::Utc).into(),
            access_count: p.access_count,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyEvictionPolicy {
    pub policy_type: String,
}

impl From<PyEvictionPolicy> for EvictionPolicy {
    fn from(p: PyEvictionPolicy) -> Self {
        match p.policy_type.as_str() {
            "lru" => EvictionPolicy::LRU,
            "arc" => EvictionPolicy::ARC,
            "tinylfu" => EvictionPolicy::TinyLFU,
            _ => EvictionPolicy::LRU,
        }
    }
}
