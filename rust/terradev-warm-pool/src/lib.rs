use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use moka::future::Cache;
use chrono::Utc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmInstance {
    pub instance_id: String,
    pub model_name: String,
    pub gpu_type: String,
    pub region: String,
    pub last_accessed: i64,
    pub priority: i32,
    pub cost_usd_per_hour: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionCandidate {
    pub instance_id: String,
    pub reason: String,
    pub score: f64,
}

#[pyclass]
pub struct WarmPoolManager {
    instances: Arc<RwLock<HashMap<String, WarmInstance>>>,
    cache: Cache<String, WarmInstance>,
    max_instances: usize,
    max_idle_seconds: i64,
}

#[allow(non_local_definitions)]
#[pymethods]
impl WarmPoolManager {
    #[new]
    #[pyo3(signature = (max_instances=100, max_idle_seconds=3600))]
    fn new(max_instances: usize, max_idle_seconds: i64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_instances as u64)
            .time_to_idle(std::time::Duration::from_secs(max_idle_seconds as u64))
            .build();

        Self {
            instances: Arc::new(RwLock::new(HashMap::new())),
            cache,
            max_instances,
            max_idle_seconds,
        }
    }

    fn add_instance(&self, instance: &PyDict) -> PyResult<()> {
        let instance_id: String = instance.get_item("instance_id")?.unwrap().extract()?;
        let model_name: String = instance.get_item("model_name")?.unwrap().extract()?;
        let gpu_type: String = instance.get_item("gpu_type")?.unwrap().extract()?;
        let region: String = instance.get_item("region")?.unwrap().extract()?;
        let priority: i32 = instance.get_item("priority")
            .map(|p| p.extract())
            .unwrap_or(Ok(0))?;
        let cost_usd_per_hour: f64 = instance.get_item("cost_usd_per_hour")
            .map(|c| c.extract())
            .unwrap_or(Ok(0))?;

        let warm_instance = WarmInstance {
            instance_id: instance_id.clone(),
            model_name,
            gpu_type,
            region,
            last_accessed: Utc::now().timestamp(),
            priority,
            cost_usd_per_hour,
        };

        self.instances.write().insert(instance_id.clone(), warm_instance.clone());
        self.cache.insert(instance_id, warm_instance);
        Ok(())
    }

    fn get_instance(&self, instance_id: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let instances = self.instances.read();
            
            if let Some(mut instance) = instances.get(instance_id).cloned() {
                instance.last_accessed = Utc::now().timestamp();
                self.cache.insert(instance_id.to_string(), instance.clone());
                
                let dict = PyDict::new(py);
                dict.set_item("instance_id", &instance.instance_id)?;
                dict.set_item("model_name", &instance.model_name)?;
                dict.set_item("gpu_type", &instance.gpu_type)?;
                dict.set_item("region", &instance.region)?;
                dict.set_item("last_accessed", instance.last_accessed)?;
                dict.set_item("priority", instance.priority)?;
                dict.set_item("cost_usd_per_hour", instance.cost_usd_per_hour)?;
                
                Ok(dict.into())
            } else {
                Ok(py.None())
            }
        })
    }

    fn get_eviction_candidates(&self, count: usize) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let instances = self.instances.read();
            let now = Utc::now().timestamp();
            
            let mut candidates: Vec<EvictionCandidate> = instances.iter()
                .map(|(id, instance)| {
                    let idle_seconds = now - instance.last_accessed;
                    let idle_score = (idle_seconds as f64) / (self.max_idle_seconds as f64);
                    let priority_score = (instance.priority as f64) / 100.0;
                    let cost_score = instance.cost_usd_per_hour / 10.0;
                    
                    EvictionCandidate {
                        instance_id: id.clone(),
                        reason: format!("Idle for {}s, priority {}, cost ${:.2}/hr", 
                                       idle_seconds, instance.priority, instance.cost_usd_per_hour),
                        score: idle_score * 0.5 + priority_score * 0.3 + cost_score * 0.2,
                    }
                })
                .collect();
            
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            candidates.truncate(count);
            
            let list = pyo3::types::PyList::empty(py);
            for candidate in candidates {
                let dict = PyDict::new(py);
                dict.set_item("instance_id", &candidate.instance_id)?;
                dict.set_item("reason", &candidate.reason)?;
                dict.set_item("score", candidate.score)?;
                list.append(dict)?;
            }
            
            Ok(list.into())
        })
    }

    fn evict(&self, instance_id: &str) -> PyResult<bool> {
        let removed = self.instances.write().remove(instance_id).is_some();
        self.cache.invalidate(instance_id);
        Ok(removed)
    }

    fn get_pool_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let instances = self.instances.read();
            let now = Utc::now().timestamp();
            
            let total_instances = instances.len();
            let idle_instances = instances.values()
                .filter(|i| now - i.last_accessed > self.max_idle_seconds / 2)
                .count();
            
            let total_cost: f64 = instances.values()
                .map(|i| i.cost_usd_per_hour)
                .sum();
            
            let dict = PyDict::new(py);
            dict.set_item("total_instances", total_instances)?;
            dict.set_item("idle_instances", idle_instances)?;
            dict.set_item("active_instances", total_instances - idle_instances)?;
            dict.set_item("total_cost_usd_per_hour", total_cost)?;
            dict.set_item("max_instances", self.max_instances)?;
            dict.set_item("utilization_pct", total_instances as f64 / self.max_instances as f64 * 100.0)?;
            
            Ok(dict.into())
        })
    }

    fn clear(&self) {
        self.instances.write().clear();
        self.cache.invalidate_all();
    }
}

#[pymodule]
fn terradev_warm_pool(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WarmPoolManager>()?;
    Ok(())
}
