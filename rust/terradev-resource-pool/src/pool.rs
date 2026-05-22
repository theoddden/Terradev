use crate::types::{EvictionPolicy, PoolError, PooledResource};
use chrono::{Duration, Utc};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct ResourcePool {
    resources: Arc<RwLock<HashMap<String, PooledResource>>>,
    max_size: usize,
    policy: EvictionPolicy,
    #[allow(dead_code)]
    pool_name: String,
}

impl ResourcePool {
    pub fn new(pool_name: String, max_size: usize, policy: EvictionPolicy) -> Self {
        Self {
            resources: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            policy,
            pool_name,
        }
    }
    
    pub fn add(&self, resource: PooledResource) -> Result<(), PoolError> {
        let mut resources = self.resources.write();
        
        if resources.len() >= self.max_size {
            self.evict_if_needed(&mut resources)?;
        }
        
        resources.insert(resource.id.clone(), resource);
        Ok(())
    }
    
    pub fn get(&self, id: &str) -> Option<PooledResource> {
        let mut resources = self.resources.write();
        if let Some(mut resource) = resources.remove(id) {
            resource.last_used = Utc::now();
            resources.insert(id.to_string(), resource.clone());
            Some(resource)
        } else {
            None
        }
    }
    
    pub fn remove(&self, id: &str) -> Option<PooledResource> {
        let mut resources = self.resources.write();
        resources.remove(id)
    }
    
    pub fn contains(&self, id: &str) -> bool {
        let resources = self.resources.read();
        resources.contains_key(id)
    }
    
    pub fn size(&self) -> usize {
        let resources = self.resources.read();
        resources.len()
    }
    
    pub fn clear(&self) {
        let mut resources = self.resources.write();
        resources.clear();
    }
    
    pub fn list(&self) -> Vec<PooledResource> {
        let resources = self.resources.read();
        resources.values().cloned().collect()
    }
    
    fn evict_if_needed(&self, resources: &mut HashMap<String, PooledResource>) -> Result<(), PoolError> {
        if resources.len() < self.max_size {
            return Ok(());
        }
        
        match &self.policy {
            EvictionPolicy::Lru => {
                let oldest = resources
                    .iter()
                    .min_by_key(|(_, r)| r.last_used)
                    .map(|(id, _)| id.clone());
                
                if let Some(id) = oldest {
                    resources.remove(&id);
                }
            }
            EvictionPolicy::IdleTimeout { seconds } => {
                let threshold = Utc::now() - Duration::seconds(*seconds as i64);
                let expired: Vec<String> = resources
                    .iter()
                    .filter(|(_, r)| r.last_used < threshold)
                    .map(|(id, _)| id.clone())
                    .collect();
                
                for id in expired {
                    resources.remove(&id);
                }
                
                if resources.len() >= self.max_size {
                    let oldest = resources
                        .iter()
                        .min_by_key(|(_, r)| r.last_used)
                        .map(|(id, _)| id.clone());
                    
                    if let Some(id) = oldest {
                        resources.remove(&id);
                    }
                }
            }
            EvictionPolicy::Priority => {
                let lowest_priority = resources
                    .iter()
                    .min_by_key(|(_, r)| r.priority)
                    .map(|(id, _)| id.clone());
                
                if let Some(id) = lowest_priority {
                    resources.remove(&id);
                }
            }
            EvictionPolicy::Lfu => {
                // Simplified LFU - evict oldest (would need frequency tracking)
                let oldest = resources
                    .iter()
                    .min_by_key(|(_, r)| r.last_used)
                    .map(|(id, _)| id.clone());
                
                if let Some(id) = oldest {
                    resources.remove(&id);
                }
            }
        }
        
        Ok(())
    }
}

impl Drop for ResourcePool {
    fn drop(&mut self) {
        self.clear();
    }
}
