use crate::types::{CacheEntry, EvictionPolicy};
use moka::sync::Cache;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

pub struct CacheEngine {
    cache: Cache<String, Arc<CacheEntry>>,
    policy: EvictionPolicy,
    access_counts: Arc<RwLock<HashMap<String, u64>>>,
}

impl CacheEngine {
    pub fn new(max_capacity: u64, policy: EvictionPolicy) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(Duration::from_secs(3600))
            .build();
        
        Self {
            cache,
            policy,
            access_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn put(&self, entry: CacheEntry) {
        let key = entry.key.clone();
        let entry_arc = Arc::new(entry);
        self.cache.insert(key.clone(), entry_arc);
    }
    
    pub fn get(&self, key: &str) -> Option<Arc<CacheEntry>> {
        if let Some(entry) = self.cache.get(key) {
            let mut counts = self.access_counts.write();
            *counts.entry(key.to_string()).or_insert(0) += 1;
            Some(entry)
        } else {
            None
        }
    }
    
    pub fn remove(&self, key: &str) {
        self.cache.invalidate(key);
        let mut counts = self.access_counts.write();
        counts.remove(key);
    }
    
    pub fn contains(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }
    
    pub fn size(&self) -> u64 {
        self.cache.entry_count()
    }
    
    pub fn clear(&self) {
        self.cache.invalidate_all();
        let mut counts = self.access_counts.write();
        counts.clear();
    }
    
    pub fn access_count(&self, key: &str) -> u64 {
        let counts = self.access_counts.read();
        *counts.get(key).unwrap_or(&0)
    }
    
    pub fn policy(&self) -> &EvictionPolicy {
        &self.policy
    }
}
