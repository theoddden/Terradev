use crate::types::{LockError, LockGrant, LockRequest};
use chrono::{Duration, Utc};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

pub struct DistributedLock {
    locks: Arc<RwLock<HashMap<String, LockGrant>>>,
}

impl DistributedLock {
    pub fn new() -> Self {
        Self {
            locks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn acquire(&self, request: LockRequest) -> Result<LockGrant, LockError> {
        let mut locks = self.locks.write();
        let now = Utc::now();
        
        // Check if lock exists and is still valid
        if let Some(existing) = locks.get(&request.key) {
            if existing.expires_at > now {
                return Err(LockError::AlreadyHeld {
                    holder: existing.holder.clone(),
                });
            }
            // Lock expired, remove it
            locks.remove(&request.key);
        }
        
        // Grant new lock
        let lease_id = Uuid::new_v4().to_string();
        let grant = LockGrant {
            key: request.key.clone(),
            holder: request.holder.clone(),
            acquired_at: now,
            expires_at: now + Duration::seconds(request.ttl_seconds as i64),
            lease_id: lease_id.clone(),
        };
        
        locks.insert(request.key, grant.clone());
        Ok(grant)
    }
    
    pub async fn release(&self, key: &str, holder: &str, lease_id: &str) -> Result<(), LockError> {
        let mut locks = self.locks.write();
        
        if let Some(lock) = locks.get(key) {
            if lock.holder != holder {
                return Err(LockError::InvalidHolder {
                    holder: holder.to_string(),
                });
            }
            if lock.lease_id != lease_id {
                return Err(LockError::InvalidHolder {
                    holder: holder.to_string(),
                });
            }
            locks.remove(key);
            Ok(())
        } else {
            Err(LockError::NotFound {
                key: key.to_string(),
            })
        }
    }
    
    pub async fn renew(&self, key: &str, holder: &str, lease_id: &str, ttl_seconds: u64) -> Result<LockGrant, LockError> {
        let mut locks = self.locks.write();
        let now = Utc::now();
        
        if let Some(lock) = locks.get(key) {
            if lock.holder != holder || lock.lease_id != lease_id {
                return Err(LockError::InvalidHolder {
                    holder: holder.to_string(),
                });
            }
            
            let renewed = LockGrant {
                key: key.to_string(),
                holder: holder.to_string(),
                acquired_at: lock.acquired_at,
                expires_at: now + Duration::seconds(ttl_seconds as i64),
                lease_id: lease_id.to_string(),
            };
            
            locks.insert(key.to_string(), renewed.clone());
            Ok(renewed)
        } else {
            Err(LockError::NotFound {
                key: key.to_string(),
            })
        }
    }
    
    pub async fn is_held(&self, key: &str) -> bool {
        let locks = self.locks.read();
        let now = Utc::now();
        
        if let Some(lock) = locks.get(key) {
            lock.expires_at > now
        } else {
            false
        }
    }
    
    pub async fn get_holder(&self, key: &str) -> Option<String> {
        let locks = self.locks.read();
        let now = Utc::now();
        
        if let Some(lock) = locks.get(key) {
            if lock.expires_at > now {
                Some(lock.holder.clone())
            } else {
                None
            }
        } else {
            None
        }
    }
    
    pub async fn cleanup_expired(&self) -> usize {
        let mut locks = self.locks.write();
        let now = Utc::now();
        
        let expired: Vec<String> = locks
            .iter()
            .filter(|(_, lock)| lock.expires_at <= now)
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in expired {
            locks.remove(&key);
        }
        
        expired.len()
    }
}
