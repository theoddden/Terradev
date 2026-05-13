use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PooledResource {
    pub id: String,
    pub resource_type: String,
    pub endpoint: String,
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
    pub priority: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    Priority,
    IdleTimeout { seconds: u64 },
}

#[derive(Debug, Error)]
pub enum PoolError {
    #[error("Pool exhausted: {0}")]
    PoolExhausted(String),
    
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    
    #[error("Invalid pool configuration")]
    InvalidConfiguration,
}
