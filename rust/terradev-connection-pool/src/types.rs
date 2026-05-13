use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    pub base_url: String,
    pub max_connections: usize,
    pub timeout_seconds: u64,
    pub keep_alive: bool,
}

#[derive(Debug, Error)]
pub enum PoolError {
    #[error("Connection pool exhausted")]
    Exhausted,
    
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Invalid configuration")]
    InvalidConfig,
}
