use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuotaError {
    #[error("Quota exceeded: {resource} limit {limit}, requested {requested}")]
    QuotaExceeded { resource: String, limit: u64, requested: u64 },
    
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quota {
    pub resource: String,
    pub limit: u64,
    pub used: u64,
    pub remaining: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaRequest {
    pub resource: String,
    pub amount: u64,
}
