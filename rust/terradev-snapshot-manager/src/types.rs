use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    pub job_id: String,
    pub step: u32,
    pub model_weights: Vec<u8>,
    pub optimizer_state: Vec<u8>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),
    
    #[error("Compression failed: {0}")]
    CompressionFailed(String),
    
    #[error("IO error: {0}")]
    IoError(String),
}
