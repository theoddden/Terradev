use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockRequest {
    pub key: String,
    pub holder: String,
    pub ttl_seconds: u64,
    pub requested_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockGrant {
    pub key: String,
    pub holder: String,
    pub acquired_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub lease_id: String,
}

#[derive(Debug, Error)]
pub enum LockError {
    #[error("Lock already held by {holder}")]
    AlreadyHeld {
        holder: String,
    },

    #[error("Lock not found: {key}")]
    NotFound {
        key: String,
    },

    #[error("Lock expired")]
    #[allow(dead_code)]
    Expired,

    #[error("Invalid holder: {holder}")]
    InvalidHolder { holder: String },
}
