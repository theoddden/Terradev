use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    JobStarted {
        job_id: String,
        timestamp: DateTime<Utc>,
    },
    JobFailed {
        job_id: String,
        error: String,
        timestamp: DateTime<Utc>,
    },
    JobCompleted {
        job_id: String,
        timestamp: DateTime<Utc>,
    },
    CheckpointCreated {
        job_id: String,
        step: u32,
        timestamp: DateTime<Utc>,
    },
    ResourceAcquired {
        resource_id: String,
        timestamp: DateTime<Utc>,
    },
    ResourceReleased {
        resource_id: String,
        timestamp: DateTime<Utc>,
    },
    Custom {
        name: String,
        data: serde_json::Value,
        timestamp: DateTime<Utc>,
    },
}
