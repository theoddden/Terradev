use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    pub name: String,
    pub framework: String,
    pub nodes: Vec<String>,
    pub total_steps: u32,
    pub gpus_per_node: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobTopology {
    pub nodes: HashMap<String, NodeInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub gpus: Vec<GPUInfo>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    pub index: u32,
    pub name: String,
    pub memory_mb: u64,
}

#[derive(Debug, Error)]
pub enum StateTransitionError {
    #[error("Invalid transition from {from:?} to {to:?}")]
    InvalidTransition { from: JobState, to: JobState },
    
    #[error("Job is in terminal state {state:?}")]
    TerminalState { state: JobState },
    
    #[error("Checkpoint required for transition")]
    CheckpointRequired,
    
    #[error("Job not found: {id}")]
    JobNotFound { id: String },
}
