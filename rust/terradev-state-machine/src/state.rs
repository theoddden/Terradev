#![allow(clippy::wrong_self_convention)]
#![allow(clippy::result_large_err)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Type-safe job state with compile-time enforced transitions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobState {
    Created {
        created_at: DateTime<Utc>,
    },
    Preflight {
        created_at: DateTime<Utc>,
        started_at: DateTime<Utc>,
    },
    Launching {
        created_at: DateTime<Utc>,
        preflight_completed_at: DateTime<Utc>,
        nodes: Vec<String>,
    },
    Running {
        created_at: DateTime<Utc>,
        started_at: DateTime<Utc>,
        checkpoint_id: Option<String>,
        current_step: u32,
        total_steps: u32,
    },
    Checkpointing {
        created_at: DateTime<Utc>,
        started_at: DateTime<Utc>,
        checkpoint_step: u32,
    },
    Paused {
        created_at: DateTime<Utc>,
        paused_at: DateTime<Utc>,
        checkpoint_id: String,
    },
    Completed {
        created_at: DateTime<Utc>,
        started_at: DateTime<Utc>,
        finished_at: DateTime<Utc>,
        final_step: u32,
    },
    Failed {
        created_at: DateTime<Utc>,
        started_at: Option<DateTime<Utc>>,
        failed_at: DateTime<Utc>,
        error: String,
        step: u32,
    },
    Cancelled {
        created_at: DateTime<Utc>,
        cancelled_at: DateTime<Utc>,
    },
    Preempted {
        created_at: DateTime<Utc>,
        started_at: DateTime<Utc>,
        preempted_at: DateTime<Utc>,
        reason: String,
        checkpoint_id: Option<String>,
    },
}

impl JobState {
    pub fn created() -> Self {
        JobState::Created {
            created_at: Utc::now(),
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobState::Completed { .. } | JobState::Failed { .. } | JobState::Cancelled { .. }
        )
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self,
            JobState::Running { .. } | JobState::Checkpointing { .. }
        )
    }

    pub fn status_str(&self) -> &'static str {
        match self {
            JobState::Created { .. } => "created",
            JobState::Preflight { .. } => "preflight",
            JobState::Launching { .. } => "launching",
            JobState::Running { .. } => "running",
            JobState::Checkpointing { .. } => "checkpointing",
            JobState::Paused { .. } => "paused",
            JobState::Completed { .. } => "completed",
            JobState::Failed { .. } => "failed",
            JobState::Cancelled { .. } => "cancelled",
            JobState::Preempted { .. } => "preempted",
        }
    }

    /// Transition to Preflight state
    pub fn to_preflight(self) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Created { created_at } => Ok(JobState::Preflight {
                created_at,
                started_at: Utc::now(),
            }),
            _ => Err(crate::types::StateTransitionError::InvalidTransition {
                from: self,
                to: JobState::Preflight {
                    created_at: Utc::now(),
                    started_at: Utc::now(),
                },
            }),
        }
    }

    /// Transition to Launching state
    pub fn to_launching(
        self,
        nodes: Vec<String>,
    ) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Preflight {
                created_at,
                started_at: _,
            } => Ok(JobState::Launching {
                created_at,
                preflight_completed_at: Utc::now(),
                nodes,
            }),
            _ => Err(crate::types::StateTransitionError::InvalidTransition {
                from: self,
                to: JobState::Launching {
                    created_at: Utc::now(),
                    preflight_completed_at: Utc::now(),
                    nodes: vec![],
                },
            }),
        }
    }

    /// Transition to Running state
    pub fn to_running(self, total_steps: u32) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Launching { created_at, .. } => Ok(JobState::Running {
                created_at,
                started_at: Utc::now(),
                checkpoint_id: None,
                current_step: 0,
                total_steps,
            }),
            JobState::Paused {
                created_at,
                checkpoint_id,
                ..
            } => Ok(JobState::Running {
                created_at,
                started_at: Utc::now(),
                checkpoint_id: Some(checkpoint_id),
                current_step: 0,
                total_steps,
            }),
            _ => Err(crate::types::StateTransitionError::InvalidTransition {
                from: self,
                to: JobState::Running {
                    created_at: Utc::now(),
                    started_at: Utc::now(),
                    checkpoint_id: None,
                    current_step: 0,
                    total_steps: 0,
                },
            }),
        }
    }

    /// Transition to Checkpointing state
    pub fn to_checkpointing(self, step: u32) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Running {
                created_at,
                started_at,
                checkpoint_id: _,
                current_step: _,
                total_steps: _,
            } => Ok(JobState::Checkpointing {
                created_at,
                started_at,
                checkpoint_step: step,
            }),
            _ => Err(crate::types::StateTransitionError::InvalidTransition {
                from: self,
                to: JobState::Checkpointing {
                    created_at: Utc::now(),
                    started_at: Utc::now(),
                    checkpoint_step: 0,
                },
            }),
        }
    }

    /// Transition from Checkpointing back to Running
    pub fn from_checkpointing(
        self,
        new_checkpoint_id: String,
        total_steps: u32,
    ) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Checkpointing {
                created_at,
                started_at,
                checkpoint_step,
            } => Ok(JobState::Running {
                created_at,
                started_at,
                checkpoint_id: Some(new_checkpoint_id),
                current_step: checkpoint_step,
                total_steps,
            }),
            _ => Err(crate::types::StateTransitionError::InvalidTransition {
                from: self,
                to: JobState::Running {
                    created_at: Utc::now(),
                    started_at: Utc::now(),
                    checkpoint_id: None,
                    current_step: 0,
                    total_steps: 0,
                },
            }),
        }
    }

    /// Transition to Paused state
    pub fn to_paused(
        self,
        checkpoint_id: String,
    ) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Running {
                created_at,
                ..
            } => Ok(JobState::Paused {
                created_at,
                paused_at: Utc::now(),
                checkpoint_id,
            }),
            _ => Err(crate::types::StateTransitionError::CheckpointRequired),
        }
    }

    /// Transition to Completed state
    pub fn to_completed(self, final_step: u32) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Running {
                created_at,
                started_at,
                ..
            } => Ok(JobState::Completed {
                created_at,
                started_at,
                finished_at: Utc::now(),
                final_step,
            }),
            JobState::Checkpointing {
                created_at,
                started_at,
                checkpoint_step,
            } => Ok(JobState::Completed {
                created_at,
                started_at,
                finished_at: Utc::now(),
                final_step: checkpoint_step,
            }),
            _ => Err(crate::types::StateTransitionError::TerminalState { state: self }),
        }
    }

    /// Transition to Failed state
    pub fn to_failed(
        self,
        error: String,
        step: u32,
    ) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Running {
                created_at,
                started_at,
                ..
            } => Ok(JobState::Failed {
                created_at,
                started_at: Some(started_at),
                failed_at: Utc::now(),
                error,
                step,
            }),
            JobState::Launching { created_at, .. } => Ok(JobState::Failed {
                created_at,
                started_at: None,
                failed_at: Utc::now(),
                error,
                step: 0,
            }),
            JobState::Preflight {
                created_at,
                started_at,
            } => Ok(JobState::Failed {
                created_at,
                started_at: Some(started_at),
                failed_at: Utc::now(),
                error,
                step: 0,
            }),
            _ => Err(crate::types::StateTransitionError::TerminalState { state: self }),
        }
    }

    /// Transition to Cancelled state
    pub fn to_cancelled(self) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Created { created_at }
            | JobState::Preflight { created_at, .. }
            | JobState::Launching { created_at, .. } => Ok(JobState::Cancelled {
                created_at,
                cancelled_at: Utc::now(),
            }),
            _ => Err(crate::types::StateTransitionError::TerminalState { state: self }),
        }
    }

    /// Transition to Preempted state
    pub fn to_preempted(
        self,
        reason: String,
        checkpoint_id: Option<String>,
    ) -> Result<Self, crate::types::StateTransitionError> {
        match self {
            JobState::Running {
                created_at,
                started_at,
                ..
            } => Ok(JobState::Preempted {
                created_at,
                started_at,
                preempted_at: Utc::now(),
                reason,
                checkpoint_id,
            }),
            _ => Err(crate::types::StateTransitionError::InvalidTransition {
                from: self,
                to: JobState::Preempted {
                    created_at: Utc::now(),
                    started_at: Utc::now(),
                    preempted_at: Utc::now(),
                    reason: String::new(),
                    checkpoint_id: None,
                },
            }),
        }
    }
}
