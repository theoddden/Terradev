use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidationError {
    #[allow(dead_code)]
    #[error("Schema validation failed: {0}")]
    SchemaError(String),

    #[allow(dead_code)]
    #[error("Missing required field: {0}")]
    MissingField(String),

    #[allow(dead_code)]
    #[error("Invalid value for field {field}: {message}")]
    InvalidValue {
        field: String,
        message: String,
    },

    #[allow(dead_code)]
    #[error("Type mismatch for field {field}: expected {expected}, got {actual}")]
    TypeMismatch {
        field: String,
        expected: String,
        actual: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}
