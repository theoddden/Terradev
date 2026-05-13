use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Schema validation failed: {0}")]
    SchemaError(String),
    
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    #[error("Invalid value for field {field}: {message}")]
    InvalidValue { field: String, message: String },
    
    #[error("Type mismatch for field {field}: expected {expected}, got {actual}")]
    TypeMismatch { field: String, expected: String, actual: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}
