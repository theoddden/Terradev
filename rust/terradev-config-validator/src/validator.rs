use crate::types::{ValidationError, ValidationReport};
use serde_json::Value;

pub struct ConfigValidator {
    schema: Value,
}

impl ConfigValidator {
    pub fn new(schema: Value) -> Self {
        Self { schema }
    }

    pub fn validate(&self, config: &Value) -> Result<ValidationReport, ValidationError> {
        let mut errors = Vec::new();
        let warnings = Vec::new();

        // Basic validation logic
        if let Some(required_fields) = self.schema.get("required") {
            if let Some(arr) = required_fields.as_array() {
                for field in arr {
                    if let Some(field_str) = field.as_str() {
                        if config.get(field_str).is_none() {
                            errors.push(format!("Missing required field: {}", field_str));
                        }
                    }
                }
            }
        }

        // Type validation
        if let Some(properties) = self.schema.get("properties") {
            if let Some(obj) = properties.as_object() {
                for (field, field_schema) in obj {
                    if let Some(config_value) = config.get(field) {
                        if let Some(expected_type) = field_schema.get("type") {
                            if let Some(type_str) = expected_type.as_str() {
                                let actual_type = match config_value {
                                    Value::String(_) => "string",
                                    Value::Number(_) => "number",
                                    Value::Bool(_) => "boolean",
                                    Value::Array(_) => "array",
                                    Value::Object(_) => "object",
                                    Value::Null => "null",
                                };

                                if type_str != actual_type {
                                    errors.push(format!(
                                        "Type mismatch for field {}: expected {}, got {}",
                                        field, type_str, actual_type
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    #[allow(dead_code)]
    pub fn validate_str(&self, config_str: &str) -> Result<ValidationReport, ValidationError> {
        let config: Value = serde_json::from_str(config_str)
            .map_err(|e| ValidationError::SchemaError(e.to_string()))?;
        self.validate(&config)
    }
}
