mod types;
mod validator;

use pyo3::prelude::*;
use serde_json::Value;
use types::ValidationReport;
use validator::ConfigValidator;

#[pymodule]
fn terradev_config_validator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyConfigValidator>()?;
    m.add_class::<PyValidationReport>()?;
    Ok(())
}

#[pyclass]
pub struct PyConfigValidator {
    inner: ConfigValidator,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyConfigValidator {
    #[new]
    fn new(schema_json: String) -> PyResult<Self> {
        let schema: Value = serde_json::from_str(&schema_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self {
            inner: ConfigValidator::new(schema),
        })
    }
    
    fn validate(&self, config_json: String) -> PyResult<PyValidationReport> {
        let config: Value = serde_json::from_str(&config_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.inner.validate(&config)
            .map(|r| r.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyValidationReport {
    #[pyo3(get)]
    pub is_valid: bool,
    #[pyo3(get)]
    pub errors: Vec<String>,
    #[pyo3(get)]
    pub warnings: Vec<String>,
}

impl From<ValidationReport> for PyValidationReport {
    fn from(r: ValidationReport) -> Self {
        Self {
            is_valid: r.is_valid,
            errors: r.errors,
            warnings: r.warnings,
        }
    }
}
