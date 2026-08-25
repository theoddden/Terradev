#![allow(non_local_definitions)]

mod types;
mod verifier;

use pyo3::prelude::*;
use types::VerificationResult;
use verifier::ArtifactVerifier;

#[pymodule]
fn terradev_artifact_verification(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyArtifactVerifier>()?;
    m.add_class::<PyVerificationResult>()?;
    Ok(())
}

#[pyclass]
pub struct PyArtifactVerifier;

#[allow(non_local_definitions)]
#[pymethods]
impl PyArtifactVerifier {
    #[new]
    fn new() -> Self {
        Self
    }

    fn compute_sha256(&self, data: Vec<u8>) -> String {
        ArtifactVerifier::compute_sha256(&data)
    }

    fn verify_artifact(
        &self,
        data: Vec<u8>,
        expected_checksum: String,
        algorithm: String,
    ) -> PyResult<PyVerificationResult> {
        ArtifactVerifier::verify_artifact(&data, &expected_checksum, &algorithm)
            .map(|r| r.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn verify_file(
        &self,
        path: String,
        expected_checksum: String,
        algorithm: String,
    ) -> PyResult<PyVerificationResult> {
        ArtifactVerifier::verify_file(&path, &expected_checksum, &algorithm)
            .map(|r| r.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyVerificationResult {
    #[pyo3(get)]
    pub is_valid: bool,
    #[pyo3(get)]
    pub computed_checksum: String,
    #[pyo3(get)]
    pub expected_checksum: String,
    #[pyo3(get)]
    pub algorithm: String,
}

impl From<VerificationResult> for PyVerificationResult {
    fn from(r: VerificationResult) -> Self {
        Self {
            is_valid: r.is_valid,
            computed_checksum: r.computed_checksum,
            expected_checksum: r.expected_checksum,
            algorithm: r.algorithm,
        }
    }
}
