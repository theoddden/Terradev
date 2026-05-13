mod types;
mod vault;

use pyo3::prelude::*;
use types::CredentialMetadata;
use vault::CredentialVault;

#[pymodule]
fn terradev_credential_vault(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCredentialVault>()?;
    m.add_class::<PyCredentialMetadata>()?;
    Ok(())
}

#[pyclass]
pub struct PyCredentialVault {
    inner: CredentialVault,
}

#[pymethods]
impl PyCredentialVault {
    #[new]
    fn new() -> Self {
        Self {
            inner: CredentialVault::new(),
        }
    }
    
    fn store(&self, name: String, value: Vec<u8>, provider: String) {
        self.inner.store(name, value, provider);
    }
    
    fn retrieve(&self, name: String) -> Option<Vec<u8>> {
        self.inner.retrieve(&name)
    }
    
    fn get_metadata(&self, name: String) -> Option<PyCredentialMetadata> {
        self.inner.get_metadata(&name).map(|m| m.into())
    }
    
    fn delete(&self, name: String) -> bool {
        self.inner.delete(&name)
    }
    
    fn list(&self) -> Vec<String> {
        self.inner.list()
    }
    
    fn clear(&mut self) {
        self.inner.clear();
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyCredentialMetadata {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub last_accessed: String,
}

impl From<CredentialMetadata> for PyCredentialMetadata {
    fn from(m: CredentialMetadata) -> Self {
        Self {
            name: m.name,
            provider: m.provider,
            created_at: m.created_at,
            last_accessed: m.last_accessed,
        }
    }
}
