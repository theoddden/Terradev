mod encoding;
mod hmac;
mod types;

use hmac::{AlibabaSigner, OVHSigner};
use pyo3::prelude::*;
use types::{AlibabaCredentials, OVHCredentials, SignatureResult};

#[pymodule]
fn terradev_authentication(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAlibabaSigner>()?;
    m.add_class::<PyOVHSigner>()?;
    m.add_class::<PyAlibabaCredentials>()?;
    m.add_class::<PyOvhCredentials>()?;
    m.add_class::<PySignatureResult>()?;
    Ok(())
}

#[pyclass]
pub struct PyAlibabaSigner;

#[allow(non_local_definitions)]
#[pymethods]
impl PyAlibabaSigner {
    #[new]
    fn new() -> Self {
        Self
    }
    
    fn sign_request(
        &self,
        credentials: PyAlibabaCredentials,
        http_method: String,
        url: String,
        params: Vec<(String, String)>,
    ) -> PySignatureResult {
        let creds = AlibabaCredentials {
            access_key_id: credentials.access_key_id,
            access_key_secret: credentials.access_key_secret,
        };
        
        AlibabaSigner::sign_request(&creds, &http_method, &url, &params).into()
    }
}

#[pyclass]
pub struct PyOVHSigner;

#[allow(non_local_definitions)]
#[pymethods]
impl PyOVHSigner {
    #[new]
    fn new() -> Self {
        Self
    }
    
    fn sign_request(
        &self,
        credentials: PyOvhCredentials,
        http_method: String,
        url: String,
        body: String,
        timestamp: String,
    ) -> PySignatureResult {
        let creds = OVHCredentials {
            application_key: credentials.application_key,
            application_secret: credentials.application_secret,
            consumer_key: credentials.consumer_key,
        };
        
        OVHSigner::sign_request(&creds, &http_method, &url, &body, &timestamp).into()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyAlibabaCredentials {
    #[pyo3(get, set)]
    pub access_key_id: String,
    #[pyo3(get, set)]
    pub access_key_secret: String,
}

#[pyclass]
#[derive(Clone)]
pub struct PyOvhCredentials {
    #[pyo3(get, set)]
    pub application_key: String,
    #[pyo3(get, set)]
    pub application_secret: String,
    #[pyo3(get, set)]
    pub consumer_key: String,
}

#[pyclass]
#[derive(Clone)]
pub struct PySignatureResult {
    #[pyo3(get)]
    pub signature: String,
    #[pyo3(get)]
    pub timestamp: String,
}

impl From<SignatureResult> for PySignatureResult {
    fn from(s: SignatureResult) -> Self {
        Self {
            signature: s.signature,
            timestamp: s.timestamp,
        }
    }
}
