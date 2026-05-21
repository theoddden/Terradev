mod snapshot;
mod types;

use pyo3::prelude::*;
use snapshot::SnapshotManager;
use types::ModelState;

#[pymodule]
fn terradev_snapshot_manager(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySnapshotManager>()?;
    m.add_class::<PyModelState>()?;
    Ok(())
}

#[pyclass]
#[allow(non_local_definitions)]
pub struct PySnapshotManager {
    inner: SnapshotManager,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PySnapshotManager {
    #[new]
    fn new(compression_level: i32) -> Self {
        Self {
            inner: SnapshotManager::new(compression_level),
        }
    }
    
    fn save_snapshot(&self, state: PyModelState) -> PyResult<Vec<u8>> {
        self.inner.save_snapshot(&state.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn load_snapshot(&self, data: Vec<u8>) -> PyResult<PyModelState> {
        self.inner.load_snapshot(&data)
            .map(|s| s.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn save_snapshot_to_file(&self, state: PyModelState, path: String) -> PyResult<()> {
        self.inner.save_snapshot_to_file(&state.into(), &path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn load_snapshot_from_file(&self, path: String) -> PyResult<PyModelState> {
        self.inner.load_snapshot_from_file(&path)
            .map(|s| s.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn get_compression_ratio(&self, state: PyModelState) -> PyResult<f64> {
        self.inner.get_compression_ratio(&state.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyModelState {
    #[pyo3(get, set)]
    pub job_id: String,
    #[pyo3(get, set)]
    pub step: u32,
    #[pyo3(get, set)]
    pub model_weights: Vec<u8>,
    #[pyo3(get, set)]
    pub optimizer_state: Vec<u8>,
    #[pyo3(get, set)]
    pub metadata: String,
    #[pyo3(get, set)]
    pub created_at: String,
}

impl From<PyModelState> for ModelState {
    fn from(p: PyModelState) -> Self {
        Self {
            job_id: p.job_id,
            step: p.step,
            model_weights: p.model_weights,
            optimizer_state: p.optimizer_state,
            metadata: serde_json::from_str(&p.metadata).unwrap_or(serde_json::Value::Null),
            created_at: chrono::DateTime::parse_from_rfc3339(&p.created_at).unwrap().with_timezone(&chrono::Utc),
        }
    }
}

impl From<ModelState> for PyModelState {
    fn from(s: ModelState) -> Self {
        Self {
            job_id: s.job_id,
            step: s.step,
            model_weights: s.model_weights,
            optimizer_state: s.optimizer_state,
            metadata: serde_json::to_string(&s.metadata).unwrap_or_default(),
            created_at: s.created_at.to_rfc3339(),
        }
    }
}
