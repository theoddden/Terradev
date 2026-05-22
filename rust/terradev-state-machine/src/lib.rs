#![allow(non_local_definitions)]
#![allow(clippy::wrong_self_convention)]

mod state;
mod types;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use state::JobState;
use types::{JobConfig, JobTopology};

#[pymodule]
fn terradev_state_machine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<JobStateMachine>()?;
    m.add_function(wrap_pyfunction!(create_job_py, m)?)?;
    Ok(())
}

#[pyclass]
pub struct JobStateMachine {
    id: String,
    state: JobState,
    #[allow(dead_code)]
    config: Option<JobConfig>,
    #[allow(dead_code)]
    topology: Option<JobTopology>,
}

#[allow(non_local_definitions)]
#[pymethods]
impl JobStateMachine {
    #[new]
    fn new(id: String) -> Self {
        Self {
            id,
            state: JobState::created(),
            config: None,
            topology: None,
        }
    }
    
    #[getter]
    fn id(&self) -> String {
        self.id.clone()
    }
    
    #[getter]
    fn status(&self) -> String {
        self.state.status_str().to_string()
    }
    
    #[getter]
    fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
    
    #[getter]
    fn is_active(&self) -> bool {
        self.state.is_active()
    }
    
    fn to_preflight(&mut self) -> PyResult<()> {
        self.state = self.state.clone().to_preflight()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn to_launching(&mut self, nodes: Vec<String>) -> PyResult<()> {
        self.state = self.state.clone().to_launching(nodes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn to_running(&mut self, total_steps: u32) -> PyResult<()> {
        self.state = self.state.clone().to_running(total_steps)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn to_checkpointing(&mut self, step: u32) -> PyResult<()> {
        self.state = self.state.clone().to_checkpointing(step)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn from_checkpointing(&mut self, checkpoint_id: String, total_steps: u32) -> PyResult<()> {
        self.state = self.state.clone().from_checkpointing(checkpoint_id, total_steps)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn to_paused(&mut self, checkpoint_id: String) -> PyResult<()> {
        self.state = self.state.clone().to_paused(checkpoint_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn to_completed(&mut self, final_step: u32) -> PyResult<()> {
        self.state = self.state.clone().to_completed(final_step)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn to_failed(&mut self, error: String, step: u32) -> PyResult<()> {
        self.state = self.state.clone().to_failed(error, step)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn to_cancelled(&mut self) -> PyResult<()> {
        self.state = self.state.clone().to_cancelled()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn to_preempted(&mut self, reason: String, checkpoint_id: Option<String>) -> PyResult<()> {
        self.state = self.state.clone().to_preempted(reason, checkpoint_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("id", &self.id)?;
            dict.set_item("status", self.state.status_str())?;
            dict.set_item("is_terminal", self.state.is_terminal())?;
            dict.set_item("is_active", self.state.is_active())?;
            Ok(dict.into())
        })
    }
}

#[pyfunction]
fn create_job_py(id: String) -> PyResult<JobStateMachine> {
    Ok(JobStateMachine::new(id))
}
