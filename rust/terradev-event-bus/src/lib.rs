#![allow(non_local_definitions)]

mod bus;
mod types;

use bus::EventBus;
use pyo3::prelude::*;
use types::Event;

#[pymodule]
fn terradev_event_bus(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEventBus>()?;
    m.add_class::<PyEvent>()?;
    Ok(())
}

#[pyclass]
pub struct PyEventBus {
    inner: EventBus,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyEventBus {
    #[new]
    fn new() -> Self {
        Self {
            inner: EventBus::new(),
        }
    }
    
    fn publish(&self, event: PyEvent) -> PyResult<()> {
        self.inner.publish(event.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn subscribe(&self) -> String {
        self.inner.subscribe()
    }
    
    fn unsubscribe(&self, id: String) {
        self.inner.unsubscribe(&id)
    }
    
    fn subscriber_count(&self) -> usize {
        self.inner.subscriber_count()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyEvent {
    #[pyo3(get, set)]
    pub event_type: String,
    #[pyo3(get, set)]
    pub data: PyObject,
}

impl From<PyEvent> for Event {
    fn from(p: PyEvent) -> Self {
        Python::with_gil(|py| {
            match p.event_type.as_str() {
                "job_started" => {
                    let data: PyObject = p.data;
                    let job_id = data.getattr(py, "job_id").unwrap().extract(py).unwrap();
                    Event::JobStarted {
                        job_id,
                        timestamp: chrono::Utc::now(),
                    }
                }
                "job_failed" => {
                    let data: PyObject = p.data;
                    let job_id = data.getattr(py, "job_id").unwrap().extract(py).unwrap();
                    let error = data.getattr(py, "error").unwrap().extract(py).unwrap();
                    Event::JobFailed {
                        job_id,
                        error,
                        timestamp: chrono::Utc::now(),
                    }
                }
                "job_completed" => {
                    let data: PyObject = p.data;
                    let job_id = data.getattr(py, "job_id").unwrap().extract(py).unwrap();
                    Event::JobCompleted {
                        job_id,
                        timestamp: chrono::Utc::now(),
                    }
                }
                "checkpoint_created" => {
                    let data: PyObject = p.data;
                    let job_id = data.getattr(py, "job_id").unwrap().extract(py).unwrap();
                    let step = data.getattr(py, "step").unwrap().extract(py).unwrap();
                    Event::CheckpointCreated {
                        job_id,
                        step,
                        timestamp: chrono::Utc::now(),
                    }
                }
                "resource_acquired" => {
                    let data: PyObject = p.data;
                    let resource_id = data.getattr(py, "resource_id").unwrap().extract(py).unwrap();
                    Event::ResourceAcquired {
                        resource_id,
                        timestamp: chrono::Utc::now(),
                    }
                }
                "resource_released" => {
                    let data: PyObject = p.data;
                    let resource_id = data.getattr(py, "resource_id").unwrap().extract(py).unwrap();
                    Event::ResourceReleased {
                        resource_id,
                        timestamp: chrono::Utc::now(),
                    }
                }
                _ => {
                    let data_json = Python::with_gil(|py| {
                        // Convert PyObject to a string representation for serialization
                        let py_obj = p.data.as_ref(py);
                        let data_str = if let Ok(repr) = py_obj.repr() {
                            repr.to_string()
                        } else {
                            py_obj.str().unwrap_or_else(|_| pyo3::types::PyString::new(py, "null")).to_string()
                        };
                        serde_json::to_value(data_str).unwrap_or(serde_json::Value::Null)
                    });
                    Event::Custom {
                        name: p.event_type,
                        data: data_json,
                        timestamp: chrono::Utc::now(),
                    }
                }
            }
        })
    }
}
