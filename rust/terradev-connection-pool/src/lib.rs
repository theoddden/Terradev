mod pool;
mod types;

use pool::ConnectionPool;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use types::{ConnectionConfig, PoolError};

#[pymodule]
fn terradev_connection_pool(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyConnectionPool>()?;
    m.add_class::<PyConnectionConfig>()?;
    Ok(())
}

#[pyclass]
pub struct PyConnectionPool {
    inner: ConnectionPool,
}

#[pymethods]
impl PyConnectionPool {
    #[new]
    fn new(config: PyConnectionConfig) -> PyResult<Self> {
        Ok(Self {
            inner: ConnectionPool::new(config.into())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        })
    }
    
    fn max_connections(&self) -> usize {
        self.inner.max_connections()
    }
    
    fn active_connections(&self) -> usize {
        self.inner.active_connections()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyConnectionConfig {
    #[pyo3(get, set)]
    pub base_url: String,
    #[pyo3(get, set)]
    pub max_connections: usize,
    #[pyo3(get, set)]
    pub timeout_seconds: u64,
    #[pyo3(get, set)]
    pub keep_alive: bool,
}

impl From<PyConnectionConfig> for ConnectionConfig {
    fn from(p: PyConnectionConfig) -> Self {
        Self {
            base_url: p.base_url,
            max_connections: p.max_connections,
            timeout_seconds: p.timeout_seconds,
            keep_alive: p.keep_alive,
        }
    }
}
