mod pipeline;
mod types;

use pipeline::TelemetryPipeline;
use pyo3::prelude::*;
use types::{HistogramSnapshot, Metric};

#[pymodule]
fn terradev_telemetry(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTelemetryPipeline>()?;
    m.add_class::<PyMetric>()?;
    m.add_class::<PyHistogramSnapshot>()?;
    Ok(())
}

#[pyclass]
pub struct PyTelemetryPipeline {
    inner: TelemetryPipeline,
}

#[pymethods]
impl PyTelemetryPipeline {
    #[new]
    fn new() -> Self {
        Self {
            inner: TelemetryPipeline::new(),
        }
    }
    
    fn record(&self, metric: PyMetric) -> PyResult<()> {
        self.inner.record(metric.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn record_value(&self, name: String, value: f64, tags: Vec<(String, String)>) -> PyResult<()> {
        self.inner.record_value(name, value, tags)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn get_histogram(&self, name: String) -> Option<PyHistogramSnapshot> {
        self.inner.get_histogram(&name).map(|s| s.into())
    }
    
    fn list_histograms(&self) -> Vec<String> {
        self.inner.list_histograms()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyMetric {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub value: f64,
    #[pyo3(get, set)]
    pub timestamp: String,
    #[pyo3(get, set)]
    pub tags: Vec<(String, String)>,
}

impl From<PyMetric> for Metric {
    fn from(p: PyMetric) -> Self {
        Self {
            name: p.name,
            value: p.value,
            timestamp: chrono::DateTime::parse_from_rfc3339(&p.timestamp).unwrap().with_timezone(&chrono::Utc).into(),
            tags: p.tags,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyHistogramSnapshot {
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub p50: f64,
    #[pyo3(get)]
    pub p95: f64,
    #[pyo3(get)]
    pub p99: f64,
    #[pyo3(get)]
    pub count: u64,
    #[pyo3(get)]
    pub sum: f64,
}

impl From<HistogramSnapshot> for PyHistogramSnapshot {
    fn from(s: HistogramSnapshot) -> Self {
        Self {
            min: s.min,
            max: s.max,
            mean: s.mean,
            p50: s.p50,
            p95: s.p95,
            p99: s.p99,
            count: s.count,
            sum: s.sum,
        }
    }
}
