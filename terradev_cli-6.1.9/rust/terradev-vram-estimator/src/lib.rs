#![allow(non_local_definitions)]

mod estimator;
mod types;

use estimator::VRAMEstimator;
use pyo3::prelude::*;
use types::{ModelArchitecture, Precision, VRAMBreakdown};

#[pymodule]
fn terradev_vram_estimator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVRAMEstimator>()?;
    m.add_class::<PyModelArchitecture>()?;
    m.add_class::<PyVRAMBreakdown>()?;
    m.add_class::<PyPrecision>()?;
    Ok(())
}

#[pyclass]
#[allow(non_local_definitions)]
pub struct PyVRAMEstimator;

#[allow(non_local_definitions)]
#[pymethods]
impl PyVRAMEstimator {
    #[new]
    fn new() -> Self {
        Self
    }

    fn estimate_vram(
        &self,
        architecture: PyModelArchitecture,
        context_tokens: u32,
        batch_size: u32,
        precision: String,
        use_mla: bool,
    ) -> PyVRAMBreakdown {
        let arch = ModelArchitecture {
            name: architecture.name,
            hidden_size: architecture.hidden_size,
            num_layers: architecture.num_layers,
            num_heads: architecture.num_heads,
            vocab_size: architecture.vocab_size,
            max_sequence_length: architecture.max_sequence_length,
        };

        let prec = match precision.as_str() {
            "fp32" => Precision::FP32,
            "fp16" => Precision::FP16,
            "bf16" => Precision::BF16,
            "fp8" => Precision::FP8,
            "int8" => Precision::INT8,
            "int4" => Precision::INT4,
            _ => Precision::FP16,
        };

        VRAMEstimator::estimate_vram(&arch, context_tokens, batch_size, prec, use_mla).into()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyModelArchitecture {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub hidden_size: u32,
    #[pyo3(get, set)]
    pub num_layers: u32,
    #[pyo3(get, set)]
    pub num_heads: u32,
    #[pyo3(get, set)]
    pub vocab_size: u32,
    #[pyo3(get, set)]
    pub max_sequence_length: u32,
}

#[pyclass]
#[derive(Clone)]
pub struct PyVRAMBreakdown {
    #[pyo3(get)]
    pub model_weights_gb: f64,
    #[pyo3(get)]
    pub kv_cache_gb: f64,
    #[pyo3(get)]
    pub activation_cache_gb: f64,
    #[pyo3(get)]
    pub overhead_gb: f64,
    #[pyo3(get)]
    pub total_gb: f64,
    #[pyo3(get)]
    pub per_gpu_gb: f64,
    #[pyo3(get)]
    pub gpu_count: u32,
    #[pyo3(get)]
    pub architecture: String,
    #[pyo3(get)]
    pub context_tokens: u32,
    #[pyo3(get)]
    pub batch_size: u32,
}

impl From<VRAMBreakdown> for PyVRAMBreakdown {
    fn from(b: VRAMBreakdown) -> Self {
        Self {
            model_weights_gb: b.model_weights_gb,
            kv_cache_gb: b.kv_cache_gb,
            activation_cache_gb: b.activation_cache_gb,
            overhead_gb: b.overhead_gb,
            total_gb: b.total_gb,
            per_gpu_gb: b.per_gpu_gb,
            gpu_count: b.gpu_count,
            architecture: b.architecture,
            context_tokens: b.context_tokens,
            batch_size: b.batch_size,
        }
    }
}

#[pyclass]
pub struct PyPrecision;
