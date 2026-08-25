#![allow(non_local_definitions)]

use lz4::block::decompress;
use lz4::block::{compress, CompressionMode};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass]
pub struct ResultCompressor {
    #[allow(dead_code)]
    compression_level: u32,
}

#[allow(non_local_definitions)]
#[pymethods]
impl ResultCompressor {
    #[new]
    #[pyo3(signature = (compression_level=1))]
    fn new(compression_level: u32) -> Self {
        Self {
            compression_level: compression_level.min(16),
        }
    }

    fn compress(&self, data: &[u8]) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mode = CompressionMode::default();
            let compressed = compress(data, Some(mode), false)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyBytes::new(py, &compressed).into())
        })
    }

    fn decompress(&self, compressed: &[u8]) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let decompressed = decompress(compressed, None)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyBytes::new(py, &decompressed).into())
        })
    }

    fn compress_json(&self, json_str: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mode = CompressionMode::default();
            let compressed = compress(json_str.as_bytes(), Some(mode), false)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("compressed", PyBytes::new(py, &compressed))?;
            dict.set_item("original_size", json_str.len())?;
            dict.set_item("compressed_size", compressed.len())?;
            dict.set_item(
                "compression_ratio",
                json_str.len() as f64 / compressed.len() as f64,
            )?;
            Ok(dict.into())
        })
    }

    fn decompress_json(&self, compressed: &[u8]) -> PyResult<String> {
        let decompressed = decompress(compressed, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        String::from_utf8(decompressed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyUnicodeDecodeError, _>(e.to_string()))
    }
}

#[pymodule]
fn terradev_result_compressor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ResultCompressor>()?;
    Ok(())
}
