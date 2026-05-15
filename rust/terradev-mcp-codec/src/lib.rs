use pyo3::prelude::*;
use pyo3::types::PyDict, PyBytes;
use serde::{Deserialize, Serialize};
use simd_json::from_str;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolCall {
    pub id: String,
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolResult {
    pub id: String,
    pub content: Vec<serde_json::Value>,
    pub is_error: bool,
}

#[pyclass]
pub struct MCPCodec {
    use_simd: bool,
}

#[pymethods]
impl MCPCodec {
    #[new]
    #[pyo3(signature = (use_simd=true))]
    fn new(use_simd: bool) -> Self {
        Self { use_simd }
    }

    fn decode_tool_call(&self, json_str: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            if self.use_simd {
                let call: MCPToolCall = from_str(json_str)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                
                let dict = PyDict::new(py);
                dict.set_item("id", &call.id)?;
                dict.set_item("name", &call.name)?;
                dict.set_item("arguments", serde_json::to_value(&call.arguments).unwrap())?;
                Ok(dict.into())
            } else {
                let call: MCPToolCall = serde_json::from_str(json_str)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                
                let dict = PyDict::new(py);
                dict.set_item("id", &call.id)?;
                dict.set_item("name", &call.name)?;
                dict.set_item("arguments", serde_json::to_value(&call.arguments).unwrap())?;
                Ok(dict.into())
            }
        })
    }

    fn encode_tool_result(&self, id: String, content: Vec<serde_json::Value>, is_error: Option<bool>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let result = MCPToolResult {
                id,
                content,
                is_error: is_error.unwrap_or(false),
            };

            let json_str = if self.use_simd {
                simd_json::to_string(&result)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            } else {
                serde_json::to_string(&result)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            };

            Ok(PyBytes::new(py, json_str.as_bytes()).into())
        })
    }

    fn decode_batch(&self, json_str: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let calls: Vec<MCPToolCall> = if self.use_simd {
                from_str(json_str)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            } else {
                serde_json::from_str(json_str)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            };

            let list = pyo3::types::PyList::empty(py);
            for call in calls {
                let dict = PyDict::new(py);
                dict.set_item("id", &call.id)?;
                dict.set_item("name", &call.name)?;
                dict.set_item("arguments", serde_json::to_value(&call.arguments).unwrap())?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    fn encode_batch(&self, results: Vec<(String, Vec<serde_json::Value>, bool)>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mcp_results: Vec<MCPToolResult> = results.into_iter()
                .map(|(id, content, is_error)| MCPToolResult { id, content, is_error })
                .collect();

            let json_str = if self.use_simd {
                simd_json::to_string(&mcp_results)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            } else {
                serde_json::to_string(&mcp_results)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            };

            Ok(PyBytes::new(py, json_str.as_bytes()).into())
        })
    }
}

#[pymodule]
fn terradev_mcp_codec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MCPCodec>()?;
    Ok(())
}
