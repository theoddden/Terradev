#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
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
#[allow(non_local_definitions)]
pub struct MCPCodec {
    use_simd: bool,
}

#[allow(non_local_definitions)]
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
                let mut json_str_mut = json_str.to_string();
                let call: MCPToolCall = unsafe { from_str(&mut json_str_mut) }
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

                let dict = PyDict::new(py);
                dict.set_item("id", &call.id)?;
                dict.set_item("name", &call.name)?;
                let args_dict = serde_json::to_string(&call.arguments).unwrap();
                dict.set_item("arguments", pyo3::types::PyString::new(py, &args_dict))?;
                Ok(dict.into())
            } else {
                let call: MCPToolCall = serde_json::from_str(json_str)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

                let dict = PyDict::new(py);
                dict.set_item("id", &call.id)?;
                dict.set_item("name", &call.name)?;
                let args_dict = serde_json::to_string(&call.arguments).unwrap();
                dict.set_item("arguments", pyo3::types::PyString::new(py, &args_dict))?;
                Ok(dict.into())
            }
        })
    }

    fn encode_tool_result(
        &self,
        id: String,
        content: PyObject,
        is_error: Option<bool>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let content_vec: Vec<serde_json::Value> =
                if let Ok(list) = content.extract::<Vec<PyObject>>(py) {
                    list.iter()
                        .map(|obj| {
                            if let Ok(s) = obj.extract::<String>(py) {
                                serde_json::from_str(&s)
                                    .ok()
                                    .unwrap_or(serde_json::Value::String(s))
                            } else if let Ok(n) = obj.extract::<i64>(py) {
                                serde_json::Value::Number(n.into())
                            } else if let Ok(f) = obj.extract::<f64>(py) {
                                serde_json::json!(f)
                            } else if let Ok(b) = obj.extract::<bool>(py) {
                                serde_json::Value::Bool(b)
                            } else {
                                serde_json::Value::Null
                            }
                        })
                        .collect()
                } else {
                    vec![]
                };

            let result = MCPToolResult {
                id,
                content: content_vec,
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
                let mut json_str_mut = json_str.to_string();
                unsafe { from_str(&mut json_str_mut) }
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
                let args_dict = serde_json::to_string(&call.arguments).unwrap();
                dict.set_item("arguments", pyo3::types::PyString::new(py, &args_dict))?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    fn encode_batch(&self, results: PyObject) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let results_vec: Vec<(String, Vec<serde_json::Value>, bool)> =
                if let Ok(list) = results.extract::<Vec<PyObject>>(py) {
                    list.iter()
                        .map(|obj| {
                            if let Ok(tuple) = obj.extract::<(String, PyObject, bool)>(py) {
                                let content_vec: Vec<serde_json::Value> =
                                    if let Ok(list) = tuple.1.extract::<Vec<PyObject>>(py) {
                                        list.iter()
                                            .map(|item| {
                                                if let Ok(s) = item.extract::<String>(py) {
                                                    serde_json::from_str(&s)
                                                        .ok()
                                                        .unwrap_or(serde_json::Value::String(s))
                                                } else if let Ok(n) = item.extract::<i64>(py) {
                                                    serde_json::Value::Number(n.into())
                                                } else if let Ok(f) = item.extract::<f64>(py) {
                                                    serde_json::json!(f)
                                                } else if let Ok(b) = item.extract::<bool>(py) {
                                                    serde_json::Value::Bool(b)
                                                } else {
                                                    serde_json::Value::Null
                                                }
                                            })
                                            .collect()
                                    } else {
                                        vec![]
                                    };
                                (tuple.0, content_vec, tuple.2)
                    } else {
                        (String::new(), vec![], false)
                    }
                }).collect()
            } else {
                vec![]
            };

            let mcp_results: Vec<MCPToolResult> = results_vec
                .into_iter()
                .map(|(id, content, is_error)| MCPToolResult {
                    id,
                    content,
                    is_error,
                })
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
