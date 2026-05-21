#![allow(non_local_definitions)]

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct CompressedTool {
    pub name: String,
    pub namespace: Option<String>,
    pub compressed_schema: Vec<u8>,
}

pub struct MCPOptimizer {
    enable_compression: bool,
    strip_optional: bool,
    #[allow(dead_code)]
    enable_parallel: bool,
}

impl MCPOptimizer {
    pub fn new(enable_compression: bool, strip_optional: bool, enable_parallel: bool) -> Self {
        Self {
            enable_compression,
            strip_optional,
            enable_parallel,
        }
    }

    pub fn compress_tools(&self, tools: Vec<Tool>) -> Vec<Tool> {
        if !self.enable_compression {
            return tools;
        }

        tools
            .into_iter()
            .map(|mut tool| {
                if self.strip_optional {
                    tool = self.strip_optional_fields(tool);
                }
                tool
            })
            .collect()
    }

    fn strip_optional_fields(&self, mut tool: Tool) -> Tool {
        if let Some(obj) = tool.input_schema.as_object_mut() {
            let required_clone = obj.get("required").and_then(|r| r.as_array()).cloned();
            if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
                if let Some(required) = required_clone {
                    let required_set: std::collections::HashSet<&str> =
                        required.iter().filter_map(|v| v.as_str()).collect();

                    props.retain(|k, _| required_set.contains(k.as_str()));
                }
            }
        }
        tool
    }

    pub fn expand_call(
        &self,
        tool_name: String,
        arguments: HashMap<String, serde_json::Value>,
    ) -> (String, HashMap<String, serde_json::Value>) {
        // If namespace is present, extract original tool name
        if tool_name.contains('.') {
            let parts: Vec<&str> = tool_name.split('.').collect();
            if parts.len() == 2 {
                return (parts[1].to_string(), arguments);
            }
        }
        (tool_name, arguments)
    }
}

#[pyclass]
pub struct PyMCPOptimizer {
    inner: MCPOptimizer,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyMCPOptimizer {
    #[new]
    fn new(enable_compression: bool, strip_optional: bool, enable_parallel: bool) -> Self {
        Self {
            inner: MCPOptimizer::new(enable_compression, strip_optional, enable_parallel),
        }
    }

    fn compress_tools(&self, tools_py: Vec<PyObject>) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let tools: Vec<Tool> = tools_py
                .into_iter()
                .map(|obj| {
                    let tool_dict: &pyo3::types::PyDict = obj.extract(py)?;
                    let name = tool_dict.get_item("name")?;
                    let description = tool_dict.get_item("description")?;
                    let input_schema = tool_dict.get_item("inputSchema")?;
                    Ok(Tool {
                        name: name.unwrap().extract()?,
                        description: description.unwrap().extract()?,
                        input_schema: {
                            let schema_str: String = input_schema.unwrap().extract()?;
                            serde_json::from_str(&schema_str)
                                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                        },
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;

            let compressed = self.inner.compress_tools(tools);

            compressed
                .into_iter()
                .map(|tool| {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("name", tool.name)?;
                    dict.set_item("description", tool.description)?;
                    let schema_str = serde_json::to_string(&tool.input_schema)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    let schema_py: PyObject = pyo3::types::PyString::new(py, &schema_str).into();
                    let schema_value: PyObject = pyo3::types::PyModule::import(py, "json")?
                        .getattr("loads")?
                        .call1((schema_py,))?
                        .into();
                    dict.set_item("inputSchema", schema_value)?;
                    Ok(dict.into())
                })
                .collect()
        })
    }

    fn expand_call(
        &self,
        tool_name: String,
        arguments: HashMap<String, PyObject>,
    ) -> PyResult<(String, HashMap<String, PyObject>)> {
        let args_json: HashMap<String, serde_json::Value> = Python::with_gil(|py| {
            arguments
                .into_iter()
                .map(|(k, v)| {
                    let v_str: String = v.extract(py)?;
                    let v_json: serde_json::Value = serde_json::from_str(&v_str)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    Ok::<(String, serde_json::Value), pyo3::PyErr>((k, v_json))
                })
                .collect::<PyResult<HashMap<_, _>>>()
        })?;

        let (new_name, new_args) = self.inner.expand_call(tool_name, args_json);

        let result_args = Python::with_gil(|py| {
            new_args
                .into_iter()
                .map(|(k, v)| {
                    let v_str = serde_json::to_string(&v)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    let v_py: PyObject = pyo3::types::PyString::new(py, &v_str).into();
                    let v_value: PyObject = pyo3::types::PyModule::import(py, "json")?
                        .getattr("loads")?
                        .call1((v_py,))?
                        .into();
                    Ok::<(String, Py<PyAny>), pyo3::PyErr>((k, v_value))
                })
                .collect::<PyResult<HashMap<_, _>>>()
        })?;

        Ok((new_name, result_args))
    }
}

#[pymodule]
fn terradev_mcp_optimizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMCPOptimizer>()?;
    Ok(())
}
