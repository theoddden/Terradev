#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[pyclass]
pub struct ToolRegistry {
    tools: HashMap<String, ToolSchema>,
}

#[allow(non_local_definitions)]
#[pymethods]
impl ToolRegistry {
    #[new]
    fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    fn register_tool(&mut self, name: String, description: String, input_schema: &PyDict) -> PyResult<()> {
        let schema: serde_json::Value = Python::with_gil(|py| {
            // Convert PyDict to string representation, then parse as JSON
            let schema_str = input_schema.str().unwrap_or_else(|_| pyo3::types::PyString::new(py, "{}")).to_string();
            serde_json::from_str(&schema_str).unwrap_or_else(|_| serde_json::json!({}))
        });

        self.tools.insert(name.clone(), ToolSchema {
            name,
            description,
            input_schema: schema,
        });
        Ok(())
    }

    fn get_tool(&self, name: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            if let Some(tool) = self.tools.get(name) {
                let dict = PyDict::new(py);
                dict.set_item("name", &tool.name)?;
                dict.set_item("description", &tool.description)?;
                let schema_str = serde_json::to_string(&tool.input_schema).unwrap_or_else(|_| "{}".to_string());
                dict.set_item("input_schema", schema_str)?;
                Ok(dict.into())
            } else {
                Ok(py.None())
            }
        })
    }

    fn get_all_tools(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty(py);
            for tool in self.tools.values() {
                let dict = PyDict::new(py);
                dict.set_item("name", &tool.name)?;
                dict.set_item("description", &tool.description)?;
                let schema_str = serde_json::to_string(&tool.input_schema).unwrap_or_else(|_| "{}".to_string());
                dict.set_item("input_schema", schema_str)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    fn get_tool_names(&self) -> PyResult<Vec<String>> {
        Ok(self.tools.keys().cloned().collect())
    }

    fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    fn count(&self) -> usize {
        self.tools.len()
    }

    fn unregister_tool(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    fn clear(&mut self) {
        self.tools.clear();
    }
}

#[pymodule]
fn terradev_tool_registry(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ToolRegistry>()?;
    Ok(())
}
