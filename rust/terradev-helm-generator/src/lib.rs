use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use tera::{Tera, Context};
use std::collections::HashMap;

#[pyclass]
pub struct HelmGenerator {
    tera: Tera,
}

#[pymethods]
impl HelmGenerator {
    #[new]
    #[pyo3(signature = (template_dir=None))]
    fn new(template_dir: Option<String>) -> PyResult<Self> {
        let tera = if let Some(dir) = template_dir {
            Tera::new(&format!("{}/**/*.yaml", dir))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        } else {
            Tera::default()
        };
        
        Ok(Self { tera })
    }

    fn add_template(&mut self, name: String, content: String) -> PyResult<()> {
        self.tera.add_raw_template(&name, &content)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    fn render_template(&self, name: String, context: Option<&PyDict>) -> PyResult<String> {
        let mut ctx = Context::new();
        
        if let Some(py_ctx) = context {
            for (key, value) in py_ctx.iter() {
                let key_str: String = key.extract()?;
                let value_json: serde_json::Value = value.extract()?;
                ctx.insert(&key_str, &value_json);
            }
        }
        
        self.tera.render(&name, &ctx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn render_values(&self, values: &PyDict) -> PyResult<String> {
        let mut ctx = Context::new();
        
        for (key, value) in values.iter() {
            let key_str: String = key.extract()?;
            let value_json: serde_json::Value = value.extract()?;
            ctx.insert(&key_str, &value_json);
        }
        
        // Render as YAML
        let yaml_str = serde_yaml::to_string(&ctx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        Ok(yaml_str)
    }

    fn generate_deployment(&self, name: String, image: String, replicas: Option<u32>, resources: Option<&PyDict>) -> PyResult<String> {
        let mut ctx = Context::new();
        ctx.insert("name", &name);
        ctx.insert("image", &image);
        ctx.insert("replicas", &replicas.unwrap_or(1));
        
        if let Some(res) = resources {
            let res_json: serde_json::Value = res.extract()?;
            ctx.insert("resources", &res_json);
        }
        
        let template = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ name }}
spec:
  replicas: {{ replicas }}
  selector:
    matchLabels:
      app: {{ name }}
  template:
    metadata:
      labels:
        app: {{ name }}
    spec:
      containers:
      - name: {{ name }}
        image: {{ image }}
        {{#if resources}}
        resources:
          {{ resources }}
        {{/if}}
"#;
        
        self.tera.add_raw_template("deployment", template)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        self.tera.render("deployment", &ctx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn generate_service(&self, name: String, port: u32, service_type: Option<String>) -> PyResult<String> {
        let mut ctx = Context::new();
        ctx.insert("name", &name);
        ctx.insert("port", &port);
        ctx.insert("type", &service_type.unwrap_or_else(|| "ClusterIP".to_string()));
        
        let template = r#"
apiVersion: v1
kind: Service
metadata:
  name: {{ name }}
spec:
  type: {{ type }}
  selector:
    app: {{ name }}
  ports:
  - port: {{ port }}
    targetPort: {{ port }}
"#;
        
        self.tera.add_raw_template("service", template)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        self.tera.render("service", &ctx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

#[pymodule]
fn terradev_helm_generator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HelmGenerator>()?;
    Ok(())
}
