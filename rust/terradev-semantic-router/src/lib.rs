use pyo3::prelude::*;
use pyo3::types::PyDict, PyList;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub name: String,
    pub value: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub name: String,
    pub score: f64,
    pub reason: String,
}

#[pyclass]
pub struct SemanticRouter {
    routes: HashMap<String, RouteConfig>,
}

#[derive(Debug, Clone)]
struct RouteConfig {
    name: String,
    keywords: Vec<String>,
    threshold: f64,
}

#[pymethods]
impl SemanticRouter {
    #[new]
    fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }

    fn add_route(&mut self, name: String, keywords: Vec<String>, threshold: Option<f64>) {
        self.routes.insert(name.clone(), RouteConfig {
            name,
            keywords,
            threshold: threshold.unwrap_or(0.5),
        });
    }

    fn route(&self, query: &str, signals: Option<&PyDict>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut scores: HashMap<String, f64> = HashMap::new();
            let query_lower = query.to_lowercase();

            for (route_name, config) in &self.routes {
                let mut score = 0.0;
                let mut matches = 0;

                for keyword in &config.keywords {
                    let keyword_lower = keyword.to_lowercase();
                    if query_lower.contains(&keyword_lower) {
                        score += 1.0;
                        matches += 1;
                    }
                }

                // Normalize score
                if !config.keywords.is_empty() {
                    score = score / config.keywords.len() as f64;
                }

                // Apply signals if provided
                if let Some(sigs) = signals {
                    if let Ok(signal_value) = sigs.get_item(route_name) {
                        if let Ok(val) = signal_value.extract::<f64>() {
                            score = score * 0.7 + val * 0.3; // Weighted combination
                        }
                    }
                }

                if score >= config.threshold {
                    scores.insert(route_name.clone(), score);
                }
            }

            // Find best match
            let best_route = scores.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(name, score)| Route {
                    name: name.clone(),
                    score: *score,
                    reason: format!("Matched with {:.2} confidence", score),
                });

            let dict = PyDict::new(py);
            if let Some(route) = best_route {
                dict.set_item("route", &route.name)?;
                dict.set_item("score", route.score)?;
                dict.set_item("reason", &route.reason)?;
            } else {
                dict.set_item("route", "default")?;
                dict.set_item("score", 0.0)?;
                dict.set_item("reason", "No route matched threshold")?;
            }

            dict.set_item("all_scores", serde_json::to_value(&scores).unwrap())?;

            Ok(dict.into())
        })
    }

    fn batch_route(&self, queries: Vec<String>, signals: Option<Vec<&PyDict>>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let results = PyList::empty(py);

            for (i, query) in queries.iter().enumerate() {
                let sigs = signals.as_ref().and_then(|s| s.get(i).copied());
                let result = self.route(query, sigs)?;
                results.append(result)?;
            }

            Ok(results.into())
        })
    }

    fn clear(&mut self) {
        self.routes.clear();
    }
}

#[pymodule]
fn terradev_semantic_router(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SemanticRouter>()?;
    Ok(())
}
