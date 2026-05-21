use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyAny};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam::thread;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DAGNode {
    pub name: String,
    pub dependencies: Vec<String>,
    pub output: Option<serde_json::Value>,
    pub status: String,
    pub latency_ms: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionWave {
    pub depth: usize,
    pub nodes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub waves: Vec<ExecutionWave>,
    pub total_nodes: usize,
    pub max_parallelism: usize,
    pub critical_path_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub outputs: HashMap<String, serde_json::Value>,
    pub node_latencies: HashMap<String, f64>,
    pub node_statuses: HashMap<String, String>,
    pub total_latency_ms: f64,
    pub wall_clock_ms: f64,
    pub parallelism_achieved: f64,
    pub errors: HashMap<String, String>,
}

#[pyclass]
pub struct DAGExecutor {
    name: String,
    nodes: Arc<RwLock<HashMap<String, DAGNode>>>,
    max_workers: usize,
}

#[allow(non_local_definitions)]
#[pymethods]
impl DAGExecutor {
    #[new]
    #[pyo3(signature = (name="dag_executor", max_workers=None))]
    fn new(name: String, max_workers: Option<usize>) -> Self {
        let max_workers = max_workers.unwrap_or_else(|| {
            let cpu_count = num_cpus::get();
            std::cmp::min(32, std::cmp::max(4, cpu_count * 2))
        });
        
        Self {
            name,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            max_workers,
        }
    }

    fn add_node(
        &self,
        name: String,
        dependencies: Option<Vec<String>>,
    ) -> PyResult<()> {
        let mut nodes = self.nodes.write();
        if nodes.contains_key(&name) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Node '{}' already exists in DAG '{}'", name, self.name)
            ));
        }
        
        nodes.insert(name, DAGNode {
            name: name.clone(),
            dependencies: dependencies.unwrap_or_default(),
            output: None,
            status: "pending".to_string(),
            latency_ms: 0.0,
            error: None,
        });
        Ok(())
    }

    fn plan(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let nodes = self.nodes.read();
            
            // Build dependency graph
            let mut graph = DiGraph::new();
            let mut node_indices: HashMap<String, NodeIndex> = HashMap::new();
            
            // Add all nodes
            for name in nodes.keys() {
                let idx = graph.add_node(name.clone());
                node_indices.insert(name.clone(), idx);
            }
            
            // Add edges
            for (name, node) in nodes.iter() {
                if let Some(from_idx) = node_indices.get(name) {
                    for dep in &node.dependencies {
                        if let Some(to_idx) = node_indices.get(dep) {
                            graph.add_edge(*to_idx, *from_idx);
                        }
                    }
                }
            }
            
            // Topological sort
            let sorted = match toposort(&graph, None) {
                Ok(sorted) => sorted,
                Err(_) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Cycle detected in DAG '{}'", self.name)
                    ));
                }
            };
            
            // Build execution waves
            let mut waves: Vec<ExecutionWave> = Vec::new();
            let mut in_degree: HashMap<String, usize> = HashMap::new();
            let mut forward: HashMap<String, Vec<String>> = HashMap::new();
            
            // Initialize in-degree
            for (name, node) in nodes.iter() {
                in_degree.insert(name.clone(), node.dependencies.len());
                for dep in &node.dependencies {
                    forward.entry(dep.clone()).or_default().push(name.clone());
                }
            }
            
            // Kahn's algorithm
            let mut queue: Vec<String> = nodes.keys()
                .filter(|k| in_degree.get(*k) == Some(&0))
                .cloned()
                .collect();
            
            let mut depth = 0;
            let mut processed = 0;
            
            while !queue.is_empty() {
                waves.push(ExecutionWave {
                    depth,
                    nodes: queue.clone(),
                });
                
                let mut next_queue: Vec<String> = Vec::new();
                
                for name in &queue {
                    processed += 1;
                    if let Some(successors) = forward.get(name) {
                        for succ in successors {
                            if let Some(deg) = in_degree.get_mut(succ) {
                                *deg -= 1;
                                if *deg == 0 {
                                    next_queue.push(succ.clone());
                                }
                            }
                        }
                    }
                }
                
                queue = next_queue;
                depth += 1;
            }
            
            if processed != nodes.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("DAG '{}' has unreachable nodes", self.name)
                ));
            }
            
            let max_parallelism = waves.iter().map(|w| w.nodes.len()).max().unwrap_or(0);
            
            let plan = ExecutionPlan {
                waves,
                total_nodes: nodes.len(),
                max_parallelism,
                critical_path_depth: depth,
            };
            
            let dict = PyDict::new(py);
            dict.set_item("waves", serde_json::to_string(&plan.waves).unwrap())?;
            dict.set_item("total_nodes", plan.total_nodes)?;
            dict.set_item("max_parallelism", plan.max_parallelism)?;
            dict.set_item("critical_path_depth", plan.critical_path_depth)?;
            
            Ok(dict.into())
        })
    }

    fn apply(&self, initial_context: Option<&PyDict>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let plan = self.plan()?;
            let plan_dict: &PyDict = plan.extract(py)?;
            
            let waves_json: String = plan_dict.get_item("waves").unwrap().extract()?;
            let waves: Vec<ExecutionWave> = serde_json::from_str(&waves_json).unwrap();
            
            let mut context: HashMap<String, serde_json::Value> = if let Some(ctx) = initial_context {
                ctx.extract()?
            } else {
                HashMap::new()
            };
            
            let wall_start = std::time::Instant::now();
            let mut result = ExecutionResult {
                outputs: HashMap::new(),
                node_latencies: HashMap::new(),
                node_statuses: HashMap::new(),
                total_latency_ms: 0.0,
                wall_clock_ms: 0.0,
                parallelism_achieved: 0.0,
                errors: HashMap::new(),
            };
            
            let nodes = self.nodes.clone();
            
            for wave in waves {
                let mut wave_results: HashMap<String, (serde_json::Value, f64)> = HashMap::new();
                
                thread::scope(|s| {
                    let handles: Vec<_> = wave.nodes.iter().map(|node_name| {
                        let nodes_ref = nodes.clone();
                        let node_name = node_name.clone();
                        s.spawn(move |_| {
                            let nodes = nodes_ref.read();
                            let node = nodes.get(&node_name).unwrap();
                            
                            let start = std::time::Instant::now();
                            
                            // Simulate node execution (in real implementation, this would call Python callable)
                            let output = serde_json::json!({
                                "node": node_name,
                                "status": "done"
                            });
                            
                            let latency = start.elapsed().as_secs_f64() * 1000.0;
                            
                            (node_name, output, latency)
                        })
                    }).collect();
                    
                    for handle in handles {
                        let (name, output, latency) = handle.join().unwrap();
                        wave_results.insert(name, (output, latency));
                    }
                });
                
                // Merge wave results into context
                for (name, (output, latency)) in wave_results {
                    context.insert(name.clone(), output.clone());
                    result.outputs.insert(name.clone(), output);
                    result.node_latencies.insert(name.clone(), latency);
                    result.node_statuses.insert(name, "done".to_string());
                }
            }
            
            let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
            let total_node_ms: f64 = result.node_latencies.values().sum();
            
            result.wall_clock_ms = wall_ms;
            result.total_latency_ms = total_node_ms;
            result.parallelism_achieved = if wall_ms > 0.0 { total_node_ms / wall_ms } else { 1.0 };
            
            let dict = PyDict::new(py);
            dict.set_item("outputs", serde_json::to_string(&result.outputs).unwrap())?;
            dict.set_item("node_latencies", serde_json::to_string(&result.node_latencies).unwrap())?;
            dict.set_item("node_statuses", serde_json::to_string(&result.node_statuses).unwrap())?;
            dict.set_item("total_latency_ms", result.total_latency_ms)?;
            dict.set_item("wall_clock_ms", result.wall_clock_ms)?;
            dict.set_item("parallelism_achieved", result.parallelism_achieved)?;
            dict.set_item("errors", serde_json::to_string(&result.errors).unwrap())?;
            
            Ok(dict.into())
        })
    }
}

#[pymodule]
fn terradev_dag_executor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DAGExecutor>()?;
    Ok(())
}
