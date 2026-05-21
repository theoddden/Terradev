#![allow(non_local_definitions)]

mod graph;
mod types;

use graph::EgressGraph;
use pyo3::prelude::*;
use types::{EgressEdge, Region};

#[pymodule]
fn terradev_egress_optimizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEgressGraph>()?;
    m.add_class::<PyRegion>()?;
    m.add_class::<PyEgressEdge>()?;
    m.add_class::<PyTransferPlan>()?;
    Ok(())
}

#[pyclass]
pub struct PyEgressGraph {
    inner: EgressGraph,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyEgressGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: EgressGraph::new(),
        }
    }
    
    fn add_region(&mut self, region: PyRegion) {
        self.inner.add_region(&region.into());
    }
    
    fn add_edge(&mut self, edge: PyEgressEdge) {
        self.inner.add_edge(&edge.into());
    }
    
    fn find_cheapest_route(&self, from: String, to: String) -> Option<PyTransferPlan> {
        self.inner.find_cheapest_route(&from, &to).map(|(route, cost)| {
            PyTransferPlan {
                total_cost_per_gb: cost,
                estimated_time_hours: 0.0, // Would need data size to calculate
                hops: route.len() - 1,
                route,
            }
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyRegion {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub provider: String,
    #[pyo3(get, set)]
    pub continent: String,
}

impl From<PyRegion> for Region {
    fn from(p: PyRegion) -> Self {
        Self {
            id: p.id,
            name: p.name,
            provider: p.provider,
            continent: p.continent,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyEgressEdge {
    #[pyo3(get, set)]
    pub from_region: String,
    #[pyo3(get, set)]
    pub to_region: String,
    #[pyo3(get, set)]
    pub cost_per_gb: f64,
    #[pyo3(get, set)]
    pub bandwidth_gbps: f64,
}

impl From<PyEgressEdge> for EgressEdge {
    fn from(p: PyEgressEdge) -> Self {
        Self {
            from_region: p.from_region,
            to_region: p.to_region,
            cost_per_gb: p.cost_per_gb,
            bandwidth_gbps: p.bandwidth_gbps,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyTransferPlan {
    #[pyo3(get)]
    pub route: Vec<String>,
    #[pyo3(get)]
    pub total_cost_per_gb: f64,
    #[pyo3(get)]
    pub estimated_time_hours: f64,
    #[pyo3(get)]
    pub hops: usize,
}
