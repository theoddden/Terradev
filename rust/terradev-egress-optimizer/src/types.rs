use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub continent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgressEdge {
    pub from_region: String,
    pub to_region: String,
    pub cost_per_gb: f64,
    pub bandwidth_gbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferPlan {
    pub route: Vec<String>,
    pub total_cost_per_gb: f64,
    pub estimated_time_hours: f64,
    pub hops: usize,
}
