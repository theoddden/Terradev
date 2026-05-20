use crate::types::{EgressEdge, Region};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::dijkstra;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

pub struct EgressGraph {
    graph: DiGraph<String, f64>,
    region_to_index: HashMap<String, NodeIndex>,
    index_to_region: HashMap<NodeIndex, String>,
}

impl EgressGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            region_to_index: HashMap::new(),
            index_to_region: HashMap::new(),
        }
    }
    
    pub fn add_region(&mut self, region: &Region) {
        if !self.region_to_index.contains_key(&region.id) {
            let idx = self.graph.add_node(region.id.clone());
            self.region_to_index.insert(region.id.clone(), idx);
            self.index_to_region.insert(idx, region.id.clone());
        }
    }
    
    pub fn add_edge(&mut self, edge: &EgressEdge) {
        if let (Some(&from_idx), Some(&to_idx)) = (
            self.region_to_index.get(&edge.from_region),
            self.region_to_index.get(&edge.to_region),
        ) {
            self.graph.update_edge(from_idx, to_idx, edge.cost_per_gb);
        }
    }
    
    pub fn find_cheapest_route(&self, from: &str, to: &str) -> Option<(Vec<String>, f64)> {
        let from_idx = *self.region_to_index.get(from)?;
        let to_idx = *self.region_to_index.get(to)?;
        
        let distances = dijkstra(&self.graph, from_idx, Some(to_idx), |e| *e.weight());
        
        let cost = *distances.get(&to_idx)?;
        
        // Reconstruct path using distances
        let mut path = vec![to.to_string()];
        let mut current = to_idx;
        
        while current != from_idx {
            let mut found = false;
            for edge in self.graph.edges_directed(current, petgraph::Direction::Incoming) {
                let pred_idx = edge.source();
                if let Some(&pred_cost) = distances.get(&pred_idx) {
                    let edge_cost = edge.weight();
                    if (pred_cost + edge_cost - cost).abs() < 0.001 {
                        current = pred_idx;
                        path.insert(0, self.index_to_region.get(&current)?.clone());
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                break;
            }
        }
        
        Some((path, cost))
    }
}
