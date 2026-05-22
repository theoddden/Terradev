use crate::types::{GPUDevice, NICDevice, GPUNICPair, PciLocality};

pub struct GPUNICOptimizer;

impl GPUNICOptimizer {
    pub fn compute_optimal_pairs(
        gpus: Vec<GPUDevice>,
        nics: Vec<NICDevice>,
    ) -> Vec<GPUNICPair> {
        let mut pairs = Vec::new();
        let mut available_nics = nics.clone();
        
        let locality_score = |loc: &PciLocality| -> u32 {
            match loc {
                PciLocality::Pix => 0,
                PciLocality::Pxb => 1,
                PciLocality::Phb => 2,
                PciLocality::Sys => 3,
            }
        };
        
        for gpu in &gpus {
            if available_nics.is_empty() {
                break;
            }
            
            // Find best NIC for this GPU
            let mut best_idx = 0;
            let mut best_score = u32::MAX;
            
            for (idx, nic) in available_nics.iter().enumerate() {
                let mut score = locality_score(&gpu.locality);
                
                // Prefer RDMA-capable NICs
                if !nic.rdma_capable {
                    score += 10;
                }
                
                // Prefer same NUMA node
                if gpu.numa_node != nic.numa_node {
                    score += 5;
                }
                
                if score < best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }
            
            let nic = available_nics.remove(best_idx);
            pairs.push(GPUNICPair {
                gpu_index: gpu.index,
                nic_name: nic.name.clone(),
                locality: gpu.locality.clone(),
                score: best_score,
            });
        }
        
        pairs
    }
}
