use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub name: String,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
    pub max_sequence_length: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRAMBreakdown {
    pub model_weights_gb: f64,
    pub kv_cache_gb: f64,
    pub activation_cache_gb: f64,
    pub overhead_gb: f64,
    pub total_gb: f64,
    pub per_gpu_gb: f64,
    pub gpu_count: u32,
    pub architecture: String,
    pub context_tokens: u32,
    pub batch_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    FP8,
    INT8,
    INT4,
}

impl Precision {
    pub fn bytes_per_parameter(&self) -> f64 {
        match self {
            Precision::FP32 => 4.0,
            Precision::FP16 => 2.0,
            Precision::BF16 => 2.0,
            Precision::FP8 => 1.0,
            Precision::INT8 => 1.0,
            Precision::INT4 => 0.5,
        }
    }
}
