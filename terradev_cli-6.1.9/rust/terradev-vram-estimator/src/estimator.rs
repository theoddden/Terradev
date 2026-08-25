use crate::types::{ModelArchitecture, Precision, VRAMBreakdown};

pub struct VRAMEstimator;

impl VRAMEstimator {
    pub fn estimate_vram(
        architecture: &ModelArchitecture,
        context_tokens: u32,
        batch_size: u32,
        precision: Precision,
        use_mla: bool,
    ) -> VRAMBreakdown {
        // Calculate model weights
        let params_per_layer = architecture.hidden_size * architecture.hidden_size * 4; // Approximate
        let total_params = params_per_layer * architecture.num_layers;
        let model_weights_gb = total_params as f64 * precision.bytes_per_parameter() / 1e9;

        // Calculate KV cache
        let kv_cache_gb = if use_mla {
            // MLA compression: 1/8th of standard KV cache
            Self::calculate_kv_cache(architecture, context_tokens, batch_size, &precision) * 0.125
        } else {
            Self::calculate_kv_cache(architecture, context_tokens, batch_size, &precision)
        };

        // Calculate activation cache
        let activation_cache_gb =
            Self::calculate_activation_cache(architecture, context_tokens, batch_size, &precision);

        // Framework overhead (vLLM, CUDA, etc.)
        let overhead_gb = 2.0; // Conservative estimate

        let total_gb = model_weights_gb + kv_cache_gb + activation_cache_gb + overhead_gb;

        // Calculate required GPU count
        let gpu_vram_gb = 80.0; // A100 80GB
        let gpu_count = ((total_gb / gpu_vram_gb).ceil() as u32).max(1);
        let per_gpu_gb = total_gb / gpu_count as f64;

        VRAMBreakdown {
            model_weights_gb,
            kv_cache_gb,
            activation_cache_gb,
            overhead_gb,
            total_gb,
            per_gpu_gb,
            gpu_count,
            architecture: architecture.name.clone(),
            context_tokens,
            batch_size,
        }
    }

    fn calculate_kv_cache(
        architecture: &ModelArchitecture,
        context_tokens: u32,
        batch_size: u32,
        precision: &Precision,
    ) -> f64 {
        // KV cache size = 2 * num_layers * num_heads * head_dim * seq_len * batch_size * bytes
        let head_dim = architecture.hidden_size / architecture.num_heads;
        let elements = (2
            * architecture.num_layers
            * architecture.num_heads
            * head_dim
            * context_tokens
            * batch_size) as f64;
        elements * precision.bytes_per_parameter() / 1e9
    }

    fn calculate_activation_cache(
        architecture: &ModelArchitecture,
        context_tokens: u32,
        batch_size: u32,
        precision: &Precision,
    ) -> f64 {
        // Activation cache = hidden_size * seq_len * batch_size * bytes
        let elements = (architecture.hidden_size * context_tokens * batch_size) as f64;
        elements * precision.bytes_per_parameter() / 1e9
    }
}
