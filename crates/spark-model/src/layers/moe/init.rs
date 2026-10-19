// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::new constructor.

use super::*;

impl MoeLayer {
    pub fn new(
        weights: MoeWeights,
        num_experts: usize,
        gate_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        Self::new_with_hash(weights, num_experts, gate_nvfp4, None, gpu, config)
    }

    /// Opt in to row-invariant BF16 routing for decode/verify (`types.rs`).
    /// Called only by the qwen4_exp FP4-Local builder, which ships mlp.gate
    /// as BF16; other models keep the scalar `dense_gemm` router arm.
    pub fn set_row_invariant_bf16_router(&mut self) {
        self.row_invariant_bf16_router = true;
    }
}
