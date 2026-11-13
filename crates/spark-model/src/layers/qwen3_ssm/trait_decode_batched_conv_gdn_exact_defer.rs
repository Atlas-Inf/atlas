// SPDX-License-Identifier: AGPL-3.0-only

//! Whether a verify takes the deferred fused GDN rows (`ATLAS_QWEN38_VERIFY_DEFER`).

use super::*;

impl Qwen3SsmLayer {
    /// Whether a verify of `num_tokens` rows on this layer takes the deferred
    /// fused rows (`ATLAS_QWEN38_VERIFY_DEFER`): the lever, the exact verify
    /// on the fused rows' four-kernel FP32 arm, both kernels and the geometry,
    /// and a window the staging holds. The dispatch entry marks the step
    /// pending from this (`mark_gdn_deferred_commit`, through
    /// `gdn_exact_defer`), outside any capture.
    pub(in crate::layers::qwen3_ssm) fn q38_defer_arm(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        levers: &crate::layers::ops::ModelLevers,
        config: &atlas_core::config::ModelConfig,
        num_tokens: usize,
    ) -> bool {
        if !ops::qwen38_decode_fuse::defer_on() {
            return false;
        }
        let (kv, kc) = ops::qwen38_decode_fuse::defer_kernels(gpu);
        kv.0 != 0
            && kc.0 != 0
            && crate::layers::qwen3_ssm::verify_exact_for(levers)
            && self.conv1d_l2norm_f32_k.0 != 0
            && self.gdn_f32_k.0 != 0
            && self.gated_rms_norm_f32_k.0 != 0
            && !crate::layers::qwen3_ssm::gdn_fused_norm_enabled()
            && (1..=ops::qwen38_decode_fuse::DEFER_ROWS_MAX).contains(&num_tokens)
            && config.linear_key_head_dim == 128
            && config.linear_value_head_dim == 128
            && config.linear_conv_kernel_dim == 4
            && config.linear_num_key_heads != 0
            && config.linear_num_value_heads == 3 * config.linear_num_key_heads
    }
}
