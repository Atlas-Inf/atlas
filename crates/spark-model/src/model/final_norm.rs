// SPDX-License-Identifier: AGPL-3.0-only

//! Model-level final RMSNorm dispatch.
//!
//! qwen4_exp's last layer already runs the model-level hyper-connection mixer
//! (`hc_head_site`), whose `hc_norm` IS the model's final normalization — the
//! HF reference feeds the mixer output straight to lm_head. Atlas's `final_norm`
//! for that model type is a ones-filled placeholder and the rms_norm kernel
//! computes `x * (1 + w) / rms(x)`, so applying it rescaled every position's
//! logits by `2 / rms(x)`: argmax survived but every probability was wrong.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::layers::ops;

/// Models whose final normalization happens inside the last layer (qwen4_exp's
/// hyper-connection mixer), so the model-level final RMSNorm must be skipped.
pub(crate) fn final_norm_is_identity(model_type: &str) -> bool {
    model_type == "qwen4_exp"
}

impl TransformerModel {
    /// Final norm of `rows` hidden rows into `output`: the usual RMSNorm, or a
    /// plain copy when the model's final normalization already happened
    /// (`final_norm_identity`).
    pub(super) fn final_norm_rows(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        if self.final_norm_identity {
            return self
                .gpu
                .copy_d2d_async(input, output, rows as usize * h * 2, stream);
        }
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            input,
            &self.final_norm,
            output,
            rows,
            h as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::final_norm_is_identity;

    #[test]
    fn qwen4_exp_final_norm_is_identity() {
        assert!(final_norm_is_identity("qwen4_exp"));
    }

    #[test]
    fn other_models_keep_final_rms_norm() {
        assert!(!final_norm_is_identity("qwen3_next"));
        assert!(!final_norm_is_identity("deepseek_v4"));
        assert!(!final_norm_is_identity("qwen3_5_moe"));
    }
}
