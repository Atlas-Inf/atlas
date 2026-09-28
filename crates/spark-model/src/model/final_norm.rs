// SPDX-License-Identifier: AGPL-3.0-only

//! Model-level final RMSNorm dispatch.
//!
//! qwen4_exp's last layer already runs the model-level hyper-connection mixer
//! (`hc_head_site`), whose `hc_norm` IS the model's final normalization — the
//! HF reference feeds the mixer output straight to lm_head. Atlas's `final_norm`
//! for that model type is a ones-filled placeholder and the rms_norm kernel
//! computes `x * (1 + w) / rms(x)`, so applying it rescaled every position's
//! logits by `2 / rms(x)`: argmax survived but every probability was wrong.

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::layers::ops;

/// The hidden rows handed to the final norm are BF16 — the same buffers
/// `rms_norm_kernel` reads.
const BF16_BYTES: usize = 2;

/// Refuses a model whose final norm is the identity unless its last layer is
/// the one that runs the mixer.
///
/// The identity is only correct if the LAST model layer ran the hyper-connection
/// mixer that collapses the `hc_mult` streams into `hidden_size`-wide rows:
/// today only the attention layers call `ops::hc_head_site`, so a config ending
/// on a GDN layer would silently feed the first `hidden_size` values of an
/// un-mixed `hc_mult*hidden_size` stream to lm_head.
pub(crate) fn check_final_norm_identity(config: &ModelConfig) -> Result<()> {
    if !config.final_norm_is_identity() {
        return Ok(());
    }
    if config.hc_mult.max(config.hc_count) == 0 {
        bail!(
            "{}: final norm is the identity but no hyper-connection streams are configured \
             (hc_mult/hc_count are both 0), so no layer collapses them",
            config.model_type
        );
    }
    let last = config.num_hidden_layers - 1;
    match config.layer_type(last) {
        LayerType::FullAttention => Ok(()),
        other => bail!(
            "{}: final norm is the identity but the last layer {} is {other:?}, not full \
             attention — only the attention layers run the hyper-connection mixer",
            config.model_type,
            last
        ),
    }
}

impl TransformerModel {
    /// Final norm of `rows` hidden rows into `output`: the usual RMSNorm, or a
    /// plain copy when the model's final normalization already happened
    /// (`final_norm_identity`).
    ///
    /// Precondition: the rows are BF16 `[rows, hidden_size]` — post-mixer
    /// (collapsed to `hidden_size` by the last layer) when the identity applies.
    pub(super) fn final_norm_rows(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        if self.final_norm_identity {
            if input == output {
                return Ok(());
            }
            return self
                .gpu
                .copy_d2d_async(input, output, rows as usize * h * BF16_BYTES, stream);
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
    use super::check_final_norm_identity;
    use atlas_core::config::{LayerType, ModelConfig};

    /// A qwen4_exp-shaped config: hyper-connections on, last layer full attention.
    fn qwen4_exp_like() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        config.hc_mult = 4;
        config
    }

    #[test]
    fn identity_ok_when_last_layer_runs_the_mixer() {
        // The factory layout (full_attention_interval = 4, 48 layers) ends on a
        // full-attention layer.
        let config = qwen4_exp_like();
        assert_eq!(
            config.layer_type(config.num_hidden_layers - 1),
            LayerType::FullAttention
        );
        assert!(check_final_norm_identity(&config).is_ok());
    }

    #[test]
    fn identity_refuses_a_gdn_last_layer_naming_its_index() {
        let mut config = qwen4_exp_like();
        let last = config.num_hidden_layers - 1;
        config.layer_types[last] = LayerType::LinearAttention;
        let err = check_final_norm_identity(&config).unwrap_err().to_string();
        assert!(err.contains("qwen4_exp"), "{err}");
        assert!(err.contains(&last.to_string()), "{err}");
        assert!(err.contains("LinearAttention"), "{err}");
    }

    #[test]
    fn identity_refuses_no_hyper_connections() {
        let mut config = qwen4_exp_like();
        config.hc_mult = 0;
        let err = check_final_norm_identity(&config).unwrap_err().to_string();
        assert!(err.contains("qwen4_exp"), "{err}");
    }

    #[test]
    fn other_models_are_not_checked() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(check_final_norm_identity(&config).is_ok());
    }
}
