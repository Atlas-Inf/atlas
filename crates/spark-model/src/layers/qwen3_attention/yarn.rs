// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3AttentionLayer;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Called by the shared attention constructor, so native MTP gets exactly
    /// the same positional policy as the target, in Strict and Spark modes.
    pub(super) fn configure_qwen_yarn(
        &mut self,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if config.model_type != "qwen4_exp" || config.yarn_factor == 0.0 {
            return Ok(());
        }
        anyhow::ensure!(
            config.mrope_interleaved
                && config.mrope_section == [11, 11, 10]
                && config.rotary_dim() == 64,
            "Qwen YaRN requires the native 64-dimension interleaved MRoPE layout"
        );
        let frequencies = atlas_core::qwen_yarn::inverse_frequencies(config)?;
        let kernel = gpu.kernel(
            "rope_mrope_interleaved",
            "rope_forward_mrope_interleaved_yarn",
        )?;
        let bytes: Vec<u8> = frequencies.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ptr = gpu.alloc(bytes.len())?;
        if let Err(e) = gpu.copy_h2d(&bytes, ptr) {
            let _ = gpu.free(ptr);
            return Err(e);
        }
        self.set_yarn_rope(ptr, config.yarn_attention_factor);
        self.rope_yarn_mrope_k = kernel;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_scaled_rope(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        k: DevicePtr,
        pos_t: DevicePtr,
        pos_h: DevicePtr,
        pos_w: DevicePtr,
        n: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        rot: u32,
        stream: u64,
    ) -> Result<()> {
        if self.rope_yarn_mrope_k.0 != 0 {
            ops::rope_yarn_mrope(
                gpu,
                self.rope_yarn_mrope_k,
                q,
                k,
                pos_t,
                pos_h,
                pos_w,
                n,
                nq,
                nkv,
                hd,
                rot,
                self.yarn_inv_freq,
                self.yarn_attention_factor,
                stream,
            )
        } else {
            ops::rope_yarn_scaled(
                gpu,
                self.rope_yarn_scaled_k,
                q,
                k,
                pos_t,
                n,
                nq,
                nkv,
                hd,
                rot,
                self.yarn_inv_freq,
                self.yarn_attention_factor,
                stream,
            )
        }
    }
}
