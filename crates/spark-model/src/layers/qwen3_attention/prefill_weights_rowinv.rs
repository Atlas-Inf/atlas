// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ`: the attention prefill weights at decode
//! precision (split out of `prefill_weights.rs` for the 500-LoC cap).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::types::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ`: Q (+gate) / K / V / O as BF16 from the
    /// NVFP4 weights decode reads (`lut * e4m3 * scale2`, decode's weight
    /// values rounded once to BF16), so prefill runs BF16 x BF16 with FP32
    /// accumulation like decode, on the row-invariant k-chain -- in place of
    /// the FP8 copies (E4M3 weights x E4M3 activations).
    pub(super) fn dequant_bf16_for_rowinv(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let k_dq = crate::layers::try_kernel(gpu, "dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16");
        anyhow::ensure!(
            k_dq.0 != 0,
            "ATLAS_QWEN4EXP_PREFILL_BF16_PROJ: no dequant_nvfp4_to_bf16"
        );
        // This rank's heads (TP), as the prefill projections read them.
        let h = config.hidden_size;
        let hd = self.head_dim_override.unwrap_or(config.head_dim);
        let q_dim = self
            .num_q_heads_override
            .unwrap_or(config.num_attention_heads)
            * hd;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = self
            .num_kv_heads_override
            .unwrap_or(config.num_key_value_heads)
            * hd;
        let o = self.o_nvfp4_t.is_some().then_some(&self.attn.o_proj);
        let sources = [
            (
                self.q_weight.as_ref().and_then(|w| w.as_nvfp4()),
                q_proj_dim,
                h,
            ),
            (self.k_weight.as_ref().and_then(|w| w.as_nvfp4()), kv_dim, h),
            (self.v_weight.as_ref().and_then(|w| w.as_nvfp4()), kv_dim, h),
            (o, h, q_dim),
        ];
        for (slot, (w, n, k)) in sources.into_iter().enumerate() {
            let Some(w) = w else { continue };
            let buf = gpu.alloc(n * k * 2)?;
            crate::layers::ops::dequant_nvfp4_to_bf16(
                gpu,
                k_dq,
                w.weight,
                w.weight_scale,
                buf,
                w.weight_scale_2,
                n as u32,
                k as u32,
                stream,
            )?;
            self.rowinv_bf16[slot] = Some(buf);
        }
        gpu.synchronize(stream)?;
        Ok(())
    }
}
