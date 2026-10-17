// SPDX-License-Identifier: AGPL-3.0-only

//! KV rows only ([`crate::layer::TransformerLayer::decode_rows_kv_only`]):
//! each row's attention input on the mHC highway — the attention site's
//! collapse and norm, QKV, RoPE — written to the paged cache, and nothing
//! after it. The same phases, in the same order, as the start of
//! `decode_multi_seq_inner_hc`, so a row's K/V bytes are the ones a full
//! multi-row decode of it writes.
//!
//! The qwen4_exp MTP absorb (`qwen4exp_mtp_absorb.rs`) rewrites drafter KV
//! rows this way: the drafter is ONE layer, so its KV row is a function of
//! its own input row, and the attention read, the FFN and the stream update
//! a full decode would add change nothing it keeps.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// `Ok(false)`: not a middle mHC layer with plain attention (no MLA,
    /// indexer or adapter), or no metadata — the caller decodes in full.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_rows_kv_only_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n: usize,
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(hc) = self.hc.as_ref() else {
            return Ok(false);
        };
        let Some(meta) = ctx.attn_metadata else {
            return Ok(false);
        };
        if n == 0
            || self.mla.is_some()
            || self.qsa.is_some()
            || self.lora.is_some()
            || hc.is_first_model_layer
            || ctx.config.model_type == "glm5_next"
        {
            return Ok(false);
        }
        let bs = kv_cache.block_size() as u32;
        let c = MultiSeqCtx::new(self, ctx, hidden, residual, n, seq_lens, bs, stream);
        let eps = ctx.config.rms_norm_eps as f32;
        self.ms_hc_pre_site(
            &hc.attn,
            hc,
            ctx.buffers.hc_streams(),
            c.hidden,
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            n,
            eps,
            ctx,
            stream,
        )?;
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                c.hidden,
                &self.input_norm,
                c.normed,
                n as u32,
                c.h as u32,
                eps,
                stream,
            )?;
        } else {
            ctx.gpu
                .copy_d2d_async(c.hidden, c.normed, n * c.h * 2, stream)?;
        }
        self.ms_phase_qkv(&c)?;
        self.ms_phase_rope(&c, meta)?;
        self.ms_phase_cache_write(&c, kv_cache, meta)?;
        Ok(true)
    }
}
