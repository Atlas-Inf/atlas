// SPDX-License-Identifier: AGPL-3.0-only
//!
//! Ported from Enntity/atlas (AGPL-3.0-only, sparkqwen drafter series).

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
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::model::qwen4exp_step_copies::copy_rows;

impl Qwen3AttentionLayer {
    /// `Ok(false)`: not a middle mHC layer with plain attention (no MLA,
    /// indexer or adapter), or no metadata — the caller decodes in full.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_rows_kv_only_inner(
        &self,
        rows: crate::layer::KvRows<'_>,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
    ) -> Result<bool> {
        let crate::layer::KvRows {
            hidden,
            residual,
            n,
            seq_lens,
            stream,
        } = rows;
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
        // K and V only where the exact lane's K/V arm serves: a KV row never
        // reads Q, the bulk of the projection's weight bytes.
        if self.ms_kv_proj_only(&c)? {
            self.ms_qkv_norms(&c)?;
        } else {
            self.ms_phase_qkv(&c)?;
        }
        self.ms_phase_rope(&c, meta)?;
        self.ms_phase_cache_write(&c, kv_cache, meta)?;
        Ok(true)
    }

    /// The K and V projections of every row into their `[Q|K|V]` slots, as
    /// the exact lane computes them (`qkv_exact4.rs`); the Q slots keep
    /// whatever they held. `Ok(false)`: not NVFP4, or no scalar row tiers.
    fn ms_kv_proj_only(&self, c: &MultiSeqCtx<'_>) -> Result<bool> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nkv,
            hd,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let (Some(k), Some(v)) = (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) else {
            return Ok(false);
        };
        if self.w4a16_batchm.scalar_kernel(4).0 == 0 {
            return Ok(false);
        }
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        let k_out = qkv_buf.offset(q_proj_bytes);
        for (w, out, dst) in [
            (k, k_scratch, k_out),
            (v, v_scratch, k_out.offset(kv_bytes)),
        ] {
            self.wide_rows.w4a16_rows(
                fwd.gpu,
                &self.w4a16_batchm,
                normed,
                w,
                out,
                (n as u32, kv_dim, h as u32),
                stream,
            )?;
            copy_rows(
                fwd.gpu,
                out,
                kv_bytes,
                dst,
                per_seq_qkv,
                kv_bytes,
                n,
                true,
                stream,
            )?;
        }
        Ok(true)
    }
}
