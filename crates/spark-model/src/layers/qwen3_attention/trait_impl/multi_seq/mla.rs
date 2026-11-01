// SPDX-License-Identifier: AGPL-3.0-only

//! MLA (Multi-head Latent Attention) branch of multi-sequence batched
//! decode — the batched analogue of `decode::attention_forward_mla`.
//!
//! GitHub issue #84: the standard `ms_phase_qkv` path unconditionally
//! reads `attn.q_proj` / `q_weight`, which the Mistral MLA loader leaves
//! as a NULL `DevicePtr` stub (the real projections live in `self.mla`).
//! Routing an MLA model through the non-MLA `decode_multi_seq` body
//! launched `dense_gemv` against a NULL pointer → illegal address.
//! Commit 9e68dc2 stopped the crash with an `is_mla_dispatch()` per-seq
//! `decode()` fallback, but that fallback shares one `logits` buffer
//! across the loop and `decode()`'s `zero_all` wipes it — cross-seq
//! contamination. This module is the proper fix.
//!
//! ## Design
//!
//! The MLA decode chain (Q latent → norm → expand → absorbed-Q → Q_rope
//! → K latent → K_rope+RoPE → cache assemble+write → paged decode → V
//! extract → O proj) is run **once per sequence**, each iteration using
//!
//!   * a distinct per-sequence slice of the `normed` input and the
//!     `o_out` output buffer (stride `h` elements), and
//!   * per-sequence attention metadata — `positions[i]` (u32, +4 bytes),
//!     `slot[i]` (i64, +8 bytes), `seq_len[i]` (i32, +4 bytes) and
//!     `block_table` row `i` (`max_blocks_per_seq` i32 entries).
//!
//! Every sequence therefore reads and writes ONLY its own compressed
//! latent-KV history — no cross-contamination. The transient scratch
//! buffers (`ssm_ba`, `ssm_deinterleaved`, `expert_up_out`, …) are
//! reused across iterations: each iteration fully overwrites them before
//! reading, and all work is serialized on a single CUDA stream, so the
//! reuse is sound. Unlike the per-seq `decode()` fallback this stays in
//! ONE forward pass — no `Buffers::zero_all`, no host round-trip — so
//! the assembled `[n, h]` `o_out` is handed straight to `ms_phase_ffn`.
//!
//! The paged-decode attention kernel (`paged_decode_mla_k`) is itself
//! multi-seq capable (`grid[num_q_heads, num_seqs, 1]`); we still invoke
//! it per-sequence here so each sequence's absorbed-Q (built in shared
//! head-strided scratch) is consumed before the next iteration reuses
//! that scratch. N ≤ 8, the chain is GEMV-bound, so the per-seq launch
//! overhead is negligible.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use super::mla_gemv::MlaDims;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// Batched MLA decode for `c.n` sequences. Writes each sequence's
    /// O-projection output into `moe_output[i*h .. (i+1)*h]` and returns
    /// the `moe_output` base pointer for `ms_phase_ffn`.
    ///
    /// `c.normed` already holds the RMS-normed hidden state for all `n`
    /// tokens (phase 1 ran before dispatch).
    pub(super) fn ms_mla_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let mla = self
            .mla
            .as_ref()
            .expect("ms_mla_decode called without MLA config");

        let h = c.h as u32;
        let nq = c.nq;
        let hd = c.hd;
        let eps = c.eps;
        let bf16 = c.bf16;
        let stream = c.stream;
        let bs = c.bs as usize;

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        let mla_cache_dim = kv_lora + mla_rope;
        let q_dim = nq * hd;
        let inv_sqrt_d = self.effective_attn_scale(hd);

        // O-projection output destination. `ms_phase_o_proj` (the non-MLA
        // sibling) returns `moe_output`; match it so `ms_phase_ffn`
        // consumes the same buffer for both paths.
        let o_out = c.fwd.buffers.moe_output();

        // GLM-5's zero-RoPE MLA shape can keep the N=2/3 rows together for
        // its large stateless projections. Cache mutation and attention stay
        // on the proven sequence-private path; other MLA variants retain the
        // fully conservative implementation below.
        let long_verify = self.validate_glm_long_verify(c, kv_cache, meta)?;
        if !long_verify && self.glm_mla_multi_seq_eligible(c, mla)? {
            return self.ms_glm_mla_decode(c, kv_cache, meta, mla, o_out);
        }

        let o_stage = self.glm_k3_o_stage(c, mla, long_verify, o_out)?;
        let query_stage = self.glm_k3_query_stage(c, mla, long_verify)?;
        if let Some(plan) = query_stage {
            plan.project(self, c, mla)?;
        }
        for i in 0..c.n {
            let normed_i = c.normed.offset(i * c.h * bf16);
            // Per-sequence metadata views. The batched metadata packs
            // positions as `[n]` u32, slot as `[n]` i64, seq_len as `[n]`
            // i32 and block_table as `[n * max_blocks_per_seq]` i32 —
            // identical to the layout `ms_phase_rope` / `ms_phase_cache_write`
            // index for the non-MLA path.
            let meta_i = AttnMetadataDev {
                positions: meta.positions.offset(i * 4),
                positions_h: meta.positions_h.offset(i * 4),
                positions_w: meta.positions_w.offset(i * 4),
                slot: meta.slot.offset(i * 8),
                seq_len: meta.seq_len.offset(i * 4),
                block_table: meta
                    .block_table
                    .offset(i * meta.max_blocks_per_seq as usize * 4),
                max_blocks_per_seq: meta.max_blocks_per_seq,
                num_seqs: 1,
                seq_slot: spark_runtime::gpu::DevicePtr(0),
                moe_row_adapter: spark_runtime::gpu::DevicePtr::NULL,
            };
            let o_out_i = o_out.offset(i * c.h * bf16);

            // DeepSeek-V4-Flash (o_lora_rank > 0) uses the DIRECT-KV
            // attention algorithm, NOT the absorbed-MLA chain. Its
            // V3-style absorption weights (`w_uk_t` / `w_uv` / `wkv_b`)
            // are loaded as NULL `DevicePtr` stubs (see
            // `deepseek_v4::load_layers`: `is_v4_flash` branch), so the
            // absorbed `ms_mla_decode_one` here would dereference a NULL
            // weight in the Q-absorb / V-extract GEMVs → CUDA illegal
            // address on the K=2 MTP verify. Drive the single-token
            // V4-Flash decode chain (`attention_forward_v4`, the same one
            // the n=1 path uses) once per verify token instead — SSOT
            // with the correct algorithm and buffer layout.
            if mla.o_lora_rank > 0 {
                // Per-token forward context carrying this token's sliced
                // attention metadata (positions / slot / seq_len /
                // block_table). All other ctx fields are copied verbatim.
                let ctx_i = crate::layer::ForwardContext {
                    ssm_batch: None,
                    attn_metadata: Some(meta_i),
                    midchunk_capture: None,
                    ..*c.fwd
                };
                // Q/K/V projection destinations inside `qkv_output`,
                // matching the single-token `attention_forward` layout:
                // Q `[nq*hd]`, then K `[nkv*hd]`, then V `[nkv*hd]`. V4 is
                // ungated MLA, so `q_proj_dim == q_dim`.
                let qkv = c.fwd.buffers.qkv_output();
                let q_proj_bytes = q_dim as usize * bf16;
                let kv_bytes = (c.nkv * hd) as usize * bf16;
                let k_out = qkv.offset(q_proj_bytes);
                let v_out = k_out.offset(kv_bytes);
                let args = super::super::super::decode::attention_forward_mla::DecodeMlaArgs {
                    normed: normed_i,
                    q_out: qkv,
                    k_out,
                    v_out,
                    q_dim,
                    h,
                    nq,
                    hd,
                    eps,
                    bs,
                    stream,
                    // Batched / MTP-verify path: skip the inc-3 compressed-pool
                    // append (a shared per-layer position counter can't track
                    // interleaved verify tokens) → frozen inc-2 pool here.
                    pos: None,
                };
                let o_v4 = self.attention_forward_v4(kv_cache, &ctx_i, &args)?;
                // `attention_forward_v4` writes its O projection into the
                // shared `qkv_output` buffer and returns it; copy this
                // token's row into its dedicated `o_out` slot before the
                // next iteration reuses `qkv_output`.
                c.fwd
                    .gpu
                    .copy_d2d_async(o_v4, o_out_i, c.h * bf16, stream)?;
                continue;
            }

            self.ms_mla_decode_one(
                c,
                kv_cache,
                &meta_i,
                normed_i,
                o_out_i,
                mla,
                MlaDims {
                    h,
                    nq,
                    hd,
                    q_dim,
                    q_lora,
                    kv_lora,
                    mla_nope,
                    mla_v_dim,
                    mla_rope,
                    mla_cache_dim,
                    eps,
                    bs,
                    inv_sqrt_d,
                    o_lora_rank: mla.o_lora_rank as u32,
                },
                stream,
                c.seq_lens[i],
                o_stage.map(|p| p.row(i)).transpose()?,
                query_stage.map(|p| p.row(i)).transpose()?,
            )?;
            if let Some(plan) = o_stage {
                plan.capture_scalar(c.fwd.gpu, self.dense_gemv_k, &mla.wo, i, stream)?;
            }
        }

        if let Some(plan) = o_stage {
            plan.project(c.fwd.gpu, self.dense_gemv_batchm_k, &mla.wo, stream)?;
            plan.finish_compare(c.fwd.gpu, stream, self.attn_layer_idx, c.fwd.config.tp_rank)?;
        }

        // ATLAS_MLA_HSD: per-seq diagnostic — scans each sequence's full
        // `o_out` row for NaN/Inf and reports magnitude, to localize
        // cross-sequence corruption in the batched MLA decode.
        if std::env::var("ATLAS_MLA_HSD").is_ok_and(|v| v == "1") && self.attn_layer_idx == 0 {
            c.fwd.gpu.synchronize(stream)?;
            for i in 0..c.n {
                let mut row = vec![0u8; c.h * bf16];
                let _ = c.fwd.gpu.copy_d2h(o_out.offset(i * c.h * bf16), &mut row);
                let vals: Vec<f32> = row
                    .chunks_exact(2)
                    .map(|x| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16))
                    .collect();
                let bad = vals.iter().filter(|v| !v.is_finite()).count();
                let absmax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                tracing::info!(
                    "MLA_HSD L0 s{i}: o_out non-finite={bad}/{} absmax={absmax:.4}",
                    vals.len(),
                );
            }
        }
        Ok(o_out)
    }
}

#[path = "mla_glm_verify.rs"]
mod glm_verify;
#[path = "mla_serial.rs"]
mod serial;

#[cfg(test)]
#[path = "mla_long_context_tests.rs"]
mod long_context_tests;

#[path = "mla_k3_o.rs"]
mod k3_o;
#[path = "mla_k3_query.rs"]
mod k3_query;
