// SPDX-License-Identifier: AGPL-3.0-only

//! The 4-row (K=4 verify) NVFP4 Q/K/V projections under the qwen4_exp exact
//! verify (`ATLAS_QWEN4EXP_EXACT_VERIFY=1`, `model/qwen4exp_exact_verify.rs`),
//! and every width from 4 rows up under the exact batching lane
//! (`ATLAS_QWEN4EXP_BATCH_FAST=1`, `model/qwen4exp_batch_fast.rs`): there the
//! Q+gate rows run `w4a16_gemv_qg_batch4/3/2` (or `w4a16_gemv_qg`) in chunks
//! of at most 4 rows, and K/V the scalar `w4a16_gemv_batchN` tiers in chunks
//! of at most 8.
//!
//! The default 4-row arm (`ms_qkv_batchn`) is the batched template GEMV for
//! all three projections. Its rows equal `w4a16_gemv`, but serial decode
//! projects Q+gate with `w4a16_gemv_qg`, a different accumulation (one
//! accumulator, scale folded into the weight), and K/V with
//! `w4a16_gemv_dual`. So row `i` here runs serial decode's arithmetic:
//!
//! - Q+gate: `w4a16_gemv_qg_batch4` (row `r` byte-identical to
//!   `w4a16_gemv_qg`), else `w4a16_gemv_qg` per row;
//! - K and V: one scalar `w4a16_gemv_batch4` pass each (rows byte-identical
//!   to `w4a16_gemv`, i.e. to `w4a16_gemv_dual` per projection), else
//!   `w4a16_gemv_dual` per row.
//!
//! Parity of every pair: `scripts/dev/qwen4exp_exact_verify_bench.cu` (M=4).

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// Run the exact projections when they apply (`Ok(true)`), else leave
    /// the step to the default dispatch (`Ok(false)`).
    pub(super) fn ms_qkv_exact4(&self, c: &MultiSeqCtx<'_>) -> Result<bool> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let (Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4)) = (
            self.q_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) else {
            return Ok(false);
        };
        // 2 and 3 rows: the qg_batch2/3 arms and their exact K/V already are
        // serial decode's arithmetic.
        let lane = fwd.levers.qwen4exp_batch_fast && n >= 4;
        if !(lane || fwd.levers.qwen4exp_exact_verify && n == 4) {
            return Ok(false);
        }
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let row = |i: usize| {
            let q = qkv_buf.offset(i * per_seq_qkv);
            let k = q.offset(q_proj_bytes);
            (q, k, k.offset(kv_bytes))
        };

        // Q+gate. The fused deinterleave is the serial arm only without a q
        // adapter; `ms_qkv_seq_q` covers the adapter case exactly as serial.
        if self.gated && !self.q_lora_active() && self.w4a16_gemv_qg_batch4_k.0 != 0 {
            let q_scratch = fwd.buffers.ssm_qkvz();
            // At most 4 rows a launch: the qg template has no wider tier.
            for first in (0..n).step_by(4) {
                let m = (n - first).min(4);
                let (src, dst) = (
                    normed.offset(first * h * bf16),
                    q_scratch.offset(first * q_proj_bytes),
                );
                match m {
                    4 => ops::w4a16_gemv_qg_batch4(
                        fwd.gpu,
                        self.w4a16_gemv_qg_batch4_k,
                        src,
                        q_nvfp4,
                        dst,
                        q_proj_dim,
                        h as u32,
                        nq,
                        hd,
                        stream,
                    )?,
                    3 => ops::w4a16_gemv_qg_batch3(
                        fwd.gpu,
                        self.w4a16_gemv_qg_batch3_k,
                        src,
                        q_nvfp4,
                        dst,
                        q_proj_dim,
                        h as u32,
                        nq,
                        hd,
                        stream,
                    )?,
                    2 => ops::w4a16_gemv_qg_batch2(
                        fwd.gpu,
                        self.w4a16_gemv_qg_batch2_k,
                        src,
                        q_nvfp4,
                        dst,
                        q_proj_dim,
                        h as u32,
                        nq,
                        hd,
                        stream,
                    )?,
                    _ => ops::w4a16_gemv_qg(
                        fwd.gpu,
                        self.w4a16_gemv_qg_k,
                        src,
                        q_nvfp4,
                        dst,
                        q_proj_dim,
                        h as u32,
                        nq,
                        hd,
                        stream,
                    )?,
                }
            }
            for i in 0..n {
                fwd.gpu.copy_d2d_async(
                    q_scratch.offset(i * q_proj_bytes),
                    row(i).0,
                    q_proj_bytes,
                    stream,
                )?;
            }
        } else {
            for i in 0..n {
                let normed_i = normed.offset(i * h * bf16);
                self.ms_qkv_seq_q(
                    fwd,
                    normed_i,
                    row(i).0,
                    q_proj_dim,
                    q_dim,
                    nq,
                    hd,
                    h,
                    stream,
                )?;
            }
        }

        // K and V.
        let batch4 = self.w4a16_batchm.scalar_kernel(4);
        if batch4.0 != 0 {
            let k_scratch = fwd.buffers.attn_output();
            let v_scratch = k_scratch.offset(n * kv_bytes);
            // The widest scalar tier the build carries (8, else 4) a launch.
            let chunk = if self.w4a16_batchm.scalar_kernel(8).0 != 0 {
                8
            } else {
                4
            };
            for (w, out) in [(k_nvfp4, k_scratch), (v_nvfp4, v_scratch)] {
                for first in (0..n).step_by(chunk) {
                    let m = (n - first).min(chunk) as u32;
                    ops::w4a16_gemv_batchm(
                        fwd.gpu,
                        self.w4a16_batchm.scalar_kernel(m),
                        normed.offset(first * h * bf16),
                        w,
                        out.offset(first * kv_bytes),
                        m,
                        kv_dim,
                        h as u32,
                        stream,
                    )?;
                }
            }
            for i in 0..n {
                let (_, k_out, v_out) = row(i);
                fwd.gpu
                    .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out, kv_bytes, stream)?;
                fwd.gpu
                    .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out, kv_bytes, stream)?;
            }
        } else {
            for i in 0..n {
                let (_, k_out, v_out) = row(i);
                let normed_i = normed.offset(i * h * bf16);
                self.ms_qkv_seq_kv(fwd, normed_i, k_out, v_out, nkv, hd, h, stream)?;
            }
        }
        Ok(true)
    }
}
