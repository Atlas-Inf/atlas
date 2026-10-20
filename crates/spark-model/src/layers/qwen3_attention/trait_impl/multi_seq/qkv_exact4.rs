// SPDX-License-Identifier: AGPL-3.0-only

//! The 4-row (K=4 verify) NVFP4 Q/K/V projections under the qwen4_exp exact
//! verify (`ATLAS_QWEN4EXP_EXACT_VERIFY=1`, `model/qwen4exp_exact_verify.rs`),
//! and every width from 4 rows up under the exact batching lane
//! (`ATLAS_QWEN4EXP_BATCH_FAST=1`, `model/qwen4exp_batch_fast.rs`): there the
//! Q+gate rows run `w4a16_gemv_qg_batch4/3/2` (or `w4a16_gemv_qg`) in chunks
//! of at most 4 rows, past 8 rows `qwen4exp_qg_rows16/32`, and K/V the scalar
//! `w4a16_gemv_batchN` tiers in chunks of at most 8, past 8 rows `batch16`
//! (`ops::Qwen4ExpWideRows`).
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
use crate::model::qwen4exp_step_copies::{copy_rows, qkv_rows_2d};

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
        let rows_2d = qkv_rows_2d();
        let row = |i: usize| {
            let q = qkv_buf.offset(i * per_seq_qkv);
            let k = q.offset(q_proj_bytes);
            (q, k, k.offset(kv_bytes))
        };

        // Q+gate. The fused deinterleave is the serial arm only without a q
        // adapter; `ms_qkv_seq_q` covers the adapter case exactly as serial.
        if self.gated && !self.q_lora_active() && self.w4a16_gemv_qg_batch4_k.0 != 0 {
            let q_scratch = fwd.buffers.ssm_qkvz();
            // The lane's 16/32-row tiers past 8 rows, else at most 4 a launch.
            let mut first = 0usize;
            while first < n {
                let (rows, wide) = self.wide_rows.qg_chunk((n - first) as u32);
                let m = rows as usize;
                let (src, dst) = (
                    normed.offset(first * h * bf16),
                    q_scratch.offset(first * q_proj_bytes),
                );
                first += m;
                if let Some(t) = wide {
                    self.wide_rows.qg_launch(
                        fwd.gpu,
                        t,
                        src,
                        q_nvfp4,
                        dst,
                        (rows, q_proj_dim, h as u32),
                        (nq, hd),
                        stream,
                    )?;
                    continue;
                }
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
            // `ATLAS_QWEN4EXP_QKV_ROWS_2D`: the n row copies as one pitched copy.
            copy_rows(
                fwd.gpu,
                q_scratch,
                q_proj_bytes,
                row(0).0,
                per_seq_qkv,
                q_proj_bytes,
                n,
                rows_2d,
                stream,
            )?;
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
            // The widest scalar tier (batch16 past 8 rows, else 8 or 4) a launch.
            for (w, out) in [(k_nvfp4, k_scratch), (v_nvfp4, v_scratch)] {
                self.wide_rows.w4a16_rows(
                    fwd.gpu,
                    &self.w4a16_batchm,
                    normed,
                    w,
                    out,
                    (n as u32, kv_dim, h as u32),
                    stream,
                )?;
            }
            let (_, k_out, v_out) = row(0);
            if rows_2d {
                // K rows then V rows: disjoint destinations, so the order the
                // per-row loop interleaved them in does not change a byte.
                for (src, dst) in [(k_scratch, k_out), (v_scratch, v_out)] {
                    copy_rows(
                        fwd.gpu,
                        src,
                        kv_bytes,
                        dst,
                        per_seq_qkv,
                        kv_bytes,
                        n,
                        true,
                        stream,
                    )?;
                }
            } else {
                for i in 0..n {
                    let (_, k_out, v_out) = row(i);
                    fwd.gpu.copy_d2d_async(
                        k_scratch.offset(i * kv_bytes),
                        k_out,
                        kv_bytes,
                        stream,
                    )?;
                    fwd.gpu.copy_d2d_async(
                        v_scratch.offset(i * kv_bytes),
                        v_out,
                        kv_bytes,
                        stream,
                    )?;
                }
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
