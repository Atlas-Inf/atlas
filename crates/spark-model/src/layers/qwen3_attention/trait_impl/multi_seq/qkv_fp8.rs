// SPDX-License-Identifier: AGPL-3.0-only

//! Native FP8 q/k/v (and, from `attn/o_proj.rs`, o) over several rows: one
//! weight pass per launch of [`ops::fp8_rows_plan`]. Up to 4 rows that is one
//! `w8a16_gemv_batch4`, whose rows are serial decode's `w8a16_gemv` bytes, so
//! the K=2/3/4 verify is exact (`model/exact_verify.rs`). Past 16 rows the
//! plan chunks by 16 on NVIDIA under `ATLAS_DENSE_NATIVE_FP8_ATTN` (without it,
//! the per-row loop); on HIP it runs 8-row slices past 8 rows.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::Fp8Weight;

/// `ATLAS_HIP_FP8_WIDE_ROWS=0` keeps gfx1151's per-row loop past 8 rows (A/B).
fn hip_wide_rows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_HIP_FP8_WIDE_ROWS").as_deref() != Ok("0"))
}

impl Qwen3AttentionLayer {
    /// The multi-row FP8 launches for `n` rows, `None` where the caller keeps
    /// its per-row `w8a16_gemv` loop.
    pub(in crate::layers::qwen3_attention) fn fp8_rows_plan(
        &self,
        fwd: &ForwardContext<'_>,
        n: usize,
    ) -> Option<Vec<ops::Fp8RowsLaunch>> {
        let kernels = ops::Fp8RowsKernels {
            batch4: self.w8a16_gemv_batch4_k,
            batch8: self.w8a16_gemv_batch8_k,
            batch8_dyn_vl2: self.w8a16_gemv_batch8_dyn_vl2_k,
            batch8_dyn_vl2_n2: self.w8a16_gemv_batch8_dyn_vl2_n2_k,
            batch16: self.w8a16_gemv_batch16_k,
        };
        if cfg!(atlas_hip) && n > 8 && hip_wide_rows() {
            // gfx1151: 8-row slices on the batch-8 tiers (each row byte-identical to
            // serial `w8a16_gemv`) instead of one `w8a16_gemv` per row. A C=4 27B
            // DFlash2 verify (36 rows) re-read every q/k/v/o weight 36 times:
            // per-row `w8a16_gemv` was 20% of decode GPU time (rocprof 43627623271).
            let mut plan = Vec::with_capacity(n.div_ceil(8));
            for first in (0..n).step_by(8) {
                let rows = (n - first).min(8);
                let slice = ops::fp8_rows_plan(rows, false, ops::gemv_vl2_enabled(), kernels)?;
                plan.extend(slice.into_iter().map(|mut l| {
                    l.first += first;
                    l
                }));
            }
            return Some(plan);
        }
        let wide = !cfg!(atlas_hip) && fwd.levers.dense_native_fp8_attn;
        ops::fp8_rows_plan(n, wide, ops::gemv_vl2_enabled(), kernels)
    }

    /// `plan`'s launches of `w` (`[n_out, k]`) over `input` (`[rows, k]`)
    /// into `output` (`[rows, n_out]`), rows contiguous.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn fp8_rows(
        &self,
        fwd: &ForwardContext<'_>,
        plan: &[ops::Fp8RowsLaunch],
        input: DevicePtr,
        w: &Fp8Weight,
        output: DevicePtr,
        [n_out, k]: [u32; 2],
        stream: u64,
    ) -> Result<()> {
        let (in_row, out_row) = (k as usize * 2, n_out as usize * 2);
        for l in plan {
            ops::w8a16_gemv_batch4_div(
                fwd.gpu,
                l.kernel,
                input.offset(l.first * in_row),
                w.weight,
                w.row_scale,
                output.offset(l.first * out_row),
                l.rows as u32,
                n_out,
                k,
                l.div,
                stream,
            )?;
        }
        Ok(())
    }

    /// Whether q/k/v are native FP8 and [`Self::fp8_rows_plan`] serves `n`
    /// rows; then [`Self::ms_qkv_batchm_fp8`] projects them.
    pub(super) fn ms_qkv_fp8_plan(
        &self,
        fwd: &ForwardContext<'_>,
        n: usize,
    ) -> Option<Vec<ops::Fp8RowsLaunch>> {
        let fp8 = |w: &Option<crate::weight_map::QuantWeight>| {
            w.as_ref().and_then(|w| w.as_fp8()).is_some()
        };
        if n < 2 || !(fp8(&self.q_weight) && fp8(&self.k_weight) && fp8(&self.v_weight)) {
            return None;
        }
        self.fp8_rows_plan(fwd, n)
    }

    pub(super) fn ms_qkv_batchm_fp8(
        &self,
        c: &MultiSeqCtx<'_>,
        plan: &[ops::Fp8RowsLaunch],
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q = self.q_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
        let k = self.k_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
        let v = self.v_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
        let q_scratch = fwd.buffers.ssm_qkvz();
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);

        self.fp8_rows(
            fwd,
            plan,
            normed,
            q,
            q_scratch,
            [q_proj_dim, h as u32],
            stream,
        )?;
        self.fp8_rows(fwd, plan, normed, k, k_scratch, [kv_dim, h as u32], stream)?;
        self.fp8_rows(fwd, plan, normed, v, v_scratch, [kv_dim, h as u32], stream)?;

        let q_lora_active = self
            .lora
            .as_ref()
            .and_then(|weights| weights.q.as_ref())
            .is_some();
        if self.gated && !q_lora_active {
            ops::deinterleave_qg(
                fwd.gpu,
                self.deinterleave_qg_k,
                q_scratch,
                n as u32,
                nq,
                hd,
                q_proj_dim,
                stream,
            )?;
        }

        for row in 0..n {
            let q_out = qkv_buf.offset(row * per_seq_qkv);
            let k_out = q_out.offset(q_proj_bytes);
            let v_out = k_out.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(row * q_proj_bytes),
                q_out,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(row * kv_bytes), k_out, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(row * kv_bytes), v_out, kv_bytes, stream)?;
        }
        Ok(())
    }
}
