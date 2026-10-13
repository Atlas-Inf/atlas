// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Glm5KdaLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};
#[path = "projection_fp8.rs"]
mod fp8;

/// Balanced `(first_row, rows)` chunks of at most the batch-M GEMV cap, so an
/// owner-batched verify of `m > 8` rows reuses the exact short-batch kernels.
fn verify_row_chunks(m: u32) -> impl Iterator<Item = (u32, u32)> {
    // `ATLAS_GLM_LONG_BATCH_PROJ_ROWS=3` pins owner-exact 3-row chunks.
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| {
        std::env::var("ATLAS_GLM_LONG_BATCH_PROJ_ROWS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| (2..=ops::DENSE_GEMV_BATCHM_MAX_M).contains(v))
            .unwrap_or(ops::DENSE_GEMV_BATCHM_MAX_M)
    });
    let chunks = m.div_ceil(cap);
    (0..chunks).map(move |c| {
        let start = c * m / chunks;
        (start, (c + 1) * m / chunks - start)
    })
}

/// Memory-tight GLM KDA projection.
///
/// The checkpoint-native BF16 matrix is quantized once at load time and then
/// released. Decode uses W4A16 GEMV; prefill uses W4A16 GEMM.
pub struct Glm5Projection {
    pub nvfp4: QuantizedWeight,
    /// Optional `[K/2, N]` twin for Atlas' tensor-core prefill GEMMs.
    pub prefill_nvfp4_t: Option<QuantizedWeight>,
}

pub struct Glm5KdaWeights {
    pub q_proj: Glm5Projection,
    pub k_proj: Glm5Projection,
    pub v_proj: Glm5Projection,
    pub b_proj: DenseWeight,
    pub f_a_proj: DenseWeight,
    pub f_b_proj: DenseWeight,
    pub g_a_proj: DenseWeight,
    pub g_b_proj: DenseWeight,
    pub conv: DenseWeight,
    pub a_log: DenseWeight,
    pub dt_bias: DenseWeight,
    pub o_norm: DenseWeight,
    pub o_proj: Glm5Projection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProjectionPath {
    PrefillTransposedM128,
    PrefillBase,
    DecodeGemv,
}

impl ProjectionPath {
    pub(super) fn for_forward(decode: bool, has_transposed: bool, tokens: u32) -> Self {
        if decode {
            Self::DecodeGemv
        } else if has_transposed && tokens > 128 {
            Self::PrefillTransposedM128
        } else {
            Self::PrefillBase
        }
    }
}

/// Kernel family for one prefill/decode dense (BF16) KDA projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DensePath {
    Gemv,
    /// `ops::bf16_gemm` (CUTLASS 128x128x64): f_b / g_b, K = 128.
    Cutlass,
    /// `dense_gemm_bf16_pipelined` (or its scalar fallback).
    Pipelined,
}

impl DensePath {
    /// `cutlass_rows` is [`ops::bf16_gemm_cutlass_rows`] for `m`. CUTLASS only
    /// replaces the pipelined kernel where the two are bit-identical.
    pub(super) fn for_forward(m: u32, k: u32, cutlass_rows: bool) -> Self {
        if m == 1 {
            Self::Gemv
        } else if cutlass_rows && k <= 128 && ops::bf16_gemm_matches_pipelined(k) {
            Self::Cutlass
        } else {
            Self::Pipelined
        }
    }
}

impl Glm5KdaLayer {
    /// Project a short speculative-verification batch with the decode kernels
    /// that amortize one weight read across the candidate rows.  Atlas only
    /// ships tuned NVFP4 variants for M=2..=8. Widths 4..=8 use the shared
    /// exact-M tier resolver, so K=5 does not fall off the decode path into a
    /// small prefill GEMM.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_hot_verify(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if (2..=8).contains(&m) {
            self.project_hot_multi_decode(input, weight, output, m, n, k, ctx, stream)
        } else if m > 8
            && let tc = crate::layers::w4a16_gemv_tiers::tc_kernel(m)
            && tc.0 != 0
        {
            // One tensor-core pass reads the weight once for all owner rows.
            ops::w4a16_verify_touch(ctx.gpu, tc, input, &weight.nvfp4, output, m, n, k, stream)
                .unwrap_or_else(|| {
                    ops::w4a16_gemv_batchm(
                        ctx.gpu,
                        tc,
                        input,
                        &weight.nvfp4,
                        output,
                        m,
                        n,
                        k,
                        stream,
                    )
                })
        } else if m > 8 {
            // Owner-batched verify: the batch-M GEMV tiers stream the weight
            // at near-peak bandwidth where the M64-tile prefill GEMM mostly
            // pads. One weight read per balanced <=8-row chunk.
            for (row, rows) in verify_row_chunks(m) {
                self.project_hot_multi_decode(
                    input.offset(row as usize * k as usize * 2),
                    weight,
                    output.offset(row as usize * n as usize * 2),
                    rows,
                    n,
                    k,
                    ctx,
                    stream,
                )?;
            }
            Ok(())
        } else {
            self.project_hot(input, weight, output, m, n, k, false, ctx, stream)
        }
    }

    /// BF16 side projections already have a general batch-M GEMV kernel, so
    /// speculative verification need not fall through to a prefill GEMM.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_dense_verify(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if m == 1 {
            self.project_dense(input, weight, output, m, n, k, ctx, stream)
        } else if m <= ops::DENSE_GEMV_BATCHM_MAX_M {
            self.project_dense_multi_decode(input, weight, output, m, n, k, ctx, stream)
        } else if let tc = ops::dense_tc_kernel(ctx.gpu, m)
            && tc.0 != 0
            && k.is_multiple_of(8)
        {
            ops::dense_gemv_bf16_tc(ctx.gpu, tc, input, weight, output, m, n, k, n, stream)
        } else {
            // Owner-batched verify rows exceed the batch-M kernel's cap. These
            // side projections are small; read them once per <=8-row chunk.
            for (row, rows) in verify_row_chunks(m) {
                self.project_dense_multi_decode(
                    input.offset(row as usize * k as usize * 2),
                    weight,
                    output.offset(row as usize * n as usize * 2),
                    rows,
                    n,
                    k,
                    ctx,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    /// Read one NVFP4 projection once for a short decode/verify batch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_hot_multi_decode(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // ATLAS_GLM_DECODE_GEMV_BATCH: the same bodies behind a weight touch.
        let tier = self.w4a16_gemv_batchm.kernel(m);
        let nvfp4 = &weight.nvfp4;
        if let Some(done) =
            ops::w4a16_verify_touch(ctx.gpu, tier, input, nvfp4, output, m, n, k, stream)
        {
            return done;
        }
        match m {
            2 => ops::w4a16_gemv_batch2(
                ctx.gpu,
                self.w4a16_gemv_batch2_k,
                input,
                &weight.nvfp4,
                output,
                n,
                k,
                stream,
            ),
            3 => ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3_k,
                input,
                &weight.nvfp4,
                output,
                n,
                k,
                stream,
            ),
            4..=8 => {
                let kernel = self.w4a16_gemv_batchm.kernel(m);
                anyhow::ensure!(
                    kernel.0 != 0,
                    "GLM KDA M={m} decode requires a matching w4a16 batch-M tier"
                );
                ops::w4a16_gemv_batchm(
                    ctx.gpu,
                    kernel,
                    input,
                    &weight.nvfp4,
                    output,
                    m,
                    n,
                    k,
                    stream,
                )
            }
            _ => anyhow::bail!("GLM KDA multi-decode projection requires M=2..=8, got {m}"),
        }
    }

    /// Read one BF16 projection once for all concurrent decode rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_dense_multi_decode(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let kernel = if m == 5
            && self.dense_gemv_batch5_k.0 != 0
            && std::env::var("ATLAS_GLM_K5_DENSE_EXACT").as_deref() == Ok("1")
        {
            self.dense_gemv_batch5_k
        } else {
            self.dense_gemv_batchm_k
        };
        ops::dense_gemv_batchm(ctx.gpu, kernel, input, weight, output, m, n, k, n, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_hot(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        decode: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.project_hot_cast(input, weight, output, m, n, k, decode, true, ctx, stream)
            .map(|_| ())
    }

    /// `project_hot`, returning whether it took the Lt FP8 route. `cast` false
    /// reuses the E4M3 input the previous Lt FP8 projection of this same
    /// `input` and `m` left in the scratch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_hot_cast(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        decode: bool,
        cast: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !decode
            && m >= 2048
            && fp8::try_project(
                ctx.gpu,
                input,
                weight,
                output,
                m,
                n,
                k,
                decode,
                ctx.graph_capture,
                ctx.buffers.expert_gate_out(),
                ctx.buffers.sizes().expert_gate_out,
                ctx.buffers.max_batch_tokens(),
                std::env::var("ATLAS_GLM_KDA_PREFILL_LT_FP8").as_deref() == Ok("1"),
                cast,
                stream,
            )?
        {
            return Ok(true);
        }
        let () = match ProjectionPath::for_forward(decode, weight.prefill_nvfp4_t.is_some(), m) {
            ProjectionPath::DecodeGemv => ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                input,
                &weight.nvfp4,
                output,
                n,
                k,
                stream,
            ),
            ProjectionPath::PrefillTransposedM128 => ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128_k,
                input,
                weight.prefill_nvfp4_t.as_ref().unwrap(),
                output,
                m,
                n,
                k,
                stream,
            ),
            ProjectionPath::PrefillBase => ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                input,
                &weight.nvfp4,
                output,
                m,
                n,
                k,
                stream,
            ),
        }?;
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_dense(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match DensePath::for_forward(m, k, ops::bf16_gemm_cutlass_rows(m)) {
            DensePath::Gemv => ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                input,
                weight,
                output,
                n,
                k,
                stream,
            ),
            // ~10% faster than the pipelined kernel on this store-bound shape.
            DensePath::Cutlass => ops::bf16_gemm(input, weight.weight.0, output, m, n, k, stream),
            DensePath::Pipelined => ops::dense_gemm_prefill(
                ctx.gpu,
                self.dense_gemm_k,
                self.dense_gemm_pipelined_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm_kda_selects_nvfp4_gemv_for_decode_and_gemm_for_prefill() {
        assert_eq!(
            ProjectionPath::for_forward(true, true, 1000),
            ProjectionPath::DecodeGemv
        );
        assert_eq!(
            ProjectionPath::for_forward(false, true, 1000),
            ProjectionPath::PrefillTransposedM128
        );
        assert_eq!(
            ProjectionPath::for_forward(false, true, 128),
            ProjectionPath::PrefillBase
        );
        assert_eq!(
            ProjectionPath::for_forward(false, false, 1000),
            ProjectionPath::PrefillBase
        );
    }

    #[test]
    fn glm_kda_dense_takes_cutlass_only_for_short_k_in_whole_k16_steps() {
        // f_b / g_b (K = dim = 128) on CUTLASS rows.
        assert_eq!(DensePath::for_forward(8196, 128, true), DensePath::Cutlass);
        assert_eq!(DensePath::for_forward(300, 64, true), DensePath::Cutlass);
        // CUTLASS off or short rows (`bf16_gemm_cutlass_rows` false).
        assert_eq!(
            DensePath::for_forward(8196, 128, false),
            DensePath::Pipelined
        );
        // beta / f_a / g_a (K = hidden) stay on the pipelined kernel.
        assert_eq!(
            DensePath::for_forward(8196, 4096, true),
            DensePath::Pipelined
        );
        // A K residue groups differently in CUTLASS: not bit-identical.
        for k in [40, 104, 120] {
            assert_eq!(DensePath::for_forward(8196, k, true), DensePath::Pipelined);
        }
        assert_eq!(DensePath::for_forward(1, 128, true), DensePath::Gemv);
    }
}

#[cfg(test)]
mod verify_row_chunk_tests {
    use super::verify_row_chunks;

    #[test]
    fn verify_row_chunks_cover_rows_within_the_kernel_cap() {
        for m in 9..=96u32 {
            let chunks: Vec<_> = verify_row_chunks(m).collect();
            let mut next = 0;
            for &(start, rows) in &chunks {
                assert_eq!(start, next, "m={m} chunks must be contiguous");
                assert!((2..=8).contains(&rows), "m={m} chunk of {rows} rows");
                next += rows;
            }
            assert_eq!(next, m, "m={m} chunks must cover every row");
        }
        assert_eq!(verify_row_chunks(12).collect::<Vec<_>>(), [(0, 6), (6, 6)]);
    }
}
