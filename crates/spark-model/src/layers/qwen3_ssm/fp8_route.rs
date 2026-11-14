// SPDX-License-Identifier: AGPL-3.0-only

//! Batched-decode GDN FP8 projections past the GEMV tiers (multi-seq n > 16,
//! per-sequence verify num_tokens > 4) on gfx1151: the block-scaled
//! `w8a16_gemm_n_m128` route (its small-M twins for <= 64 rows) instead of the
//! base `w8a16_gemm`, byte-identical and 2.3-6.2x faster (devlab kbench_w8base
//! 43627626354). At C=4 with the 27B DFlash2 drafter (4 x 9 rows) the base
//! kernel was 35.7% of decode GPU time at ~5 ms per call (rocprof 43616462758).
//! `ATLAS_SSM_BATCHED_NM128=0` keeps the base kernel.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

impl super::Qwen3SsmLayer {
    /// `out[m, n] = input[m, k] @ fp8^T` through the n_m128 route when the weight
    /// is block-scaled and the kernel is linked, else the base `w8a16_gemm`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ssm_fp8_gemm(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        fp8: &Fp8Weight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let nm128 = self.w8a16_gemm_n_m128_k.0 != 0
            && fp8.scale_format == WeightQuantFormat::Fp8BlockScaled
            && *ON.get_or_init(|| std::env::var("ATLAS_SSM_BATCHED_NM128").as_deref() != Ok("0"));
        let (f, kernel) = match nm128 {
            true => (ops::w8a16_gemm_n_m128 as Proj, self.w8a16_gemm_n_m128_k),
            false => (ops::w8a16_gemm as Proj, self.w8a16_gemm_k),
        };
        f(
            ctx.gpu,
            kernel,
            input,
            fp8.weight,
            fp8.row_scale,
            out,
            m,
            n,
            k,
            stream,
        )
    }
}

type Proj = fn(
    &dyn spark_runtime::gpu::GpuBackend,
    spark_runtime::gpu::KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;
