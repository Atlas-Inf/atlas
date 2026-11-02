// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel pick for `w4a16_gemm_s64`, the bit-identical 64-deep-K-stage twin
//! of the untransposed `w4a16_gemm` (the MoE router GEMM, N=512 K=2560).
//!
//! The twin has the SAME signature, grid `(ceil(N/64), ceil(M/64))` and
//! block 128 as `w4a16_gemm`, so the pick is a handle swap at init and every
//! dispatch site keeps its launch code. Why it exists: the base kernel
//! gathers per-element byte loads across 64 rows and pays two barriers per
//! 16-K step; `_s64` stages 64 K per stage (one barrier pair for 4 WMMA
//! k-steps) with 16-byte A loads and one packed 16-byte B load plus its 2
//! scale bytes per thread. See the kernel header in
//! `kernels/strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu`.
//!
//! The twin lookup is issued only under `cfg!(atlas_hip)`: the kernel exists
//! only in `kernels/strix-hip/`, and a lookup that is never made leaves no
//! failed row in the boot audit (the #101 / #127 class). It ships in every
//! strix-hip `w4a16` module (qwen3.6-27b, qwen3.6-35b-a3b, and the
//! flash-next symlink), so on native HIP it always resolves.

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use crate::layers::try_kernel_gated;

/// `w4a16_gemm`, or its bit-identical `_s64` twin on gfx1151. Required —
/// every target that reaches this resolver ships `w4a16::w4a16_gemm`.
#[track_caller]
pub fn w4a16_gemm_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    let h = try_kernel_gated(cfg!(atlas_hip), gpu, "w4a16", "w4a16_gemm_s64");
    if h.0 != 0 {
        return Ok(h);
    }
    gpu.kernel("w4a16", "w4a16_gemm")
}
