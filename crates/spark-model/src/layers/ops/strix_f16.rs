// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel picks for the gfx1151 F16-operand prefill GEMM twins
//! (`*_f16a` in `kernels/strix-hip/qwen3.6-{27b,35b-a3b}/nvfp4/`).
//!
//! Each twin has the SAME signature, grid and launcher as the BF16 kernel it
//! replaces, so the pick is a handle swap and every caller keeps its launch
//! code. Why the twins exist: gfx1151 has no f32->bf16 instruction, and the
//! BF16 kernels spend most of their hot loop rounding dequantized weights to
//! bf16 in software (829 instructions per 32-K step for 32 WMMA in
//! `w4a16_gemm_t_m128`; 688 for 16 in the routed MoE kernels). The twins
//! dequantize NVFP4 to f16 exactly (no rounding) and convert the bf16 A tile in
//! LDS (378 and 293 instructions). See the kernel headers.
//!
//! The twin lookup is issued only under `cfg!(atlas_hip)`: the kernels exist
//! only in `kernels/strix-hip/`, and a lookup that is never made leaves no
//! failed row in the boot audit (the #101 / #127 class).

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use crate::layers::try_kernel;

fn f16_twin(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    if cfg!(atlas_hip) {
        try_kernel(gpu, module, &format!("{func}_f16a"))
    } else {
        KernelHandle(0)
    }
}

/// `w4a16_gemm_t_m128`, or its F16 twin on gfx1151. Optional (0 if absent).
#[track_caller]
pub fn w4a16_m128_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    let h = f16_twin(gpu, "w4a16", "w4a16_gemm_t_m128");
    if h.0 != 0 {
        return h;
    }
    try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128")
}

/// [`w4a16_m128_kernel`] for callers that require the kernel.
#[track_caller]
pub fn w4a16_m128_kernel_required(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    let h = f16_twin(gpu, "w4a16", "w4a16_gemm_t_m128");
    if h.0 != 0 {
        return Ok(h);
    }
    gpu.kernel("w4a16", "w4a16_gemm_t_m128")
}

/// A routed-expert MoE prefill kernel (`moe_w4a16_fused_gate_up_t`,
/// `moe_w4a16_grouped_gemm_ptrtable_t`), or its F16 twin on gfx1151.
#[track_caller]
pub fn moe_w4a16_t_kernel(gpu: &dyn GpuBackend, func: &str) -> Result<KernelHandle> {
    let h = f16_twin(gpu, "moe_w4a16", func);
    if h.0 != 0 {
        return Ok(h);
    }
    gpu.kernel("moe_w4a16", func)
}

/// `w8a16_gemm_n_m128` (FP8 weights, 128x128 block scales), or its F16 twin on
/// gfx1151. The kernel exists only in `kernels/strix-hip/common`, so off HIP no
/// lookup is issued at all (0, like the gated lookup this replaces).
#[track_caller]
pub fn w8a16_n_m128_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if !cfg!(atlas_hip) {
        return KernelHandle(0);
    }
    let h = f16_twin(gpu, "w8a16_gemm_n_m128", "w8a16_gemm_n_m128");
    if h.0 != 0 {
        return h;
    }
    try_kernel(gpu, "w8a16_gemm_n_m128", "w8a16_gemm_n_m128")
}
