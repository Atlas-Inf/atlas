// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel and scratch resolution added to `from_weights` with the GLM-5.3
//! DFlash2 drafter: the unclamped SwiGLU choice, the tensor-core small-M and
//! MXFP8 GEMV sets, and the batched tail's candidate-selector scratch.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::weight_loader::DflashWeights;

/// The drafter's SwiGLU kernel. `#[track_caller]` so the kernel audit keeps
/// naming the constructor.
#[track_caller]
pub(super) fn silu_mul_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    // The drafter checkpoint declares no SwiGLU limit, but a target's
    // `moe_silu_mul` may be a clamping shadow (GLM-5.3 inherits
    // DeepSeek-V4's gate/up <= 10; the DFlash2 drafter reaches 27/38).
    // Targets that ship the never-shadowed `silu_mul_plain` use it.
    Ok(
        match crate::layers::try_kernel(gpu, "silu_mul_plain", "silu_mul_plain") {
            plain if plain.0 != 0 => plain,
            _ => gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
        },
    )
}

/// Zeroed selector scratch for `gamma` rows, or NULL without a candidate
/// selector.
pub(super) fn batch_selector_scratch(
    gpu: &dyn GpuBackend,
    weights: &DflashWeights,
    gamma: usize,
) -> Result<DevicePtr> {
    // The batched tail's per-sequence selector launches run in stream
    // order, so they share one selector scratch (ticket starts at zero).
    Ok(if weights.candidate_selector.is_some() {
        let bytes = crate::layers::ops::dflash2_selector_scratch_bytes(gamma);
        let p = gpu.alloc(bytes)?;
        gpu.memset(p, 0, bytes)?;
        p
    } else {
        DevicePtr::NULL
    })
}

/// Tensor-core BF16 GEMVs for 9..=16 / 17..=32 rows (0 when absent).
pub(super) fn dense_gemv_tc_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 2] {
    ["dense_gemv_bf16_tc16", "dense_gemv_bf16_tc32"]
        .map(|name| crate::layers::try_kernel(gpu, "dense_gemv_bf16_batchm", name))
}

/// Startup small-M GEMV policy given the tensor-core GEMV set.
pub(super) fn small_m_gemv(dense_gemv_tc: &[KernelHandle; 2]) -> bool {
    super::small_m_gemm::small_m_gemv_enabled(dense_gemv_tc.iter().all(|k| k.0 != 0))
}

/// The 8/16/32-row MXFP8 tensor-core GEMVs (0 when absent).
pub(super) fn mxfp8_gemv_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 3] {
    ["mxfp8_gemv_tc8", "mxfp8_gemv_tc16", "mxfp8_gemv_tc32"]
        .map(|name| crate::layers::try_kernel(gpu, "mxfp8_gemv", name))
}
