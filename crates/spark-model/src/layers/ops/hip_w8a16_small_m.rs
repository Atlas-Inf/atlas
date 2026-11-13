// SPDX-License-Identifier: AGPL-3.0-only

//! gfx1151 FP8 block-scaled projections at small M (score passes, verify
//! blocks): the bit-identical `w8a16_gemm_n_m32` / `_m64` twins of
//! `w8a16_gemm_n_m128` (same module, `kernels/strix-hip/common`).
//!
//! At M <= 64 `w8a16_gemm_n_m128` launches ceil(N/128) CTAs whose 128-row
//! tiles are mostly padding, and walks K one 32-step per barrier (20-50 GB/s
//! at the 27B FP8 shapes). The twins tile 32 or 64 rows x 64 cols and stage
//! four 32-K steps per barrier. A, the dequantized B and the WMMA order per
//! output are unchanged, so the output is byte-identical (kbench_w8m32: 0
//! differing outputs; M17 1.5-2.2x, M56 1.2-1.7x). Above 64 rows they lose,
//! so the fused kernel keeps M > 64. `ATLAS_W8A16_SMALL_M=0` turns the route
//! off.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// `(m32, m64)` twin handles, `None` when the route is off or a kernel is
/// missing from this build.
fn kernels(gpu: &dyn GpuBackend) -> Option<(KernelHandle, KernelHandle)> {
    static K: OnceLock<Option<(KernelHandle, KernelHandle)>> = OnceLock::new();
    *K.get_or_init(|| {
        if std::env::var("ATLAS_W8A16_SMALL_M").as_deref() == Ok("0") {
            return None;
        }
        let m32 = gpu.kernel("w8a16_gemm_n_m128", "w8a16_gemm_n_m32").ok()?;
        let m64 = gpu.kernel("w8a16_gemm_n_m128", "w8a16_gemm_n_m64").ok()?;
        Some((m32, m64))
    })
}

/// Launch the small-M twin for `m <= 64`. `Ok(false)`: not taken (M > 64,
/// route off, or no twin) and nothing was launched.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_w8a16_n(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<bool> {
    if m == 0 || m > 64 {
        return Ok(false);
    }
    let Some((m32, m64)) = kernels(gpu) else {
        return Ok(false);
    };
    let (kernel, mt) = if m <= 32 { (m32, 32) } else { (m64, 64) };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, mt), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    Ok(true)
}
