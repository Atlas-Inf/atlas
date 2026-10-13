// SPDX-License-Identifier: AGPL-3.0-only
//! MXFP8 (E4M3 + E8M0 per 32 values) weights for small-row projections.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Values per E8M0 scale block.
pub const MXFP8_BLOCK: usize = 32;

/// Quantize a BF16 `[n, k]` weight into E4M3 `data` (`n*k` bytes) and E8M0
/// `scales` (`n*k/32` bytes).
pub fn mxfp8_quantize(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    weight: DevicePtr,
    data: DevicePtr,
    scales: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(MXFP8_BLOCK),
        "MXFP8 needs K % 32 == 0 (K={k})"
    );
    let blocks = (n * k / MXFP8_BLOCK) as u64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(blocks as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight)
        .arg_ptr(data)
        .arg_ptr(scales)
        .arg_u64(blocks)
        .launch(stream)
}

/// `C[m, n] = A[m, k] · dequant(W)[n, k]ᵀ` on the tensor cores; `kernel` is
/// `mxfp8_gemv_tc8/16/32` for up to 8/16/32 rows.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    data: DevicePtr,
    scales: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=32).contains(&m) && (k as usize).is_multiple_of(MXFP8_BLOCK),
        "mxfp8_gemv: m={m} k={k} unsupported"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(data)
        .arg_ptr(scales)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// Widest row count [`mxfp8_gemv_grouped`] serves.
pub const MXFP8_GROUPED_MAX_M: u32 = 16;

/// Head-grouped [`mxfp8_gemv`] for per-head weights: for each of `g` heads,
/// `C[:, h*n..][m, n] = A[:, h*k..][m, k] · dequant(W_h)[n, k]ᵀ`, with all
/// heads' `data` / `scales` contiguous (`[g, n, k]` / `[g, n, k/32]`) and
/// token-row strides `a_stride` / `c_stride`. `tiers` holds
/// `mxfp8_gemv_tc{8,16}_grouped`.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_gemv_grouped(
    gpu: &dyn GpuBackend,
    tiers: &[KernelHandle; 2],
    input: DevicePtr,
    data: DevicePtr,
    scales: DevicePtr,
    output: DevicePtr,
    m: u32,
    g: u32,
    k: u32,
    n: u32,
    a_stride: u32,
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=MXFP8_GROUPED_MAX_M).contains(&m) && (k as usize).is_multiple_of(MXFP8_BLOCK),
        "mxfp8_gemv_grouped: m={m} k={k} unsupported"
    );
    // Rows are read as 16-byte vectors and must not overlap across tokens.
    let (gk, gn) = (u64::from(g) * u64::from(k), u64::from(g) * u64::from(n));
    ensure!(
        g > 0
            && u64::from(a_stride) >= gk
            && u64::from(c_stride) >= gn
            && a_stride.is_multiple_of(8),
        "mxfp8_gemv_grouped: g={g} n={n} k={k} a_stride={a_stride} c_stride={c_stride}"
    );
    KernelLaunch::new(gpu, tiers[usize::from(m > 8)])
        .grid([div_ceil(n, 16), 1, g])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(data)
        .arg_ptr(scales)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(a_stride)
        .arg_u32(c_stride)
        .launch(stream)
}

#[cfg(test)]
#[path = "mxfp8_tests.rs"]
mod tests;
