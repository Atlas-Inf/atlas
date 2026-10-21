// SPDX-License-Identifier: AGPL-3.0-only
//! MXFP8 operands for the cuBLASLt W8A8 prefill GEMM
//! ([`spark_runtime::cublaslt::fp8_gemm_act_weight_t_mx`]), from
//! `kernels/gb10/common/mxfp8_act_quant.cu` (module `mxfp8_act_quant`):
//!
//! * [`mxfp8_quant_act`] (`mxfp8_quant_act_bf16`): BF16 activations -> E4M3
//!   and UE8M0 scales in cuBLASLt's tiled layout, ModelOpt's rounding
//!   (`mxfp8_act_quant.cuh` documents the rule and cites its source);
//! * [`mxfp8_lt_tile_scales`] (`mxfp8_lt_scale_tile`): the checkpoint's
//!   row-major `weight_scale` `[n, k/32]` -> the tiled layout, once at load.
//!
//! A producer kernel can emit the same bytes itself through the header's
//! device functions (`hc_up_mix_bf16_nt_mx` does).
//!
//! Ported from Enntity/atlas (branch `sparkqwen` @ 92dd543, AGPL-3.0-only),
//! `crates/spark-model/src/layers/ops/mxfp8_act.rs`, with Atlas-Inf/atlas#198's integration changes.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::MXFP8_BLOCK;

/// Kernel module of [`mxfp8_quant_act`] / [`mxfp8_lt_tile_scales`].
pub const MXFP8_ACT_MODULE: &str = "mxfp8_act_quant";
/// Rows of one tile of the tiled scale layout.
pub const MX_LT_TILE_ROWS: usize = 128;

/// Scale columns per row in the tiled layout: `k / 32` rounded up to 4.
pub fn mx_lt_inner_pad(k: usize) -> usize {
    (k / MXFP8_BLOCK).div_ceil(4) * 4
}

/// Bytes of the tiled scales of a `[rows, k]` operand.
pub fn mx_lt_scale_bytes(rows: usize, k: usize) -> usize {
    rows.div_ceil(MX_LT_TILE_ROWS) * MX_LT_TILE_ROWS * mx_lt_inner_pad(k)
}

/// Byte offset of scale `(outer, inner)` in the tiled layout (the host twin
/// of `mx_sf_offset`): 128 x 4 tiles of 512 bytes, row-major over
/// `(outer / 128, inner_pad / 4)`. Tests only: the device code has its own.
#[cfg(test)]
pub fn mx_lt_sf_offset(outer: usize, inner: usize, inner_pad: usize) -> usize {
    let tile = (outer / 128) * (inner_pad / 4) + inner / 4;
    tile * 512 + (outer % 32) * 16 + ((outer % 128) / 32) * 4 + inner % 4
}

/// Quantize BF16 `x` (`m` rows of `k`, row stride `lda` elements) into E4M3
/// `q` (`[m, k]`, contiguous) and tiled UE8M0 `sf`
/// ([`mx_lt_scale_bytes`]`(m, k)` bytes; the padding columns of the `m`
/// rows are written 0, rows past `m` are left as they are). `kernel` is
/// `mxfp8_quant_act_bf16`.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_quant_act(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    lda: u32,
    q: DevicePtr,
    sf: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        m > 0
            && (k as usize).is_multiple_of(MXFP8_BLOCK)
            && k > 0
            && lda >= k
            && lda.is_multiple_of(8),
        "mxfp8_quant_act: m={m} k={k} lda={lda} (need K % 32 == 0, lda >= K, lda % 8 == 0)"
    );
    let threads = u64::from(m) * u64::from(k / 8);
    ensure!(
        threads <= u64::from(u32::MAX),
        "mxfp8_quant_act: {m} x {k} is too large for one launch"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(threads as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(q)
        .arg_ptr(sf)
        .arg_u32(m)
        .arg_u32(k)
        .arg_u32(lda)
        .arg_u32(mx_lt_inner_pad(k as usize) as u32)
        .launch(stream)
}

/// Relayout row-major UE8M0 scales (`rows` rows of `k / 32`, row stride
/// `src_ld` bytes: a TP rank's K slice is `src + k0 / 32` with the full
/// width's stride) into the tiled layout at `dst`
/// ([`mx_lt_scale_bytes`]`(rows, k)` bytes, every byte written, padding 0).
/// `kernel` is `mxfp8_lt_scale_tile`.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_lt_tile_scales(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    src_ld: u32,
    dst: DevicePtr,
    rows: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let inner = k as usize / MXFP8_BLOCK;
    ensure!(
        rows > 0 && (k as usize).is_multiple_of(MXFP8_BLOCK) && k > 0 && src_ld as usize >= inner,
        "mxfp8_lt_tile_scales: rows={rows} k={k} src_ld={src_ld}"
    );
    let total = mx_lt_scale_bytes(rows as usize, k as usize);
    ensure!(
        total <= u32::MAX as usize,
        "mxfp8_lt_tile_scales: {rows} x {k} is too large for one launch"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(rows)
        .arg_u32(inner as u32)
        .arg_u32(src_ld)
        .arg_u32(mx_lt_inner_pad(k as usize) as u32)
        .launch(stream)
}

#[cfg(test)]
#[path = "mxfp8_act_tests.rs"]
pub(crate) mod tests;

#[cfg(test)]
#[path = "mxfp8_act_test_gpu.rs"]
pub(crate) mod gpu_tests;
