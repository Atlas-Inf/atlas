// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

#[path = "quant_dispatch/w4a16_gemv_fused.rs"]
mod w4a16_gemv_fused;
pub use w4a16_gemv_fused::*;

/// Unified GEMV dispatch: select kernel based on weight quantization format.
///
/// Eliminates cascading if/else chains in layer forward methods. The enum
/// branch (~1 cycle) is negligible vs GPU kernel launch overhead (~5μs).
#[allow(clippy::too_many_arguments)]
pub fn quant_gemv(
    gpu: &dyn GpuBackend,
    gemv_nvfp4: KernelHandle,
    gemv_fp8: KernelHandle,
    gemv_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemv(gpu, gemv_nvfp4, input, w, output, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemv(
            gpu,
            gemv_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemv(gpu, gemv_dense, input, w, output, n, k, stream),
        // PackedQ2 has no companion kernel handle here (its GEMV is
        // `q2_0_gemv_vec`, dispatched at the layer's own sites, not via this
        // generic 3-kernel helper). Bail rather than misdispatch.
        QuantWeight::PackedQ2(_) => anyhow::bail!(
            "quant_gemv: PackedQ2 not routed through the generic dispatcher; use q2_0_gemv_vec"
        ),
        QuantWeight::Exl3(_) => anyhow::bail!(
            "quant_gemv: EXL3 is not routed through the generic dispatcher yet (reconstruct + GEMM lands in M3)"
        ),
    }
}

/// Unified GEMM dispatch: select kernel based on weight quantization format.
///
/// For M>1 prefill projections (Q/K/V/O). Falls back to dense GEMM for BF16.
#[allow(clippy::too_many_arguments)]
pub fn quant_gemm(
    gpu: &dyn GpuBackend,
    gemm_nvfp4: KernelHandle,
    gemm_fp8: KernelHandle,
    gemm_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemm(gpu, gemm_nvfp4, input, w, output, m, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemm(
            gpu,
            gemm_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            m,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemm(gpu, gemm_dense, input, w, output, m, n, k, stream),
        QuantWeight::PackedQ2(_) => anyhow::bail!(
            "quant_gemm: PackedQ2 not routed through the generic dispatcher; \
             use the layer's transient-dequant prefill path"
        ),
        QuantWeight::Exl3(_) => anyhow::bail!(
            "quant_gemm: EXL3 is not routed through the generic dispatcher yet (reconstruct + GEMM lands in M3)"
        ),
    }
}

/// W4A16 GEMV (M=1): C = A @ dequant(B) for single-row activations.
///
/// A: [1, K] BF16, B: NVFP4 packed, C: [1, N] BF16.
/// 4 outputs/block, 64 threads (2 warps) per output. Cross-warp smem reduction.
///
/// Kernel: `w4a16_gemv(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_grid_x(n), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 double-GEMV (M=2): reads weights once, computes 2 outputs.
///
/// A: [2, K] BF16 contiguous, B: NVFP4 packed, C: [2, N] BF16 contiguous.
/// Same weight bandwidth as single GEMV — eliminates GEMM M=2 tile waste.
///
/// Kernel: `w4a16_gemv_batch2(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 triple-GEMV (M=3): reads weights once, computes 3 outputs.
///
/// A: [3, K] BF16 contiguous, B: NVFP4 packed, C: [3, N] BF16 contiguous.
/// For K=3 speculative verification.
///
/// Kernel: `w4a16_gemv_batch3(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 batched GEMV (M<=MAX_M) — the NVFP4 sibling of `w8a16_gemv_batch4/16`.
///
/// Reads the NVFP4 weight matrix ONCE and computes `m` outputs (one per seq),
/// amortizing the weight read across the batch. `kernel` is `w4a16_gemv_batch4`
/// (M<=4), `w4a16_gemv_batch8` (M<=8, chain verify),
/// `w4a16_gemv_batch16` (M<=16), or `w4a16_gemv_batch32` (M<=32).
/// A:`[m,K]` BF16, C:`[m,N]` BF16.
///
/// Kernel: `w4a16_gemv_batch4/8/16/32(A, B_packed, B_scale, scale2, C, M, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // Largest template is w4a16_gemv_batch32 (MAX_M=32). Refuse rather than
    // silently leaving rows above a selected template's bound unwritten.
    anyhow::ensure!(
        (1..=32).contains(&m),
        "w4a16_gemv_batchm: m={m} outside 1..=32"
    );
    // Tensor-core tiers own 16 outputs per 8-warp CTA.
    let tc_rows = crate::layers::w4a16_gemv_tiers::tc_rows(kernel);
    let tc = tc_rows.is_some();
    anyhow::ensure!(
        tc_rows.is_none_or(|rows| m <= rows && k.is_multiple_of(16)),
        "w4a16 tensor-core GEMV: m={m} k={k} exceeds {tc_rows:?} rows"
    );
    let (grid, block) = if tc {
        (div_ceil(n, 16), 256)
    } else {
        (div_ceil(n, 4), 256)
    };
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([block, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Tensor-core `C[m, n] = A[m, k] · W[:, col0..col0+k]ᵀ` for a K-slice of a
/// wider NVFP4 weight: `weight` points at the slice's first column (packed
/// byte and scale group), rows `ld_half` / `ld_groups` bytes apart. `kernel`
/// is a `w4a16_gemv_tc{8,16,32}_ld` tier ([`crate::layers::w4a16_gemv_tiers::tc_ld_kernel`]).
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_tc_ld(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    ld_half: u32,
    ld_groups: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        kernel.0 != 0
            && (1..=32).contains(&m)
            && k.is_multiple_of(16)
            && ld_half >= k / 2
            && ld_groups >= k / 16,
        "w4a16 strided tensor-core GEMV: m={m} k={k} ld={ld_half}/{ld_groups}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(ld_half)
        .arg_u32(ld_groups)
        .launch(stream)
}

/// Exact-M=5 native-NVFP4 Q/K/V projections in one three-plane launch;
/// `touch` swaps in the PDL touch twin (`w4a16_gemv_batch5_qkv_touch`).
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batch5_qkv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    touch: Option<GemvTouch>,
    input: DevicePtr,
    q: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    debug_assert_eq!(m, 5);
    let launch = KernelLaunch::new(gpu, touch.map_or(kernel, |t| t.kernel))
        .grid([div_ceil(n, 4), 1, 3])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(q.weight)
        .arg_ptr(q.weight_scale)
        .arg_f32(q.weight_scale_2)
        .arg_ptr(k_weight.weight)
        .arg_ptr(k_weight.weight_scale)
        .arg_f32(k_weight.weight_scale_2)
        .arg_ptr(v.weight)
        .arg_ptr(v.weight_scale)
        .arg_f32(v.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k);
    match touch.map(|t| t.batch5_qkv_args(n, k)) {
        Some((rows, ctas)) => launch.arg_u32(rows).arg_u32(ctas).launch(stream),
        None => launch.launch(stream),
    }
}

/// Exact-M=5 pair of same-shape native-NVFP4 projections in one launch.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batch5_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    first: &QuantizedWeight,
    second: &QuantizedWeight,
    first_output: DevicePtr,
    second_output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    debug_assert_eq!(m, 5);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(first.weight)
        .arg_ptr(first.weight_scale)
        .arg_f32(first.weight_scale_2)
        .arg_ptr(second.weight)
        .arg_ptr(second.weight_scale)
        .arg_f32(second.weight_scale_2)
        .arg_ptr(first_output)
        .arg_ptr(second_output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Strided-output variant (`w4a16_gemv_batch{4,8}_os`): identical GEMV, but
/// output row `t` lands at `output[t*out_stride + col]` — the K=4..8
/// attention QKV path writes each projection straight into its interleaved
/// `qkv_buf` slice with this, removing the per-row D2D scatter.
/// `out_stride` is in BF16 ELEMENTS.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batchm_os(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    debug_assert!(m <= 8, "w4a16_gemv_batchm_os caps at M=8 (m={m})");
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

// ── Position embeddings ────────────────────────────────────────────
