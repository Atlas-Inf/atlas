// SPDX-License-Identifier: AGPL-3.0-only

//! Rust launchers for the `exl3_int8` CUDA module (kernels/gb10/qwen3.8-flash-next/
//! nvfp4/exl3_int8.cu): exllamav3's int8-activation "sq" GEMV for EXL3 mul1
//! weights, and the bf16 linear composed from it.
//!
//! Unlike upstream — a cooperative launch with grid.sync — each entry here is
//! one REGULAR launch: block 256, host-chosen grid and dynamic shared memory,
//! no cooperative requirement. Cross-block work happens through `locks`, a
//! zeroed int32 workspace whose counters reset themselves at the end of every
//! call, so one buffer serves any number of back-to-back launches (it must be
//! re-zeroed only if a launch failed midway).
//!
//! Numerics: the activations are quantized to int8 and folded into dp4a
//! (exact int32 accumulation), so this path is NOT bit-comparable with the
//! fp16 `exl3_linear_bf16`; its error against an f64 reference is pinned by
//! the GPU tests instead.
//!
//! The host-side decomposition / shared-memory / workspace math below mirrors
//! upstream's `exl3_gemv_int8_sq` (exllamav3/exllamav3_ext/quant/
//! exl3_gemv_int8.cu) for m <= 2, residual = false, half_k = false.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::exl3::Exl3Weight;

use super::exl3_ops::{Exl3Kernels, exl3_convert};

/// Max k-split of the sq decomposition.
const SQ_KSPLIT_CAP: u32 = 64;
/// Floor for rows_per.
const SQ_MINROWS: u32 = 16;
/// Cap for rows_per (shared-memory budget of the kernel's own accounting).
const SQ_ROWS_MAX: u32 = 512;
/// Counters at the head of the workspace; must stay zero between calls.
const SQ_COUNTERS_CAP: usize = 4096;
/// Counter area plus the per-split {float, int, int, pad} rows upstream reserves.
const SQ_WS_RESERVED: usize = SQ_COUNTERS_CAP + 4 * SQ_KSPLIT_CAP as usize * 4;
/// Upstream's `GEMV_STAGE_D` (exl3_gemv_int8_kernel.cuh): the staging depth of
/// the odd-rate (5-bit) inner loop, in 16-lane rows per split.
const GEMV_STAGE_D: u32 = 4;

/// rows_per cap from the shared-memory budget: `min((80K / (32 + 64*m)) & !7, 512)`.
pub fn sq_rows_max(m: u32) -> u32 {
    (((80 * 1024) / (32 + 64 * m)) & !7).min(SQ_ROWS_MAX)
}

/// The staged 5-bit round-trip buffer: nonzero only for the odd integer rate.
pub fn sq_stage_bytes(bits: u32) -> u32 {
    if bits % 2 == 1 {
        8 * GEMV_STAGE_D * 16 * bits * 4
    } else {
        0
    }
}

/// The launch geometry one `exl3_int8_sq_*` call needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqPlan {
    pub grid: u32,
    pub rows_per: u32,
    pub ksplit: u32,
    pub smem: u32,
    pub ws_ints: usize,
}

/// Upstream's decomposition: split the `k/16` trellis rows into `rows_per`
/// chunks so the `rows_total * nb256` units spread over `grid` blocks with at
/// most `SQ_KSPLIT_CAP` splits, then derive the shared memory and the
/// workspace size from the result.
pub fn sq_plan(k: usize, n: usize, m: u32, bits: u32, grid: u32) -> Result<SqPlan> {
    anyhow::ensure!(
        (1..=2).contains(&m)
            && (4..=6).contains(&bits)
            && k.is_multiple_of(128)
            && n.is_multiple_of(256),
        "EXL3 int8 sq: unsupported shape (k={k}, n={n}, m={m}, bits={bits}) — \
         need m in 1..=2, bits in 4..=6, k % 128 == 0, n % 256 == 0"
    );
    let rows_total = (k / 16) as u32;
    let nb256 = (n / 256) as u32;
    let r = div_ceil(rows_total * nb256, grid);
    let mut rows_per = (r.max((2 * r).min(32)) + 7) & !7;
    rows_per = rows_per.max(SQ_MINROWS);
    rows_per = rows_per.min(sq_rows_max(m));
    rows_per = rows_per.min(div_ceil(rows_total, 8) * 8);
    let ksplit = div_ceil(rows_total, rows_per);
    anyhow::ensure!(
        ksplit <= SQ_KSPLIT_CAP && nb256 <= SQ_COUNTERS_CAP as u32,
        "EXL3 int8 sq: decomposition exceeds the caps (k={k}, n={n}, m={m}, bits={bits}): \
         ksplit={ksplit} (cap {SQ_KSPLIT_CAP}), n/256={nb256} (cap {SQ_COUNTERS_CAP})"
    );
    let smem = rows_per * 16 * 2 + rows_per * 16 * 4 * m + sq_stage_bytes(bits) + 2 * m * 128 * 4;
    Ok(SqPlan {
        grid,
        rows_per,
        ksplit,
        smem,
        ws_ints: SQ_WS_RESERVED + ksplit as usize * m as usize * n,
    })
}

/// `ATLAS_EXL3_SQ_BLOCKS_PER_SM`, clamped to 1..=8, read once — Atlas has no
/// occupancy query, so the blocks-per-SM of upstream's occupancy call is an
/// env knob (upstream's natural allocation is 2-3 resident blocks/SM).
fn sq_blocks_per_sm() -> u32 {
    static ONCE: OnceLock<u32> = OnceLock::new();
    *ONCE.get_or_init(|| {
        std::env::var("ATLAS_EXL3_SQ_BLOCKS_PER_SM")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(2)
            .clamp(1, 8)
    })
}

/// The grid for every sq call on this device: `min(blocks_per_sm * sm_count, 1024)`.
/// The SAME grid serves m = 1 and m = 2.
pub fn sq_grid(sm_count: u32) -> u32 {
    (sq_blocks_per_sm() * sm_count).min(1024)
}

/// The `exl3_int8` module's kernels, resolved once per backend.
pub struct Exl3Int8Kernels {
    /// The sq GEMV entries: index = `[bits - 4][m - 1]`.
    pub sq: [[KernelHandle; 2]; 3],
    /// f32 -> bf16 grid-stride conversion, for the bf16 linear's epilogue.
    pub f32_to_bf16: KernelHandle,
    /// f32 [rows, cols] → bf16 rows at `out_stride` elements apart (M6d batched verify).
    pub f32_to_bf16_rows: KernelHandle,
}

impl Exl3Int8Kernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: &str| gpu.kernel("exl3_int8", name);
        Ok(Self {
            sq: [
                [k("exl3_int8_sq_k4_m1")?, k("exl3_int8_sq_k4_m2")?],
                [k("exl3_int8_sq_k5_m1")?, k("exl3_int8_sq_k5_m2")?],
                [k("exl3_int8_sq_k6_m1")?, k("exl3_int8_sq_k6_m2")?],
            ],
            f32_to_bf16: k("exl3_f32_to_bf16")?,
            f32_to_bf16_rows: k("exl3_f32_to_bf16_rows")?,
        })
    }
}

/// The zeroed int32 `locks` workspace of the sq kernel; see the module doc for
/// the self-reset property.
pub struct Exl3Int8Workspace {
    pub ptr: DevicePtr,
    pub ints: usize,
}

impl Exl3Int8Workspace {
    pub fn new(gpu: &dyn GpuBackend, ints: usize) -> Result<Self> {
        let ptr = gpu.alloc(ints * 4)?;
        gpu.memset(ptr, 0, ints * 4)?;
        Ok(Self { ptr, ints })
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.ptr)
    }
}

/// `C = A · W` through the sq GEMV: fp16 `A [m, k]`, EXL3 mul1 weight `w`,
/// fp32 `C [m, n]`. `a_had` is `m * k` fp16 of scratch, `grid` from [`sq_grid`].
pub fn exl3_int8_gemv(
    gpu: &dyn GpuBackend,
    k: &Exl3Int8Kernels,
    ws: &Exl3Int8Workspace,
    a_f16: DevicePtr,
    m: u32,
    w: &Exl3Weight,
    a_had: DevicePtr,
    c_f32: DevicePtr,
    grid: u32,
    stream: u64,
) -> Result<()> {
    let (kdim, n) = (w.shape.in_features, w.shape.out_features);
    let plan = sq_plan(kdim, n, m, w.shape.bits, grid)?;
    anyhow::ensure!(
        ws.ints >= plan.ws_ints,
        "EXL3 int8 sq: workspace has {} ints, this call needs {}",
        ws.ints,
        plan.ws_ints
    );
    KernelLaunch::new(gpu, k.sq[w.shape.bits as usize - 4][m as usize - 1])
        .grid([plan.grid, 1, 1])
        .block([256, 1, 1])
        .shared_mem(plan.smem)
        .arg_ptr(a_f16)
        .arg_ptr(w.trellis)
        .arg_ptr(c_f32)
        .arg_i32(m as i32)
        .arg_i32(kdim as i32)
        .arg_i32(n as i32)
        .arg_ptr(ws.ptr)
        .arg_ptr(w.suh)
        .arg_ptr(a_had)
        .arg_ptr(w.svh)
        .launch(stream)
}

/// `y = x · diag(suh) H W_inner H diag(svh)` for bf16 activations, through the
/// int8 sq GEMV — the analogue of `exl3_linear_bf16` (see its doc for the
/// formula), but the kernel folds both Hadamard passes and the quantization:
///
/// ```text
/// xh = bf16_to_f16(x)      exl3_convert, xh [m, in] fp16
/// y  = sq(xh, w)           exl3_int8_gemv, fp32
/// y  = f32_to_bf16(y)      grid-stride, into out_bf16
/// ```
///
/// `x_f16` is `m * in` fp16, `a_had` `m * in` fp16 scratch, `c_f32` `m * n`
/// fp32, `out_bf16` `m * n` bf16.
pub fn exl3_int8_linear_bf16(
    gpu: &dyn GpuBackend,
    k8: &Exl3Int8Kernels,
    k: &Exl3Kernels,
    ws: &Exl3Int8Workspace,
    x_bf16: DevicePtr,
    m: u32,
    w: &Exl3Weight,
    x_f16: DevicePtr,
    a_had: DevicePtr,
    c_f32: DevicePtr,
    out_bf16: DevicePtr,
    grid: u32,
    stream: u64,
) -> Result<()> {
    let n = w.shape.out_features;
    exl3_int8_linear_bf16_rows(
        gpu, k8, k, ws, x_bf16, m, w, x_f16, a_had, c_f32, out_bf16, n, grid, stream,
    )
}

/// [`exl3_int8_linear_bf16`] with the bf16 result written one projection's rows
/// into a wider `[m, out_stride]` buffer (M6d MTP-verify: the qkvz half lands in
/// a `[num_tokens, qkvz_size]` row) instead of contiguously.
#[allow(clippy::too_many_arguments)]
pub fn exl3_int8_linear_bf16_rows(
    gpu: &dyn GpuBackend,
    k8: &Exl3Int8Kernels,
    k: &Exl3Kernels,
    ws: &Exl3Int8Workspace,
    x_bf16: DevicePtr,
    m: u32,
    w: &Exl3Weight,
    x_f16: DevicePtr,
    a_had: DevicePtr,
    c_f32: DevicePtr,
    out_bf16: DevicePtr,
    out_stride: usize,
    grid: u32,
    stream: u64,
) -> Result<()> {
    let (kdim, n) = (w.shape.in_features, w.shape.out_features);
    exl3_convert(
        gpu,
        k.bf16_to_f16,
        x_bf16,
        x_f16,
        (m as usize * kdim) as u32,
        stream,
    )?;
    exl3_int8_gemv(gpu, k8, ws, x_f16, m, w, a_had, c_f32, grid, stream)?;
    KernelLaunch::new(gpu, k8.f32_to_bf16_rows)
        .grid([div_ceil(m * n as u32, 256).min(1024), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(c_f32)
        .arg_ptr(out_bf16)
        .arg_u32(m)
        .arg_u32(n as u32)
        .arg_u32(out_stride as u32)
        .launch(stream)
}

#[cfg(test)]
#[path = "exl3_int8_ops_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "exl3_int8_gpu_tests.rs"]
mod gpu_tests;
