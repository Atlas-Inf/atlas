// SPDX-License-Identifier: AGPL-3.0-only
//! Native MXFP8 projections over checkpoint bytes (`mxfp8_proj.cu`).
//!
//! NVIDIA's Qwen3.8-Flash-Next-FP4-Local ships the GDN, attention, indexer
//! and shared-expert projections as MXFP8: E4M3 `[n, k]` plus one E8M0 scale
//! per 32 inputs `[n, k / 32]`. [`mxfp8_proj`] multiplies by those bytes as
//! they are:
//!
//! * 1 row: `mxfp8_proj_gemv`, one pass over the weight;
//! * 2..=32 rows (verify windows, small batches, short prefill chunks):
//!   `mxfp8_proj_batchm`, one weight pass for the whole window — the launch
//!   buckets rows at 8/16/24/32 and every row is bit-identical to the 1-row
//!   kernel;
//! * more rows (prefill): `mxfp8_proj_dequant` expands the weight into a BF16
//!   scratch, exactly (E4M3 times a power of two is a BF16 value), and
//!   [`super::bf16_gemm`] multiplies by it. The scratch is transient and
//!   shared, so no BF16 copy of the weight stays resident.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

pub use crate::weight_map::Mxfp8Proj;

/// Widest row count one weight-streaming GEMV launch serves.
pub const MXFP8_PROJ_MAX_GEMV_ROWS: usize = 32;

/// Up to this many rows run as chunked GEMVs; beyond it the expansion plus a
/// tensor-core GEMM reads fewer bytes than re-streaming the weight per chunk.
pub const MXFP8_PROJ_MAX_CHUNKED_ROWS: usize = 32;

/// Kernel handles of `mxfp8_proj.cu`.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8ProjKernels {
    pub gemv: KernelHandle,
    pub batchm: KernelHandle,
    pub dequant: KernelHandle,
    /// Register-tiled 2..=32-row tier (`mxfp8_proj_rows32`); optional —
    /// targets without it fall back to `batchm`.
    pub rows32: KernelHandle,
}

impl Mxfp8ProjKernels {
    /// Resolve all three or fail: an MXFP8 weight with no kernel to read it
    /// must not load.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemv: gpu.kernel("mxfp8_proj", "mxfp8_proj_gemv")?,
            batchm: gpu.kernel("mxfp8_proj", "mxfp8_proj_batchm")?,
            dequant: gpu.kernel("mxfp8_proj", "mxfp8_proj_dequant")?,
            rows32: crate::layers::kernel_pick::try_kernel(gpu, "mxfp8_proj", "mxfp8_proj_rows32"),
        })
    }
}

/// A BF16 scratch for the prefill expansion: device pointer and capacity.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Scratch {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

/// `out[m, n] = act[m, k] @ W[n, k]^T` for a contiguous `act` (`k` BF16 per
/// row) and `out` (`n` BF16 per row).
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_proj(
    gpu: &dyn GpuBackend,
    kernels: &Mxfp8ProjKernels,
    proj: &Mxfp8Proj,
    act: DevicePtr,
    out: DevicePtr,
    m: usize,
    scratch: Mxfp8Scratch,
    stream: u64,
) -> Result<()> {
    let Mxfp8Proj { n, k, .. } = *proj;
    ensure!(
        m > 0 && n > 0 && k.is_multiple_of(32),
        "mxfp8_proj: m={m} n={n} k={k} (k must be a multiple of 32)"
    );
    if m == 1 {
        return gemv_rows(gpu, kernels, proj, act, out, 1, stream);
    }
    if m <= MXFP8_PROJ_MAX_CHUNKED_ROWS {
        return gemv_rows(gpu, kernels, proj, act, out, m, stream);
    }
    expand_and_gemm(gpu, kernels, proj, act, out, m, scratch, stream)
}

/// The bit-exact variant for decode and speculative verify: every row is the
/// 1-row GEMV's bytes at ANY row count, served as consecutive `<= 32`-row
/// `gemv_rows` launches — never the dequant + GEMM path, which is a
/// different (prefill-only) numerics family. A verify batch can exceed 32
/// rows (e.g. 8 sequences x 8 speculated rows at C8/D7); routing it through
/// [`mxfp8_proj`] would silently drop into the GEMM and its rows would no
/// longer equal serial decode's.
pub fn mxfp8_proj_exact_rows(
    gpu: &dyn GpuBackend,
    kernels: &Mxfp8ProjKernels,
    proj: &Mxfp8Proj,
    act: DevicePtr,
    out: DevicePtr,
    m: usize,
    stream: u64,
) -> Result<()> {
    let Mxfp8Proj { n, k, .. } = *proj;
    ensure!(
        m > 0 && n > 0 && k.is_multiple_of(32),
        "mxfp8_proj_exact_rows: m={m} n={n} k={k} (k must be a multiple of 32)"
    );
    for first in (0..m).step_by(MXFP8_PROJ_MAX_GEMV_ROWS) {
        let rows = (m - first).min(MXFP8_PROJ_MAX_GEMV_ROWS);
        gemv_rows(
            gpu,
            kernels,
            proj,
            act.offset(first * k as usize * 2),
            out.offset(first * n as usize * 2),
            rows,
            stream,
        )?;
    }
    Ok(())
}

/// One weight pass for `m` in 1..=32 contiguous rows: bit-identical to the
/// 1-row kernel on every row.
fn gemv_rows(
    gpu: &dyn GpuBackend,
    kernels: &Mxfp8ProjKernels,
    proj: &Mxfp8Proj,
    act: DevicePtr,
    out: DevicePtr,
    m: usize,
    stream: u64,
) -> Result<()> {
    let Mxfp8Proj { w, n, k } = *proj;
    if m == 1 {
        return KernelLaunch::new(gpu, kernels.gemv)
            .grid([div_ceil(n, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_ptr(w.data)
            .arg_ptr(w.scales)
            .arg_ptr(out)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream);
    }
    if kernels.rows32.0 != 0 && m > 16 {
        return KernelLaunch::new(gpu, kernels.rows32)
            .grid([div_ceil(n, 8), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_ptr(w.data)
            .arg_ptr(w.scales)
            .arg_ptr(out)
            .arg_u32(m as u32)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream);
    }
    KernelLaunch::new(gpu, kernels.batchm)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act)
        .arg_ptr(w.data)
        .arg_ptr(w.scales)
        .arg_ptr(out)
        .arg_u32(m as u32)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Exact BF16 expansion into `scratch`, then a tensor-core GEMM.
#[allow(clippy::too_many_arguments)]
fn expand_and_gemm(
    gpu: &dyn GpuBackend,
    kernels: &Mxfp8ProjKernels,
    proj: &Mxfp8Proj,
    act: DevicePtr,
    out: DevicePtr,
    m: usize,
    scratch: Mxfp8Scratch,
    stream: u64,
) -> Result<()> {
    let Mxfp8Proj { w, n, k } = *proj;
    ensure!(
        !scratch.ptr.is_null() && scratch.bytes >= proj.bf16_bytes(),
        "mxfp8_proj: {m} rows need a {} B BF16 scratch, have {} B",
        proj.bf16_bytes(),
        scratch.bytes
    );
    let chunks = u64::from(n) * u64::from(k / 16);
    ensure!(
        chunks.div_ceil(256) <= u64::from(u32::MAX),
        "mxfp8_proj: {n}x{k} is too large for one dequant grid"
    );
    KernelLaunch::new(gpu, kernels.dequant)
        .grid([chunks.div_ceil(256) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(w.data)
        .arg_ptr(w.scales)
        .arg_ptr(scratch.ptr)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    super::bf16_gemm(act, scratch.ptr.0, out, m as u32, n, k, stream)
}
