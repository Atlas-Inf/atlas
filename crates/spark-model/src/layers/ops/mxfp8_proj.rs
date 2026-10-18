// SPDX-License-Identifier: AGPL-3.0-only
//! Native MXFP8 projections over checkpoint bytes (`mxfp8_proj.cu`).
//!
//! NVIDIA's Qwen3.8-Flash-Next-FP4-Local ships the GDN, attention, indexer
//! and shared-expert projections as MXFP8: E4M3 `[n, k]` plus one E8M0 scale
//! per 32 inputs `[n, k / 32]`. [`mxfp8_proj`] multiplies by those bytes as
//! they are:
//!
//! * 1 row: `mxfp8_proj_gemv`, one pass over the weight;
//! * 2..=8 rows (verify windows, small batches): `mxfp8_proj_batchm`, still
//!   one pass, every row bit-identical to the 1-row kernel;
//! * more rows (prefill): `mxfp8_proj_dequant` expands the weight into a BF16
//!   scratch, exactly (E4M3 times a power of two is a BF16 value), and
//!   [`super::bf16_gemm`] multiplies by it. The scratch is transient and
//!   shared, so no BF16 copy of the weight stays resident.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Mxfp8Weight;

/// Widest row count the weight-streaming GEMVs serve.
pub const MXFP8_PROJ_MAX_GEMV_ROWS: usize = 8;

/// Kernel handles of `mxfp8_proj.cu`.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8ProjKernels {
    pub gemv: KernelHandle,
    pub batchm: KernelHandle,
    pub dequant: KernelHandle,
}

impl Mxfp8ProjKernels {
    /// Resolve all three or fail: an MXFP8 weight with no kernel to read it
    /// must not load.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemv: gpu.kernel("mxfp8_proj", "mxfp8_proj_gemv")?,
            batchm: gpu.kernel("mxfp8_proj", "mxfp8_proj_batchm")?,
            dequant: gpu.kernel("mxfp8_proj", "mxfp8_proj_dequant")?,
        })
    }
}

/// One MXFP8 projection and its geometry.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Proj {
    pub w: Mxfp8Weight,
    /// Output features (weight rows).
    pub n: u32,
    /// Input features; a multiple of 32.
    pub k: u32,
}

impl Mxfp8Proj {
    /// Bytes of the BF16 expansion the prefill path writes into scratch.
    pub fn bf16_bytes(&self) -> usize {
        self.n as usize * self.k as usize * 2
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
    let Mxfp8Proj { w, n, k } = *proj;
    ensure!(
        m > 0 && n > 0 && k.is_multiple_of(32),
        "mxfp8_proj: m={m} n={n} k={k} (k must be a multiple of 32)"
    );
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
    if m <= MXFP8_PROJ_MAX_GEMV_ROWS {
        return KernelLaunch::new(gpu, kernels.batchm)
            .grid([div_ceil(n, 4), 1, 1])
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
