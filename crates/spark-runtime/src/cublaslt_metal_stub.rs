// SPDX-License-Identifier: AGPL-3.0-only

//! Metal-build stub of the cuda-only `cublaslt` module.
//!
//! The real [`crate::cublaslt`] module (cuBLASLt BF16/FP8 act·weightᵀ GEMMs)
//! is gated behind `feature = "cuda"` because it links cuBLASLt, which does
//! not exist on macOS. spark-model names these entry points unconditionally,
//! so the metal build (`cargo check --features metal`, cuda off) needs the
//! symbols to resolve even though FP8 inference never runs there. The bodies
//! are `unreachable!` — reaching one on metal is a bug.

use anyhow::Result;

/// Mirrors `cublaslt::Bf16GemmFallback` for API parity; unused on metal.
pub type Bf16GemmFallback = dyn Fn(u64, u64, u64, u32, u32, u32, u64) -> Result<()> + Send + Sync;

/// Mirrors the geometry constants in `cublaslt` for API parity; the values
/// are the same single source of truth (the stub is never executed).
pub const DENSE_GEMM_BF16_PIPELINED_TILE: u32 = 128;
pub const DENSE_GEMM_BF16_PIPELINED_THREADS: u32 = 256;
pub const DENSE_GEMM_BF16_SCALAR_TILE: u32 = 16;
pub const DENSE_GEMV_BATCHM_MAX_M: u32 = 8;
pub const DENSE_GEMV_BATCHM_OUTPUTS_PER_BLOCK: u32 = 4;
pub const DENSE_GEMV_BATCHM_THREADS: u32 = 256;

pub fn install_bf16_fallback(_f: Box<Bf16GemmFallback>) -> bool {
    unreachable!("cublaslt::install_bf16_fallback is cuda-only (not built for metal)")
}

pub fn make_bf16_fallback(
    _pipelined: crate::gpu::KernelHandle,
    _scalar: crate::gpu::KernelHandle,
    _gemv_batchm: crate::gpu::KernelHandle,
) -> Box<Bf16GemmFallback> {
    unreachable!("cublaslt::make_bf16_fallback is cuda-only (not built for metal)")
}

pub fn bf16_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::bf16_gemm_act_weight_t is cuda-only (not built for metal)")
}

pub fn tf32_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::tf32_gemm_act_weight_t is cuda-only (not built for metal)")
}

pub fn fp8_gemm_act_weight_t_rowwise(
    _act_fp8: u64,
    _act_scale: u64,
    _weight_fp8: u64,
    _weight_scale: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::fp8_gemm_act_weight_t_rowwise is cuda-only (not built for metal)")
}

pub fn fp8_gemm_act_weight_t_blkscaled(
    _act_fp8: u64,
    _act_scale: u64,
    _weight_fp8: u64,
    _weight_block_scale: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::fp8_gemm_act_weight_t_blkscaled is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn bf16_grouped_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _g: u32,
    _n: u32,
    _k: u32,
    _a_stride: u32,
    _c_stride: u32,
    _stream: u64,
) -> Result<()> {
    anyhow::bail!("cublaslt::bf16_grouped_gemm_act_weight_t requires CUDA")
}

/// No CUDA FP8 library on Metal; the opt-in caller fails explicitly.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_tensorwise(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _no_split_k: bool,
    _stream: u64,
) -> anyhow::Result<()> {
    anyhow::bail!("cuBLASLt tensorwise FP8 requires CUDA")
}
