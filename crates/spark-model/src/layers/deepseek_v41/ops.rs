// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the DeepSeek-V4.1 attention kernels (`dsv41_attn.cu`,
//! `dsv41_cache.cu`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::ops::mxfp8_gemv_rows;
use crate::weight_map::Mxfp8Weight;

/// Values in one key, and in one head's query.
pub const HEAD_DIM: usize = 512;
/// The most keys one query attends to: the window and the selected
/// compressed positions.
pub const MAX_KEYS: usize = 640;

const THREADS: u32 = 256;

/// The kernels of the V4.1 attention cache.
#[derive(Clone, Copy)]
pub struct Kernels {
    pub sparse_attn: KernelHandle,
    pub rotate: KernelHandle,
    pub fp8_quant_dequant: KernelHandle,
    pub fp4_quant_dequant: KernelHandle,
}

impl Kernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            sparse_attn: gpu.kernel("dsv41_attn", "dsv41_sparse_attn")?,
            rotate: gpu.kernel("dsv41_cache", "dsv41_rotate")?,
            fp8_quant_dequant: gpu.kernel("dsv41_cache", "dsv41_fp8_quant_dequant")?,
            fp4_quant_dequant: gpu.kernel("dsv41_cache", "dsv41_fp4_quant_dequant")?,
        })
    }
}

/// Attention of `queries` queries of `heads` heads (`q`, `[queries, heads,
/// 512]` BF16) over gathered keys, into `out` of the same shape. `idx`
/// (`[queries, keys]` i32) holds row numbers, -1 for none: columns below
/// `split` read `keys_a`, the rest `keys_b`. `sink` is `[heads]` FP32.
///
/// A query's bytes depend only on its own q and on its index row, not on the
/// other queries of the launch.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    keys_a: DevicePtr,
    keys_b: DevicePtr,
    idx: DevicePtr,
    sink: DevicePtr,
    out: DevicePtr,
    queries: usize,
    heads: usize,
    keys: usize,
    split: usize,
    stream: u64,
) -> Result<()> {
    ensure!(keys <= MAX_KEYS, "dsv41_sparse_attn: {keys} keys");
    KernelLaunch::new(gpu, kernel)
        .grid([queries as u32, heads as u32, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(q)
        .arg_ptr(keys_a)
        .arg_ptr(keys_b)
        .arg_ptr(idx)
        .arg_ptr(sink)
        .arg_ptr(out)
        .arg_u32(heads as u32)
        .arg_u32(keys as u32)
        .arg_u32(split as u32)
        .arg_f32((HEAD_DIM as f32).powf(-0.5))
        .launch(stream)
}

/// Rotate the last `rope` values of every vector of `x` (`[rows, vectors,
/// dim]` BF16) to its row's position (`positions`, `[rows]` u32), in place.
/// `freqs` is `[rope / 2]` FP32, radians per position. `inverse` undoes it.
#[allow(clippy::too_many_arguments)]
pub fn rotate(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    positions: DevicePtr,
    freqs: DevicePtr,
    rows: usize,
    vectors: usize,
    dim: usize,
    rope: usize,
    inverse: bool,
    stream: u64,
) -> Result<()> {
    ensure!(
        rope <= dim && rope.is_multiple_of(2),
        "dsv41_rotate: rope {rope} of {dim}"
    );
    let pairs = (rows * vectors * rope / 2) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(pairs, THREADS), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(positions)
        .arg_ptr(freqs)
        .arg_u32(rows as u32)
        .arg_u32(vectors as u32)
        .arg_u32(dim as u32)
        .arg_u32(rope as u32)
        .arg_u32(u32::from(inverse))
        .launch(stream)
}

/// FP8 under a power-of-two scale per 32 values, as BF16: `values` BF16
/// values of `src` into `dst`, which may be `src`.
pub fn fp8_quant_dequant(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    values: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        values.is_multiple_of(32),
        "dsv41_fp8_quant_dequant: {values} values"
    );
    let blocks = (values / 32) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(blocks, THREADS), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(blocks)
        .launch(stream)
}

/// A projection with FP8 weights, as the reference runs it: `rows` rows of
/// `input` (`[rows, k]` BF16) are quantized to FP8 into `quantized` (the same
/// size, left holding them) and multiplied by `w` (`[n, k]`) into `output`
/// (`[rows, n]` BF16). `tiers` holds `mxfp8_gemv_tc{8,16,32}`.
#[allow(clippy::too_many_arguments)]
pub fn fp8_linear(
    gpu: &dyn GpuBackend,
    kernels: &Kernels,
    tiers: &[KernelHandle; 3],
    w: &Mxfp8Weight,
    input: DevicePtr,
    quantized: DevicePtr,
    output: DevicePtr,
    rows: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    fp8_quant_dequant(
        gpu,
        kernels.fp8_quant_dequant,
        input,
        quantized,
        rows * k,
        stream,
    )?;
    mxfp8_gemv_rows(gpu, tiers, w, quantized, output, rows, n, k, n, stream)
}

/// The scale an FP4 group is quantized under.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fp4Scale {
    /// A power of two per 32 values: index keys.
    PowerOfTwoPer32,
    /// An E4M3 value per 16 values: compressed keys.
    E4m3Per16,
}

/// FP4 under `scale`, back as BF16, over `values` BF16 values of `x` in place.
pub fn fp4_quant_dequant(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    values: usize,
    scale: Fp4Scale,
    stream: u64,
) -> Result<()> {
    let width = match scale {
        Fp4Scale::PowerOfTwoPer32 => 32,
        Fp4Scale::E4m3Per16 => 16,
    };
    ensure!(
        values.is_multiple_of(width),
        "dsv41_fp4_quant_dequant: {values} values"
    );
    let groups = (values / width) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(groups, THREADS), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(x)
        .arg_u32(groups)
        .arg_u32(u32::from(scale == Fp4Scale::E4m3Per16))
        .launch(stream)
}

#[cfg(test)]
#[path = "ops_test_gpu.rs"]
mod test_gpu;

#[cfg(test)]
#[path = "proj_test_gpu.rs"]
mod proj_test_gpu;
