// SPDX-License-Identifier: AGPL-3.0-only

//! The compressor and Engram in their plain forms (`dsv41_side.cu`): the two
//! paths the reference runs in FP32.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const THREADS: u32 = 256;
const MAX_K: usize = 5120;

#[derive(Clone, Copy)]
pub struct SideKernels {
    pub gemv_f32: KernelHandle,
    pub compress_pool: KernelHandle,
    pub engram_gate: KernelHandle,
}

impl SideKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let kernel = |name: &str| gpu.kernel("dsv41_side", name);
        Ok(Self {
            gemv_f32: kernel("dsv41_gemv_f32")?,
            compress_pool: kernel("dsv41_compress_pool")?,
            engram_gate: kernel("dsv41_engram_gate")?,
        })
    }
}

/// `out` (`[n]` FP32, not rounded) = `w` (`[n, k]` BF16) times `x` (`[k]`
/// BF16): a compressor projection.
pub fn gemv_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(k <= MAX_K, "dsv41_gemv_f32: k {k}");
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n as u32, THREADS), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

/// One group's pooled key (`[dim]` BF16, before the norm) from its `ratio`
/// tokens' key and gate projections (`[ratio, dim]` FP32 each).
#[allow(clippy::too_many_arguments)]
pub fn compress_pool(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    kv: DevicePtr,
    score: DevicePtr,
    pooled: DevicePtr,
    ratio: usize,
    dim: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(dim as u32, THREADS), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(kv)
        .arg_ptr(score)
        .arg_ptr(pooled)
        .arg_u32(ratio as u32)
        .arg_u32(dim as u32)
        .launch(stream)
}

/// Add an Engram lookup into `stream` (`[tokens, copies, hidden]` BF16, in
/// place). `kv` (`[tokens, (copies + 1) * hidden]` BF16) is the lookup's
/// projection: a key per copy, then the value. `weight` (`[copies, hidden]`
/// FP32) is the layer's `q_weight * k_weight`.
#[allow(clippy::too_many_arguments)]
pub fn engram_gate(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    stream_buf: DevicePtr,
    kv: DevicePtr,
    weight: DevicePtr,
    tokens: usize,
    hidden: usize,
    copies: usize,
    eps: f32,
    stream: u64,
) -> Result<()> {
    ensure!(copies <= 4, "dsv41_engram_gate: {copies} copies");
    KernelLaunch::new(gpu, kernel)
        .grid([tokens as u32, 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(stream_buf)
        .arg_ptr(kv)
        .arg_ptr(weight)
        .arg_u32(hidden as u32)
        .arg_u32(copies as u32)
        .arg_f32(eps)
        .launch(stream)
}

#[cfg(test)]
#[path = "side_test_gpu.rs"]
mod test_gpu;
