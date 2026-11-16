// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 MoE in its plain form (`dsv41_moe.cu`): the rounding points
//! of the reference, one launch per step. The tensor-core paths are held to
//! what this computes.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const THREADS: u32 = 256;
/// The widest activation a kernel here stages in shared memory.
const MAX_K: usize = 5120;

#[derive(Clone, Copy)]
pub struct MoeKernels {
    /// The plain routing, one launch.
    pub route: KernelHandle,
    /// The same in two launches, a block per (expert, token).
    pub scores: KernelHandle,
    pub pick: KernelHandle,
    /// The plain projection, one thread per row.
    pub fp4_gemv: KernelHandle,
    /// The same on tensor cores, for up to 8 rows (`dsv41_fp4_tc.cu`).
    pub fp4_gemv_tc: KernelHandle,
    /// `ATLAS_DSV41_PLAIN_FP4=1` (debug): project with the plain kernel.
    pub plain_fp4: bool,
    pub swiglu: KernelHandle,
    pub accumulate: KernelHandle,
    pub round_bf16: KernelHandle,
}

impl MoeKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let kernel = |name: &str| gpu.kernel("dsv41_moe", name);
        Ok(Self {
            route: kernel("dsv41_moe_route")?,
            scores: kernel("dsv41_moe_scores")?,
            pick: kernel("dsv41_moe_pick")?,
            fp4_gemv: kernel("dsv41_fp4_gemv")?,
            fp4_gemv_tc: gpu.kernel("dsv41_fp4_tc", "dsv41_fp4_gemv_tc8")?,
            plain_fp4: std::env::var("ATLAS_DSV41_PLAIN_FP4").is_ok_and(|v| v == "1"),
            swiglu: kernel("dsv41_swiglu")?,
            accumulate: kernel("dsv41_accumulate")?,
            round_bf16: kernel("dsv41_round_bf16")?,
        })
    }
}

/// Route `tokens` rows of `x` (`[tokens, hidden]` BF16, not quantized):
/// `chosen` (`[tokens, top_k]` u32, in order of selection) and their
/// `weights` (`[tokens, top_k]` FP32). `gate` is `[experts, hidden]` BF16 and
/// `bias` `[experts]` FP32.
#[allow(clippy::too_many_arguments)]
pub fn route(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    gate: DevicePtr,
    bias: DevicePtr,
    chosen: DevicePtr,
    weights: DevicePtr,
    tokens: usize,
    hidden: usize,
    experts: usize,
    top_k: usize,
    route_scale: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        hidden <= MAX_K && experts <= 512 && top_k <= 8,
        "dsv41_moe_route: hidden {hidden}, {experts} experts, top {top_k}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([tokens as u32, 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(gate)
        .arg_ptr(bias)
        .arg_ptr(chosen)
        .arg_ptr(weights)
        .arg_u32(hidden as u32)
        .arg_u32(experts as u32)
        .arg_u32(top_k as u32)
        .arg_f32(route_scale)
        .launch(stream)
}

/// [`route`] as the block runs it: the same choice, the scores summed in
/// another order. `scores` is `[tokens, experts]` FP32 scratch.
#[allow(clippy::too_many_arguments)]
pub fn route_fast(
    gpu: &dyn GpuBackend,
    kernels: &MoeKernels,
    x: DevicePtr,
    gate: DevicePtr,
    bias: DevicePtr,
    scores: DevicePtr,
    chosen: DevicePtr,
    weights: DevicePtr,
    tokens: usize,
    hidden: usize,
    experts: usize,
    top_k: usize,
    route_scale: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        experts <= 512 && top_k <= 8,
        "dsv41_moe_pick: {experts} experts, top {top_k}"
    );
    KernelLaunch::new(gpu, kernels.scores)
        .grid([experts as u32, tokens as u32, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(gate)
        .arg_ptr(scores)
        .arg_u32(hidden as u32)
        .arg_u32(experts as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.pick)
        .grid([tokens as u32, 1, 1])
        .block([512, 1, 1])
        .arg_ptr(scores)
        .arg_ptr(bias)
        .arg_ptr(chosen)
        .arg_ptr(weights)
        .arg_u32(experts as u32)
        .arg_u32(top_k as u32)
        .arg_f32(route_scale)
        .launch(stream)
}

/// One routed-expert projection of one vector: `out` (`[n]` BF16) = `w4`
/// (`[n, k]` FP4, two values a byte) times `x` (`[k]` BF16), under the E8M0
/// `scales` (`[n, k / 32]`).
#[allow(clippy::too_many_arguments)]
pub fn fp4_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    w4: DevicePtr,
    scales: DevicePtr,
    out: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(k <= MAX_K && k.is_multiple_of(32), "dsv41_fp4_gemv: k {k}");
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n as u32, THREADS), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w4)
        .arg_ptr(scales)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

/// [`fp4_gemv`] on tensor cores: the same weights, unrounded, and the same
/// FP32 accumulation in another order.
#[allow(clippy::too_many_arguments)]
pub fn fp4_gemv_tc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    w4: DevicePtr,
    scales: DevicePtr,
    out: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(k.is_multiple_of(32), "dsv41_fp4_gemv_tc: k {k}");
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n as u32, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w4)
        .arg_ptr(scales)
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(n as u32)
        .launch(stream)
}

/// One routed-expert projection, as the block runs it.
#[allow(clippy::too_many_arguments)]
pub fn fp4_project(
    gpu: &dyn GpuBackend,
    kernels: &MoeKernels,
    x: DevicePtr,
    w4: DevicePtr,
    scales: DevicePtr,
    out: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    match kernels.plain_fp4 {
        true => fp4_gemv(gpu, kernels.fp4_gemv, x, w4, scales, out, n, k, stream),
        false => fp4_gemv_tc(gpu, kernels.fp4_gemv_tc, x, w4, scales, out, n, k, stream),
    }
}

fn elementwise(gpu: &dyn GpuBackend, kernel: KernelHandle, n: usize) -> KernelLaunch<'_> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n as u32, THREADS), 1, 1])
        .block([THREADS, 1, 1])
}

/// `h` = BF16(`weight` * silu(min(`gate`, `limit`)) * clamp(`up`, +-`limit`)).
#[allow(clippy::too_many_arguments)]
pub fn swiglu(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    h: DevicePtr,
    n: usize,
    weight: f32,
    limit: f32,
    stream: u64,
) -> Result<()> {
    elementwise(gpu, kernel, n)
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(h)
        .arg_u32(n as u32)
        .arg_f32(weight)
        .arg_f32(limit)
        .launch(stream)
}

/// `acc` (`[n]` FP32) += `x` (`[n]` BF16).
pub fn accumulate(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    acc: DevicePtr,
    x: DevicePtr,
    n: usize,
    stream: u64,
) -> Result<()> {
    elementwise(gpu, kernel, n)
        .arg_ptr(acc)
        .arg_ptr(x)
        .arg_u32(n as u32)
        .launch(stream)
}

/// `out` (`[n]` BF16) = `acc` (`[n]` FP32), rounded.
pub fn round_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    acc: DevicePtr,
    out: DevicePtr,
    n: usize,
    stream: u64,
) -> Result<()> {
    elementwise(gpu, kernel, n)
        .arg_ptr(acc)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .launch(stream)
}

#[cfg(test)]
#[path = "moe_test_gpu.rs"]
mod test_gpu;
