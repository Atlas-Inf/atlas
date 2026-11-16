// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 MoE in its plain form (`dsv41_moe.cu`): the rounding points
//! of the reference, one launch per step. The tensor-core paths are held to
//! what this computes.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::weights::{ExpertSlab, Slice};

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
    /// A batch's scores from its logits.
    pub soften: KernelHandle,
    /// Every expert of a token in three launches (`dsv41_expert.cu`).
    pub expert_in: KernelHandle,
    pub expert_act: KernelHandle,
    pub expert_out: KernelHandle,
    /// The same for a batch of rows, grouped by expert (`moe_grouped.rs`).
    pub experts_in: KernelHandle,
    pub experts_act: KernelHandle,
    pub experts_act_codes: KernelHandle,
    pub experts_out: KernelHandle,
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
            soften: kernel("dsv41_moe_soften")?,
            expert_in: gpu.kernel("dsv41_expert", "dsv41_expert_in")?,
            expert_act: gpu.kernel("dsv41_expert", "dsv41_expert_act")?,
            expert_out: gpu.kernel("dsv41_expert", "dsv41_expert_out")?,
            experts_in: gpu.kernel("dsv41_expert", "dsv41_experts_in")?,
            experts_act: gpu.kernel("dsv41_expert", "dsv41_experts_act")?,
            experts_act_codes: gpu.kernel("dsv41_expert", "dsv41_experts_act_codes")?,
            experts_out: gpu.kernel("dsv41_expert", "dsv41_experts_out")?,
            fp4_gemv: kernel("dsv41_fp4_gemv")?,
            fp4_gemv_tc: gpu.kernel("dsv41_tc", "dsv41_fp4_gemv_tc8")?,
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
/// another order. `scores` is `[tokens, experts]` FP32 scratch. `by_expert`
/// leaves the chosen in ascending expert order, each with its weight, which
/// is the order their outputs are added in. `batch` takes the scores as one
/// GEMM, in yet another order.
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
    by_expert: bool,
    batch: bool,
    stream: u64,
) -> Result<()> {
    ensure!(
        experts <= 512 && top_k <= 8,
        "dsv41_moe_pick: {experts} experts, top {top_k}"
    );
    match batch {
        // The logits as one GEMM, in FP32.
        true => {
            let (m, n, k) = (tokens as u32, experts as u32, hidden as u32);
            crate::layers::ops::bf16_gemm_f32out(x, gate.0, scores, m, n, k, stream)?;
            elementwise(gpu, kernels.soften, tokens * experts)
                .arg_ptr(scores)
                .arg_u32((tokens * experts) as u32)
                .launch(stream)?;
        }
        false => KernelLaunch::new(gpu, kernels.scores)
            .grid([experts as u32, tokens as u32, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(x)
            .arg_ptr(gate)
            .arg_ptr(scores)
            .arg_u32(hidden as u32)
            .arg_u32(experts as u32)
            .launch(stream)?,
    }
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
        .arg_u32(u32::from(by_expert))
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

/// One part of every expert a token runs: the layer's slab of routed
/// experts, and the part's rows of the shared expert's w1, w3 and w2.
#[derive(Clone, Copy)]
pub struct ExpertPart {
    pub slab: ExpertSlab,
    pub shared: [Slice; 3],
}

impl ExpertPart {
    fn layout<'a>(&self, launch: KernelLaunch<'a>) -> KernelLaunch<'a> {
        let (data, scales) = self.slab.layout();
        launch
            .arg_u64(self.slab.stride() as u64)
            .arg_u32(data as u32)
            .arg_u32(scales as u32)
    }
}

/// One part of the way into every expert of one token, in two launches
/// (`dsv41_expert.cu`): `inner` (`[top_k + 1, rows]` BF16) = the part's
/// `rows` of each chosen expert's hidden activation in FP8, in the order of
/// `chosen` (`[top_k]` u32, with `weights` `[top_k]` FP32, as [`route_fast`]
/// leaves them by expert), then of the shared expert's. `x` is the token's
/// input in FP8, and `gate_up` scratch of `[top_k + 1, 2, rows]` BF16.
#[allow(clippy::too_many_arguments)]
pub fn experts_in(
    gpu: &dyn GpuBackend,
    kernels: &MoeKernels,
    part: &ExpertPart,
    x: DevicePtr,
    chosen: DevicePtr,
    weights: DevicePtr,
    gate_up: DevicePtr,
    inner: DevicePtr,
    top_k: usize,
    rows: usize,
    hidden: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows.is_multiple_of(32) && hidden.is_multiple_of(32) && hidden <= super::parts::STAGED_K,
        "dsv41_expert: {rows} rows of {hidden}"
    );
    let slots = top_k as u32 + 1;
    let [w1, w3, _] = part.shared;
    part.layout(
        KernelLaunch::new(gpu, kernels.expert_in)
            .grid([div_ceil(rows as u32, 16), 2 * slots, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(part.slab.base)
            .arg_ptr(chosen)
            .arg_ptr(w1.data)
            .arg_ptr(w1.scales)
            .arg_ptr(w3.data)
            .arg_ptr(w3.scales)
            .arg_ptr(gate_up)
            .arg_u32(top_k as u32)
            .arg_u32(rows as u32)
            .arg_u32(hidden as u32),
    )
    .launch(stream)?;
    elementwise(gpu, kernels.expert_act, slots as usize * rows / 32)
        .arg_ptr(gate_up)
        .arg_ptr(weights)
        .arg_ptr(inner)
        .arg_u32(top_k as u32)
        .arg_u32(slots)
        .arg_u32(rows as u32)
        .arg_f32(limit)
        .launch(stream)
}

/// One part of the way out of every expert of one token: `out` (`[top_k +
/// 1, n]` BF16) = the part's `n` rows of each expert's output. `inner` is
/// every part of the hidden activations, as [`experts_in`] left each and the
/// ranks gathered them: `rows` values a slot, and a part `piece_stride`
/// values after the last.
#[allow(clippy::too_many_arguments)]
pub fn experts_out(
    gpu: &dyn GpuBackend,
    kernels: &MoeKernels,
    part: &ExpertPart,
    chosen: DevicePtr,
    inner: DevicePtr,
    piece_stride: usize,
    out: DevicePtr,
    top_k: usize,
    rows: usize,
    n: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows.is_multiple_of(64) && rows * super::parts::PARTS <= super::parts::STAGED_K,
        "dsv41_expert_out: {rows} hidden rows a part"
    );
    let w2 = part.shared[2];
    part.layout(
        KernelLaunch::new(gpu, kernels.expert_out)
            .grid([div_ceil(n as u32, 16), top_k as u32 + 1, 1])
            .block([256, 1, 1])
            .arg_ptr(inner)
            .arg_ptr(part.slab.base)
            .arg_ptr(chosen)
            .arg_ptr(w2.data)
            .arg_ptr(w2.scales)
            .arg_ptr(out)
            .arg_u32(top_k as u32)
            .arg_u32(n as u32)
            .arg_u32(rows as u32),
    )
    .arg_u64(piece_stride as u64)
    .launch(stream)
}

#[cfg(test)]
#[path = "moe_test_gpu.rs"]
mod test_gpu;
