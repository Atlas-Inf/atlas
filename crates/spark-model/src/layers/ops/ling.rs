// SPDX-License-Identifier: AGPL-3.0-only

//! Small GPU dispatch wrappers for the original Ling 3.2 MoE path.
//!
//! Ling keeps the router in BF16/FP32 while its expert matrices stay in the
//! checkpoint's row-major E4M3 + group-128 scale layout.  The wrappers live in
//! their own module so the model-specific kernels cannot accidentally become a
//! fallback for another MoE implementation.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8Weight};

/// Dense BF16 GEMV with an FP32 output.  The kernel is the same scalar-order
/// GEMV used by the router family; only the final accumulator store differs.
pub(crate) fn dense_gemv_bf16_fp32out(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
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
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Ling's group-limited top-8 router.  The kernel writes eight i32 expert ids
/// followed by eight FP32 route weights for one token.
pub(crate) fn ling_grouped_top8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    bias: DevicePtr,
    ids: DevicePtr,
    weights: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(bias)
        .arg_ptr(ids)
        .arg_ptr(weights)
        .launch(stream)
}

/// Ling's original gated SwiGLU followed by per-token group-128 FP8
/// quantization.  `out_bf16` may be null when only the quantized activation is
/// needed by the down projection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ling_silu_mul_quant_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out_fp8: DevicePtr,
    out_scale: DevicePtr,
    out_bf16: DevicePtr,
    rows: u32,
    cols: u32,
    limit: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, cols / 128, 1])
        .block([128, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out_fp8)
        .arg_ptr(out_scale)
        .arg_ptr(out_bf16)
        .arg_u32(rows)
        .arg_u32(cols)
        .arg_f32(limit)
        .launch(stream)
}

/// Original Ling A16 epilogue: clamp, SiLU-gate and multiply in BF16.
/// Unlike the FP8 path this keeps the activation as BF16 for W4A16 GEMV.
pub(crate) fn ling_silu_mul_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    rows: u32,
    cols: u32,
    limit: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(rows * cols, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(cols)
        .arg_f32(limit)
        .launch(stream)
}

/// Original-FP8 Ling down projection with device-side expert dispatch.
/// `routed_limit` and `shared_limit` are applied after BF16 SiLU rounding,
/// matching the checkpoint reference's asymmetric SwiGLU clamp order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ling_silu_down_shared_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_weight_ptrs: DevicePtr,
    down_scale_ptrs: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    shared_gate_in: DevicePtr,
    shared_up_in: DevicePtr,
    shared_down: &Fp8Weight,
    shared_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    routed_limit: f32,
    shared_limit: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_weight_ptrs)
        .arg_ptr(down_scale_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(shared_gate_in)
        .arg_ptr(shared_up_in)
        .arg_ptr(shared_down.weight)
        .arg_ptr(shared_down.row_scale)
        .arg_ptr(shared_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_f32(routed_limit)
        .arg_f32(shared_limit)
        .launch(stream)
}

/// Weighted sum of the eight routed down-projection rows.  Unlike the legacy
/// `moe_weighted_sum_blend` ABI this has no shared-expert gate: Ling's shared
/// expert is added as a separate residual before this reduction.
pub(crate) fn ling_weighted_sum(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_out: DevicePtr,
    weights: DevicePtr,
    output: DevicePtr,
    hidden: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_out)
        .arg_ptr(weights)
        .arg_ptr(output)
        .arg_u32(hidden)
        .arg_u32(top_k)
        .launch(stream)
}
