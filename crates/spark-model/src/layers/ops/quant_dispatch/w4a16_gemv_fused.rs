// SPDX-License-Identifier: AGPL-3.0-only

//! W4A16 GEMVs with fused Q/Gate or QKVZ deinterleave, and dual-projection batch GEMVs.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// W4A16 GEMV with inline Q/Gate deinterleave on output write.
///
/// Same as `w4a16_gemv` but writes Q and Gate to deinterleaved positions,
/// eliminating the separate `deinterleave_qg` kernel (12 graph nodes saved).
///
/// Kernel: `w4a16_gemv_qg(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
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
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// W4A16 GEMV with inline QKVZ deinterleave on output write.
///
/// Same as `w4a16_gemv` but writes to deinterleaved output locations,
/// eliminating the separate `deinterleave_qkvz` kernel.
///
/// Kernel: `w4a16_gemv_qkvz(A, B, S, s2, C, N, K, ng, kd, vpg, vd)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qkvz(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_groups: u32,
    head_k_dim: u32,
    vheads_per_group: u32,
    head_v_dim: u32,
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
        .arg_u32(num_groups)
        .arg_u32(head_k_dim)
        .arg_u32(vheads_per_group)
        .arg_u32(head_v_dim)
        .launch(stream)
}

/// Q+Gate GEMV for 2 tokens with inline deinterleave.
///
/// Reads the Q+Gate weight matrix once, produces 2 deinterleaved output
/// vectors (Q|Gate for each token). Replaces 2× `w4a16_gemv_qg` calls.
///
/// Kernel: `w4a16_gemv_qg_batch2(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
/// Input A: [2, K], Output C: [2, N] deinterleaved [Q|G] per token.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
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
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// W4A16 GEMV batch3 with inline Q/Gate deinterleave.
///
/// Reads the Q+Gate weight matrix once, produces 3 deinterleaved output
/// vectors (Q|Gate for each token). For K=3 speculative verification.
///
/// Kernel: `w4a16_gemv_qg_batch3(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
/// Input A: [3, K], Output C: [3, N] deinterleaved [Q|G] per token.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
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
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// [`w4a16_gemv_qg_batch3`] at 4 rows (`w4a16_gemv_qg_batch4`): the same
/// launch geometry and argument list, so it shares the launcher. Row `r` is
/// byte-identical to `w4a16_gemv_qg` on row `r` (qwen4_exp exact K=4 verify).
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    w4a16_gemv_qg_batch3(
        gpu, kernel, input, weight, output, n, k, num_heads, head_dim, stream,
    )
}

/// Dual-projection GEMV for 3 tokens (K+V or any 2 weight matrices).
///
/// Reads each weight matrix once, produces 3 output vectors per projection.
/// `blockIdx.z` selects projection 0 or 1.
///
/// Kernel: `w4a16_gemv_dual_batch3(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
/// Input A: [3, K], Output C0: [3, N], C1: [3, N].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Dual-projection GEMV for 2 tokens (K+V or any 2 weight matrices).
///
/// Reads each weight matrix once, produces 2 output vectors per projection.
/// `blockIdx.z` selects projection 0 or 1.
///
/// Kernel: `w4a16_gemv_dual_batch2(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
/// Input A: [2, K], Output C0: [2, N], C1: [2, N].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
