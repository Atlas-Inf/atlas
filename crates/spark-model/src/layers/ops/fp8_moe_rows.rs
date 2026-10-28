// SPDX-License-Identifier: AGPL-3.0-only
//! Multi-row dedup wrappers for the transposed (`*_ptrs_t`) NVFP4 MoE tables:
//! `moe_expert_union_t` builds the unique-expert set of K rows on GPU, and the
//! `_rows` kernels read each unique expert's packed tile once while keeping
//! every (row, slot) output bit-identical to K single-row launches.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::weight_map::QuantizedWeight;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// `K * top_k` routed ids -> unique experts, row masks and per-row slot map.
pub fn moe_expert_union_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_indices: DevicePtr,
    u_experts: DevicePtr,
    u_rowmask: DevicePtr,
    u_rowslot: DevicePtr,
    u_count: DevicePtr,
    k_rows: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(expert_indices)
        .arg_ptr(u_experts)
        .arg_ptr(u_rowmask)
        .arg_ptr(u_rowslot)
        .arg_ptr(u_count)
        .arg_u32(k_rows)
        .arg_u32(top_k)
        .launch(stream)
}

/// Fused gate+up over the unique experts of `rows` rows (transposed tables).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_t_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr, // [rows, h]
    gate_packed_t_ptrs: DevicePtr,
    gate_scale_t_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr, // [rows * top_k, n]
    up_packed_t_ptrs: DevicePtr,
    up_scale_t_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    u_experts: DevicePtr,
    u_rowmask: DevicePtr,
    u_rowslot: DevicePtr,
    u_count: DevicePtr,
    u_max: u32, // grid.y bound: K * top_k (+1 for the shared slot)
    sh_gate_t: &QuantizedWeight,
    sh_gate_out: DevicePtr, // [rows, n]
    sh_up_t: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    h: u32,
    rows: u32,
    block: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, block), u_max + 1, 2])
        .block([block, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_t_ptrs)
        .arg_ptr(gate_scale_t_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_t_ptrs)
        .arg_ptr(up_scale_t_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(u_experts)
        .arg_ptr(u_rowmask)
        .arg_ptr(u_rowslot)
        .arg_ptr(u_count)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.weight_scale)
        .arg_f32(sh_gate_t.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.weight_scale)
        .arg_f32(sh_up_t.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(h)
        .arg_u32(rows)
        .launch(stream)
}

/// Fused SiLU+down over the unique experts of `rows` rows (transposed).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_t_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr, // [rows * top_k, k]
    up_out: DevicePtr,
    packed_t_ptrs: DevicePtr,
    scale_t_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr, // [rows * top_k, n]
    u_experts: DevicePtr,
    u_rowmask: DevicePtr,
    u_rowslot: DevicePtr,
    u_count: DevicePtr,
    u_max: u32,
    sh_gate_in: DevicePtr, // [rows, k]
    sh_up_in: DevicePtr,
    sh_down_t: &QuantizedWeight,
    sh_down_out: DevicePtr, // [rows, n]
    n: u32,
    k: u32,
    top_k: u32,
    rows: u32,
    block: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, block), u_max + 1, 1])
        .block([block, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_t_ptrs)
        .arg_ptr(scale_t_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(u_experts)
        .arg_ptr(u_rowmask)
        .arg_ptr(u_rowslot)
        .arg_ptr(u_count)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.weight_scale)
        .arg_f32(sh_down_t.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(rows)
        .launch(stream)
}
