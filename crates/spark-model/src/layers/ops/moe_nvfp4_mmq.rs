// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the GB10 grouped routed-expert NVFP4 MMQ path.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{QK_NVFP4, nvfp4_mmq_smem};

pub const MOE_NVFP4_MMQ_M_TILE: u32 = 64;
const QUANT_THREADS: u32 = 128;
const FP4_MMQ_BLOCK_VALUES: u32 = 256;

/// Equal-size MMQ representation: 32 packed bytes + four scale bytes per K64.
pub fn moe_nvfp4_mmq_weight_bytes(n: u32, k: u32) -> usize {
    n as usize * (k as usize / QK_NVFP4 as usize) * 36
}

#[allow(clippy::too_many_arguments)]
pub fn moe_nvfp4_mmq_repack_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    out_ptrs: DevicePtr,
    n: u32,
    k: u32,
    num_experts: u32,
    stream: u64,
) -> Result<()> {
    let blocks = n * (k / QK_NVFP4);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(blocks, 256), num_experts, 1])
        .block([256, 1, 1])
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(out_ptrs)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_experts)
        .launch(stream)
}

/// Quantize contiguous BF16 sorted rows into block_fp4_mmq K-block-major form.
pub fn moe_nvfp4_mmq_quantize(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    sorted_token_ids: DevicePtr,
    output: DevicePtr,
    rows: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let kpad = div_ceil(k, FP4_MMQ_BLOCK_VALUES) * FP4_MMQ_BLOCK_VALUES;
    KernelLaunch::new(gpu, kernel)
        .grid([rows, div_ceil(kpad, 16 * QUANT_THREADS), 1])
        .block([QUANT_THREADS, 1, 1])
        .arg_ptr(input)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(output)
        .arg_u64(k as u64)
        .arg_u64(k as u64)
        .arg_u64(kpad as u64)
        .arg_u32(rows)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_nvfp4_mmq_gate_up(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_ptrs: DevicePtr,
    up_ptrs: DevicePtr,
    input_fp4: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    expert_offsets: DevicePtr,
    num_experts: u32,
    rows: u32,
    n: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), max_m_tiles, num_experts * 2])
        .block([32, 8, 1])
        .shared_mem(nvfp4_mmq_smem(MOE_NVFP4_MMQ_M_TILE))
        .arg_ptr(gate_ptrs)
        .arg_ptr(up_ptrs)
        .arg_ptr(input_fp4)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(expert_offsets)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(rows)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_nvfp4_mmq_down(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    down_ptrs: DevicePtr,
    input_fp4: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    num_experts: u32,
    rows: u32,
    n: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), max_m_tiles, num_experts])
        .block([32, 8, 1])
        .shared_mem(nvfp4_mmq_smem(MOE_NVFP4_MMQ_M_TILE))
        .arg_ptr(down_ptrs)
        .arg_ptr(input_fp4)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(rows)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_nvfp4_mmq_silu_scale2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    gate_scale2: DevicePtr,
    up_scale2: DevicePtr,
    expert_offsets: DevicePtr,
    width: u32,
    max_rows: u32,
    num_experts: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([max_rows, num_experts, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(output)
        .arg_ptr(gate_scale2)
        .arg_ptr(up_scale2)
        .arg_ptr(expert_offsets)
        .arg_u32(width)
        .arg_u32(max_rows)
        .arg_u32(num_experts)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_nvfp4_mmq_scale2_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    scale2: DevicePtr,
    expert_offsets: DevicePtr,
    width: u32,
    max_rows: u32,
    num_experts: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([max_rows, num_experts, 1])
        .block([256, 1, 1])
        .arg_ptr(data)
        .arg_ptr(scale2)
        .arg_ptr(expert_offsets)
        .arg_u32(width)
        .arg_u32(max_rows)
        .arg_u32(num_experts)
        .launch(stream)
}
