// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

#[path = "moe_grouped_a/grouped_prefill.rs"]
mod grouped_prefill;
pub use grouped_prefill::*;

/// MoE grouped GEMM: per-expert W4A16 matrix multiply.
pub fn moe_w4a16_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed: DevicePtr,
    b_scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    num_experts: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_experts, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed)
        .arg_ptr(b_scale)
        .arg_f32(scale2)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Element-wise SiLU activation + multiply: `output[i] = silu(gate[i]) * up[i]`.
///
/// Grid: (ceil(total_elements/256), 1, 1)  Block: (256, 1, 1)
pub fn moe_silu_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    total_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(output)
        .arg_u32(total_elements)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::grouped_prefill::ptrtable_legacy_grid_x;

    #[test]
    fn legacy_ptrtable_grid_covers_every_64_column_tile() {
        assert_eq!(ptrtable_legacy_grid_x(1), 1);
        assert_eq!(ptrtable_legacy_grid_x(64), 1);
        assert_eq!(ptrtable_legacy_grid_x(65), 2);
        assert_eq!(ptrtable_legacy_grid_x(1024), 16);
        assert_eq!(ptrtable_legacy_grid_x(3072), 48);
    }
}
