// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-K GLM verify convolution with inline rollback snapshots.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

/// Exact-K=5 GLM verify convolution with inline rollback snapshots.
///
/// `state_inter` is a contiguous FP32 slab containing K-1 states, with
/// `inter_stride` FP32 elements between snapshots.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_prefill_tp_snap(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    bias: DevicePtr,
    output: DevicePtr,
    state_inter: DevicePtr,
    inter_stride: usize,
    d_inner: u32,
    d_conv: u32,
    seq_len: u32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 32), div_ceil(seq_len, 64), 1])
        .block([32, 8, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_ptr(state_inter)
        .arg_u64(inter_stride as u64)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(seq_len)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}
