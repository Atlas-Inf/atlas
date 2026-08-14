// SPDX-License-Identifier: AGPL-3.0-only
//! Torch-free Marlin NVFP4 launches. Opt-in via ATLAS_MOE_MARLIN=1.
//! Kernels: atlas_marlin_nvfp4_m8, atlas_marlin_moe_nvfp4_m8,
//! atlas_marlin_repack_w4, atlas_marlin_align_block8.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Persistent-grid Marlin linear GEMM. M<=8, N%64==0, K%128==0, group=16.
pub fn marlin_nvfp4_m8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    c_tmp: DevicePtr,
    scales: DevicePtr,
    global_scale: DevicePtr,
    locks: DevicePtr,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    num_groups: i32,
    sms: u32,
    smem: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([sms, 1, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_ptr(c_tmp)
        .arg_ptr(DevicePtr::NULL) // bias
        .arg_ptr(DevicePtr::NULL) // a_scales
        .arg_ptr(scales)
        .arg_ptr(global_scale)
        .arg_ptr(DevicePtr::NULL) // zp
        .arg_ptr(DevicePtr::NULL) // g_idx
        .arg_i32(num_groups)
        .arg_i32(m)
        .arg_i32(n)
        .arg_i32(k)
        .arg_i32(lda)
        .arg_ptr(locks)
        .arg_i32(0) // has_bias
        .arg_i32(1) // use_atomic_add
        .arg_i32(1) // use_fp32_reduce
        .arg_i32(smem as i32)
        .launch(stream)
}

pub fn marlin_repack_w4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qweight_int32_t: DevicePtr,
    out: DevicePtr,
    size_k: i32,
    size_n: i32,
    sms: u32,
    smem: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([sms, 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(qweight_int32_t)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(out)
        .arg_i32(size_k)
        .arg_i32(size_n)
        .launch(stream)
}

pub fn marlin_align_block8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    topk_ids: DevicePtr,
    sorted_token_ids: DevicePtr,
    expert_ids: DevicePtr,
    num_tokens_post_pad: DevicePtr,
    tokens: i32,
    top_k: i32,
    num_experts: i32,
    sorted_cap: i32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(topk_ids)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(expert_ids)
        .arg_ptr(num_tokens_post_pad)
        .arg_i32(tokens)
        .arg_i32(top_k)
        .arg_i32(num_experts)
        .arg_i32(sorted_cap)
        .launch(stream)
}
