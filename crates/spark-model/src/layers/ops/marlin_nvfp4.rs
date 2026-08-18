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

/// Sorted-row gather for prefill Marlin: dst\[row\] = src\[sorted_token_ids\[row\]\].
/// Grid: (te, 1, 1) Block: (256, 1, 1)
pub fn marlin_pack_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    sorted_token_ids: DevicePtr,
    src: DevicePtr,
    dst: DevicePtr,
    te: i32,
    hidden_size: i32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([te as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(sorted_token_ids)
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_i32(te)
        .arg_i32(hidden_size)
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

#[allow(clippy::too_many_arguments)]
pub fn marlin_moe_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    c_tmp: DevicePtr,
    scales: DevicePtr,
    global_scale: DevicePtr,
    sorted_ids: DevicePtr,
    expert_ids: DevicePtr,
    n_post: DevicePtr,
    top_k: i32,
    num_groups: i32,
    prob_m: i32,
    prob_n: i32,
    prob_k: i32,
    locks: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([SMS_DEFAULT, 1, 1])
        .block([128, 1, 1])
        .shared_mem(96 * 1024)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_ptr(c_tmp)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(scales)
        .arg_ptr(global_scale)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(sorted_ids)
        .arg_ptr(expert_ids)
        .arg_ptr(n_post)
        .arg_ptr(DevicePtr::NULL)
        .arg_i32(top_k)
        .arg_i32(0)
        .arg_i32(num_groups)
        .arg_i32(prob_m)
        .arg_i32(prob_n)
        .arg_i32(prob_k)
        .arg_ptr(locks)
        .arg_i32(0)
        .arg_i32(1)
        .arg_i32(1)
        .launch(stream)
}

// Native DSpark B8 verifies up to 32 rows × top_k=6. One fixed-M8 slot can
// hold eight hits for an expert; 128 slots cover all 128 experts plus every
// possible overflow chunk in the R<=32 product envelope. Must match CUDA.
pub const MARLIN_SLOTS: i32 = 128;
pub const MARLIN_M_TILE: i32 = 8;

#[allow(clippy::too_many_arguments)]
pub fn marlin_pack_slots(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    topk_ids: DevicePtr,
    hidden: DevicePtr,
    slot_eids: DevicePtr,
    slot_map: DevicePtr,
    slot_a: DevicePtr,
    n_slots: DevicePtr,
    tokens: i32,
    top_k: i32,
    num_experts: i32,
    hidden_size: i32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(topk_ids)
        .arg_ptr(hidden)
        .arg_ptr(slot_eids)
        .arg_ptr(slot_map)
        .arg_ptr(slot_a)
        .arg_ptr(n_slots)
        .arg_i32(tokens)
        .arg_i32(top_k)
        .arg_i32(num_experts)
        .arg_i32(hidden_size)
        .launch(stream)
}

pub fn marlin_scatter_slots(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    slot_c: DevicePtr,
    slot_map: DevicePtr,
    out: DevicePtr,
    hidden_size: i32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([MARLIN_SLOTS as u32, MARLIN_M_TILE as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(slot_c)
        .arg_ptr(slot_map)
        .arg_ptr(out)
        .arg_i32(hidden_size)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn marlin_nvfp4_m8_slot(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_base: DevicePtr,
    b_base: DevicePtr,
    c_base: DevicePtr,
    c_tmp: DevicePtr,
    scales_base: DevicePtr,
    gs_base: DevicePtr,
    expert_ids: DevicePtr,
    slot: i32,
    num_groups: i32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    locks: DevicePtr,
    sms: u32,
    smem: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([sms, 1, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(a_base)
        .arg_ptr(b_base)
        .arg_ptr(c_base)
        .arg_ptr(c_tmp)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(scales_base)
        .arg_ptr(gs_base)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(expert_ids)
        .arg_i32(slot)
        .arg_i32(num_groups)
        .arg_i32(m)
        .arg_i32(n)
        .arg_i32(k)
        .arg_i32(lda)
        .arg_ptr(locks)
        .arg_i32(0)
        .arg_i32(1)
        .arg_i32(1)
        .arg_i32(smem as i32)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn marlin_nvfp4_m8_allslots(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_base: DevicePtr,
    b_base: DevicePtr,
    c_base: DevicePtr,
    c_tmp: DevicePtr,
    scales_base: DevicePtr,
    gs_base: DevicePtr,
    expert_ids: DevicePtr,
    n_slots: DevicePtr,
    bars: DevicePtr,
    num_groups: i32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    locks: DevicePtr,
    _sms: u32,
    smem: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([16, MARLIN_SLOTS as u32, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(a_base)
        .arg_ptr(b_base)
        .arg_ptr(c_base)
        .arg_ptr(c_tmp)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(scales_base)
        .arg_ptr(gs_base)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(expert_ids)
        .arg_ptr(n_slots)
        .arg_ptr(bars)
        .arg_i32(num_groups)
        .arg_i32(m)
        .arg_i32(n)
        .arg_i32(k)
        .arg_i32(lda)
        .arg_ptr(locks)
        .arg_i32(0)
        .arg_i32(1)
        .arg_i32(1)
        .arg_i32(smem as i32)
        .launch(stream)
}

const SMS_DEFAULT: u32 = 48;
