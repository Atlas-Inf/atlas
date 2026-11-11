// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

#[allow(clippy::too_many_arguments)]
pub fn qsa_block_pool_scaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    raw_keys: DevicePtr,
    k_norm_w: DevicePtr,
    block_keys: DevicePtr,
    first_block: u32,
    n_new: u32,
    ratio: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    inv_freq: DevicePtr,
    attention_factor: f32,
    stream: u64,
) -> Result<()> {
    if inv_freq.is_null() {
        return super::qsa_block_pool(
            gpu,
            kernel,
            raw_keys,
            k_norm_w,
            block_keys,
            first_block,
            n_new,
            ratio,
            hd,
            rot,
            theta,
            eps,
            stream,
        );
    }
    if n_new == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([n_new, 1, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(raw_keys)
        .arg_ptr(k_norm_w)
        .arg_ptr(block_keys)
        .arg_u32(first_block)
        .arg_u32(ratio)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .arg_ptr(inv_freq)
        .arg_f32(attention_factor)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep_scaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_in: DevicePtr,
    q_norm_w: DevicePtr,
    q_out: DevicePtr,
    n_heads: u32,
    hd: u32,
    rot: u32,
    pos: u32,
    theta: f32,
    eps: f32,
    inv_freq: DevicePtr,
    attention_factor: f32,
    stream: u64,
) -> Result<()> {
    if inv_freq.is_null() {
        return super::qsa_qprep(
            gpu, kernel, q_in, q_norm_w, q_out, n_heads, hd, rot, pos, theta, eps, stream,
        );
    }
    KernelLaunch::new(gpu, kernel)
        .grid([n_heads, 1, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(q_in)
        .arg_ptr(q_norm_w)
        .arg_ptr(q_out)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_u32(pos)
        .arg_f32(theta)
        .arg_f32(eps)
        .arg_ptr(inv_freq)
        .arg_f32(attention_factor)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep_rows_scaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qk: DevicePtr,
    q_norm_w: DevicePtr,
    q_out: DevicePtr,
    rows: u32,
    first_pos: u32,
    qkw: u32,
    n_heads: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    inv_freq: DevicePtr,
    attention_factor: f32,
    stream: u64,
) -> Result<()> {
    if inv_freq.is_null() {
        return super::qsa_qprep_rows(
            gpu, kernel, qk, q_norm_w, q_out, rows, first_pos, qkw, n_heads, hd, rot, theta, eps,
            stream,
        );
    }
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_heads, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(qk)
        .arg_ptr(q_norm_w)
        .arg_ptr(q_out)
        .arg_u32(first_pos)
        .arg_u32(qkw)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .arg_ptr(inv_freq)
        .arg_f32(attention_factor)
        .launch(stream)
}
