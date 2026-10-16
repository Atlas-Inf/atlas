// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the device-resident QSA decode/verify kernels
//! (`qsa_indexer.cu` "Device-resident decode/verify selection" and
//! `qsa_sel_attn.cu`). Every row's position comes from device memory and
//! every grid is fixed, so each launch is CUDA-graph capturable and one
//! captured graph serves every position, inert or active.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Block-tile width of `qsa_score_rows_dev*`; must match `QSA_SD_BN`.
pub const QSA_SD_BN: u32 = 64;

/// Dynamic shared memory of `qsa_score_rows_dev{bm}`.
pub fn qsa_score_rows_dev_smem(bm: u32, n_heads: u32, hd: u32) -> u32 {
    (bm * n_heads * hd + QSA_SD_BN * (hd + 1)) * 4
}

/// Per-row q prep with positions from `pos_dev[0..rows]`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep_rows_dev(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qk: DevicePtr,
    q_norm_w: DevicePtr,
    q_out: DevicePtr,
    pos_dev: DevicePtr,
    rows: u32,
    qkw: u32,
    n_heads: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_heads, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(qk)
        .arg_ptr(q_norm_w)
        .arg_ptr(q_out)
        .arg_ptr(pos_dev)
        .arg_u32(qkw)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// Block scores for `rows <= bm` rows, grid-stride over 64-block tiles up to
/// the largest row's complete-block count (read on device). `kernel` must be
/// the `qsa_score_rows_dev{bm}` instance.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows_dev(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    bm: u32,
    grid: u32,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    pos_dev: DevicePtr,
    score_stride: u32,
    ratio: u32,
    n_heads: u32,
    hd: u32,
    rows: u32,
    block_topk: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([bm * QSA_SD_BN, 1, 1])
        .shared_mem(qsa_score_rows_dev_smem(bm, n_heads, hd))
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_ptr(pos_dev)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(rows)
        .arg_u32(block_topk)
        .launch(stream)
}

/// Per-row selection list (token positions) and its length.
#[allow(clippy::too_many_arguments)]
pub fn qsa_select_rows_dev(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    scores: DevicePtr,
    sel: DevicePtr,
    nsel: DevicePtr,
    pos_dev: DevicePtr,
    rows: u32,
    score_stride: u32,
    sel_stride: u32,
    block_topk: u32,
    ratio: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(scores)
        .arg_ptr(sel)
        .arg_ptr(nsel)
        .arg_ptr(pos_dev)
        .arg_u32(score_stride)
        .arg_u32(sel_stride)
        .arg_u32(block_topk)
        .arg_u32(ratio)
        .launch(stream)
}

/// Paged decode attention over each row's selection, read straight from
/// the paged cache (no gather). Grid `[nq, rows]`, 256 threads.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_sel(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    out: DevicePtr,
    block_tables: DevicePtr,
    sel: DevicePtr,
    nsel: DevicePtr,
    sel_stride: u32,
    max_blocks_per_seq: u32,
    rows: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([nq, rows, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(out)
        .arg_ptr(block_tables)
        .arg_ptr(sel)
        .arg_ptr(nsel)
        .arg_u32(sel_stride)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(nq)
        .arg_u32(nkv)
        .arg_u32(hd)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .launch(stream)
}
