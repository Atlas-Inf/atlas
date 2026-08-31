// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the Qwen3.8-Flash-Next QSA indexer kernels
//! (`qsa_indexer.cu`): block-key pooling, decode-query prep, block scoring
//! and the selected-token K/V gather. See the .cu header for the semantics
//! and the scratch-as-paged-cache trick that lets the EXISTING paged decode
//! attention consume the selection.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Pool `n_new` freshly complete blocks starting at `first_block`:
/// mean over `ratio` raw keys -> RMSNorm*(1+w) -> rope at block-start pos.
#[allow(clippy::too_many_arguments)]
pub fn qsa_block_pool(
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
    stream: u64,
) -> Result<()> {
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
        .launch(stream)
}

/// One decode query: per head, RMSNorm*(1+w) + partial rope at `pos` -> FP32.
#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep(
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
    stream: u64,
) -> Result<()> {
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
        .launch(stream)
}

/// `scores[b] = sum_h relu(q_h . k_b) / sqrt(hd)` over `n_blocks` blocks.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    n_blocks: u32,
    n_heads: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks, 1, 1])
        .block([hd, 1, 1])
        .shared_mem(32 * 4)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .launch(stream)
}

/// Device-side decode block selection (`qsa_select_topk`): writes the
/// expanded selection straight into `sel`, replacing the D2H + host sort +
/// H2D round trip. One block of 1024 threads; `complete <= 4096`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_select_topk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    scores: DevicePtr,
    sel: DevicePtr,
    complete: u32,
    block_topk: u32,
    ratio: u32,
    tail_start: u32,
    visible: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(scores)
        .arg_ptr(sel)
        .arg_u32(complete)
        .arg_u32(block_topk)
        .arg_u32(ratio)
        .arg_u32(tail_start)
        .arg_u32(visible)
        .launch(stream)
}

/// Pack the selected tokens' K/V rows into contiguous NHD scratch.
#[allow(clippy::too_many_arguments)]
pub fn qsa_gather(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    block_table: DevicePtr,
    sel: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    n_sel: u32,
    block_size: u32,
    nkv: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_sel, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(block_table)
        .arg_ptr(sel)
        .arg_ptr(k_out)
        .arg_ptr(v_out)
        .arg_u32(block_size)
        .arg_u32(nkv)
        .arg_u32(hd)
        .launch(stream)
}

/// Stage 2: per-row q prep for a contiguous selective row range.
#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep_rows(
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
    stream: u64,
) -> Result<()> {
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
        .launch(stream)
}

/// Stage 2: per-row block scores, -inf beyond each row's complete count.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    rows: u32,
    n_blocks_max: u32,
    first_pos: u32,
    score_stride: u32,
    ratio: u32,
    n_heads: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_blocks_max, 1])
        .block([hd, 1, 1])
        .shared_mem(32 * 4)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .launch(stream)
}

/// Block width of [`qsa_topk_rows`]; must match `QSA_TOPK_K` in
/// `qsa_indexer.cu`, and bounds the `topk` it can serve.
pub const QSA_TOPK_K: u32 = 512;

/// Whether the GPU selection can serve this shape. `topk` must fit the running
/// best-K the kernel keeps in shared memory; anything wider falls back to the
/// host path, which has no such bound.
pub fn qsa_topk_rows_ok(topk: u32) -> bool {
    topk > 0 && topk <= QSA_TOPK_K
}

/// Stage 1B: per-row top-k block selection, on the GPU.
///
/// Replaces a D2H of the whole score matrix, a host sort per row and an H2D of
/// the lists — a full stream drain per attention layer per slab, measured at
/// 7.3 s of DEAD GPU on a 30k prefill. Produces the identical list, in the
/// identical order; see the kernel note for why that is by construction.
#[allow(clippy::too_many_arguments)]
pub fn qsa_topk_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    scores: DevicePtr,
    lists: DevicePtr,
    rows: u32,
    first_pos: u32,
    score_stride: u32,
    ratio: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([QSA_TOPK_K, 1, 1])
        .arg_ptr(scores)
        .arg_ptr(lists)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(topk)
        .launch(stream)
}

/// Stage 2: per-row selected-set attention, overwriting the context rows.
#[allow(clippy::too_many_arguments)]
pub fn qsa_prefill_attn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    block_table: DevicePtr,
    lists: DevicePtr,
    attn_out: DevicePtr,
    rows: u32,
    first_pos: u32,
    topk: u32,
    ratio: u32,
    block_size: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    // 8 warps x [hd] acc partials + m/l per warp.
    let smem = (8 * hd + 16) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([rows, nq, 1])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(block_table)
        .arg_ptr(lists)
        .arg_ptr(attn_out)
        .arg_u32(first_pos)
        .arg_u32(topk)
        .arg_u32(ratio)
        .arg_u32(block_size)
        .arg_u32(nq)
        .arg_u32(nkv)
        .arg_u32(hd)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}

/// Heads per block in [`qsa_prefill_attn_g`]. Must match `QSA_PA_G` in
/// `qsa_indexer.cu`.
pub const QSA_PA_G: u32 = 4;

/// Whether the grouped kernel can serve this head geometry.
///
/// Two conditions, both structural. `nq % G == 0` so the grid divides evenly,
/// and `(nq / nkv) % G == 0` so every head in a group maps to the SAME kv head
/// -- the kernel derives `kvh` from the group's first head. The merge buffer is
/// `[8][G][hd]` floats plus `m`/`l`, which must fit the 48 KB block limit.
pub fn qsa_prefill_attn_grouped_ok(nq: u32, nkv: u32, hd: u32) -> bool {
    nkv != 0
        && nq % QSA_PA_G == 0
        && (nq / nkv) % QSA_PA_G == 0
        && qsa_prefill_attn_g_smem(hd) <= 48 * 1024
}

fn qsa_prefill_attn_g_smem(hd: u32) -> u32 {
    (8 * QSA_PA_G * hd + 2 * 8 * QSA_PA_G) * 4
}

/// Stage 2, `QSA_PA_G` q-heads per block. Same math and same accumulation
/// order as [`qsa_prefill_attn`]; it shares each K/V row across the group
/// instead of re-reading it per head.
#[allow(clippy::too_many_arguments)]
pub fn qsa_prefill_attn_g(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    block_table: DevicePtr,
    lists: DevicePtr,
    attn_out: DevicePtr,
    rows: u32,
    first_pos: u32,
    topk: u32,
    ratio: u32,
    block_size: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, nq / QSA_PA_G, 1])
        .block([256, 1, 1])
        .shared_mem(qsa_prefill_attn_g_smem(hd))
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(block_table)
        .arg_ptr(lists)
        .arg_ptr(attn_out)
        .arg_u32(first_pos)
        .arg_u32(topk)
        .arg_u32(ratio)
        .arg_u32(block_size)
        .arg_u32(nq)
        .arg_u32(nkv)
        .arg_u32(hd)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}
