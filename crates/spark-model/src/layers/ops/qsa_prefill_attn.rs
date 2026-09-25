// SPDX-License-Identifier: AGPL-3.0-only

//! QSA selected-set PREFILL ATTENTION launchers (scalar, head-grouped and
//! tensor-core arms). Split out of `ops/qsa_rows.rs` for the 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

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
        && nq.is_multiple_of(QSA_PA_G)
        && (nq / nkv).is_multiple_of(QSA_PA_G)
        // The runtime opts in to >48 KB dynamic shared automatically
        // (registry.rs sets CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
        // so the old 48 KB cap here was self-imposed, not a hardware limit.
        && qsa_prefill_attn_g_smem(hd) <= 96 * 1024
}

fn qsa_prefill_attn_g_smem(hd: u32) -> u32 {
    (8 * QSA_PA_G * hd + 2 * 8 * QSA_PA_G) * 4
}

/// Same geometry rule as [`qsa_prefill_attn_tc2_ok`]; tc3 differs only in tile
/// and buffering (BC=32, single-buffered K).
pub fn qsa_prefill_attn_tc3_ok(nq: u32, nkv: u32, hd: u32) -> bool {
    qsa_prefill_attn_tc2_ok(nq, nkv, hd)
}

/// Stage 2 on tensor cores, both kv heads, BC=32. tc2's packing with tc1's
/// tile, paid for by single-buffering K (a second K buffer at BC=32 would be
/// 121,088 B of static shared and the driver caps a block at 101,376).
#[allow(clippy::too_many_arguments)]
pub fn qsa_prefill_attn_tc3(
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
        .grid([1, rows, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(attn_out)
        .arg_ptr(block_table)
        .arg_ptr(lists)
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

/// Score-tile dimensions of [`qsa_score_rows_tc`]; must match `QSC_BM` /
/// `QSC_BN` in `qsa_score_tc.cu`.
pub const QSC_BM: u32 = 16;
pub const QSC_BN: u32 = 32;

fn qsa_score_rows_tc_smem(n_heads: u32, hd: u32) -> u32 {
    let ldq = hd + 8;
    (n_heads * QSC_BM * ldq + QSC_BN * ldq) * 2 + n_heads * QSC_BM * QSC_BN * 4
}

/// Whether the tensor-core scorer can serve this geometry.
///
/// One warp per indexer head, so exactly 4 heads (the block is 128 threads),
/// and `hd` a multiple of the 16-wide MMA k-step.
pub fn qsa_score_rows_tc_ok(n_heads: u32, hd: u32) -> bool {
    n_heads == 4 && hd.is_multiple_of(16) && qsa_score_rows_tc_smem(n_heads, hd) <= 96 * 1024
}

/// Stage 1 on TENSOR CORES. Same scores as [`qsa_score_rows_exact`](super::qsa_score_rows_exact) up to the
/// MMA's contraction order and a BF16 Q; `block_keys` is already BF16.
///
/// NOT bit-identical, so it is gated behind `ATLAS_QSA_SCORE_TC` and needs
/// `scripts/ppl.py`. TTFT_GAP.md 34 prices the BF16 Q half on its own.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows_tc(
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
        .grid([rows.div_ceil(QSC_BM), n_blocks_max.div_ceil(QSC_BN), 1])
        .block([128, 1, 1])
        .shared_mem(qsa_score_rows_tc_smem(n_heads, hd))
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(rows)
        .arg_u32(n_blocks_max)
        .launch(stream)
}

/// Whether the two-kv-head tensor-core attention can serve this geometry.
///
/// One CTA per ROW, holding BOTH kv heads: rows 0..gqa-1 are kv head 0's
/// q-heads and rows 16..16+gqa-1 kv head 1's, which is the split the kernel's
/// inherited warp mapping already makes at 16. So exactly two kv heads, and the
/// group must fit the 16-row half.
pub fn qsa_prefill_attn_tc2_ok(nq: u32, nkv: u32, hd: u32) -> bool {
    nkv == 2 && nq.is_multiple_of(nkv) && nq / nkv <= 16 && hd == 256
}

/// Stage 2 on tensor cores, both kv heads per CTA. Same selected set and same
/// per-row semantics as [`qsa_prefill_attn_tc`]; it halves that kernel's M
/// padding (2.67x -> 1.33x) by filling the second 16-row half of the tile.
#[allow(clippy::too_many_arguments)]
pub fn qsa_prefill_attn_tc2(
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
        .grid([1, rows, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(attn_out)
        .arg_ptr(block_table)
        .arg_ptr(lists)
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

/// Whether the tensor-core attention can serve this geometry.
///
/// One CTA per (row, kv head); the M tile holds that kv head's `nq / nkv`
/// q-heads, so the group must fit `BR = 32`. `hd` must be the 256 the kernel's
/// `HDIM` is compiled at -- a narrower head would read the wrong columns.
pub fn qsa_prefill_attn_tc_ok(nq: u32, nkv: u32, hd: u32) -> bool {
    nkv != 0 && nq.is_multiple_of(nkv) && nq / nkv <= 32 && hd == 256
}

/// Stage 2 on TENSOR CORES. Same selected set and same per-row semantics as
/// [`qsa_prefill_attn_g`], but `S = Q @ K^T` and `O = P @ V` run as
/// `mma.sync.m16n8k16` with the ROW held fixed and the q-heads as the M tile.
///
/// NOT bit-identical (a different summation tree), so it is gated behind
/// `ATLAS_QSA_ATTN_TC` and needs `scripts/ppl.py`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_prefill_attn_tc(
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
        .grid([nkv, rows, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(attn_out)
        .arg_ptr(block_table)
        .arg_ptr(lists)
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

/// Whether the 8-lanes-per-head variant can serve this geometry.
///
/// Everything `qsa_prefill_attn_grouped_ok` needs, plus: the warp must split
/// into `QSA_PA_G` groups of 8 (so `QSA_PA_G * 8 == 32`), and each lane's slice
/// `hd / 8` must be a whole number of 16-byte chunks and fit `QSA_L8_EV`.
pub fn qsa_prefill_attn_l8_ok(nq: u32, nkv: u32, hd: u32) -> bool {
    qsa_prefill_attn_grouped_ok(nq, nkv, hd)
        && QSA_PA_G * 8 == 32
        && hd.is_multiple_of(64)
        && hd / 8 <= 32
}

/// Stage 2, 8 lanes per head. Same selected set, same per-head online softmax
/// and same cross-warp merge as [`qsa_prefill_attn_g`]; it reduces each head
/// across 8 lanes instead of 32, which is three butterfly levels instead of
/// five and three shuffle instructions per warp-key instead of twenty.
///
/// NOT bit-identical -- the dot-product summation tree changes -- so it is
/// gated behind `ATLAS_QSA_ATTN_L8` and needs `scripts/ppl.py`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_prefill_attn_l8(
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
