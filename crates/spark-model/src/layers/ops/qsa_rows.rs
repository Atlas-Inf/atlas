// SPDX-License-Identifier: AGPL-3.0-only

//! Per-ROW (prefill) QSA launchers: row scoring, device top-k and the
//! selected-set prefill attention (scalar, grouped and tensor-core arms).
//! Split out of `ops/qsa.rs` for the 500-LoC cap; same semantics.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

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

/// Score-tile dimensions of [`qsa_score_rows_exact`]; must match `QSA_SE_BM` /
/// `QSA_SE_BN` in `qsa_indexer.cu`.
pub const QSA_SE_BM: u32 = 8;
pub const QSA_SE_BN: u32 = 32;

fn qsa_score_rows_exact_smem(n_heads: u32, hd: u32) -> u32 {
    (QSA_SE_BM * n_heads * hd + QSA_SE_BN * (hd + 1)) * 4
}

/// Whether the exact-tree scorer fits this geometry. `hd` must be a multiple of
/// 32 — the kernel replays the reference's per-warp reduction tree, and a warp
/// is 32 lanes.
pub fn qsa_score_rows_exact_ok(n_heads: u32, hd: u32) -> bool {
    hd.is_multiple_of(32) && qsa_score_rows_exact_smem(n_heads, hd) <= 48 * 1024
}

/// Per-row block scores, one thread per score, BIT-IDENTICAL to
/// [`qsa_score_rows`].
///
/// Evaluates the reference's `__shfl_down_sync` reduction tree locally in one
/// thread instead of across 128 of them, so the FP addition DAG is unchanged
/// while every block-wide reduction and `__syncthreads` disappears. See the
/// kernel note for the tree and for why the `__fmul_rn`/`__fadd_rn` intrinsics
/// are required rather than stylistic.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows_exact(
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
        .grid([
            rows.div_ceil(QSA_SE_BM),
            n_blocks_max.div_ceil(QSA_SE_BN),
            1,
        ])
        .block([QSA_SE_BM * QSA_SE_BN, 1, 1])
        .shared_mem(qsa_score_rows_exact_smem(n_heads, hd))
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

/// Score-tile dimensions of [`qsa_score_rows_gemm`]; must match `QSA_SG_BM` /
/// `QSA_SG_BN` in `qsa_indexer.cu`, and their product is the block width.
pub const QSA_SG_BM: u32 = 8;
pub const QSA_SG_BN: u32 = 32;

fn qsa_score_rows_gemm_smem(n_heads: u32, hd: u32) -> u32 {
    (QSA_SG_BM * n_heads * hd + QSA_SG_BN * (hd + 1)) * 4
}

/// Whether the tiled-GEMM scorer fits this geometry.
pub fn qsa_score_rows_gemm_ok(n_heads: u32, hd: u32) -> bool {
    qsa_score_rows_gemm_smem(n_heads, hd) <= 48 * 1024
}

/// Per-row block scores as a tiled GEMM: one thread per score, contracting
/// serially over `hd`, with no block-wide reductions at all.
///
/// NOT bit-identical to [`qsa_score_rows`] — the `d` contraction is
/// reassociated. Gate with `lc_subset.py` + `kl_drift.py --precision-change`
/// and `lc_check.py` needle recall; these scores choose which blocks a query
/// reads, so a KL check alone is not sufficient evidence.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows_gemm(
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
        .grid([
            rows.div_ceil(QSA_SG_BM),
            n_blocks_max.div_ceil(QSA_SG_BN),
            1,
        ])
        .block([QSA_SG_BM * QSA_SG_BN, 1, 1])
        .shared_mem(qsa_score_rows_gemm_smem(n_heads, hd))
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

/// `b` values per block in [`qsa_score_rows_b`]; must match `QSA_SR_B` in
/// `qsa_indexer.cu`.
pub const QSA_SR_B: u32 = 16;

/// Shared bytes [`qsa_score_rows_b`] needs: the reduction scratch plus one
/// staged copy of the row's `q`.
fn qsa_score_rows_b_smem(n_heads: u32, hd: u32) -> u32 {
    (32 + n_heads * hd) * 4
}

/// Whether the tiled scorer fits this geometry.
pub fn qsa_score_rows_b_ok(n_heads: u32, hd: u32) -> bool {
    qsa_score_rows_b_smem(n_heads, hd) <= 48 * 1024
}

/// Per-row block scores, QSA_SR_B outputs per block.
///
/// Identical arithmetic to [`qsa_score_rows`] -- same reduction over the same
/// 128 threads, same `hh` order, same scale -- with `q` staged in shared once
/// per block instead of re-read per output, and QSA_SR_B fewer blocks. See the
/// kernel note for why bit-identity is a requirement here and not a nicety.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows_b(
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
        .grid([rows, n_blocks_max.div_ceil(QSA_SR_B), 1])
        .block([hd, 1, 1])
        .shared_mem(qsa_score_rows_b_smem(n_heads, hd))
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(n_blocks_max)
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
