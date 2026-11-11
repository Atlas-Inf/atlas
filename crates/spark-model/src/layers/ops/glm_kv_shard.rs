// SPDX-License-Identifier: AGPL-3.0-only
//! Launchers for `glm_kv_shard.cu`: the device half of
//! `spark_runtime::kv_cache::LatentShard`'s block-ownership rule
//! (`ATLAS_GLM_KV_SHARD=1`) and the selection split of the canonical form an
//! unsharded pair runs for the same owners (`glm_sparse_canonical.rs`). The
//! module's attention entry points (counted and paired splits, FP32,
//! extra-partition and paired merges) are launched beside their unsharded
//! twins in `glm_sparse_prefill_split.rs` and `glm_sparse_decode_split.rs`.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// The module of the merge-form kernels (sharded and canonical).
pub(super) const MODULE: &str = "glm_kv_shard";
/// The packing kernels take a row in one CTA: 256 threads of
/// `GLM_KV_SHARD_COMPACT_CHUNK` (16) IDs.
const COMPACT_MAX_WIDTH: u32 = 256 * 16;

fn kernel(gpu: &dyn GpuBackend, symbol: &'static str) -> Result<KernelHandle> {
    let k = gpu.op_cache().kernel(gpu, MODULE, symbol)?;
    ensure!(k.0 != 0, "GLM KV shard kernel {symbol} is unavailable");
    Ok(k)
}

/// Rank ownership of the pair, as the kernels take it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardRank {
    pub rank: u32,
    pub world: u32,
}

/// Global cache-write slots `[n]` i64 → this rank's local slots (`-1` where
/// the peer owns the block).
pub fn glm_kv_shard_map_slots(
    gpu: &dyn GpuBackend,
    slots: DevicePtr,
    local: DevicePtr,
    n: u32,
    block_size: u32,
    owner: ShardRank,
    stream: u64,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel(gpu, "glm_kv_shard_map_slots")?)
        .grid([div_ceil(n, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(slots)
        .arg_ptr(local)
        .arg_u32(n)
        .arg_u32(block_size)
        .arg_u32(owner.rank)
        .arg_u32(owner.world)
        .launch(stream)
}

/// Selected token IDs `[rows, width]` → this rank's local token IDs
/// (addressed through the shard's identity table) of the tokens it stores,
/// packed to each row's front in selected order and counted in `counts`
/// (`u32[rows]`); `-1` behind them. `selected == None` generates causal IDs:
/// row `r` keeps tokens `[0, causal_start + r + 1)`. `out` must not alias
/// `selected`.
#[allow(clippy::too_many_arguments)]
pub fn glm_kv_shard_localize(
    gpu: &dyn GpuBackend,
    selected: Option<DevicePtr>,
    out: DevicePtr,
    counts: DevicePtr,
    block_table: DevicePtr,
    rows: u32,
    width: u32,
    block_size: u32,
    owner: ShardRank,
    causal_start: u32,
    stream: u64,
) -> Result<()> {
    let selected = selected.unwrap_or(DevicePtr::NULL);
    check_pack(rows, width, selected, &[out])?;
    KernelLaunch::new(gpu, kernel(gpu, "glm_kv_shard_localize_compact")?)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(selected)
        .arg_ptr(out)
        .arg_ptr(counts)
        .arg_ptr(block_table)
        .arg_u32(rows)
        .arg_u32(width)
        .arg_u32(block_size)
        .arg_u32(owner.rank)
        .arg_u32(owner.world)
        .arg_u32(causal_start)
        .launch(stream)
}

/// A row-packing launch's shape: one CTA per row, out of place.
fn check_pack(rows: u32, width: u32, selected: DevicePtr, outs: &[DevicePtr]) -> Result<()> {
    ensure!(
        (1..=65535).contains(&rows)
            && (1..=COMPACT_MAX_WIDTH).contains(&width)
            && outs.iter().all(|&o| o != selected),
        "GLM merge-form packing takes 1..=65535 rows of up to {COMPACT_MAX_WIDTH} IDs, out of place"
    );
    Ok(())
}

/// The canonical form's selection split ([`super::glm_sparse_canonical`]):
/// row `r`'s selected IDs (`None`: causal from `causal_start`) whose logical
/// block's residue is `rank` packed to `own` and the rest to `peer`, global
/// IDs in selected order, with per-row counts: what
/// [`glm_kv_shard_localize`] packs on each rank of a shard.
#[allow(clippy::too_many_arguments)]
pub fn glm_kv_canonical_partition(
    gpu: &dyn GpuBackend,
    selected: Option<DevicePtr>,
    [own, own_counts]: [DevicePtr; 2],
    [peer, peer_counts]: [DevicePtr; 2],
    rows: u32,
    width: u32,
    block_size: u32,
    rank: u32,
    causal_start: u32,
    stream: u64,
) -> Result<()> {
    let selected = selected.unwrap_or(DevicePtr::NULL);
    check_pack(rows, width, selected, &[own, peer])?;
    ensure!(rank < 2, "GLM canonical partition of rank {rank} of a pair");
    KernelLaunch::new(gpu, kernel(gpu, "glm_kv_canonical_partition")?)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(selected)
        .arg_ptr(own)
        .arg_ptr(own_counts)
        .arg_ptr(peer)
        .arg_ptr(peer_counts)
        .arg_u32(rows)
        .arg_u32(width)
        .arg_u32(block_size)
        .arg_u32(rank)
        .arg_u32(causal_start)
        .launch(stream)
}

/// Copy `n` blocks of `block_bytes`: destination block `dst_idx[i]` (or `i`)
/// receives source block `src_idx[i]` (or `i`); `u32` index lists.
#[allow(clippy::too_many_arguments)]
pub fn glm_kv_shard_copy_blocks(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    src_idx: Option<DevicePtr>,
    dst: DevicePtr,
    dst_idx: Option<DevicePtr>,
    n: usize,
    block_bytes: usize,
    stream: u64,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    ensure!(
        block_bytes.is_multiple_of(16)
            && src.0.is_multiple_of(16)
            && dst.0.is_multiple_of(16)
            && u32::try_from(n).is_ok(),
        "GLM KV shard block copy needs 16-byte blocks and bases"
    );
    KernelLaunch::new(gpu, kernel(gpu, "glm_kv_shard_copy_blocks")?)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(src_idx.unwrap_or(DevicePtr::NULL))
        .arg_ptr(dst)
        .arg_ptr(dst_idx.unwrap_or(DevicePtr::NULL))
        .arg_u32(n as u32)
        .arg_u32((block_bytes / 16) as u32)
        .launch(stream)
}

#[cfg(test)]
#[path = "glm_kv_shard_test_gpu.rs"]
pub(crate) mod shard_test_gpu;

#[cfg(test)]
#[path = "glm_kv_shard_tests.rs"]
mod tests;
