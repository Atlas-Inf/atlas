// SPDX-License-Identifier: AGPL-3.0-only
//! The canonical form of few-row GLM sparse MLA attention on an unsharded TP
//! pair: the arithmetic a token-sharded pair (`ATLAS_GLM_KV_SHARD=1`, merge
//! form, `layers::glm_kv_shard`) runs for the same heads, without the
//! exchanges, so that a pair computes the same bits sharded or not.
//!
//! A shard stores logical block `l` on rank `l % 2`. Rank `r`'s heads there
//! attend in two groups: the tokens rank `r` stores, in `MERGE_SPLITS`
//! partitions on rank `r`, and the tokens its peer stores, in as many
//! partitions on the peer, merged there to one FP32 partial and sent; rank `r`
//! then merges its own partitions with that partial last. The count is the
//! same for every owner size, so a row's bits never depend on its owner's
//! rows (a DFlash verify's width). The canonical form
//! splits each row's selection by the same rule (`glm_kv_canonical_partition`,
//! with the logical block's residue, which a sharded pool's allocator keeps
//! equal to the physical block's), runs both groups' counted splits in one
//! launch (`*_split_counted_pair`: every CTA is the one the shard runs for its
//! partition) and both merges in one launch (`glm_sparse_decode_split_merge_
//! pair`: the FP32 merge, then the BF16 merge with the partial last). Only
//! where the latents sit and the exchanges differ; the GPU test
//! `canonical_matches_the_shard_bitwise` (`paged_glm_shard_merge_gpu_tests.rs`)
//! holds the two to the same bits.
//!
//! On by default for every owner of at most `MERGE_MAX_ROWS` rows (verify,
//! decode, prefill tails) on a pair with 32 heads per rank;
//! `ATLAS_GLM_KV_CANONICAL=0` restores the earlier unsharded kernels (split
//! or unsplit TC, dense, native), whose bits differ from the shard's.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::KvCacheDtype;

use super::glm_kv_shard::{MODULE, glm_kv_canonical_partition};
use super::{GlmSparsePrefillTc, PartialGroup, launch_merge_pair, launch_sparse_partial_pair};
use crate::layers::glm_kv_shard::{HEADS, LATENT, MERGE_MAX_ROWS, MERGE_SPLITS, WIDTH};

/// `ATLAS_GLM_KV_CANONICAL`: `0` turns the canonical form off.
pub const CANONICAL: &str = "ATLAS_GLM_KV_CANONICAL";
/// Where the canonical scratch starts in the MoE expert scratch the caller
/// lends: past the causal IDs a dense decode row keeps at its start.
pub const CANONICAL_SCRATCH_OFFSET: usize = 64 << 10;

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => anyhow::bail!("{CANONICAL} must be 0 or 1, got {other:?}"),
    }
}

/// Whether `ATLAS_GLM_KV_CANONICAL` leaves the canonical form on (read once).
pub fn glm_kv_canonical_requested() -> Result<bool> {
    static ON: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    ON.get_or_init(|| parse(std::env::var(CANONICAL).ok().as_deref()).map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// The rank whose heads an unsharded owner of `rows` rows over `heads` local
/// heads attends in the canonical form, or `None` where the shard has no
/// counterpart (not a GLM TP pair of 32 heads per rank, too many rows, or
/// `ATLAS_GLM_KV_CANONICAL=0`).
pub fn glm_kv_canonical_rank(config: &ModelConfig, heads: u32, rows: usize) -> Result<Option<u32>> {
    let fits = config.model_type == "glm5_next"
        && config.tp_world_size == 2
        && config.tp_rank < 2
        && heads == HEADS
        && (1..=MERGE_MAX_ROWS).contains(&rows);
    Ok((fits && glm_kv_canonical_requested()?).then_some(config.tp_rank as u32))
}

fn align(bytes: usize) -> usize {
    bytes.next_multiple_of(256)
}

/// Offsets (from the scratch start) of one canonical owner's buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalLayout {
    /// Packed global IDs `[rows, 2051]` i32 of the tokens this rank would
    /// store (`own`) and its peer would (`peer`), and their `u32[rows]` counts.
    pub own_ids: usize,
    pub peer_ids: usize,
    pub own_counts: usize,
    pub peer_counts: usize,
    /// `splits` FP32 partials `[splits, rows, 32, 512]` and LSEs of each group.
    pub own_o: usize,
    pub own_lse: usize,
    pub peer_o: usize,
    pub peer_lse: usize,
    /// The peer group's merged partial: FP32 output then LSEs.
    pub extra: usize,
    /// The merged LSEs `[rows, 32]` the BF16 merge writes.
    pub out_lse: usize,
    pub total: usize,
}

impl CanonicalLayout {
    pub fn new(rows: u32, splits: u32) -> Self {
        let (rows, splits) = (rows as usize, splits as usize);
        let part = rows * (HEADS * LATENT) as usize * 4;
        let lse = rows * HEADS as usize * 4;
        let ids = rows * WIDTH as usize * 4;
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let offset = at;
            at += align(bytes);
            offset
        };
        Self {
            own_ids: take(ids),
            peer_ids: take(ids),
            own_counts: take(rows * 4),
            peer_counts: take(rows * 4),
            own_o: take(splits * part),
            own_lse: take(splits * lse),
            peer_o: take(splits * part),
            peer_lse: take(splits * lse),
            extra: take(part + lse),
            out_lse: take(lse),
            total: at,
        }
    }

    /// Scratch bytes an owner of `rows` rows needs.
    pub fn bytes(rows: u32) -> usize {
        Self::new(rows, MERGE_SPLITS).total
    }
}

fn disjoint(a: (DevicePtr, usize), b: (DevicePtr, usize)) -> bool {
    a.0.0 + a.1 as u64 <= b.0.0 || b.0.0 + b.1 as u64 <= a.0.0
}

/// Rank `rank`'s heads' attention for `a.rows` query rows over their selected
/// IDs `selected` (`[rows, 2051]`; `None`: row `r` attends causally to tokens
/// `[0, causal_start + r + 1)`) in the canonical form, into `a.output`
/// (`[rows, 32, 512]` BF16). `a.k_cache` is the layer's whole latent pool and
/// `a.block_table` the sequence's table; `a.indices` is not read. `scratch`
/// (256-aligned, `scratch_bytes` long) holds [`CanonicalLayout`] and must not
/// overlap the query, output or selection.
#[allow(clippy::too_many_arguments)]
pub fn glm_sparse_canonical(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    selected: Option<DevicePtr>,
    causal_start: u32,
    rank: u32,
    (scratch, scratch_bytes): (DevicePtr, usize),
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=MERGE_MAX_ROWS as u32).contains(&a.rows) && a.index_width == WIDTH,
        "GLM canonical attention takes 1..={MERGE_MAX_ROWS} rows of {WIDTH} selected IDs"
    );
    let rows = a.rows as usize;
    let splits = MERGE_SPLITS;
    let m = CanonicalLayout::new(a.rows, splits);
    let region = (scratch, m.total);
    let latent = rows * (HEADS * LATENT) as usize * 2;
    ensure!(
        scratch.0 != 0
            && scratch.0.is_multiple_of(256)
            && m.total <= scratch_bytes
            && disjoint(region, (a.query, latent))
            && disjoint(region, (a.output, latent))
            && selected.is_none_or(|s| disjoint(region, (s, rows * WIDTH as usize * 4))),
        "GLM canonical attention needs {} bytes of 256-aligned scratch apart from its inputs, \
         got {scratch_bytes} at {:#x}",
        m.total,
        scratch.0
    );
    let at = |offset: usize| scratch.offset(offset);
    let (own, peer) = (
        [at(m.own_ids), at(m.own_counts)],
        [at(m.peer_ids), at(m.peer_counts)],
    );
    glm_kv_canonical_partition(
        gpu,
        selected,
        own,
        peer,
        a.rows,
        WIDTH,
        a.block_size,
        rank,
        causal_start,
        stream,
    )?;
    let extra = at(m.extra);
    let peer_parts = [at(m.peer_o), at(m.peer_lse)];
    let group =
        |[indices, counts]: [DevicePtr; 2], [part_o, part_lse]: [DevicePtr; 2]| PartialGroup {
            query: a.query,
            indices,
            counts,
            part_o,
            part_lse,
        };
    let own_parts = [at(m.own_o), at(m.own_lse)];
    let groups = [group(own, own_parts), group(peer, peer_parts)];
    launch_sparse_partial_pair(gpu, a, splits, groups, stream)?;
    launch_merge_pair(
        gpu,
        own_parts,
        [a.output, at(m.out_lse)],
        a.rows,
        splits,
        peer_parts,
        extra,
        stream,
    )
}

/// Load the canonical form's kernels for `dtype` and run each once on zero
/// rows, so that their first real launch (module load, 69 KB shared-memory
/// opt-in) never falls inside a CUDA-graph capture. A no-op where
/// [`glm_kv_canonical_rank`] never applies.
pub fn initialize_glm_kv_canonical(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    dtype: KvCacheDtype,
) -> Result<()> {
    if glm_kv_canonical_rank(config, HEADS, 1)?.is_none()
        || !matches!(dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128)
    {
        return Ok(());
    }
    let kernel = |symbol: &'static str| {
        let k = gpu.op_cache().kernel(gpu, MODULE, symbol)?;
        ensure!(k.0 != 0, "GLM canonical kernel {symbol} is unavailable");
        Ok(k)
    };
    let null = DevicePtr::NULL;
    let stream = gpu.default_stream();
    // Zero rows: every entry point returns before touching memory.
    let mut partition = KernelLaunch::new(gpu, kernel("glm_kv_canonical_partition")?)
        .grid([1, 1, 1])
        .block([256, 1, 1]);
    for _ in 0..5 {
        partition = partition.arg_ptr(null);
    }
    partition
        .arg_u32(0)
        .arg_u32(WIDTH)
        .arg_u32(16)
        .arg_u32(0)
        .arg_u32(0)
        .launch(stream)?;
    let split = match dtype {
        KvCacheDtype::Fp8G128 => {
            "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted_pair"
        }
        _ => "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split_counted_pair",
    };
    let mut pair = KernelLaunch::new(gpu, kernel(split)?)
        .grid([1, 1, 2])
        .block([256, 1, 1])
        .shared_mem(69376);
    for _ in 0..6 {
        pair = pair.arg_ptr(null);
    }
    pair = pair
        .arg_u32(0)
        .arg_u32(HEADS)
        .arg_u32(LATENT)
        .arg_u32(WIDTH)
        .arg_u32(16)
        .arg_f32(0.0625);
    for _ in 0..8 {
        pair = pair.arg_ptr(null);
    }
    pair.launch(stream)?;
    let mut merge = KernelLaunch::new(gpu, kernel("glm_sparse_decode_split_merge_pair")?)
        .grid([1, 1, 1])
        .block([256, 1, 1]);
    for _ in 0..4 {
        merge = merge.arg_ptr(null);
    }
    merge = merge.arg_u32(0).arg_u32(HEADS).arg_u32(LATENT).arg_u32(1);
    for _ in 0..3 {
        merge = merge.arg_ptr(null);
    }
    merge.launch(stream)?;
    gpu.synchronize(stream)?;
    tracing::info!(
        "GLM few-row MLA attention takes the canonical (KV-shard-exact) form; {CANONICAL}=0 restores the earlier kernels"
    );
    Ok(())
}

#[cfg(test)]
#[path = "glm_sparse_canonical_tests.rs"]
mod tests;
