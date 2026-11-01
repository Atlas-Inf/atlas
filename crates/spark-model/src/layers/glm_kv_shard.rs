// SPDX-License-Identifier: AGPL-3.0-only
//! Token-sharded GLM MLA latents over the TP pair (`ATLAS_GLM_KV_SHARD=1`).
//!
//! Each KV block's latents live on one rank
//! (`spark_runtime::kv_cache::LatentShard`): logical block `l` of every
//! sequence on rank `l % 2`, which the allocator guarantees although the
//! ranks' physical ids differ; the semantic-index keys stay
//! replicated, so both ranks still select the same top-k tokens locally with
//! no exchange. Attention then runs in one of two exact forms
//! (docs/glm-kv-shard.md):
//!
//! - **merge** (at most [`MERGE_MAX_ROWS`] rows: decode, DFlash verify,
//!   short prefill tails): the ranks swap their heads' absorbed queries, each
//!   attends ALL heads over the selected tokens it stores, the ranks swap the
//!   FP32 partial outputs + LSEs of each other's heads, and each merges its
//!   own heads' two partials (flash-decoding LSE merge);
//! - **view** (larger prefill chunks): each rank assembles the sequence's
//!   whole latent history — its own blocks plus the peer's, exchanged — in a
//!   scratch view the unchanged kernels read through an identity table.
//!
//! The merge form packs each row's owned IDs to the front so the attention
//! kernels walk only the tokens this rank stores, and merges the peer's
//! partial where it landed. Its arithmetic is the canonical form's
//! (`ops::glm_sparse_canonical`), which an unsharded pair runs for the same
//! owners, so a pair computes the same bits with the shard on or off.
//! `ATLAS_GLM_KV_SHARD_OVERLAP=1` runs the two exchanges on a side stream
//! beside the compute they do not depend on ([`overlapped_exchange`]); the
//! arithmetic is unchanged.
//!
//! This module holds the policy, the scratch layout both forms carve from
//! the shard's one allocation, and the pair exchange.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{ExchangeLane, KvCacheConfig, LatentShard, LatentShardSpec};

/// Heads per rank (GLM-5.3: 64 heads over TP2) = one sparse-kernel launch.
pub const HEADS: u32 = 32;
/// Latent width (`kv_lora_rank`).
pub const LATENT: u32 = 512;
/// Selected IDs per row (`index_topk + index_kpool - 1`).
pub const WIDTH: u32 = 2051;
/// Owners up to this many rows take the merge form; larger ones the view.
pub const MERGE_MAX_ROWS: usize = 64;
/// At most 15 local splits, so the final merge (plus the peer's partial)
/// stays within `glm_sparse_decode_split_merge`'s 16 partitions.
pub const MAX_SPLITS: u32 = 15;
/// Blocks per view-exchange round (the send and receive pieces).
pub const PIECE_BLOCKS: usize = 2048;
/// Tokens past `max_seq_len` a view must still hold (verify rows).
const VIEW_SLACK_TOKENS: usize = 64;
/// Environment opt-ins whose attention was never ported to sharded latents
/// (multi-sequence sparse decode, the repaired MTP, C4 and independent decode
/// lanes, the long-verify serial diagnostic): a shard refuses them at startup.
const UNSHARDED_LANES: [&str; 7] = [
    "ATLAS_GLM_MULTI_SEQ_SPARSE",
    "ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS",
    "ATLAS_GLM_MTP_REPAIR",
    "ATLAS_GLM_C4_DECODE",
    "ATLAS_GLM_C4_SPARSE",
    "ATLAS_GLM_INDEPENDENT_DECODE",
    "ATLAS_GLM_LONG_BATCH_SERIAL",
];

fn parse(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{name} must be 0 or 1, got {other:?}"),
    }
}

fn flag(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => parse(name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse(name, None),
        Err(error) => Err(error.into()),
    }
}

/// The first of the unported lanes `var` opts into (anything but unset,
/// empty or `0`).
pub fn unsharded_lane(var: impl Fn(&str) -> Option<String>) -> Option<&'static str> {
    UNSHARDED_LANES
        .into_iter()
        .find(|name| !matches!(var(name).as_deref(), None | Some("" | "0")))
}

/// `ATLAS_GLM_KV_SHARD=1`: store each block's latents on one rank.
pub fn requested() -> Result<bool> {
    flag("ATLAS_GLM_KV_SHARD")
}

/// The merge form's opt-in refinements and the shard's self-check, read once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MergeTuning {
    /// `ATLAS_GLM_KV_SHARD_COMPACT=1`: accepted for older profiles. The
    /// merge form is always compact now (the canonical form an unsharded pair
    /// matches bit for bit is), so the value changes nothing but is still
    /// held to 0/1, to the shard, and to its peer's.
    pub compact: bool,
    /// `ATLAS_GLM_KV_SHARD_OVERLAP=1`; off under `..._CHECK=1`, whose own
    /// exchange would run inside an overlap window. Decides at boot whether
    /// the cache gets an [`ExchangeLane`], which is what the layers consult.
    pub overlap: bool,
    /// `ATLAS_GLM_KV_SHARD_CHECK=1`: before each merge-form attention,
    /// confirm this rank's block table keeps the ownership invariant and
    /// both ranks attend the same number of blocks (one host sync + exchange
    /// each; the view form does so regardless).
    pub check: bool,
}

impl MergeTuning {
    /// Any of them set without the shard (`sharded`) is a misconfiguration.
    fn parse(
        compact: Option<&str>,
        overlap: Option<&str>,
        check: Option<&str>,
        sharded: bool,
    ) -> Result<Self> {
        let compact = parse("ATLAS_GLM_KV_SHARD_COMPACT", compact)?;
        let overlap = parse("ATLAS_GLM_KV_SHARD_OVERLAP", overlap)?;
        let check = parse("ATLAS_GLM_KV_SHARD_CHECK", check)?;
        ensure!(
            sharded || !(compact || overlap || check),
            "ATLAS_GLM_KV_SHARD_COMPACT, ATLAS_GLM_KV_SHARD_OVERLAP and ATLAS_GLM_KV_SHARD_CHECK \
             tune ATLAS_GLM_KV_SHARD=1, which is not set"
        );
        Ok(Self {
            compact,
            overlap: overlap && !check,
            check,
        })
    }

    pub fn get() -> Result<Self> {
        static TUNING: std::sync::OnceLock<Result<MergeTuning, String>> =
            std::sync::OnceLock::new();
        TUNING
            .get_or_init(|| {
                let var = |name: &str| std::env::var(name).ok();
                requested()
                    .and_then(|sharded| {
                        Self::parse(
                            var("ATLAS_GLM_KV_SHARD_COMPACT").as_deref(),
                            var("ATLAS_GLM_KV_SHARD_OVERLAP").as_deref(),
                            var("ATLAS_GLM_KV_SHARD_CHECK").as_deref(),
                            sharded,
                        )
                    })
                    .map_err(|e| format!("{e:#}"))
            })
            .clone()
            .map_err(anyhow::Error::msg)
    }
}

/// Where [`settings_word`] sits in the word the ranks gather for the pool
/// size (`factory::build::glm::agree_kv_blocks`): bits 28-31, above any block
/// count and below the spill-tier word the upper half carries
/// (`factory::build::kv_nvme::rank_word`).
const SETTINGS_SHIFT: u32 = 28;

/// `[shard, compact, overlap, check]` of a sharded rank, one bit each.
fn settings_bits(tuning: MergeTuning) -> u64 {
    1 | (tuning.compact as u64) << 1 | (tuning.overlap as u64) << 2 | (tuning.check as u64) << 3
}

/// What both ranks must run alike: a rank sharding alone would exchange
/// inside attention with a peer that does not, and the tunings change the
/// merge form's exchanges. 0 when unsharded, so the pool-size gather then
/// carries the bytes it always did.
pub fn settings_word(sharded: bool) -> Result<u64> {
    if !sharded {
        return Ok(0);
    }
    Ok(settings_bits(MergeTuning::get()?))
}

/// A rank's word for the pool-size gather: `blocks` under its `settings`.
pub fn blocks_word(blocks: usize, settings: u64) -> Result<u64> {
    ensure!(
        blocks < 1 << SETTINGS_SHIFT,
        "a KV pool of {blocks} blocks overflows the pool-size agreement word"
    );
    Ok(blocks as u64 | settings << SETTINGS_SHIFT)
}

/// The smallest pool in `gathered` (every rank's [`blocks_word`]), once every
/// rank runs the settings of this rank's word `ours`.
pub fn agreed_blocks(rank: usize, ours: u64, gathered: &[u64]) -> Result<usize> {
    let bits = |w: u64| [0, 1, 2, 3].map(|i| w >> (SETTINGS_SHIFT + i) & 1);
    ensure!(
        gathered.iter().all(|&w| bits(w) == bits(ours)),
        "ATLAS_GLM_KV_SHARD settings [shard, compact, overlap, check] differ across the pair: \
         rank {rank} has {:?}, the ranks have {:?}",
        bits(ours),
        gathered.iter().map(|&w| bits(w)).collect::<Vec<_>>()
    );
    let blocks = |w: u64| (w & ((1 << SETTINGS_SHIFT) - 1)) as usize;
    Ok(gathered
        .iter()
        .map(|&w| blocks(w))
        .fold(blocks(ours), usize::min))
}

fn align(bytes: usize) -> usize {
    bytes.next_multiple_of(256)
}

/// Sparse partitions of each token group (the tokens one rank stores, about
/// half a row's selection) of every merge-form owner, whatever its rows. A
/// row's partition boundaries then follow only its own selection, so its bits
/// do not depend on the rows beside it: a DFlash verify's width follows the
/// drafter's confidence, and a count per width made the target's output
/// follow the drafter's state. Four fills GB10's 48 SMs in one wave up to six
/// rows (two groups x rows x 4 CTAs); `scripts/dev/glm_kv_shard_bench.cu`
/// (`MERGE_SPLITS=n`) times other counts. The shard and the canonical form
/// both partition with it, which their bitwise equality needs.
pub const MERGE_SPLITS: u32 = 4;
const _: () = assert!(MERGE_SPLITS > 1 && MERGE_SPLITS <= MAX_SPLITS);

/// Offsets (from the work region) of one merge-form owner's buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeLayout {
    /// Peer heads' absorbed queries `[rows, 32, 512]` BF16.
    pub q_peer: usize,
    /// Localized IDs `[rows, 2051]` i32.
    pub ids: usize,
    /// Own heads: `splits + 1` FP32 partials (the peer's last) and LSEs.
    pub own_o: usize,
    pub own_lse: usize,
    /// Merged LSE the BF16 merge also writes `[rows, 32]`.
    pub out_lse: usize,
    /// Peer heads over this rank's tokens: `splits` partials and LSEs.
    pub peer_o: usize,
    pub peer_lse: usize,
    /// One merged FP32 partial `[rows, 32, 512]` then its LSE `[rows, 32]`.
    pub send: usize,
    pub recv: usize,
    pub total: usize,
}

impl MergeLayout {
    pub fn new(rows: u32, splits: u32) -> Self {
        let (rows, splits) = (rows as usize, splits as usize);
        let heads = HEADS as usize;
        let part = rows * heads * LATENT as usize * 4;
        let lse = rows * heads * 4;
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let offset = at;
            at += align(bytes);
            offset
        };
        let q_peer = take(rows * heads * LATENT as usize * 2);
        let ids = take(rows * WIDTH as usize * 4);
        let own_o = take((splits + 1) * part);
        let own_lse = take((splits + 1) * lse);
        let out_lse = take(lse);
        let peer_o = take(splits * part);
        let peer_lse = take(splits * lse);
        let send = take(part + lse);
        let recv = take(part + lse);
        Self {
            q_peer,
            ids,
            own_o,
            own_lse,
            out_lse,
            peer_o,
            peer_lse,
            send,
            recv,
            total: at,
        }
    }

    /// Bytes of one exchanged partial: FP32 output then LSE.
    pub fn partial_bytes(rows: u32) -> usize {
        rows as usize * HEADS as usize * (LATENT as usize + 1) * 4
    }

    /// Per-row owned-ID counts `u32[rows]` of the compact form, which merges
    /// the peer's partial in place and so leaves the last own partition free.
    pub fn counts(&self, rows: u32, splits: u32) -> usize {
        self.own_o + splits as usize * rows as usize * (HEADS * LATENT) as usize * 4
    }
}

/// The shard scratch carved by both attention forms and the cache write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLayout {
    /// Assembled view: `view_blocks` blocks of `block_bytes`.
    pub view: usize,
    /// `u32[view_blocks]` lists: own local slots, their logical positions,
    /// and the peer's logical positions.
    pub mine_slot: usize,
    pub mine_dst: usize,
    pub peer_dst: usize,
    /// Table-hash exchange words (`ATLAS_GLM_KV_SHARD_CHECK`).
    pub check: usize,
    /// Local write slots `i64[write_rows]`.
    pub slots: usize,
    /// View send/receive pieces, or one merge-form owner's buffers.
    pub work: usize,
    /// Bytes of one view piece.
    pub piece_bytes: usize,
    pub block_bytes: usize,
    pub total: usize,
}

impl ScratchLayout {
    pub fn new(view_blocks: usize, write_rows: usize, block_bytes: usize) -> Self {
        let piece_bytes = PIECE_BLOCKS * block_bytes;
        let merge = MergeLayout::new(MERGE_MAX_ROWS as u32, MERGE_SPLITS).total;
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let offset = at;
            at += align(bytes);
            offset
        };
        let view = take(view_blocks * block_bytes);
        let mine_slot = take(view_blocks * 4);
        let mine_dst = take(view_blocks * 4);
        let peer_dst = take(view_blocks * 4);
        let check = take(16);
        let slots = take(write_rows * 8);
        let work = take((2 * align(piece_bytes)).max(merge));
        Self {
            view,
            mine_slot,
            mine_dst,
            peer_dst,
            check,
            slots,
            work,
            piece_bytes,
            block_bytes,
            total: at,
        }
    }

    /// The layout of `shard`'s scratch over `config`'s widest latent block.
    pub fn of(shard: &LatentShard, config: &KvCacheConfig) -> Result<Self> {
        let layout = Self::new(
            shard.spec.view_blocks,
            shard.spec.write_rows,
            latent_block_bytes(config),
        );
        ensure!(
            layout.total == shard.spec.scratch_bytes && !shard.scratch.is_null(),
            "GLM KV shard scratch disagrees with its layout"
        );
        Ok(layout)
    }
}

/// The widest per-layer latent (K) block.
pub fn latent_block_bytes(config: &KvCacheConfig) -> usize {
    (0..config.num_layers)
        .map(|layer| config.k_block_bytes_for_layer(layer))
        .max()
        .unwrap_or(0)
}

/// The shard spec of `rank` for sequences up to `max_seq_len` tokens and
/// cache writes of up to `write_rows` rows.
pub fn spec(
    rank: usize,
    config: &KvCacheConfig,
    max_seq_len: usize,
    write_rows: usize,
) -> LatentShardSpec {
    let view_blocks = (max_seq_len + VIEW_SLACK_TOKENS).div_ceil(config.block_size);
    let layout = ScratchLayout::new(view_blocks, write_rows, latent_block_bytes(config));
    LatentShardSpec {
        rank,
        world: 2,
        scratch_bytes: layout.total,
        view_blocks,
        write_rows,
        lane: false,
    }
}

/// Exchange `bytes` with the other rank on `stream`: send from `send`,
/// receive the peer's equally sized payload into `recv`. Both ranks must
/// call it in the same order with the same `bytes`. The RDMA pair serves it
/// when up (and not capturing); NCCL point-to-point otherwise.
pub fn pair_exchange(
    comm: &dyn spark_comm::CommBackend,
    send: DevicePtr,
    recv: DevicePtr,
    bytes: usize,
    stream: u64,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    if comm.exchange_async(send.0, recv.0, bytes, false, stream)? {
        return Ok(());
    }
    let peer = 1 - comm.rank();
    comm.group_start()?;
    comm.send_to(send.0, bytes, peer, stream)?;
    comm.recv_from(recv.0, bytes, peer, stream)?;
    comm.group_end()
}

/// The communicator inside an [`overlapped_exchange`] window: the pair's
/// rank and size, and an error from every operation that would use the pair.
/// The pair orders its sends by stream order and counts landings in one
/// place, so a second user beside the lane's exchange would corrupt or hang
/// both ranks without an error of its own.
pub struct WindowComm<'a>(&'a dyn spark_comm::CommBackend);

impl WindowComm<'_> {
    fn refuse<T>(&self, what: &str) -> Result<T> {
        bail!(
            "{what} inside a GLM KV shard overlap window on rank {}: the pair is carrying the lane's exchange",
            self.0.rank()
        )
    }
}

impl spark_comm::CommBackend for WindowComm<'_> {
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        self.refuse("all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        self.refuse("all-gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        self.refuse("reduce-scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        self.refuse("broadcast")
    }
    fn barrier(&self) -> Result<()> {
        self.refuse("barrier")
    }
    fn peer_exchange_async(&self, _: u64, _: u64, _: usize, _: u64) -> Result<()> {
        self.refuse("peer exchange")
    }
    fn exchange_async(&self, _: u64, _: u64, _: usize, _: bool, _: u64) -> Result<bool> {
        self.refuse("exchange")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        self.refuse("send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        self.refuse("receive")
    }
    fn rank(&self) -> usize {
        self.0.rank()
    }
    fn world_size(&self) -> usize {
        self.0.world_size()
    }
}

/// [`pair_exchange`] on the shard's side stream, so the compute stream runs
/// `during` while the payload is in flight, and waits for it to land after.
///
/// `during` must not read `recv` or write `send`, and cannot use the pair:
/// only this fence orders the side stream against the compute stream, so
/// the communicator it is handed (give it to whatever `during` calls)
/// refuses every pair operation. The wait is enqueued even when `during`
/// fails. Both ranks run the same exchanges in the same order, so an overlap
/// only moves where each waits.
pub fn overlapped_exchange<T>(
    gpu: &dyn GpuBackend,
    comm: &dyn spark_comm::CommBackend,
    lane: ExchangeLane,
    (send, recv, bytes): (DevicePtr, DevicePtr, usize),
    stream: u64,
    during: impl FnOnce(&WindowComm) -> Result<T>,
) -> Result<T> {
    gpu.record_event(lane.begun, stream)?;
    gpu.stream_wait_event(lane.stream, lane.begun)?;
    pair_exchange(comm, send, recv, bytes, lane.stream)?;
    gpu.record_event(lane.landed, lane.stream)?;
    let out = during(&WindowComm(comm));
    gpu.stream_wait_event(stream, lane.landed)?;
    out
}

#[cfg(test)]
#[path = "glm_kv_shard_tests.rs"]
mod tests;
