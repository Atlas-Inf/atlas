// SPDX-License-Identifier: AGPL-3.0-only

//! Knowledge packs for score passes: a token block (a policy, a handbook)
//! encoded once and continued by many requests without being recomputed.
//!
//! A pack is the K/V of its P tokens in the full-attention layers and the SSM
//! (Gated DeltaNet) state and conv window after its last token, held by one
//! retained [`SequenceState`] (its KV blocks and its SSM slot). It is encoded
//! by a score pass of its own (`score_forward` over one stream, no branches),
//! under `prefix_bidir` a bidirectional block: every pack row reads every pack
//! key and nothing else, so what it holds depends on nothing that follows it.
//!
//! A request on a pack (`ScoreInput::pack`) is planned at positions shifted by
//! P (`score_plan.rs`). A stream that reads the pack directly (the request's
//! head, or a branch without one):
//!
//! - shares the pack's `P / block` whole KV blocks by reference count, as a
//!   head shares its whole blocks with its branches, and gets its own copy of
//!   the pack's partial last block (rows `[F_P, P)`), into which it writes
//!   its own rows from P on;
//! - takes the pack's SSM state and conv window as its entry state through the
//!   score pass's pointer tables, read in place (no copy) by the single-launch
//!   scan and conv1d (`score_ssm.rs`); only a layer without them copies the
//!   pack's state into the stream's slot first.
//!
//! An image pack holds images (the image pads among its tokens): its build
//! pass encodes them, and it keeps their merged grids and content hashes,
//! which a stream on it stages for its MRoPE positions (`score_vision.rs`);
//! nothing of the images is encoded again.
//!
//! Packs are content-addressed (tokens, image hashes and attention mode), held up to a
//! caller's limit and evicted least recently used first. Every score pass and
//! every build runs under the registry lock, so a pack is never dropped while
//! a pass reads it.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::ssm_batched_copy::{StateCopy, run_state_copies};
use super::super::super::types::TransformerModel;
use super::score_plan::ScorePlan;
use super::score_store::PackMeta;
use crate::layer::{BatchedAttnMetadata, SsmLayerState};
use crate::traits::SequenceState;

/// One held pack.
pub(in crate::model) struct Pack {
    pub tokens: Vec<u32>,
    pub bidir: bool,
    /// Its images' merged (t, h, w) grids and content hashes, in pad order.
    pub grids: Vec<(usize, usize, usize)>,
    pub hashes: Vec<u128>,
    /// Its KV blocks (`ceil(P / block)`) and its SSM slot.
    pub seq: SequenceState,
    last_use: u64,
}

/// The model's held packs and a use counter (for LRU).
#[derive(Default)]
pub(in crate::model) struct ScorePacks(parking_lot::Mutex<(HashMap<u64, Pack>, u64)>);

pub(in crate::model) type PackGuard<'a> = parking_lot::MutexGuard<'a, (HashMap<u64, Pack>, u64)>;

impl ScorePacks {
    pub(in crate::model) fn lock(&self) -> PackGuard<'_> {
        self.0.lock()
    }

    pub(in crate::model) fn held(&self) -> usize {
        self.0.lock().0.len()
    }
}

/// A pack's content address.
pub(in crate::model) fn pack_id(tokens: &[u32], hashes: &[u128], bidir: bool) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    tokens.hash(&mut h);
    hashes.hash(&mut h);
    bidir.hash(&mut h);
    h.finish()
}

/// Mark `id` used and return it, or [`crate::traits::UnknownScorePack`].
pub(in crate::model) fn touch<'g>(guard: &'g mut PackGuard<'_>, id: u64) -> Result<&'g Pack> {
    guard.1 += 1;
    let tick = guard.1;
    match guard.0.get_mut(&id) {
        Some(p) => {
            p.last_use = tick;
            Ok(p)
        }
        None => Err(crate::traits::UnknownScorePack(id).into()),
    }
}

/// The pass's packs: per request the index of its pack among the pass's
/// distinct packs (`ScorePlan`'s `pack_of`), and those packs' ids in order.
/// Each is marked used; an unheld id is [`crate::traits::UnknownScorePack`].
pub(super) fn resolve(
    guard: &mut PackGuard<'_>,
    reqs: &[crate::traits::ScoreInput<'_>],
) -> Result<(Vec<Option<usize>>, Vec<u64>)> {
    let mut ids: Vec<u64> = Vec::new();
    let mut of = Vec::with_capacity(reqs.len());
    for r in reqs {
        let Some(id) = r.pack else {
            of.push(None);
            continue;
        };
        let p = touch(guard, id)?;
        ensure!(
            p.bidir == r.prefix_bidir,
            "score: pack {id:016x} was built {} but the request is {}",
            if p.bidir { "prefix_bidir" } else { "causal" },
            if r.prefix_bidir {
                "prefix_bidir"
            } else {
                "causal"
            }
        );
        let k = ids.iter().position(|&x| x == id).unwrap_or_else(|| {
            ids.push(id);
            ids.len() - 1
        });
        of.push(Some(k));
    }
    Ok((of, ids))
}

/// The (h-state, conv-state) pointers of an SSM layer of a read-only
/// sequence (a pack's).
pub(super) fn ssm_ptrs_of(seq: &SequenceState, layer_idx: usize) -> Result<(DevicePtr, DevicePtr)> {
    let st = seq.layer_states[layer_idx]
        .as_any()
        .downcast_ref::<SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("score: layer {layer_idx} is not an SSM layer"))?;
    Ok((st.h_state, st.conv_state))
}

/// K and V copy plans that duplicate block `src` into block `dst` in every
/// attention layer of `kv`.
fn block_copies(kv: &PagedKvCache, src: u32, dst: u32) -> [Vec<StateCopy>; 2] {
    let mut out = [Vec::new(), Vec::new()];
    for l in 0..kv.num_layers() {
        let k = kv.k_block_stride_bytes_for_layer(l);
        let v = kv.v_block_stride_bytes_for_layer(l);
        if k > 0 {
            out[0].push(StateCopy {
                src: kv.k_cache_ptr(l, src),
                dst: kv.k_cache_ptr(l, dst),
                bytes: k,
            });
        }
        if v > 0 {
            out[1].push(StateCopy {
                src: kv.v_cache_ptr(l, src),
                dst: kv.v_cache_ptr(l, dst),
                bytes: v,
            });
        }
    }
    out
}

impl TransformerModel {
    /// Upload the per-stream knowledge-pack reader markers for rows128 GQA.
    /// Returns the scratch cursor past the aligned staging array.
    pub(super) fn score_stage_packs(
        &self,
        plan: &ScorePlan,
        meta: &mut BatchedAttnMetadata,
        cursor: usize,
        stream: u64,
    ) -> Result<usize> {
        let n = plan.streams();
        let off = (cursor + 63) & !63;
        ensure!(
            off + n * 4 <= self.buffers.scratch_bytes(),
            "score: pack-stream staging exceeds scratch ({n} streams)"
        );
        let host: Vec<bool> = plan.pack_of.iter().map(Option::is_some).collect();
        let bytes: Vec<u8> = host
            .iter()
            .flat_map(|&r| (r as i32).to_le_bytes())
            .collect();
        let dst = self.buffers.scratch().offset(off);
        self.gpu.copy_h2d_async_retained(&bytes, dst, stream)?;
        meta.pack_streams = dst;
        meta.pack_streams_host = host;
        Ok(off + ((n * 4 + 63) & !63))
    }

    /// See [`crate::traits::Model::score_pack_by_id`].
    pub(in crate::model) fn score_pack_by_id_impl(
        &self,
        id: u64,
        limit: usize,
        stream: u64,
    ) -> Result<bool> {
        let mut guard = self.score_packs.lock();
        if touch(&mut guard, id).is_ok() {
            return Ok(true);
        }
        let Some((meta, seq)) = self.store_restore(id, stream)? else {
            return Ok(false);
        };
        self.hold_pack(&mut guard, id, meta, seq, limit.max(1))?;
        Ok(true)
    }

    /// See [`crate::traits::Model::score_packs_prefetch`].
    pub(in crate::model) fn score_packs_prefetch_impl(
        &self,
        packs: &[crate::traits::ScorePackKey<'_>],
    ) -> (Vec<bool>, bool) {
        use crate::traits::ScorePackKey;
        let ids: Vec<u64> = packs
            .iter()
            .map(|k| match k {
                ScorePackKey::Id(id) => *id,
                ScorePackKey::Spec(s) => {
                    let hashes: Vec<u128> = s.images.iter().map(|it| it.content_hash).collect();
                    pack_id(s.tokens, &hashes, s.prefix_bidir)
                }
            })
            .collect();
        let held: Vec<bool> = {
            let guard = self.score_packs.lock();
            ids.iter().map(|id| guard.0.contains_key(id)).collect()
        };
        self.store_prefetch(&ids, &held)
    }

    /// Hold `seq` as pack `id`, then evict the least recently used other
    /// packs down to `limit`.
    pub(super) fn hold_pack(
        &self,
        guard: &mut PackGuard<'_>,
        id: u64,
        meta: PackMeta,
        seq: SequenceState,
        limit: usize,
    ) -> Result<()> {
        guard.1 += 1;
        let tick = guard.1;
        guard.0.insert(
            id,
            Pack {
                tokens: meta.tokens,
                bidir: meta.bidir,
                grids: meta.grids,
                hashes: meta.hashes,
                seq,
                last_use: tick,
            },
        );
        while guard.0.len() > limit {
            let Some((&old, _)) = guard
                .0
                .iter()
                .filter(|(k, _)| **k != id)
                .min_by_key(|(_, p)| p.last_use)
            else {
                break;
            };
            if let Some(mut p) = guard.0.remove(&old) {
                self.free_sequence_dispatch(&mut p.seq, false)?;
            }
        }
        Ok(())
    }

    /// The pass's tail-block copies: each stream that reads its pack directly
    /// gets the pack's partial last block, if it has one, in the block it owns
    /// at that index. Issued on `stream` before the layer loop writes.
    pub(super) fn score_pack_tail_copies(
        &self,
        plan: &ScorePlan,
        seqs: &[SequenceState],
        packs: &[&SequenceState],
        kv: &PagedKvCache,
        stream: u64,
    ) -> Result<()> {
        let (mut k_plan, mut v_plan) = (Vec::new(), Vec::new());
        for (i, seq) in seqs.iter().enumerate() {
            let Some(p) = plan.reads_pack(i) else {
                continue;
            };
            let at = plan.shared_blocks[i];
            let Some(&src) = packs[p].block_table.get(at) else {
                continue;
            };
            let [k, v] = block_copies(kv, src, seq.block_table[at]);
            k_plan.extend(k);
            v_plan.extend(v);
        }
        run_state_copies(self.gpu.as_ref(), &k_plan, stream)?;
        run_state_copies(self.gpu.as_ref(), &v_plan, stream)
    }

    /// For a layer without the in-place score scan or conv1d: copy each pack's
    /// SSM state (`conv`: its conv window) into the slots of the streams in
    /// `range` that read it directly.
    pub(super) fn score_pack_seed(
        &self,
        plan: &ScorePlan,
        seqs: &mut [&mut SequenceState],
        packs: &[&SequenceState],
        layer_idx: usize,
        conv: bool,
        range: std::ops::Range<usize>,
        stream: u64,
    ) -> Result<()> {
        let bytes = if conv {
            self.ssm_pool.conv_bytes
        } else {
            self.ssm_pool.h_stored_bytes
        };
        for i in range {
            let Some(p) = plan.reads_pack(i) else {
                continue;
            };
            let pick = |q: (DevicePtr, DevicePtr)| if conv { q.1 } else { q.0 };
            let src = pick(ssm_ptrs_of(packs[p], layer_idx)?);
            let dst = pick(ssm_ptrs_of(seqs[i], layer_idx)?);
            self.gpu.copy_d2d_async(src, dst, bytes, stream)?;
        }
        Ok(())
    }

    /// Release every held pack (model teardown, before the pools go).
    pub(in crate::model) fn score_packs_release(&self) -> Result<()> {
        let mut guard = self.score_packs.lock();
        let mut first = Ok(());
        for (_, mut p) in guard.0.drain() {
            let r = self.free_sequence_dispatch(&mut p.seq, false);
            if first.is_ok() {
                first = r;
            }
        }
        first
    }
}
