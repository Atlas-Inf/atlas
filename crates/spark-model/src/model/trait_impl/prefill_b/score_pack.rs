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
//! Packs are content-addressed (tokens and attention mode), held up to a
//! caller's limit and evicted least recently used first. Every score pass and
//! every build runs under the registry lock, so a pack is never dropped while
//! a pass reads it.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::super::ssm_batched_copy::{StateCopy, run_state_copies};
use super::super::super::types::TransformerModel;
use super::score_plan::ScorePlan;
use crate::layer::SsmLayerState;
use crate::traits::{ScorePackInfo, SequenceState};

/// One held pack.
pub(in crate::model) struct Pack {
    pub tokens: Vec<u32>,
    pub bidir: bool,
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
pub(in crate::model) fn pack_id(tokens: &[u32], bidir: bool) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    tokens.hash(&mut h);
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
    /// See [`crate::traits::Model::score_pack`].
    pub(in crate::model) fn score_pack_impl(
        &self,
        tokens: &[u32],
        bidir: bool,
        limit: usize,
        stream: u64,
    ) -> Result<ScorePackInfo> {
        ensure!(
            limit > 0,
            "score: packs are off on this server (start it with --score-packs N)"
        );
        ensure!(!tokens.is_empty(), "score: an empty pack");
        let v = self.config.vocab_size as u32;
        if let Some(t) = tokens.iter().find(|&&t| t >= v) {
            anyhow::bail!("score: pack token id {t} outside the model's vocabulary ({v} rows)");
        }
        let id = pack_id(tokens, bidir);
        let mut guard = self.score_packs.lock();
        if let Ok(p) = touch(&mut guard, id) {
            ensure!(
                p.tokens == tokens,
                "score: pack address collision ({id:016x})"
            );
            return Ok(ScorePackInfo {
                id,
                tokens: tokens.len(),
                built: false,
            });
        }
        // Build first (a failed build keeps every held pack), then evict the
        // least recently used down to the limit.
        let seq = self.score_build_pack(tokens, bidir, guard.0.len(), stream)?;
        let tick = guard.1;
        guard.0.insert(
            id,
            Pack {
                tokens: tokens.to_vec(),
                bidir,
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
        Ok(ScorePackInfo {
            id,
            tokens: tokens.len(),
            built: true,
        })
    }

    /// Encode `tokens` in a pass of their own and keep the sequence (`held`:
    /// packs already held, the registry being locked by the caller).
    fn score_build_pack(
        &self,
        tokens: &[u32],
        bidir: bool,
        held: usize,
        stream: u64,
    ) -> Result<SequenceState> {
        ensure!(
            self.score_supported(),
            "score: single-rank models with BF16 logits only"
        );
        ensure!(
            !bidir
                || self
                    .layers
                    .iter()
                    .all(|l| l.is_ssm_layer() || l.honours_state_end()),
            "score: attention \"prefix_bidir\" is not supported by this model's attention layers"
        );
        {
            let kv = self.kv_cache.lock();
            ensure!(
                (0..kv.num_layers()).all(|l| matches!(
                    kv.dtype_for_layer(l),
                    KvCacheDtype::Bf16 | KvCacheDtype::Fp8
                )),
                "score: packs need a bf16 or fp8 KV cache"
            );
        }
        let (max_rows, max_streams) = self.score_capacity_held(held);
        ensure!(
            tokens.len() <= max_rows && max_streams > 0,
            "score: a {}-token pack exceeds one pass ({max_rows} rows, {max_streams} free \
             sequences)",
            tokens.len()
        );
        let plan = ScorePlan::pack(tokens, bidir);
        let mut seqs = vec![self.alloc_sequence_reset(false)?];
        let slot = seqs[0].slot_idx;
        let run = self
            .ssm_pool
            .zero_slot(slot, self.gpu.as_ref(), stream)
            .and_then(|()| self.score_forward(&plan, &mut seqs, &[], stream))
            .and_then(|()| self.gpu.synchronize(stream));
        if let Err(e) = run {
            if let Err(s) = self.gpu.synchronize(stream) {
                tracing::error!("score pack: synchronize after a failed build: {s:#}");
            }
            if let Err(f) = self.free_sequence_dispatch(&mut seqs[0], false) {
                tracing::error!("score pack: free_sequence: {f:#}");
            }
            return Err(e);
        }
        Ok(seqs.pop().expect("one sequence"))
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
