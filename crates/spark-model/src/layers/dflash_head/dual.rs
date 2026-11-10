// SPDX-License-Identifier: AGPL-3.0-only

//! Two drafters for one target (`--draft-model-alt`): e.g. DFlash2 (8-row
//! blocks, strong on prose) beside DSpine (16-row blocks, far ahead on code,
//! JSON and tool calls).
//!
//! **Capture.** The target captures the union of both drafters' layers once
//! (`union_layers`); each head's `fc` is widened to the union with zero
//! columns for the other's layers (`widen_to_union`), so both heads read the
//! same accumulator slots unchanged.
//!
//! **State.** The primary head owns the sequence's state; the alt head's
//! state rides inside it (`DflashProposerState::alt`) with its own drafter
//! KV (block table, `ctx_committed`) and mirrors the shared context
//! accumulator in before each of its proposes and back after
//! (`mirror_from` / `mirror_to`). A head that sat out rounds catches its KV
//! up from its `ctx_committed` on its next propose; a slid accumulator
//! (slot 0 moved) resets it.
//!
//! **Routing** (`Route`), per request, by headroom: the short-block head
//! hands over when it keeps accepting its whole block, and takes back when
//! the long-block head's rounds stop going past it. Drafts only steer the
//! verify; the exact verify decides every token, so routing never changes
//! output.

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::weight_loader::DflashWeights;
use crate::weight_map::DenseWeight;

/// Rounds each head's window remembers, and the fewest that judge it.
const WINDOW: usize = 8;
const JUDGE: usize = 6;

/// `a`'s layers, then `b`'s not already in `a`.
pub fn union_layers(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut u = a.to_vec();
    for &l in b {
        if !u.contains(&l) {
            u.push(l);
        }
    }
    u
}

/// Widen `w`'s `fc` from its own capture layers to `union` (zero columns for
/// the layers it does not read) and declare the union as its layers.
pub fn widen_to_union(
    w: &mut DflashWeights,
    union: &[usize],
    target_hidden: usize,
    store: &WeightStore,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let hidden = w.config.hidden_size;
    let sub = w
        .config
        .dflash_config
        .as_mut()
        .context("dual drafter without dflash_config")?;
    let own = sub.target_layer_ids.clone();
    if own == union {
        return Ok(());
    }
    let row = target_hidden * 2;
    let (src_pitch, dst_pitch) = (own.len() * row, union.len() * row);
    let fc = gpu.alloc(hidden * dst_pitch)?;
    gpu.memset(fc, 0, hidden * dst_pitch)?;
    let stream = gpu.default_stream();
    for (j, layer) in own.iter().enumerate() {
        let at = union
            .iter()
            .position(|u| u == layer)
            .with_context(|| format!("dual drafter layer {layer} not in the union capture"))?;
        gpu.copy_d2d_2d_async(
            w.fc.weight.offset(j * row),
            src_pitch,
            fc.offset(at * row),
            dst_pitch,
            row,
            hidden,
            stream,
        )?;
    }
    gpu.synchronize(stream)?;
    for name in ["fc.weight", "model.fc.weight"] {
        if store.contains(name) && store.get(name)?.ptr == w.fc.weight {
            store.reclaim(gpu, name)?;
        }
    }
    w.fc = DenseWeight { weight: fc };
    sub.target_layer_ids = union.to_vec();
    tracing::info!("DFlash dual drafter: fc widened from layers {own:?} to the union {union:?}");
    Ok(())
}

/// One sequence's routing between the primary (0) and alt (1) head, by
/// headroom. The short head (smaller block) proposes until it runs into its
/// cap — every draft accepted — in 3 of its last 8 rounds: the text is
/// predictable enough that a longer block would go further. The long head
/// keeps the request while its blocks go past the short head's cap; when
/// at most 1 of its last 8 rounds did, the short head does the same work
/// for less (a smaller verify), and takes the request back. Each judgement
/// is over at least 6 rounds of the head since it took over. Decisions use
/// accepted tokens only (no clocks), so they are deterministic.
#[derive(Clone, Debug, Default)]
pub struct Route {
    /// The head of the last propose.
    pub last: usize,
    /// Each head's block (`gamma`): a round of `cap` tokens accepted every draft.
    caps: [usize; 2],
    /// The proposing head, and its last rounds' emitted tokens.
    chosen: usize,
    win: std::collections::VecDeque<usize>,
}

impl Route {
    /// A route over heads of blocks `caps`, starting on the short one.
    pub fn new(caps: [usize; 2]) -> Self {
        Self {
            caps,
            chosen: usize::from(caps[1] < caps[0]),
            ..Self::default()
        }
    }

    /// The short-block head.
    pub fn short(&self) -> usize {
        usize::from(self.caps[1] < self.caps[0])
    }

    /// The head of the next propose. Idempotent until the next `record`.
    pub fn pick(&mut self) -> usize {
        let short = self.short();
        let cap = self.caps[short];
        let n = self.win.len();
        if n >= JUDGE && self.caps[0] != self.caps[1] {
            let switch = if self.chosen == short {
                8 * self.win.iter().filter(|&&t| t >= cap).count() >= 3 * n
            } else {
                8 * self.win.iter().filter(|&&t| t > cap).count() <= n
            };
            if switch {
                self.chosen = 1 - self.chosen;
                self.win.clear();
            }
        }
        self.chosen
    }

    /// A verify of the last propose emitted `tokens`.
    pub fn record(&mut self, tokens: usize) {
        if self.last != self.chosen {
            return;
        }
        self.win.push_back(tokens);
        if self.win.len() > WINDOW {
            self.win.pop_front();
        }
    }
}

impl DflashProposerState {
    /// Take the primary's shared context (accumulator, its watermarks and
    /// positions, the capture-row bookkeeping, the generation owner) into
    /// this alt state. A slid or shrunk accumulator drops this head's
    /// committed drafter context so its next propose rebuilds it.
    pub(super) fn mirror_from(&mut self, p: &DflashProposerState) {
        let slid = self.ctx_committed > 0
            && (p.ctx_len < self.ctx_committed
                || p.ctx_positions.first() != self.ctx_positions.first());
        if slid {
            self.ctx_committed = 0;
        }
        self.seq_len = p.seq_len;
        self.prefill_done = p.prefill_done;
        self.ctx_hidden_acc = p.ctx_hidden_acc;
        self.ctx_len = p.ctx_len;
        self.last_num_accepted = p.last_num_accepted;
        self.skip_next_decode_append = p.skip_next_decode_append;
        self.max_ctx_len = p.max_ctx_len;
        self.ctx_acc_rows = p.ctx_acc_rows;
        self.ctx_prefill_origin = p.ctx_prefill_origin;
        self.ctx_prefill_base = p.ctx_prefill_base;
        self.ctx_slot_bytes = p.ctx_slot_bytes;
        self.lifecycle.clone_from(&p.lifecycle);
        self.ctx_positions.clone_from(&p.ctx_positions);
        self.end_floor = p.end_floor;
        self.first_append_at = p.first_append_at;
        self.own_capture = p.own_capture;
        self.own_row_at = p.own_row_at;
        self.own_row = p.own_row;
        self.warm_ctx = p.warm_ctx;
        self.ctx_floor = p.ctx_floor;
        self.ctx_run = p.ctx_run;
    }

    /// Hand what this alt state's propose advanced back to the primary.
    pub(super) fn mirror_to(&self, p: &mut DflashProposerState) {
        p.seq_len = self.seq_len;
        p.prefill_done = self.prefill_done;
        p.ctx_len = self.ctx_len;
        p.skip_next_decode_append = self.skip_next_decode_append;
        p.ctx_prefill_origin = self.ctx_prefill_origin;
        p.ctx_prefill_base = self.ctx_prefill_base;
        p.lifecycle.clone_from(&self.lifecycle);
        p.ctx_positions.clone_from(&self.ctx_positions);
        p.first_append_at = self.first_append_at;
        p.own_capture = self.own_capture;
        p.own_row_at = self.own_row_at;
        p.ctx_floor = self.ctx_floor;
        p.ctx_run = self.ctx_run;
        p.last_num_drafted = self.last_num_drafted;
        p.last_draft_conf.clone_from(&self.last_draft_conf);
    }

    /// Detach the shared accumulator before this alt state is freed: the
    /// primary owns (and frees or carries) it.
    pub(super) fn release_shared(&mut self) {
        self.ctx_hidden_acc = spark_runtime::gpu::DevicePtr(0);
        self.own_row = None;
    }
}

/// Run `f` with sequence `d`'s alt state mirrored in, and mirror it back.
pub(super) fn with_alt<T>(
    d: &mut DflashProposerState,
    f: impl FnOnce(&mut DflashProposerState) -> Result<T>,
) -> Result<T> {
    let mut a = d
        .alt
        .take()
        .context("dual drafter state without its alt state")?;
    a.mirror_from(d);
    let r = f(&mut a);
    a.mirror_to(d);
    d.alt = Some(a);
    r
}

impl BlockDiffusionDraftHead {
    /// Install `alt` as this head's second drafter.
    pub fn set_alt(&mut self, alt: BlockDiffusionDraftHead) -> Result<()> {
        anyhow::ensure!(
            alt.target_layer_ids == self.target_layer_ids
                && alt.target_hidden_size == self.target_hidden_size
                && alt.vocab_size == self.vocab_size,
            "dual drafters must share the union capture ({:?} vs {:?}) and vocabulary",
            self.target_layer_ids,
            alt.target_layer_ids
        );
        anyhow::ensure!(
            self.rank_split.is_none() && alt.rank_split.is_none() && alt.alt.is_none(),
            "dual drafters do not split across ranks or nest"
        );
        tracing::info!(
            "DFlash dual drafter: primary γ={}{}, alt γ={}{}; per-request routing",
            self.gamma,
            if self.dspine.is_some() {
                " (DSpine)"
            } else {
                ""
            },
            alt.gamma,
            if alt.dspine.is_some() {
                " (DSpine)"
            } else {
                ""
            },
        );
        self.alt = Some(Box::new(alt));
        Ok(())
    }

    /// The head of a sequence's next propose: its router's pick with at most
    /// two sequences alive, else the short-block head (the
    /// batched verify runs 8 rows a sequence there, where the long head's
    /// extra rows buy nothing; the router keeps its state for later).
    pub(super) fn dual_pick(&self, route: &mut Route) -> usize {
        let live = self.live.load(std::sync::atomic::Ordering::Relaxed);
        let head = if live > 2 {
            route.short()
        } else {
            route.pick()
        };
        route.last = head;
        head
    }

    /// The head serving route `h` (0 = this one).
    pub(super) fn routed(&self, h: usize) -> &BlockDiffusionDraftHead {
        match (h, self.alt.as_deref()) {
            (1, Some(alt)) => alt,
            _ => self,
        }
    }
}

impl BlockDiffusionDraftHead {
    /// `propose_batch` with two drafters: each sequence's router picks its
    /// head; each head proposes for its sequences (staged when two or more).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dual_propose_batch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[spark_runtime::gpu::DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn crate::speculative::ProposerState],
        expected_owners: Option<&[super::SequenceGeneration]>,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        grammar_bitmasks: Option<&[Option<Vec<i32>>]>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let n = last_tokens.len();
        if n < 2 {
            return Ok(None);
        }
        let owners = expected_owners.context("DFlash batched propose requires expected owners")?;
        let mut heads = Vec::with_capacity(n);
        for state in states.iter_mut() {
            let d = state
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .context("Invalid DFlash proposer state")?;
            heads.push(self.dual_pick(&mut d.route));
        }
        let mut out = vec![Vec::new(); n];
        for h in 0..2 {
            let idx: Vec<usize> = (0..n).filter(|&i| heads[i] == h).collect();
            if idx.is_empty() {
                continue;
            }
            let tokens: Vec<u32> = idx.iter().map(|&i| last_tokens[i]).collect();
            let hiddens: Vec<_> = idx.iter().map(|&i| target_hiddens[i]).collect();
            let pos: Vec<usize> = idx.iter().map(|&i| positions[i]).collect();
            let own: Vec<_> = idx.iter().map(|&i| owners[i]).collect();
            let masks: Option<Vec<Option<Vec<i32>>>> = grammar_bitmasks
                .map(|m| idx.iter().map(|&i| m.get(i).cloned().flatten()).collect());
            let head = self.routed(h);
            let mut picked: Vec<&mut DflashProposerState> = states
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| heads[*i] == h)
                .map(|(_, s)| {
                    s.as_any_mut()
                        .downcast_mut::<DflashProposerState>()
                        .context("Invalid DFlash proposer state")
                })
                .collect::<Result<_>>()?;
            let drafts = if h == 0 {
                let mut group: Vec<&mut dyn crate::speculative::ProposerState> =
                    picked.iter_mut().map(|d| &mut **d as _).collect();
                head.propose_group(
                    &tokens,
                    &hiddens,
                    &pos,
                    num_drafts,
                    &mut group,
                    &own,
                    ctx,
                    stream,
                    masks.as_deref(),
                )?
            } else {
                let mut alts = Vec::with_capacity(picked.len());
                for d in picked.iter_mut() {
                    let mut a = d
                        .alt
                        .take()
                        .context("dual drafter state without its alt state")?;
                    a.mirror_from(d);
                    alts.push(a);
                }
                let r = {
                    let mut group: Vec<&mut dyn crate::speculative::ProposerState> =
                        alts.iter_mut().map(|a| &mut **a as _).collect();
                    head.propose_group(
                        &tokens,
                        &hiddens,
                        &pos,
                        num_drafts,
                        &mut group,
                        &own,
                        ctx,
                        stream,
                        masks.as_deref(),
                    )
                };
                for (d, a) in picked.iter_mut().zip(alts) {
                    a.mirror_to(d);
                    d.alt = Some(a);
                }
                r?
            };
            for (k, &i) in idx.iter().enumerate() {
                out[i] = drafts[k].clone();
            }
        }
        Ok(Some(out))
    }

    /// One head's proposes for `states`: staged when the head batches them,
    /// else one by one, as the generic serial dispatch issues them.
    #[allow(clippy::too_many_arguments)]
    fn propose_group(
        &self,
        tokens: &[u32],
        hiddens: &[spark_runtime::gpu::DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn crate::speculative::ProposerState],
        owners: &[super::SequenceGeneration],
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        masks: Option<&[Option<Vec<i32>>]>,
    ) -> Result<Vec<Vec<u32>>> {
        if tokens.len() >= 2
            && let Some(drafts) = self.propose_batch_staged(
                tokens,
                hiddens,
                positions,
                num_drafts,
                states,
                Some(owners),
                ctx,
                stream,
                masks,
                None,
            )?
        {
            return Ok(drafts);
        }
        let mut out = Vec::with_capacity(tokens.len());
        for i in 0..tokens.len() {
            out.push(self.propose_drafts(
                tokens[i],
                hiddens[i],
                positions[i],
                num_drafts,
                states[i],
                Some(owners[i]),
                ctx,
                stream,
                None,
                masks.and_then(|m| m.get(i)?.as_deref()),
                Some(hiddens[i]),
            )?);
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "dual_tests.rs"]
mod tests;
