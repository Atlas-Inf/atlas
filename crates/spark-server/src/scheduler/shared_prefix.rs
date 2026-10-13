// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-prefix admission (`ATLAS_GLM_PC_INFLIGHT=1`, default off): the
//! scheduler half of in-flight shared-prefix checkpoints. The model half,
//! with the exactness caveat and the rank-lockstep argument, is
//! `spark-model/src/model/trait_impl/prefill_b/pc_inflight.rs`.
//!
//! Agentic clients open several new conversations at once with one long
//! system prompt (measured on qwen4_exp TP2: eight 22.6K-token first turns
//! sharing 22.5K tokens, TTFT 15-36 s, against 8.5 s for one cold prefill and
//! 0.5 s for a warm turn). Admitted together, they each prefill the shared
//! part, and later ones replay up to 6K tokens from the nearest chunk-boundary
//! checkpoint.
//!
//! Each tick, before new requests start, this pass compares every drained
//! request with the prefills in flight and with the requests admitted ahead
//! of it in the same tick. A request sharing at least
//! `ATLAS_GLM_PC_BRANCH_MIN` tokens (block-aligned) more than any checkpoint
//! already computed for it (`passed`) with a prefill (the leader) that has
//! not yet computed that far follows the leader:
//!
//! - the leader plants a checkpoint at the shared boundary (`Model::pc_plant`
//!   before its first chunk, or between its chunks when it is in flight), or
//!   keeps the shallower one it already plants;
//! - the follower is held: it goes back to the front of the pending queue in
//!   arrival order and is re-planned next tick, so it starts once the leader
//!   is past the checkpoint (or gone), restores there and computes its own
//!   suffix.
//!
//! The leader thus prefills alone at full speed. Fairness: a follower waits
//! only while its leader, which arrived first, advances through the shared
//! prefix; it would otherwise have been prefilling that prefix behind the
//! leader in the FIFO prefill queue. Requests sharing nothing (or too little)
//! are admitted exactly as before, and the pending queue keeps arrival order
//! (SLAI still admits the oldest first). Held requests stay counted in the
//! drain's batch cap, as they would be once admitted.
//!
//! Only on multi-rank worlds (`Model::is_ep`): every prefill chunk there runs
//! through `prefill_chunk`, which honours plants. The single-GPU fused and
//! batched prefill paths (mixed forward, co-dispatch, VARLEN) do not, so a
//! follower there would wait for a checkpoint nobody saves.

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use spark_model::traits::Model;

use super::types::{PendingQueue, PrefillInProgress};
use crate::api::InferenceRequest;

/// A prompt the planner compares: its tokens and its adapter (the cache key).
#[derive(Clone, Copy, Debug)]
pub(super) struct Prompt<'a> {
    pub tokens: &'a [u32],
    pub adapter: u64,
}

/// A prefill that requests can wait on.
#[derive(Clone, Copy, Debug)]
pub(super) struct Leader<'a> {
    pub prompt: Prompt<'a>,
    /// Prompt tokens it has computed.
    pub done: usize,
    /// The checkpoint it was asked to plant.
    pub plant: Option<usize>,
}

/// What a tick does with one drained request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Admit {
    /// Start it; `Some(at)` plants a checkpoint at `at` in its prefill.
    Now(Option<usize>),
    /// Hold it until the prefill it shares a prefix with is past it.
    Hold,
}

/// One tick's plan.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Plan {
    /// Per drained request, in order.
    pub admit: Vec<Admit>,
    /// New or moved plants for prefills in flight: (index, position).
    pub replant: Vec<(usize, usize)>,
}

/// The block-aligned prefix `a` shares with `b`, capped at `a`'s last block
/// boundary strictly below its end (a restore must leave a row to compute).
pub(super) fn shared_boundary(a: &[u32], b: &[u32], bs: usize) -> usize {
    let cap = (a.len().saturating_sub(1) / bs * bs).min(b.len() / bs * bs);
    let mut at = 0;
    while at < cap && a[at..at + bs] == b[at..at + bs] {
        at += bs;
    }
    at
}

/// The deepest checkpoint at or under `upto` believed computed for `r`.
fn known(r: Prompt<'_>, upto: usize, passed: &[(Prompt<'_>, usize)], bs: usize) -> usize {
    passed
        .iter()
        .filter(|(p, at)| {
            p.adapter == r.adapter && *at <= upto && shared_boundary(r.tokens, p.tokens, bs) >= *at
        })
        .map(|&(_, at)| at)
        .max()
        .unwrap_or(0)
}

/// The leader `r` waits on and the plant that needs, if any: the one that
/// leaves `r` the deepest restore, gaining at least `min` tokens over what is
/// already computed for it.
fn follow(
    r: Prompt<'_>,
    leaders: &[Leader<'_>],
    passed: &[(Prompt<'_>, usize)],
    min: usize,
    bs: usize,
) -> Option<(usize, Option<usize>)> {
    let mut best: Option<(usize, usize, Option<usize>)> = None;
    for (j, c) in leaders.iter().enumerate() {
        if c.prompt.adapter != r.adapter {
            continue;
        }
        let s = shared_boundary(r.tokens, c.prompt.tokens, bs);
        if s < min.max(bs) {
            continue;
        }
        // (restore depth for r, leader progress it waits for, plant needed)
        let (depth, wait, plant) = if s + 2 * bs < c.prompt.tokens.len() {
            match c.plant {
                Some(p) if p <= s && p > c.done => (p, p, None),
                _ => (s, s, Some(s)),
            }
        } else {
            // At the leader's tail: its tail checkpoint serves, once it ends.
            (s, c.prompt.tokens.len(), None)
        };
        if wait <= c.done || depth < known(r, depth, passed, bs) + min {
            continue;
        }
        if best.is_none_or(|(d, ..)| depth > d) {
            best = Some((depth, j, plant));
        }
    }
    best.map(|(_, j, plant)| (j, plant))
}

/// Plan one tick: `inflight` are the prefills in flight in service order,
/// `new` the drained requests in admission order (`None`: not plannable,
/// e.g. vision), `passed` the checkpoints believed computed.
pub(super) fn plan(
    inflight: &[Leader<'_>],
    new: &[Option<Prompt<'_>>],
    passed: &[(Prompt<'_>, usize)],
    min: usize,
    bs: usize,
) -> Plan {
    let mut leaders = inflight.to_vec();
    let mut owner = Vec::new();
    let mut admit = Vec::with_capacity(new.len());
    for (i, r) in new.iter().enumerate() {
        let Some(r) = *r else {
            admit.push(Admit::Now(None));
            continue;
        };
        match follow(r, &leaders, passed, min, bs) {
            Some((j, plant)) => {
                if plant.is_some() {
                    leaders[j].plant = plant;
                }
                admit.push(Admit::Hold);
            }
            None => {
                admit.push(Admit::Now(None));
                owner.push(i);
                leaders.push(Leader {
                    prompt: r,
                    done: 0,
                    plant: None,
                });
            }
        }
    }
    for (&i, l) in owner.iter().zip(&leaders[inflight.len()..]) {
        admit[i] = Admit::Now(l.plant);
    }
    let replant = inflight
        .iter()
        .zip(&leaders)
        .enumerate()
        .filter_map(|(j, (was, now))| (was.plant != now.plant).then_some((j, now.plant?)))
        .collect();
    Plan { admit, replant }
}

/// How many computed checkpoints the scheduler remembers.
const PASSED_CAP: usize = 16;

/// A leader prompt and a checkpoint position in it.
type Anchor = (Arc<Vec<u32>>, u64, usize);

/// The scheduler's memory of the checkpoints it asked for.
#[derive(Default)]
pub(super) struct SharedPrefix {
    /// Plants sent whose leader has not yet been seen past them.
    planted: Vec<Anchor>,
    /// Checkpoints believed computed, most recent last.
    passed: VecDeque<Anchor>,
    /// Requests held last tick (logs on change).
    held: usize,
}

impl SharedPrefix {
    /// Move the plants whose leader is past them, or gone, to `passed`.
    fn settle(&mut self, prefilling: &[PrefillInProgress]) {
        let (passed, waiting): (Vec<Anchor>, Vec<Anchor>) =
            self.planted.drain(..).partition(|(t, _, at)| {
                prefilling
                    .iter()
                    .find(|p| Arc::ptr_eq(&p.prompt_tokens, t))
                    .is_none_or(|p| p.chunk_offset >= *at)
            });
        self.planted = waiting;
        for a in passed {
            if self.passed.len() == PASSED_CAP {
                self.passed.pop_front();
            }
            self.passed.push_back(a);
        }
    }

    /// Record a plant sent for `tokens`, replacing an earlier one.
    fn plant(&mut self, tokens: &Arc<Vec<u32>>, adapter: u64, at: usize) {
        self.planted.retain(|(t, ..)| !Arc::ptr_eq(t, tokens));
        self.planted.push((tokens.clone(), adapter, at));
    }
}

/// Hold this tick's followers (back to the front of `pending`) and plant
/// their leaders' checkpoints. Returns the requests to start now, each with
/// the checkpoint its own prefill plants. A no-op unless the model takes
/// plants (`Model::pc_inflight_min_tokens`) on a multi-rank world with
/// chunked prefill.
pub(super) fn admit(
    model: &dyn Model,
    pending: &Arc<(Mutex<PendingQueue>, Condvar)>,
    new_reqs: Vec<InferenceRequest>,
    prefilling: &mut [PrefillInProgress],
    chunked: bool,
    state: &mut SharedPrefix,
) -> (Vec<InferenceRequest>, Vec<Option<usize>>) {
    // Cheapest first: this runs every tick.
    let on = !new_reqs.is_empty() && chunked && model.is_ep();
    let min = if on {
        model.pc_inflight_min_tokens()
    } else {
        None
    };
    let knobs = min.and_then(|min| model.kv_block_size().map(|bs| (min, bs)));
    let Some((min, bs)) = knobs else {
        let none = vec![None; new_reqs.len()];
        return (new_reqs, none);
    };
    state.settle(prefilling);
    let new_tokens: Vec<_> = new_reqs.iter().map(|r| r.prompt_tokens_arc()).collect();
    let adapters: Vec<_> = new_reqs
        .iter()
        .map(|r| model.adapter_id_for(r.adapter_slot()))
        .collect();
    let plan = {
        let inflight: Vec<_> = prefilling
            .iter()
            .map(|p| Leader {
                prompt: Prompt {
                    tokens: &p.prompt_tokens,
                    adapter: p.seq.adapter_id,
                },
                done: p.chunk_offset,
                plant: p.seq.pc_plant_at,
            })
            .collect();
        let new: Vec<_> = new_reqs
            .iter()
            .zip(new_tokens.iter().zip(&adapters))
            .map(|(r, (t, &adapter))| {
                (!r.has_image_pixels()).then_some(Prompt { tokens: t, adapter })
            })
            .collect();
        let passed: Vec<_> = state
            .passed
            .iter()
            .map(|(t, adapter, at)| {
                let prompt = Prompt {
                    tokens: t,
                    adapter: *adapter,
                };
                (prompt, *at)
            })
            .collect();
        plan(&inflight, &new, &passed, min, bs)
    };
    for &(j, at) in &plan.replant {
        let p = &mut prefilling[j];
        tracing::info!(
            "shared-prefix admission: planting a checkpoint at token {at} in a prefill \
             in flight ({} of {} tokens done)",
            p.chunk_offset,
            p.prompt_tokens.len(),
        );
        if let Err(e) = model.pc_plant(&mut p.seq, at) {
            tracing::error!("shared-prefix admission: plant failed: {e:#}");
            continue;
        }
        state.plant(&p.prompt_tokens, p.seq.adapter_id, at);
    }
    let (mut now, mut plants, mut held) = (Vec::new(), Vec::new(), Vec::new());
    for (i, (req, admit)) in new_reqs.into_iter().zip(&plan.admit).enumerate() {
        match *admit {
            Admit::Hold => held.push(req),
            Admit::Now(plant) => {
                if let Some(at) = plant {
                    tracing::info!(
                        "shared-prefix admission: a new {}-token prefill plants a \
                         checkpoint at token {at}",
                        new_tokens[i].len(),
                    );
                    state.plant(&new_tokens[i], adapters[i], at);
                }
                now.push(req);
                plants.push(plant);
            }
        }
    }
    if held.len() != state.held {
        tracing::info!(
            "shared-prefix admission: {} request(s) wait for a prefill in flight to pass \
             their shared prefix",
            held.len(),
        );
        state.held = held.len();
    }
    if !held.is_empty() {
        let mut g = pending.0.lock();
        for (i, req) in held.into_iter().enumerate() {
            g.requests.insert(i, req);
        }
    }
    (now, plants)
}

#[cfg(test)]
#[path = "shared_prefix_tests.rs"]
mod tests;
