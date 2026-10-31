// SPDX-License-Identifier: AGPL-3.0-only

//! Prompt-lookup ("copy") drafts beside DFlash2 (`ATLAS_DFLASH_COPY_DRAFTS=1`).
//!
//! A reply that quotes or edits its prompt, or repeats itself, is cheap to
//! draft: when the context's last `match` tokens (the pending token
//! included) occurred earlier in the prompt or the reply, the tokens that
//! followed that occurrence are verified in place of DFlash2's block.
//! Copies only propose: every row still verifies against the target's own
//! pick, so an accepted copy is the token the target chose there.
//!
//! **Exact verification, window-dependent numerics; an experiment.** The
//! accept rule is unchanged, so every emitted token is the target's pick at
//! its own verify row whatever proposed it. But a copy changes how many
//! tokens a step accepts, so later positions are verified in a different
//! window (width and start), and this engine's verify numerics depend on the
//! window: greedy text can differ from flag-off at near-ties, as it does
//! under `ATLAS_DFLASH_CONF_WIDTH` (2026-09-30 decode campaign). On the
//! GLM profile the known width-dependent kernels are the sparse-MLA verify
//! split count, which follows the owner's rows unless
//! `ATLAS_GLM_SPARSE_VERIFY_SPLIT_PIN=1` (`glm_sparse_prefill_split`), and
//! the per-row-count tiers of the cuBLAS / MXFP8 projections. The kernels
//! documented as per-row (causal attention, the indexer's top-k, MoE
//! routing, per-row activation scales) keep a rejected row's content out of
//! the kept rows, so a copy should differ from a drafter's block only
//! through the window it leaves. Keep it out of serving profiles until a
//! hardware A/B has measured it.
//!
//! - **Index.** Each request keeps an incremental n-gram index of its
//!   context on the host: the newest end of every `match`-gram, and per
//!   position a link to the previous end of the same gram. A step indexes
//!   only what it committed (well under a microsecond), and a long prompt
//!   [`INDEX_BUDGET`] positions a step; a rewritten context (rollback) is
//!   re-indexed. At most
//!   [`MAX_INDEXED`] positions: the links take 4 bytes a position (2 MiB at
//!   most), the gram table 10–21 bytes a distinct gram (9 MiB at most, 14 MiB
//!   while it grows), freed with the request.
//! - **Proposal.** The latest earlier occurrence with `k` tokens after it,
//!   else the latest one continued over its own copy (a repeat whose period
//!   is shorter than `k`). Candidates are checked token by token, so a hash
//!   collision only costs a candidate, and the walk is capped at
//!   [`MAX_HOPS`].
//! - **Merge.** A copy that differs from DFlash2's block replaces it; one
//!   DFlash2 already agrees with leaves it alone. `k` never exceeds the
//!   drafts DFlash2 proposed, so a verify stays within the widths DFlash2
//!   uses (and the M16/M32 verify kernels it runs on). Copies carry
//!   [`COPY_CONF`] (of the prompt) or [`COPY_REPLY_CONF`] (of the reply) for
//!   confidences, so `ATLAS_DFLASH_CONF_WIDTH` prices them by the copies'
//!   own measured acceptance and never by DFlash2's.
//!
//! Knobs (read once): `ATLAS_DFLASH_COPY_MATCH` (default 8, 2..=64) tokens
//! that must match; `ATLAS_DFLASH_COPY_MAX` (default and at most
//! [`MAX_DRAFTS`]) drafts a copy; `ATLAS_DFLASH_COPY_REPLY_MATCH` (0 = off,
//! else more than the match, up to 64) tokens an occurrence inside the
//! reply must match (code replies repeat short boilerplate whose
//! continuations differ); `ATLAS_DFLASH_COPY_MISS_MAX` (0 = off) drafts a
//! copy after a copy that was not kept whole, until one is.
//!
//! Each request's `Done:` line is followed by a `COPY DRAFTS` line with its
//! copy rounds, drafts and accepted drafts, of the prompt and of the reply.
//! They count the verifies that settle through `verify_dflash_tail` (the
//! per-sequence and GLM owner-batched paths); the single-GPU batched eager
//! verify drops confidences, the drafter's as well, and books neither.
//!
//! Prior art: the policy (match length, latest occurrence with a full
//! continuation, copy over the drafter's block, the reply-match and
//! narrow-after-miss refinements) follows MiaAI-Lab's TensorFold recipe
//! (<https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks>,
//! `patches/0007-glm-copy-drafts.patch` and
//! `patches/0032-glm-code-copy-drafts.patch`, Apache-2.0; patches of
//! TensorFold by ashhart), as do the copy bins' priors. Ideas, no code; see
//! docs/glm-prior-art.md. Ours: the incremental index, the overlap
//! continuation, the calibrated copy bins in the confidence width and the
//! merge with DFlash2's block.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::OnceLock;

use super::ActiveSeq;
use super::dflash_conf_width::{COPY_CONF, COPY_REPLY_CONF, is_copy};
use super::dflash_width::MAX_DRAFTS;

/// Positions indexed at most: the 512K context limit (a longer context
/// copies from its first 512K positions).
const MAX_INDEXED: usize = 1 << 19;
/// Positions one offer indexes at most, so a long prompt is indexed over
/// its first steps instead of stalling one: 0.2–0.3 ms a step, the first
/// (which allocates the index) up to about 1 ms, on an Apple M-series core;
/// on a GB10 host core 8.5 ms over the 32 steps of a 512K prompt, the first
/// 2.5 ms, and 0.1 µs a decode step (`copy_drafts_timing`, 2026-10-08).
const INDEX_BUDGET: usize = 1 << 14;
/// Earlier occurrences a proposal checks at most.
const MAX_HOPS: usize = 32;
const NONE: u32 = u32::MAX;

/// The copy-draft knobs; `None` from [`settings`] with copy drafts off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Settings {
    match_len: usize,
    max: usize,
    reply_match: usize,
    miss_max: usize,
}

impl Settings {
    /// Parse the knobs from `var` (an environment lookup); out-of-range
    /// values take their defaults.
    fn parse(var: impl Fn(&str) -> Option<String>) -> Option<Self> {
        if var("ATLAS_DFLASH_COPY_DRAFTS").as_deref() != Some("1") {
            return None;
        }
        let num = |name: &str, default: usize, ok: &dyn Fn(usize) -> bool| {
            var(name)
                .and_then(|v| v.trim().parse().ok())
                .filter(|&v| ok(v))
                .unwrap_or(default)
        };
        let match_len = num("ATLAS_DFLASH_COPY_MATCH", 8, &|v| (2..=64).contains(&v));
        let max = num("ATLAS_DFLASH_COPY_MAX", MAX_DRAFTS, &|v| {
            (1..=MAX_DRAFTS).contains(&v)
        });
        Some(Self {
            match_len,
            max,
            reply_match: num("ATLAS_DFLASH_COPY_REPLY_MATCH", 0, &|v| {
                v > match_len && v <= 64
            }),
            miss_max: num("ATLAS_DFLASH_COPY_MISS_MAX", 0, &|v| v <= max),
        })
    }
}

fn settings() -> Option<&'static Settings> {
    static S: OnceLock<Option<Settings>> = OnceLock::new();
    S.get_or_init(|| Settings::parse(|name| std::env::var(name).ok()))
        .as_ref()
}

/// A request's context: the committed tokens, then the pending one.
#[derive(Clone, Copy)]
struct Context<'a> {
    history: &'a [u32],
    pending: u32,
}

impl Context<'_> {
    fn len(&self) -> usize {
        self.history.len() + 1
    }

    fn at(&self, i: usize) -> u32 {
        self.history.get(i).copied().unwrap_or(self.pending)
    }

    /// Key of the `n` tokens ending at `end`.
    fn key(&self, end: usize, n: usize) -> u32 {
        let h = crate::ngram::hash_tokens((end + 1 - n..=end).map(|i| self.at(i)));
        (h ^ (h >> 32)) as u32
    }

    /// Whether the `n` tokens ending at `a` equal those ending at `b`.
    fn same(&self, a: usize, b: usize, n: usize) -> bool {
        a + 1 >= n && b + 1 >= n && (0..n).all(|i| self.at(a - i) == self.at(b - i))
    }
}

/// The gram table's hasher. Its keys are FNV-1a hashes already, so one
/// multiply spreads them over the bits the table reads (the bucket from the
/// low bits, the tag from the top seven) without SipHash's cost.
#[derive(Default)]
struct KeyHasher(u64);

const MIX: u64 = 0x9E37_79B9_7F4A_7C15;

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Only `u32` keys reach the table; other input still hashes.
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(b)).wrapping_mul(MIX);
        }
    }

    fn write_u32(&mut self, key: u32) {
        self.0 = u64::from(key).wrapping_mul(MIX);
    }
}

/// One request's n-gram index: the newest end of each gram, and per end
/// the previous end of the same gram.
#[derive(Debug, Default)]
struct Index {
    n: usize,
    /// Context positions before this are the prompt.
    prompt: usize,
    head: HashMap<u32, u32, BuildHasherDefault<KeyHasher>>,
    prev: Vec<u32>,
    /// Grams ending before this position are indexed.
    indexed: usize,
    /// Key of the gram ending at `indexed - 1` (detects a rewrite).
    check: u32,
}

impl Index {
    fn new(n: usize, prompt: usize) -> Self {
        Self {
            n: n.max(1),
            prompt,
            ..Self::default()
        }
    }

    /// Index the grams that have a token after it (end before the last
    /// position), [`INDEX_BUDGET`] at most, from scratch when the indexed
    /// part was rewritten.
    fn sync(&mut self, ctx: Context) {
        let n = self.n;
        let last = ctx.len() - 1;
        if self.indexed > 0 && (self.indexed > last || ctx.key(self.indexed - 1, n) != self.check) {
            *self = Self::new(n, self.prompt);
        }
        let end = last
            .min(MAX_INDEXED)
            .min(self.indexed.max(n - 1) + INDEX_BUDGET);
        if end < n {
            return;
        }
        if self.indexed == 0 {
            // Room for the whole context up front: a table that grew while
            // indexing a long prompt would rehash on the step's critical path.
            let room = last.min(MAX_INDEXED);
            self.head.reserve(room);
            self.prev.reserve_exact(room);
        }
        if self.prev.len() < end {
            // Doubling, but never past what can be indexed.
            let room = end.max(2 * self.prev.len()).min(MAX_INDEXED);
            self.prev.reserve_exact(room - self.prev.len());
            self.prev.resize(end, NONE);
        }
        for e in self.indexed.max(n - 1)..end {
            let key = ctx.key(e, n);
            self.prev[e] = self.head.insert(key, e as u32).unwrap_or(NONE);
            self.check = key;
        }
        self.indexed = self.indexed.max(end);
    }

    /// Whether the occurrence of a gram ending at `e` starts inside the reply.
    fn in_reply(&self, e: usize) -> bool {
        e + 1 >= self.prompt + self.n
    }

    /// Whether the occurrence ending at `e` matches the context's tail,
    /// with `reply_match` tokens when it starts inside the reply.
    fn matches(&self, ctx: Context, e: usize, reply_match: usize) -> bool {
        let tail = ctx.len() - 1;
        let n = if reply_match > self.n && self.in_reply(e) {
            reply_match
        } else {
            self.n
        };
        ctx.same(e, tail, n)
    }

    /// The end of the earlier occurrence of the context's last `n` tokens to
    /// copy `k` drafts after (module doc); none when it never occurred.
    fn find(&mut self, ctx: Context, k: usize, reply_match: usize) -> Option<usize> {
        let len = ctx.len();
        if k == 0 || len <= self.n {
            return None;
        }
        self.sync(ctx);
        let mut e = self
            .head
            .get(&ctx.key(len - 1, self.n))
            .copied()
            .unwrap_or(NONE);
        let mut latest = None;
        for _ in 0..MAX_HOPS {
            if e == NONE {
                break;
            }
            let at = e as usize;
            if self.matches(ctx, at, reply_match) {
                if at + k < len {
                    return Some(at);
                }
                latest.get_or_insert(at);
            }
            e = self.prev[at];
        }
        latest
    }
}

/// The `k` tokens after position `e`, continuing over the copy itself past
/// the context's end.
fn copy_after(ctx: Context, e: usize, k: usize) -> Vec<u32> {
    let len = ctx.len();
    let mut out = Vec::with_capacity(k);
    for p in e + 1..e + 1 + k {
        let token = if p < len { ctx.at(p) } else { out[p - len] };
        out.push(token);
    }
    out
}

/// Put `copy` (whose drafts carry `copy_conf`) in place of DFlash2's
/// `drafts` with confidences `conf`, unless it is empty or DFlash2's block
/// already starts with it. Returns whether it did.
fn merge(copy: &[u32], copy_conf: f32, drafts: &mut Vec<u32>, conf: &mut Vec<f32>) -> bool {
    if copy.is_empty() || drafts.starts_with(copy) {
        return false;
    }
    drafts.clear();
    drafts.extend_from_slice(copy);
    conf.clear();
    conf.resize(copy.len(), copy_conf);
    true
}

/// A request's copy-draft state, embedded in its `AdaptState`.
#[derive(Debug, Default)]
pub(crate) struct CopyState {
    index: Option<Index>,
    /// Context length of the last offer (one offer a round).
    offered: usize,
    /// The last settled copy round kept fewer than all of its drafts.
    missed: bool,
    /// Copy rounds, drafts and accepted drafts: of the prompt, of the reply.
    rounds: [u64; 2],
    drafted: [u64; 2],
    accepted: [u64; 2],
}

impl CopyState {
    /// The state a preempted request resumes with: the index and counters are
    /// host-side and its context is unchanged, so only the offer is renewed.
    pub(super) fn resumed(self) -> Self {
        Self { offered: 0, ..self }
    }
}

/// Offer a copy in place of `a`'s pending DFlash2 drafts (no-op with copy
/// drafts off).
pub(super) fn offer(a: &mut ActiveSeq) {
    if let Some(s) = settings() {
        offer_with(a, s);
    }
}

fn offer_with(a: &mut ActiveSeq, s: &Settings) {
    if a.grammar_state.is_some() || a.pending_drafts.is_empty() {
        return;
    }
    let ctx = Context {
        history: &a.seq.tokens,
        pending: a.last_token,
    };
    let state = &mut a.spec_adapt.copy;
    if state.offered == ctx.len() {
        return;
    }
    state.offered = ctx.len();
    let mut k = s.max.min(a.pending_drafts.len());
    if state.missed && s.miss_max > 0 {
        k = k.min(s.miss_max);
    }
    let prompt = ctx.len().saturating_sub(a.output_tokens.len());
    let index = state
        .index
        .get_or_insert_with(|| Index::new(s.match_len, prompt));
    if let Some(at) = index.find(ctx, k, s.reply_match) {
        let copy_conf = if index.in_reply(at) {
            COPY_REPLY_CONF
        } else {
            COPY_CONF
        };
        let copy = copy_after(ctx, at, k);
        merge(
            &copy,
            copy_conf,
            &mut a.pending_drafts,
            &mut a.pending_draft_conf,
        );
    }
}

/// After `a`'s pending drafts were cut, cut a copy's confidences with them.
/// A drafter's are left as they were (stale), as before copies existed.
pub(super) fn cut_conf(a: &mut ActiveSeq) {
    if is_copy(&a.pending_draft_conf) {
        a.pending_draft_conf.truncate(a.pending_drafts.len());
    }
}

/// Book a verify of `drafted` drafts with confidences `conf` that accepted
/// `accepted`; only copy rounds count.
pub(super) fn settle(state: &mut CopyState, conf: &[f32], drafted: usize, accepted: usize) {
    if !is_copy(conf) {
        return;
    }
    let source = usize::from(conf[0] >= COPY_REPLY_CONF);
    state.missed = accepted < drafted;
    state.rounds[source] += 1;
    state.drafted[source] += drafted as u64;
    state.accepted[source] += accepted as u64;
}

/// The request's copy line after its `Done:` line (copy rounds only).
pub(super) fn log_done(state: &CopyState) {
    if state.rounds.iter().any(|&r| r > 0) {
        let [p, r] = [0, 1].map(|i| (state.rounds[i], state.drafted[i], state.accepted[i]));
        tracing::info!(
            "COPY DRAFTS prompt rounds={} drafted={} accepted={} reply rounds={} drafted={} accepted={}",
            p.0,
            p.1,
            p.2,
            r.0,
            r.1,
            r.2
        );
    }
}

#[cfg(test)]
#[path = "copy_drafts_tests.rs"]
mod tests;
