// SPDX-License-Identifier: AGPL-3.0-only

//! The fixed overhead of a prefix-cached ("warm") turn: what a request pays
//! before its first token that does not depend on its new tokens.
//!
//! # A warm turn, step by step
//!
//! `N` prompt tokens, of which the radix tree matches `matched` and an SSM
//! snapshot restores `restored <= matched`; chunks of `C` tokens
//! (`--max-prefill-tokens`), 16-token blocks. Both ranks run every model step
//! in lockstep; the worker follows the head's commands. Cost class in
//! brackets: `fixed` per request, `chunk` per prefill chunk, `pass` per chunk
//! that computes, `cached` per cached token or block, `new` per new row.
//!
//! 1. API thread: parse, render the template, tokenize the whole prompt
//!    [cached + new, host]. A client sees it; the scheduler's `TTFT=` (from
//!    the scheduler taking the request to its first token) does not.
//!    `ATLAS_CHAT_PHASE_TIMING=1` logs it (`template_render_and_tokenize`).
//!    Nothing here reduces it.
//! 2. Scheduler: allocate the sequence, send the request preamble (native
//!    fence, vision state) [fixed, a few 4-byte broadcasts].
//! 3. Per chunk, the head sends the chunk command: slot, command, chunk
//!    length, chunk start, prompt length [chunk, five 4-byte broadcasts, each
//!    a stream sync and a device read on the worker], then the whole prompt
//!    [chunk x cached: a pageable copy to the device, one broadcast, a sync
//!    and a read of `4 N` bytes on the worker].
//! 4. Per chunk, both ranks: zero the whole buffer arena [chunk, fixed
//!    size], embed the chunk [chunk x chunk rows]. A chunk below `restored`
//!    then computes nothing: it reserves its blocks (one min-vote, two 4-byte
//!    broadcasts), appends its tokens and returns.
//!    `ATLAS_GLM_WARM_SKIP_CACHED` does not zero or embed for such a chunk
//!    (`prefill_b::warm`; multi-rank worlds).
//! 5. Chunk 0 only: the radix walk and its references [cached blocks, host],
//!    the match min-vote [fixed], the restore-depth agreement with
//!    `ATLAS_GLM_PC_EVICT` or `_BRANCH` [fixed, one to three votes], the
//!    snapshot restore [fixed: one copy of the SSM state per rank].
//! 6. After every chunk, cached or not, both ranks normalize the sequence's
//!    SSM state [chunk: one launch per SSM layer], and the scheduler ends its
//!    tick. While another sequence decodes, the next chunk therefore waits
//!    for that sequence's decode or verify step [chunk x one step of the
//!    others]: five steps for the cached chunks of a 45K-token turn, over
//!    sixty at 512K. `ATLAS_GLM_WARM_CHUNK_RUN` (`spark-server`,
//!    `phase_continue_prefills`) runs the chunk after a cached one in the
//!    same tick.
//! 7. The chunk holding `restored` and every later one computes rows
//!    `[max(start, restored), end)`: the block-table upload [cached blocks on
//!    the first pass, new blocks after], positions and slots [new], a stream
//!    sync, then all layers [pass: fixed launches and collectives per layer;
//!    new: the weights each row's experts sweep, which dominates; new x
//!    context: attention and the index over the cached rows].
//! 8. The last chunk is split at the tail cut (`pc_policy::tail_cut`), so a
//!    warm turn runs two passes: `[restored, cut)`, then the checkpoint save
//!    at `cut` [fixed: one state copy, and the radix insert over the cached
//!    blocks], then `[cut, N)`. Each pass pays step 4 again. A turn whose
//!    restore depth is `cut` already has an empty first half, which
//!    `ATLAS_GLM_WARM_SKIP_CACHED` makes free. The cut sits 17 to 32 rows
//!    under `N`, and the next turn replays them [a fixed number of rows,
//!    each priced as `new`]; `ATLAS_GLM_TAIL_CUT_DEEP` ([`tail_cut_at`])
//!    moves it one block up. That changes pass shapes, so it is its own
//!    switch.
//! 9. Final norm and LM head on the last row [fixed: one sweep of the head],
//!    the radix insert of the prompt [cached blocks, host], then the
//!    scheduler reads the logits and samples [fixed].
//!
//! `ATLAS_GLM_WARM_TRACE` logs one line per request and rank with the time
//! of steps 3 to 9 and a hash of the logits step 9 leaves ([`Trace`]). Its
//! `wall` minus its spans is the time between the request's chunks: command
//! words, the normalization and the wait of step 6. `ATLAS_PROFILE_PREFILL`
//! (per chunk, host submit time) and the scheduler's `Done: ... TTFT=` line
//! were there before; the per-request sum, the prompt transfer, the lookup,
//! the finish and any fingerprint of a warm prefill's result were not.
//! Steps 1 and 2 are outside the line (`CHAT_PHASE`, and `TTFT=` minus
//! `wall`).
//!
//! # What is not removed, and why
//!
//! * The arena zero of a chunk that computes, by default. The arena is
//!   shared scratch with layouts that do not follow the row count (the index
//!   logits of a row group over the whole context, the FlashKDA workspace,
//!   the MLA absorbed query), and `decode_a` documents a path that read rows
//!   it had not written. Zeroing "the rows this chunk uses" is therefore not
//!   the zero every pass starts from today. [`ZeroRows`] zeroes what earlier
//!   passes may have dirtied instead, behind `ATLAS_GLM_ZERO_ROWS`, with a
//!   check mode that measures whether that bound holds on a workload.
//! * The second pass of the tail split. One pass over `[restored, N)` runs
//!   the same math with other GEMM shapes, MoE groups and KDA pieces
//!   (`pc_policy`, "Accumulation order"), and the checkpoint at `cut` would
//!   need the KDA recurrence split in-pass, which only the `atlas_scale`
//!   build has (`midchunk_capture`).
//! * The full-chunk embed of a chunk that computes only its tail. Its rows
//!   past the computed ones are what a kernel reading beyond its rows sees
//!   today; skipping it would change them.
//! * The prompt in every chunk command (step 3). Sending only what the
//!   worker lacks changes the wire for what is derived as 0.3 ms a chunk at
//!   45K tokens and 2 ms at 512K; the trace's `transfer` span measures it.
//!
//! # Rank parity
//!
//! `ATLAS_GLM_TAIL_CUT_DEEP` sets the rows of every rank's passes: ranks
//! with different values run collectives of different sizes and the pair
//! hangs. `ATLAS_GLM_WARM_SKIP_CACHED` and `ATLAS_GLM_ZERO_ROWS` add no
//! command and no collective, and a request's own passes start from the same
//! arena either way; but a decode step of another sequence that runs between
//! two chunks finds what the last chunk left, so ranks with different values
//! hand that step different leftovers. All three must be the same on every
//! rank ([`tail_cut_deep`], [`skip_cached_requested`], [`ZeroRows::from_env`]
//! are what a startup agreement reads). `ATLAS_GLM_WARM_TRACE` and the
//! scheduler's `ATLAS_GLM_WARM_CHUNK_RUN` are local to a rank.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};
use parking_lot::Mutex;

#[cfg(test)]
#[path = "warm_turn_tests.rs"]
mod tests;

fn env_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// `ATLAS_GLM_WARM_SKIP_CACHED=1`: see "Rank parity" above.
pub(in crate::model) fn skip_cached_requested() -> bool {
    env_on("ATLAS_GLM_WARM_SKIP_CACHED")
}

/// `ATLAS_GLM_TAIL_CUT_DEEP=1`: see [`tail_cut_at`] and "Rank parity" above.
/// Read once; a model other than GLM-5 refuses it at load.
pub(in crate::model) fn tail_cut_deep() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_on("ATLAS_GLM_TAIL_CUT_DEEP"))
}

/// The tail-split cut for a `total`-token prompt
/// (`prefill_chunk_dispatch_with`): where the last chunk is split and the
/// checkpoint the conversation's next turn restores is saved.
///
/// Base: one block below `spark_runtime::ssm_tail_boundary`, the last block
/// boundary strictly under `total`. That is at or below the next turn's
/// match even when the template's generation-only suffix is not reproduced in
/// the re-rendered history and crosses the boundary; the next turn then
/// replays the 17 to 32 rows from there to this prompt's end, plus what is
/// new.
///
/// `deep` (`ATLAS_GLM_TAIL_CUT_DEEP=1`, default off, GLM-5 only): that
/// boundary itself, so 2 to 16 rows, 16 fewer on a warm turn, and a final
/// pass of this prompt of 2 to 16 rows instead of 18 to 32. It is where
/// `ATLAS_SSM_TAIL_CKPT` put its checkpoint, which stayed off because the
/// extra pass to land it cost what the replay it saved was worth; here that
/// pass exists already (the base cut's) and only moves. The next turn can
/// restore at the deep cut only when its match reaches this prompt's last
/// whole block, that is when its history reproduces this prompt to its end.
/// GLM-5's template does (`<|assistant|><think>` is rendered again, whatever
/// follows); a template that does not, or a turn that edits the last
/// message, leaves the checkpoint above the match, and the turn falls back
/// to the one before and replays that whole turn.
///
/// Two prompts keep the base cut: one that ends one row past a boundary (its
/// final pass would be a single row, which takes the decode path of
/// `forward_layers`; that path serves a prefill today only when a chunk
/// boundary leaves a one-row last chunk), and one of 32 tokens or fewer (the
/// base cut does not split it, and a checkpoint at 16 is under the restore
/// floor).
///
/// Same kernels and math, other pass shapes: like any change of where a
/// pass starts, the result can differ bitwise from the base cut's (see
/// `pc_policy`, "Accumulation order"), on cold prefills too. A warm turn
/// also hands the DFlash drafter 16 fewer context rows, since its context is
/// filled from the rows a prefill processes; what that costs in acceptance
/// is measured, not derived.
pub(in crate::model) fn tail_cut_at(total: usize, bs: usize, deep: bool) -> usize {
    let tail = spark_runtime::ssm_tail_boundary(total, bs).unwrap_or(0);
    if deep && tail > bs && total - tail > 1 {
        tail
    } else {
        tail.saturating_sub(bs)
    }
}

/// The warm-turn switches and the state they keep, one per model.
pub(in crate::model) struct WarmTurn {
    /// `ATLAS_GLM_WARM_SKIP_CACHED=1`: a chunk that computes nothing does not
    /// zero the arena or embed (`prefill_b::warm`).
    pub(in crate::model) skip_cached: bool,
    /// `ATLAS_GLM_WARM_TRACE`: the per-request line of [`Trace`].
    pub(in crate::model) trace: TraceMode,
    /// `ATLAS_GLM_ZERO_ROWS`: how a multi-rank chunk zeroes the arena.
    pub(in crate::model) zero_rows: ZeroRows,
    /// Bulk token broadcast time not yet charged to a chunk.
    transfer: Mutex<Duration>,
    traces: Mutex<HashMap<usize, Trace>>,
}

impl WarmTurn {
    /// The switches of a `model_type` model, from the environment.
    pub(in crate::model) fn from_env(model_type: &str) -> Result<Self> {
        ensure!(
            !tail_cut_deep() || model_type == "glm5_next",
            "ATLAS_GLM_TAIL_CUT_DEEP requires glm5_next (got {model_type}): the deep cut is \
             restorable only under a template that re-renders its generation prompt"
        );
        Ok(Self {
            skip_cached: skip_cached_requested(),
            trace: TraceMode::parse(std::env::var("ATLAS_GLM_WARM_TRACE").ok().as_deref())?,
            zero_rows: ZeroRows::from_env()?,
            transfer: Mutex::default(),
            traces: Mutex::default(),
        })
    }
}

/// `ATLAS_GLM_WARM_TRACE`: whether a request's prefill logs its [`Trace`]
/// line, and how its spans are timed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) enum TraceMode {
    /// Unset or `0`.
    Off,
    /// `hash`: the line, for its logits hash. One stream sync and one read of
    /// the logits row per request; the spans are host time (a launch is
    /// charged where the host waits for it, mostly `finish`).
    Hash,
    /// `1`: the line with a stream sync at every span boundary, so a span
    /// holds the device time of its own work. Slows the prefill it times.
    Spans,
}

impl TraceMode {
    pub(super) fn parse(mode: Option<&str>) -> Result<Self> {
        match mode {
            None | Some("0") => Ok(Self::Off),
            Some("hash") => Ok(Self::Hash),
            Some("1") => Ok(Self::Spans),
            Some(v) => bail!("ATLAS_GLM_WARM_TRACE must be 0, 1 or hash, got {v:?}"),
        }
    }
}

/// How a multi-rank prefill chunk zeroes the buffer arena
/// (`ATLAS_GLM_ZERO_ROWS`, default off; `spark_runtime::buffers`,
/// `zero_dirty`). The payload is the floor in rows
/// (`ATLAS_GLM_ZERO_ROWS_FLOOR`, default [`ZeroRows::FLOOR`]).
///
/// Not exact by construction: `Trim` leaves the arena `zero_all` leaves only
/// while no pass writes past the rows it noted plus the floor. Run a
/// workload under `Check` first; it serves exactly as `Off` does and logs an
/// error for every byte `Trim` would have left. A clean check qualifies the
/// arena size (`--max-prefill-tokens`), context lengths and concurrency it
/// ran with, not others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) enum ZeroRows {
    /// Unset or `0`: the whole arena, as always.
    Off,
    /// `1`: what may be dirty.
    Trim(usize),
    /// `check`: the whole arena, after reading what `Trim` would leave.
    Check(usize),
}

impl ZeroRows {
    /// Rows a trimmed zero always covers: past a batched verify's 128 rows,
    /// a row group of index logits over a 512K context and the routed
    /// experts' 32-row padding (9,216 slots, 1,152 rows' worth).
    pub(super) const FLOOR: usize = 2048;

    /// This process's `ATLAS_GLM_ZERO_ROWS` and `_FLOOR`.
    pub(in crate::model) fn from_env() -> Result<Self> {
        let var = |name| std::env::var(name).ok();
        Self::parse(
            var("ATLAS_GLM_ZERO_ROWS").as_deref(),
            var("ATLAS_GLM_ZERO_ROWS_FLOOR").as_deref(),
        )
    }

    pub(super) fn parse(mode: Option<&str>, floor: Option<&str>) -> Result<Self> {
        let rows = match floor {
            None => Self::FLOOR,
            Some(v) => v.parse().ok().filter(|&rows| rows >= 256).ok_or_else(|| {
                anyhow::anyhow!("ATLAS_GLM_ZERO_ROWS_FLOOR must be at least 256 rows, got {v:?}")
            })?,
        };
        match mode {
            None | Some("0") => Ok(Self::Off),
            Some("1") => Ok(Self::Trim(rows)),
            Some("check") => Ok(Self::Check(rows)),
            Some(v) => bail!("ATLAS_GLM_ZERO_ROWS must be 0, 1 or check, got {v:?}"),
        }
    }
}

/// One request's prefill on one rank, for the `ATLAS_GLM_WARM_TRACE` line.
/// Times are host wall clock ([`TraceMode`] says what a span then holds).
#[derive(Default)]
pub(in crate::model) struct Trace {
    started: Option<Instant>,
    chunks: usize,
    cached_chunks: usize,
    rows: usize,
    spans: [Duration; PHASES.len()],
}

/// The spans of one chunk, in [`Trace`] order. `transfer` is the rank's bulk
/// token broadcasts since the chunk before: the prompt of the chunk command,
/// and while other sequences decode, their verify rows too.
pub(in crate::model) const PHASES: [&str; 8] = [
    "transfer", "zero", "embed", "lookup", "blocks", "meta", "forward", "finish",
];

/// Charges the time it lives to the next chunk's `transfer` span.
pub(in crate::model) struct TransferSpan<'a>(&'a WarmTurn, Instant);

impl Drop for TransferSpan<'_> {
    fn drop(&mut self) {
        *self.0.transfer.lock() += self.1.elapsed();
    }
}

/// FNV-1a over `bytes`, eight at a time: the logits fingerprint of the line.
pub(in crate::model) fn logits_hash(bytes: &[u8]) -> u64 {
    let step = |h: u64, w: u64| (h ^ w).wrapping_mul(0x0000_0100_0000_01b3);
    let words = bytes.chunks_exact(8);
    let tail = words
        .remainder()
        .iter()
        .fold(0u64, |w, &b| w << 8 | b as u64);
    let h = words.fold(0xcbf2_9ce4_8422_2325, |h, w| {
        step(h, u64::from_le_bytes(w.try_into().unwrap()))
    });
    step(step(h, tail), bytes.len() as u64)
}

impl WarmTurn {
    /// Time a bulk token broadcast for the trace; `None` with the switch off.
    pub(in crate::model) fn transfer_span(&self) -> Option<TransferSpan<'_>> {
        (self.trace != TraceMode::Off).then(|| TransferSpan(self, Instant::now()))
    }

    /// Add one chunk of `slot`'s request: `rows` computed rows (0 for a
    /// cached chunk) and its spans (`PHASES[1..]`), `began` when the chunk
    /// started. `first` (the chunk at token 0) starts the request's trace
    /// over, so one that failed before its last chunk leaves nothing behind.
    /// Returns the request's line when `last`; its `wall` runs from the
    /// first chunk's transfer to now.
    pub(in crate::model) fn note_chunk(
        &self,
        slot: usize,
        began: Instant,
        first: bool,
        rows: usize,
        spans: [Duration; PHASES.len() - 1],
        last: Option<RequestShape>,
    ) -> Option<String> {
        let mut traces = self.traces.lock();
        if first {
            traces.remove(&slot);
        }
        let t = traces.entry(slot).or_default();
        let transfer = std::mem::take(&mut *self.transfer.lock());
        t.started
            .get_or_insert(began.checked_sub(transfer).unwrap_or(began));
        t.chunks += 1;
        t.cached_chunks += usize::from(rows == 0);
        t.rows += rows;
        t.spans[0] += transfer;
        for (sum, span) in t.spans[1..].iter_mut().zip(spans) {
            *sum += span;
        }
        let shape = last?;
        let t = traces.remove(&slot)?;
        Some(t.line(slot, &shape))
    }
}

/// What the request's line says about the prompt and its result.
pub(in crate::model) struct RequestShape {
    pub rank: usize,
    pub prompt: usize,
    pub matched: usize,
    pub restored: usize,
    /// [`logits_hash`] of the last row's logits.
    pub logits: u64,
}

impl Trace {
    fn line(&self, slot: usize, s: &RequestShape) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let spans: Vec<String> = PHASES
            .iter()
            .zip(self.spans)
            .map(|(name, d)| format!("{name}={:.1}", ms(d)))
            .collect();
        format!(
            "warm-turn rank={} slot={slot} tokens={} matched={} restored={} chunks={} \
             cached_chunks={} rows={} ms: {} wall={:.1} logits={:016x}",
            s.rank,
            s.prompt,
            s.matched,
            s.restored,
            self.chunks,
            self.cached_chunks,
            self.rows,
            spans.join(" "),
            self.started.map_or(0.0, |t| ms(t.elapsed())),
            s.logits,
        )
    }
}
