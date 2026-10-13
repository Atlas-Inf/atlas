// SPDX-License-Identifier: AGPL-3.0-only

//! Per-step decode shape trace (`ATLAS_MTP_STEP_TRACE=1`, default off).
//!
//! # Why this exists
//!
//! The accept telemetry (`mtp_accept_debug`) reports tokens per verify per
//! sequence over steady windows at one width. A C=8 wave also pays for steps
//! that window never sees: the admission ramp and the tail (n < 8), the first
//! steps of each sequence, draftless sequences (decode rows, bootstraps), and
//! plain decode steps. Over a whole sparkDash code wave those took the
//! delivered tokens per step from ~3.65 to ~3.23. This trace logs every
//! decode tick's shape so `scripts/dev/step_trace_summary.py` can attribute
//! the drop.
//!
//! # The line
//!
//! `MTP STEP t=<us> dur=<us> kind=<k> n=<seqs> rows=<r> fwd=<f> nd=<d>
//! deep=<0|1> dl=<draftless> s=<slot>:<gen>:<path><drafts>:<emitted>,...`
//!
//! * `t`: microseconds since the first traced step's process epoch (monotonic)
//!   when the tick began; `dur` covers the tick's prefill continuation
//!   (mixed/rode steps, multi-prompt prefill) and its decode dispatch; the
//!   trailing `pf=<us>` is the part before the decode dispatch.
//! * `kind`: `mtp` (`step_mtp` ran), `dec` (plain batch decode), `gate` (the
//!   MTP gate's serial measurement step), `mixed` (decode rode a prefill
//!   chunk).
//! * `rows`: target rows the step's verify forwards carried; `fwd`: target
//!   forwards (one per batched verify chunk, per serial verify, per
//!   per-sequence bootstrap; a batched bootstrap is one).
//! * `nd`: the step's ladder depth; `deep`: the dynamic-depth arm.
//! * per sequence: `gen` = tokens it had generated before the step, `path` =
//!   `v` batched verify, `d` decode row in the batched verify, `s` serial
//!   verify, `B` batched bootstrap, `b` per-sequence bootstrap, `n` plain
//!   decode, `r` verify rode a prefill chunk, `m` mixed decode, `?` not
//!   recorded; `drafts` it verified; `emitted` = its `seq_len` advance.
//!
//! Everything is host state the scheduler already holds: no device sync, no
//! D2H. Disarmed, each hook costs one `OnceLock` load.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::time::Instant;

use super::ActiveSeq;

/// `ATLAS_MTP_STEP_TRACE=1` arms the trace. Read once per process.
pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MTP_STEP_TRACE").as_deref() == Ok("1"))
}

static EPOCH: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);

/// The shape one tick recorded through the hooks below.
#[derive(Default)]
struct Rec {
    kind: Option<&'static str>,
    /// `step_mtp` ran this tick.
    stepped: bool,
    nd: usize,
    deep: bool,
    fwd: u32,
    rows: usize,
    /// (slot, path, drafts), first record per slot wins.
    seqs: Vec<(usize, u8, usize)>,
}

impl Rec {
    fn seq(&mut self, slot: usize, path: u8, drafts: usize) {
        if self.path_of(slot).is_none() {
            self.seqs.push((slot, path, drafts));
        }
    }

    fn forward(&mut self, rows: usize) {
        self.fwd += 1;
        self.rows += rows;
    }

    fn path_of(&self, slot: usize) -> Option<(u8, usize)> {
        self.seqs
            .iter()
            .find(|&&(s, _, _)| s == slot)
            .map(|&(_, p, d)| (p, d))
    }

    fn kind(&self) -> &'static str {
        self.kind
            .unwrap_or(if self.stepped { "mtp" } else { "dec" })
    }

    /// The path a sequence with no record took, from the tick's kind.
    fn default_path(&self) -> u8 {
        match self.kind() {
            "mtp" => b'?',
            "mixed" => b'm',
            _ => b'n',
        }
    }
}

thread_local! {
    static REC: RefCell<Option<Rec>> = const { RefCell::new(None) };
}

fn with_rec(f: impl FnOnce(&mut Rec)) {
    if enabled() {
        REC.with(|r| {
            if let Some(rec) = r.borrow_mut().as_mut() {
                f(rec);
            }
        });
    }
}

/// One tick's snapshot: when it began and each decoding sequence's
/// (slot, seq_len, generated) before it.
pub(super) struct Trace {
    t0: Instant,
    /// When the decode dispatch began (after the tick's prefill work).
    t_dec: Option<Instant>,
    before: Vec<(usize, usize, usize)>,
}

fn snapshot(a: &ActiveSeq) -> (usize, usize, usize) {
    (a.seq.slot_idx, a.seq.seq_len, a.output_tokens.len())
}

/// Open a tick over the decoding sequences `active`. `None` when disarmed.
pub(super) fn begin(active: &[ActiveSeq]) -> Option<Trace> {
    if !enabled() {
        return None;
    }
    REC.with(|r| *r.borrow_mut() = Some(Rec::default()));
    let t0 = Instant::now();
    let _ = *EPOCH;
    Some(Trace {
        t0,
        t_dec: None,
        before: active.iter().map(snapshot).collect(),
    })
}

/// The decode dispatch starts: sequences the tick's prefill work promoted
/// into `active` since [`begin`] join the snapshot at their current state.
pub(super) fn mark_decode(trace: &mut Option<Trace>, active: &[ActiveSeq]) {
    let Some(tr) = trace.as_mut() else { return };
    tr.t_dec = Some(Instant::now());
    for a in active {
        if !tr.before.iter().any(|b| b.0 == a.seq.slot_idx) {
            tr.before.push(snapshot(a));
        }
    }
}

/// `step_mtp` ran with ladder depth `nd` (`deep`: the dynamic-depth arm).
pub(super) fn note_step(nd: usize, deep: bool) {
    with_rec(|r| {
        r.stepped = true;
        r.nd = nd;
        r.deep = deep;
    });
}

/// Override the tick's kind (`gate`, `mixed`).
pub(super) fn note_kind(kind: &'static str) {
    with_rec(|r| r.kind = Some(kind));
}

/// Sequence `a` took `path` verifying `drafts` drafts.
pub(super) fn note_seq(a: &ActiveSeq, path: u8, drafts: usize) {
    with_rec(|r| r.seq(a.seq.slot_idx, path, drafts));
}

/// Sequences whose verify rode this tick's prefill chunk (slots).
pub(super) fn note_rode(slots: &[usize]) {
    with_rec(|r| slots.iter().for_each(|&s| r.seq(s, b'r', 0)));
}

/// One target forward of `rows` rows.
pub(super) fn note_forward(rows: usize) {
    with_rec(|r| r.forward(rows));
}

/// Close the tick and log its line (nothing when no sequence decoded).
pub(super) fn finish(trace: Option<Trace>, active: &[ActiveSeq]) {
    let Some(tr) = trace else { return };
    let Some(rec) = REC.with(|r| r.borrow_mut().take()) else {
        return;
    };
    let after: Vec<(usize, usize)> = active
        .iter()
        .map(|a| (a.seq.slot_idx, a.seq.seq_len))
        .collect();
    let t = tr.t0.saturating_duration_since(*EPOCH).as_micros();
    let dur = tr.t0.elapsed().as_micros();
    let pf = tr
        .t_dec
        .map_or(0, |d| d.saturating_duration_since(tr.t0).as_micros());
    if let Some(line) = format_line(t, dur, &rec, &tr.before, &after) {
        tracing::info!("{line} pf={pf}");
    }
}

/// The trace line for one tick (pure; see the module docs for the format).
/// `before`: (slot, seq_len, generated); `after`: (slot, seq_len). A
/// sequence missing from `after` (preempted) is dropped.
fn format_line(
    t: u128,
    dur: u128,
    rec: &Rec,
    before: &[(usize, usize, usize)],
    after: &[(usize, usize)],
) -> Option<String> {
    let mut s = String::new();
    let (mut n, mut dl, mut emitted_any) = (0usize, 0usize, false);
    for &(slot, len0, generated) in before {
        let Some(&(_, len1)) = after.iter().find(|&&(sl, _)| sl == slot) else {
            continue;
        };
        let emitted = len1.saturating_sub(len0);
        emitted_any |= emitted > 0;
        let (path, drafts) = rec.path_of(slot).unwrap_or((rec.default_path(), 0));
        dl += usize::from(matches!(path, b'd' | b'b' | b'B'));
        if n > 0 {
            s.push(',');
        }
        let _ = write!(s, "{slot}:{generated}:{}{drafts}:{emitted}", path as char);
        n += 1;
    }
    if !emitted_any && !rec.stepped {
        return None;
    }
    Some(format!(
        "MTP STEP t={t} dur={dur} kind={} n={n} rows={} fwd={} nd={} deep={} dl={dl} s={s}",
        rec.kind(),
        rec.rows,
        rec.fwd,
        rec.nd,
        u8::from(rec.deep),
    ))
}

#[cfg(test)]
#[path = "mtp_step_trace_tests.rs"]
mod tests;
