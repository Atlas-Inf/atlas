// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-side PLE prefetch. The n-gram row ids are a pure host function of
//! the prompt tokens, so the whole prompt's working set is known the moment
//! a prefill is dispatched — but `gather_host` only faults it in when the
//! layer's own forward runs, synchronously on the model thread. A per-
//! sequence [`PleWarm`] session instead streams the ids into the row cache
//! from a worker thread, overlapped with the earlier chunks and layers.
//!
//! The worker paces itself a bounded window ahead of consumption so it can
//! never outrun the cache slack the loader provisions beside it:
//! `WarmShared::cursor` is the next prompt position the model has not yet
//! consumed — set by the per-chunk hook (which also JUMPS it over mid-prompt
//! prefix-cache hits) and advanced per gather span — and the worker warms
//! positions only while they lie within `cursor + ahead`. Prefetched rows
//! are NOT pinned (see `NgramRowCache::prefetch`), so a row evicted before
//! its consume costs one refault, never a wrong row.
//!
//! Child module of `layer` (the layer and the sequence state read the
//! fields directly); split out for the <=500 LoC cap.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::Result;

use super::{PleLayer, PleSeqState};
use crate::layers::ngram_embed::NgramTable;
use crate::layers::ple::ids::{PleIdDims, ple_ngram_ids};

/// How far ahead of the consume cursor the worker may warm, in PROMPT
/// POSITIONS. Two scratch spans: during one chunk's layer loop it can put
/// the whole NEXT chunk's rows in flight, which is where the TTFT win is.
/// `ATLAS_PLE_WARM_AHEAD` overrides; the loader sizes the row cache's slack
/// from the same helper so the two never drift apart.
pub(crate) fn warm_ahead_tokens(scratch_tokens: usize) -> usize {
    if !warm_enabled() {
        return 0;
    }
    std::env::var("ATLAS_PLE_WARM_AHEAD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2 * scratch_tokens)
}

/// Positions per prefetch call — the bound on how long the worker holds the
/// table mutex, i.e. how long a concurrent gather can wait behind it. 1024
/// positions = 16K rows ≈ 2.5 MB of NVMe at FP8 rows.
fn warm_quantum() -> usize {
    std::env::var("ATLAS_PLE_WARM_QUANTUM")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(1024)
}

/// `ATLAS_PLE_WARM=0` restores the old behaviour — the gather faults its own
/// rows on the forward path.
pub(crate) fn warm_enabled() -> bool {
    std::env::var("ATLAS_PLE_WARM").ok().as_deref() != Some("0")
}

/// Row ids for prompt positions `[from..prompt.len())`, flattened
/// `[position][head]`. A position's n-gram reads up to `context_len`
/// predecessors, so the hash needs the prompt's lookbehind — EOS-padded when
/// `from` sits inside the first `context_len` positions, the same fill
/// `reset` applies to a fresh sequence's history. `all[ctx]` is always the
/// first wanted row (pad + lookbehind occupy exactly `ctx` rows).
fn prompt_rows(dims: &PleIdDims, prompt: &[u32], from: usize) -> Vec<u64> {
    let ctx = dims.context_len();
    let ctx_start = from.saturating_sub(ctx);
    let pad = ctx - (from - ctx_start);
    let mut window = Vec::with_capacity(prompt.len() - ctx_start + pad);
    window.resize(pad, dims.eos_token_id);
    window.extend_from_slice(&prompt[ctx_start..]);
    let all = ple_ngram_ids(dims, &window);
    all[ctx..].iter().flatten().copied().collect()
}

/// Pacing state shared by the consuming forwards and the warm worker.
struct WarmShared {
    /// Next prompt position not yet consumed — the worker may warm up to
    /// `cursor + ahead`. The per-chunk hook `fetch_max`es it to each chunk's
    /// proc_start (jumping mid-prompt cache hits); each gather span
    /// `fetch_add`s its width, which keeps the monolithic twophase forward
    /// warming a rolling window ahead of itself.
    cursor: AtomicUsize,
    /// Set when the session is dropped/aborted; the worker exits at the next
    /// quantum boundary.
    done: AtomicBool,
    /// Worker exited — the session covers nothing more; a later
    /// `prefill_warm` replaces it rather than trusting `advance` to a dead
    /// thread.
    finished: AtomicBool,
}

/// One sequence's in-flight prefill warm. Created by [`PleLayer::prefill_warm`]
/// at the first prefill dispatch and lives in the sequence's [`PleSeqState`].
pub struct PleWarm {
    shared: Arc<WarmShared>,
    thread: Option<JoinHandle<()>>,
    /// Prompt position the id stream covers from.
    covered_from: usize,
    /// Identity guard for `covers`: a leftover session on a reused sequence
    /// must not suppress the new prompt's warm (worst case otherwise is a
    /// silently cold cache — correctness never depends on the session).
    prompt_len: usize,
    edge_toks: (u32, u32),
}

impl PleWarm {
    fn start(
        ids: Vec<u64>,
        covered_from: usize,
        prompt: &[u32],
        table: Arc<Mutex<NgramTable>>,
        heads: usize,
        ahead: usize,
    ) -> Option<Self> {
        let shared = Arc::new(WarmShared {
            cursor: AtomicUsize::new(covered_from),
            done: AtomicBool::new(false),
            finished: AtomicBool::new(false),
        });
        let worker = {
            let shared = shared.clone();
            let quantum = warm_quantum();
            match std::thread::Builder::new()
                .name("ple-warm".into())
                .spawn(move || run(ids, covered_from, heads, ahead, quantum, shared, table))
            {
                Ok(w) => w,
                Err(e) => {
                    // No warm is a throughput loss, never a correctness one —
                    // the gathers fault on demand exactly as before.
                    tracing::warn!("PLE warm: worker spawn failed ({e}); gathers fault on demand");
                    return None;
                }
            }
        };
        Some(Self {
            shared,
            thread: Some(worker),
            covered_from,
            prompt_len: prompt.len(),
            edge_toks: (prompt[covered_from], prompt[prompt.len() - 1]),
        })
    }

    /// Whether this session still serves a `prefill_warm` call for `prompt`
    /// starting at `from`: same prompt, coverage began at or before it, and
    /// the worker is still alive.
    fn covers(&self, prompt: &[u32], from: usize) -> bool {
        !self.shared.finished.load(Ordering::Relaxed)
            && self.covered_from <= from
            && self.prompt_len == prompt.len()
            && self.edge_toks == (prompt[self.covered_from], prompt[prompt.len() - 1])
    }

    /// Advance the consume cursor to `pos` (the current chunk's proc_start)
    /// and wake the worker. `fetch_max`: the cursor never walks backwards.
    fn advance(&self, pos: usize) {
        self.shared.cursor.fetch_max(pos, Ordering::Relaxed);
        self.unpark();
    }

    /// `n` positions were just consumed by a gather span (in-forward pacing
    /// for the monolithic forwards, which get no further hook calls).
    /// `pub(super)`: the span loop in `layer` calls it.
    pub(super) fn note(&self, n: usize) {
        self.shared.cursor.fetch_add(n, Ordering::Relaxed);
        self.unpark();
    }

    fn unpark(&self) {
        if let Some(t) = &self.thread {
            t.thread().unpark();
        }
    }

    fn abort(&mut self) {
        self.shared.done.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            // Bounded by one quantum's fault time — the worker checks `done`
            // between prefetch calls.
            let _ = t.join();
        }
    }
}

impl Drop for PleWarm {
    fn drop(&mut self) {
        self.abort();
    }
}

/// The worker loop: warm positions `[pos, lim)` while `lim = cursor + ahead`,
/// one quantum per prefetch call so a concurrent gather waits behind at most
/// one quantum of fault I/O. Parks when caught up; `advance`/`note` unpark.
#[allow(clippy::too_many_arguments)]
fn run(
    ids: Vec<u64>,
    covered_from: usize,
    heads: usize,
    ahead: usize,
    quantum: usize,
    shared: Arc<WarmShared>,
    table: Arc<Mutex<NgramTable>>,
) {
    let end_pos = covered_from + ids.len() / heads;
    let mut pos = covered_from;
    while pos < end_pos && !shared.done.load(Ordering::Relaxed) {
        // Everything below `cursor` is consumed or skipped — consumed rows
        // are already resident (pinned by their in-flight gather), skipped
        // ones are never gathered at all. A mid-prompt `advance` can jump
        // the cursor tens of thousands of positions past `pos`; walking the
        // dead range would fault hundreds of MB nobody reads.
        pos = pos.max(shared.cursor.load(Ordering::Relaxed));
        if pos >= end_pos {
            break;
        }
        let lim = shared.cursor.load(Ordering::Relaxed).saturating_add(ahead);
        if pos >= lim {
            // Caught up to the lookahead bound. park_timeout (not park): a
            // missed unpark then costs one timeout, not a dead session.
            std::thread::park_timeout(Duration::from_millis(20));
            continue;
        }
        let upto = lim.min(pos + quantum).min(end_pos);
        let rows = &ids[(pos - covered_from) * heads..(upto - covered_from) * heads];
        match table.lock().map(|mut t| t.prefetch(rows)) {
            Ok(Ok(_)) => pos = upto,
            Ok(Err(e)) => {
                // A warm fault failure must NOT fail the request — the
                // consuming resolve refaults and surfaces its own error.
                tracing::warn!("PLE warm: prefetch stopped at position {pos}: {e:#}");
                break;
            }
            Err(_) => break, // poisoned mutex
        }
    }
    shared.finished.store(true, Ordering::Relaxed);
}

impl PleLayer {
    /// Begin or pace this sequence's prefill warm session. Called once per
    /// prefill dispatch with the FULL prompt and `from`, the first position
    /// this dispatch will process — the ids are a pure function of `prompt`,
    /// so the worker streams rows for every position the request still has
    /// to run, not just this chunk's. No-op for resident tables (nothing to
    /// warm into) and for `from >= prompt.len()` (nothing left).
    pub fn prefill_warm(&self, st: &mut PleSeqState, prompt: &[u32], from: usize) -> Result<()> {
        if !warm_enabled() || from >= prompt.len() {
            return Ok(());
        }
        if let Some(w) = st.warm.as_ref() {
            if w.covers(prompt, from) {
                w.advance(from);
                return Ok(());
            }
            // A different prompt on a reused sequence, or a finished worker:
            // drop (which aborts) and rebuild for the rest of this prompt.
            st.warm = None;
        }
        let is_cached = {
            let table = self
                .table
                .lock()
                .map_err(|_| anyhow::anyhow!("PLE table mutex poisoned"))?;
            table.is_row_cached()
        };
        if !is_cached {
            return Ok(());
        }
        let heads = self.dims.ngram_heads();
        let ids = prompt_rows(&self.dims, prompt, from);
        let ahead = warm_ahead_tokens(self.scratch_tokens);
        st.warm = PleWarm::start(ids, from, prompt, self.table.clone(), heads, ahead);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small synthetic geometry — 4 heads over 2 n-gram orders — enough to
    /// exercise the slicing and the EOS-boundary shift. `prompt_rows` is a
    /// pure re-slice of `ple_ngram_ids`, so no checkpoint fixture is needed.
    fn test_dims() -> PleIdDims {
        PleIdDims {
            ngram_size: 3,
            heads_per_ngram: 2,
            multipliers: vec![
                0x9E37_79B9_7F4A_7C15,
                0xC2B2_AE3D_27D4_EB4F,
                0x1656_67B1_9E37_79F9,
            ],
            head_vocab_sizes: vec![31, 37, 41, 43],
            head_offsets: vec![0, 31, 68, 109],
            eos_token_id: 999,
        }
    }

    /// `prompt_rows(dims, prompt, from)` must equal the tail slice of the
    /// one-shot computation over `[eos; context_len] ++ prompt` — at EVERY
    /// `from`, including mid-prompt boundaries where the lookbehind is real
    /// tokens and early ones where it is EOS pad.
    #[test]
    fn prompt_rows_matches_full_window_at_every_from() {
        let d = test_dims();
        let ctx = d.context_len();
        let heads = d.ngram_heads();
        // Interior EOS positions exercise the segment boundary the shift enforces.
        let prompt: Vec<u32> = vec![7, 8, 999, 11, 12, 999, 21, 22, 23];
        let mut window = vec![d.eos_token_id; ctx];
        window.extend_from_slice(&prompt);
        let all = ple_ngram_ids(&d, &window);

        for from in 0..prompt.len() {
            let want: Vec<u64> = all[ctx + from..]
                .iter()
                .flat_map(|r| r.iter().copied())
                .collect();
            let got = prompt_rows(&d, &prompt, from);
            assert_eq!(got, want, "from={from}");
            assert_eq!(got.len(), (prompt.len() - from) * heads);
        }
    }

    /// A session covers only the prompt it was started for: `covers` keys on
    /// the coverage start, the length, and both edge tokens, and must also
    /// refuse once the worker has exited.
    #[test]
    fn session_covers_and_cursor_semantics() {
        let prompt: Vec<u32> = vec![7, 8, 999, 11, 12, 13, 14, 15];
        let w = PleWarm {
            shared: Arc::new(WarmShared {
                cursor: AtomicUsize::new(2),
                done: AtomicBool::new(false),
                finished: AtomicBool::new(false),
            }),
            thread: None,
            covered_from: 2,
            prompt_len: prompt.len(),
            edge_toks: (prompt[2], prompt[prompt.len() - 1]),
        };

        assert!(w.covers(&prompt, 2));
        assert!(w.covers(&prompt, 6), "later chunks of the same prompt");
        assert!(!w.covers(&prompt, 1), "before the covered window");
        let mut other = prompt.clone();
        other[7] = 42;
        assert!(
            !w.covers(&other, 5),
            "different tail token = different prompt"
        );
        let mut shorter = prompt.clone();
        shorter.pop();
        assert!(!w.covers(&shorter, 5), "different length");

        // `advance` is fetch_max — a stale chunk hook must not walk the
        // cursor backwards — and `note` adds the consumed span width.
        w.advance(6);
        assert_eq!(w.shared.cursor.load(Ordering::Relaxed), 6);
        w.advance(4);
        assert_eq!(w.shared.cursor.load(Ordering::Relaxed), 6);
        w.note(3);
        assert_eq!(w.shared.cursor.load(Ordering::Relaxed), 9);

        w.shared.finished.store(true, Ordering::Relaxed);
        assert!(!w.covers(&prompt, 6), "a dead worker covers nothing");
    }
}
