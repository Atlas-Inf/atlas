// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_WARM_CTX`: the parser, the legacy placement verbatim, and a
//! prefill simulation on the mock backend (chunks, the tail split, branch
//! pieces, prefix-cache skips) whose rows encode their own position, so the
//! window a prefill leaves can be checked slot by slot against its labels
//! and against a cold prefill of the same prompt.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::first_append_tests::{SLOT, dstate};
use super::free_state_tests::zero_head;
use super::warm_ctx::{CapturePlan, CtxRun, Pass, WarmCtx, capture_plan, parse_store_slots};
use super::warm_ctx_store::WarmCtxStore;
use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::speculative::{DraftProposer, ProposerState};

const MODES: [WarmCtx; 4] = [
    WarmCtx::Legacy,
    WarmCtx::Exact,
    WarmCtx::Seed,
    WarmCtx::Restore,
];
/// Window, chunk and block of the simulation (2048 / 8192 / 16 scaled down).
pub(super) const W: usize = 16;
const CHUNK: usize = 48;
const BS: usize = 4;

#[test]
fn parser_defaults_to_legacy_and_rejects_unknown_values() {
    assert_eq!(WarmCtx::parse(None).unwrap(), WarmCtx::Legacy);
    for (raw, mode) in ["legacy", "exact", "seed", "restore"]
        .into_iter()
        .zip(MODES)
    {
        assert_eq!(WarmCtx::parse(Some(raw)).unwrap(), mode, "{raw}");
    }
    for raw in ["", "1", "Seed", "restore ", "on"] {
        let error = WarmCtx::parse(Some(raw)).unwrap_err().to_string();
        assert!(error.contains("ATLAS_DFLASH_WARM_CTX"), "{raw:?}: {error}");
    }
    assert_eq!(parse_store_slots(None).unwrap(), 8);
    assert_eq!(parse_store_slots(Some("0")).unwrap(), 0);
    assert!(parse_store_slots(Some("eight")).is_err());
}

/// `legacy` places every pass exactly as the capture did before the switch.
#[test]
fn legacy_plan_is_the_old_placement() {
    for (tokens_len, start, rows, sp_row0, max_ctx) in itertools(&[0, 5, 40, 96], &[0, 3, 32, 90]) {
        let pass = Pass {
            tokens_len,
            prompt_len: 1000,
            start,
            rows,
            sp_row0,
            branch_at: Some(start + rows),
        };
        let end = start + rows;
        let window_start = tokens_len.max(end).saturating_sub(max_ctx);
        let first = window_start.max(start).max(start + sp_row0);
        let old =
            (first < end).then(|| (first - start - sp_row0, end - first, first - window_start));
        let new = capture_plan(WarmCtx::Legacy, max_ctx, 0, &pass)
            .map(|p| (p.src_row(&pass), p.rows(), p.dst_slot()));
        assert_eq!(new, old, "{pass:?} max_ctx={max_ctx}");
    }
}

fn itertools(a: &[usize], b: &[usize]) -> Vec<(usize, usize, usize, usize, usize)> {
    let mut out = Vec::new();
    for &tokens_len in a {
        for &start in b {
            for rows in [1, 7, 16, 64] {
                for sp_row0 in [0, rows / 2] {
                    for max_ctx in [0, 8, 2048] {
                        out.push((tokens_len, start, rows, sp_row0, max_ctx));
                    }
                }
            }
        }
    }
    out
}

#[test]
fn a_run_extends_only_contiguously_in_one_layout() {
    let mut run = None;
    CtxRun::note(&mut run, 10, 12, 16);
    CtxRun::note(&mut run, 10, 12, 16); // every capture layer writes it again
    CtxRun::note(&mut run, 10, 16, 18);
    assert_eq!(
        CtxRun::rows_to(run, 18).map(|r| (r.lo, r.hi)),
        Some((12, 18))
    );
    assert_eq!(
        CtxRun::rows_to(run, 16),
        None,
        "rows must end at the checkpoint"
    );
    CtxRun::note(&mut run, 10, 20, 22); // a hole
    assert_eq!(run.map(|r| (r.lo, r.hi)), Some((20, 22)));
    CtxRun::note(&mut run, 14, 22, 24); // another layout
    assert_eq!(run.map(|r| (r.window_start, r.lo)), Some((14, 22)));
}

/// A store of `slots` windows at the simulation's shape.
pub(super) fn store(gpu: &dyn GpuBackend, slots: usize) -> WarmCtxStore {
    WarmCtxStore::reserve(gpu, slots, W, SLOT).unwrap()
}

fn head(mode: WarmCtx) -> BlockDiffusionDraftHead {
    let mut head = zero_head();
    head.target_layer_ids = vec![0, 1];
    head.target_hidden_size = SLOT / 4;
    head.max_seq_len = W;
    // Local adaptation: our ctx_window_len() reads `ctx_window` (the
    // configured ctx window), not `window_size` — the real head gets it
    // from config; the fixture pins it to W.
    head.ctx_window = W;
    head.startup.diagnostics.warm_ctx = mode;
    head
}

pub(super) fn tokens(n: usize, conversation: u32) -> Vec<u32> {
    (0..n as u32).map(|i| i * 7 + conversation).collect()
}

/// One prefill request.
#[derive(Clone, Copy)]
pub(super) struct Turn {
    pub(super) prompt: usize,
    pub(super) skip: usize,
    pub(super) branch_at: Option<usize>,
    /// Sequence-parallel passes keep their rows from half their length on.
    pub(super) sp: bool,
}

impl Turn {
    pub(super) fn cold(prompt: usize) -> Self {
        Self {
            prompt,
            skip: 0,
            branch_at: None,
            sp: false,
        }
    }

    fn tail(&self) -> usize {
        (self.prompt - 1) / BS * BS
    }

    /// The prompt tail split's cut (`pc_policy::tail_cut`).
    pub(super) fn cut(&self) -> usize {
        self.tail().saturating_sub(BS)
    }

    /// The computing passes, as `prefill_chunk_dispatch` runs them.
    fn passes(&self) -> Vec<Pass> {
        let mut out = Vec::new();
        let mut cs = 0;
        while cs < self.prompt {
            let ce = (cs + CHUNK).min(self.prompt);
            let mut cuts: Vec<usize> = self
                .branch_at
                .into_iter()
                .filter(|&b| b > cs && b < ce)
                .collect();
            if ce == self.prompt && self.cut() > cs && self.cut() < self.prompt {
                cuts.push(self.cut());
            }
            cuts.sort_unstable();
            cuts.dedup();
            // A split piece is its own dispatch: its tokens are recorded
            // before the next piece runs; a chunk's only after it ran.
            let mut tokens_len = cs;
            let mut s = cs;
            for e in cuts.into_iter().chain([ce]) {
                let start = s.max(self.skip);
                if start < e {
                    let rows = e - start;
                    let sp_row0 = if self.sp && rows >= 2 { rows / 2 } else { 0 };
                    out.push(Pass {
                        tokens_len,
                        prompt_len: self.prompt,
                        start,
                        rows,
                        sp_row0,
                        branch_at: self.branch_at,
                    });
                }
                (tokens_len, s) = (e, e);
            }
            cs = ce;
        }
        out
    }

    fn checkpoint_at(&self, end: usize) -> bool {
        end < self.prompt
            && (end == self.tail() || end == self.cut() || self.branch_at == Some(end))
    }
}

/// The capture row of position `p`: its position + 1 (a zero row reads 0).
fn write_rows(gpu: &dyn GpuBackend, d: &DflashProposerState, plan: &CapturePlan) {
    for t in 0..plan.rows() {
        let tag = (plan.first + t + 1) as u32;
        let row: Vec<u8> = (0..SLOT / 4).flat_map(|_| tag.to_le_bytes()).collect();
        gpu.copy_h2d(&row, d.ctx_hidden_acc.offset((plan.dst_slot() + t) * SLOT))
            .unwrap();
    }
}

/// Run `turn` like the model does; returns the state after the prefill.
pub(super) fn prefill(
    gpu: &dyn GpuBackend,
    store: &WarmCtxStore,
    mode: WarmCtx,
    toks: &[u32],
    turn: Turn,
) -> Box<dyn ProposerState> {
    let h = head(mode);
    let mut state = h.alloc_state(gpu).unwrap();
    // Local adaptation: the model allocates the ctx accumulator lazily at
    // its first prefill capture (acquire_ctx_acc); the fixture drives the
    // state directly, so it takes the same step here — straight to the
    // backend, bypassing the global pool (a pooled pointer belongs to
    // whichever backend allocated it, not this fixture's).
    {
        let d0 = dstate(&mut state);
        let own = usize::from(h.startup.diagnostics.first_append != super::FirstAppend::Legacy);
        // Sized like alloc_state's accumulator; the fixture head's
        // ctx_window_len() can be 0, so fall back to W rows at SLOT bytes.
        // Mock allocs come back zeroed (MockAlloc data is a vec![0u8]).
        let total = ((d0.ctx_acc_rows.max(W)) + own) * d0.ctx_slot_bytes.max(SLOT);
        d0.ctx_hidden_acc = gpu.alloc(total).unwrap();
    }
    let d = dstate(&mut state);
    let prompt = &toks[..turn.prompt];
    d.begin_warm_ctx(store, 0, Some(prompt), turn.skip, turn.prompt, gpu, 0)
        .unwrap();
    for pass in turn.passes() {
        if let Some(plan) = capture_plan(d.warm_ctx, d.max_ctx_len, d.ctx_floor, &pass) {
            d.note_capture(&plan);
            write_rows(gpu, d, &plan);
        }
        if turn.checkpoint_at(pass.end()) {
            d.save_warm_ctx(store, 0, &prompt[..pass.end()], gpu, 0)
                .unwrap();
        }
    }
    d.seed_prefill_ctx(turn.prompt, turn.prompt, turn.prompt);
    state
}

/// `(label, row)` of every window slot; a zero row reads `None`.
pub(super) fn window(
    gpu: &dyn GpuBackend,
    state: &mut Box<dyn ProposerState>,
) -> Vec<(usize, Option<usize>)> {
    let d = dstate(state);
    let mut raw = vec![0u8; d.ctx_len * SLOT];
    gpu.copy_d2h(d.ctx_hidden_acc, &mut raw).unwrap();
    raw.chunks_exact(SLOT)
        .zip(&d.ctx_positions)
        .map(|(row, &label)| {
            let tag = u32::from_le_bytes(row[..4].try_into().unwrap()) as usize;
            (label as usize, tag.checked_sub(1))
        })
        .collect()
}

pub(super) fn exact(win: &[(usize, Option<usize>)]) -> bool {
    win.iter().all(|&(label, row)| row == Some(label))
}

/// A cold prompt longer than one chunk: only the prompt-anchored modes hold
/// the rows their labels name (legacy anchors each pass at its own end).
#[test]
fn a_cold_multi_chunk_window_is_exact_off_legacy() {
    for prompt in [W + 1, CHUNK + 3, CHUNK + 18, 3 * CHUNK + 7] {
        for mode in MODES {
            let gpu = MockGpuBackend::new();
            let store = store(&gpu, 4);
            let toks = tokens(prompt, 1);
            let mut state = prefill(&gpu, &store, mode, &toks, Turn::cold(prompt));
            let win = window(&gpu, &mut state);
            assert_eq!(win.len(), W, "{mode:?} {prompt}");
            let labels: Vec<usize> = (prompt - W..prompt).collect();
            assert_eq!(win.iter().map(|w| w.0).collect::<Vec<_>>(), labels);
            assert_eq!(exact(&win), mode.exact(), "{mode:?} {prompt}: {win:?}");
        }
    }
}

/// The next turn of a conversation restores the target at the previous
/// prompt's tail cut. `restore` rebuilds exactly the cold window of the new
/// prompt; `seed` starts at the cut without zero rows; the others attend
/// zero rows under the cut.
#[test]
fn a_warm_turn_window_matches_its_cold_prefill_under_restore() {
    for (first, second) in [(40, 44), (40, 41), (70, 75), (90, 120), (9, 13)] {
        let toks = tokens(second, 2);
        let cold = {
            let gpu = MockGpuBackend::new();
            let store = store(&gpu, 4);
            window(
                &gpu,
                &mut prefill(&gpu, &store, WarmCtx::Exact, &toks, Turn::cold(second)),
            )
        };
        for mode in MODES {
            let gpu = MockGpuBackend::new();
            let store = store(&gpu, 4);
            let turn1 = Turn::cold(first);
            prefill(&gpu, &store, mode, &toks, turn1);
            assert_eq!(store.len(), mode.restores() as usize, "{mode:?}");
            let turn2 = Turn {
                skip: turn1.cut(),
                ..Turn::cold(second)
            };
            let win = window(&gpu, &mut prefill(&gpu, &store, mode, &toks, turn2));
            let case = format!("{mode:?} {first}->{second}: {win:?}");
            let zeros = win.iter().any(|w| w.1.is_none());
            match mode {
                WarmCtx::Restore => assert_eq!(win, cold, "{case}"),
                WarmCtx::Seed => {
                    assert!(exact(&win), "{case}");
                    assert_eq!(win[0].0, turn2.skip.max(second.saturating_sub(W)), "{case}");
                }
                _ => assert_eq!(zeros, second.saturating_sub(W) < turn2.skip, "{case}"),
            }
        }
    }
}

/// A miss (another conversation's prefix, an evicted entry, no store)
/// falls back to the seeded window: never a zero row.
#[test]
fn a_restore_without_its_entry_seeds_the_window() {
    for slots in [0, 1, 4] {
        let gpu = MockGpuBackend::new();
        let store = store(&gpu, slots);
        let (a, b) = (tokens(60, 3), tokens(60, 4));
        prefill(&gpu, &store, WarmCtx::Restore, &a, Turn::cold(40));
        // Same positions, other tokens: the content key differs. Its own
        // tail checkpoint then takes the only window of a 1-slot store.
        let turn = Turn {
            skip: Turn::cold(40).cut(),
            ..Turn::cold(44)
        };
        let win = window(&gpu, &mut prefill(&gpu, &store, WarmCtx::Restore, &b, turn));
        assert!(exact(&win), "slots={slots}: {win:?}");
        assert_eq!(win[0].0, turn.skip, "slots={slots}");
        let mut hit = prefill(&gpu, &store, WarmCtx::Restore, &a, turn);
        let floor = if slots > 1 { 44 - W } else { turn.skip };
        assert_eq!(dstate(&mut hit).ctx_floor, floor, "slots={slots}");
        assert!(exact(&window(&gpu, &mut hit)), "slots={slots}");
    }
}

/// A next prompt shorter than the one that stored the rows (an edit or a
/// branch) restores the rows the entry holds: a shorter window, no zeros.
#[test]
fn a_shorter_next_prompt_takes_the_rows_the_entry_holds() {
    let gpu = MockGpuBackend::new();
    let store = store(&gpu, 4);
    let toks = tokens(64, 5);
    // Window [44, 60); its tail cut 52 stores [44, 52).
    let turn1 = Turn::cold(60);
    prefill(&gpu, &store, WarmCtx::Restore, &toks, turn1);
    let turn2 = Turn {
        skip: turn1.cut(),
        ..Turn::cold(turn1.cut() + 2)
    };
    let win = window(
        &gpu,
        &mut prefill(&gpu, &store, WarmCtx::Restore, &toks, turn2),
    );
    assert!(exact(&win), "{win:?}");
    assert_eq!((win[0].0, win.len()), (44, 10));
}

/// A branch checkpoint below the window keeps the branch piece's tail for
/// its entry, and the window the prefill leaves is still exact.
#[test]
fn a_branch_checkpoint_below_the_window_stores_its_own_tail() {
    let gpu = MockGpuBackend::new();
    let store = store(&gpu, 4);
    let toks = tokens(120, 6);
    let turn = Turn {
        branch_at: Some(40),
        ..Turn::cold(100)
    };
    let win = window(
        &gpu,
        &mut prefill(&gpu, &store, WarmCtx::Restore, &toks, turn),
    );
    assert!(exact(&win), "{win:?}");
    assert_eq!(win[0].0, 100 - W);
    assert_eq!(store.len(), 2, "the branch at 40 and the tail at 96");
    // Another conversation sharing the first 40 tokens restores the branch.
    let other = Turn {
        skip: 40,
        ..Turn::cold(45)
    };
    let mut toks2 = toks[..40].to_vec();
    toks2.extend(tokens(5, 99));
    let win = window(
        &gpu,
        &mut prefill(&gpu, &store, WarmCtx::Restore, &toks2, other),
    );
    assert_eq!((win[0].0, win.len()), (45 - W, W));
    assert!(exact(&win), "{win:?}");
}

/// Rows a sequence-parallel pass leaves to the other rank are a hole: an
/// entry holds only the real rows that end at its checkpoint.
#[test]
fn an_entry_never_spans_a_hole() {
    let gpu = MockGpuBackend::new();
    let store = store(&gpu, 4);
    let toks = tokens(64, 7);
    // Passes [0, 4) and [4, 12); rank 0 keeps rows 2.. and 8.. of them.
    let turn = Turn {
        sp: true,
        ..Turn::cold(12)
    };
    let mut state = prefill(&gpu, &store, WarmCtx::Restore, &toks, turn);
    let run = dstate(&mut state).ctx_run.unwrap();
    assert_eq!((run.lo, run.hi), (8, 12));
    // The cut checkpoint (4) stored [2, 4).
    let next = Turn {
        skip: turn.cut(),
        ..Turn::cold(14)
    };
    let mut state = prefill(&gpu, &store, WarmCtx::Restore, &toks, next);
    assert_eq!(dstate(&mut state).ctx_floor, 2);
}

#[test]
fn the_store_replaces_equal_keys_and_evicts_the_least_recent() {
    let gpu = MockGpuBackend::new();
    let store = store(&gpu, 2);
    for conversation in [10, 11, 10, 12] {
        prefill(
            &gpu,
            &store,
            WarmCtx::Restore,
            &tokens(30, conversation),
            Turn::cold(30),
        );
    }
    assert_eq!(store.len(), 2);
    let turn = Turn {
        skip: Turn::cold(30).cut(),
        ..Turn::cold(31)
    };
    let floor = |c| {
        dstate(&mut prefill(
            &gpu,
            &store,
            WarmCtx::Restore,
            &tokens(31, c),
            turn,
        ))
        .ctx_floor
    };
    assert_eq!(floor(11), turn.skip, "11 was the least recently used");
    assert_eq!(floor(10), 31 - W);
    assert_eq!(floor(12), 31 - W);
    store.free(&gpu);
    assert!(store.is_empty());
}
