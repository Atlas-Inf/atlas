// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_FIRST_APPEND`, the request-local half: a sequence's own
//! decode capture lives in its own row, so what a propose appends never
//! depends on what another sequence left in the model-global capture row.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::first_append::{FirstAppend, keep_own_capture, note_own_capture, own_capture_row};
use super::first_append_tests::{
    ALL, NON_LEGACY, PROMPT, SLOT, after_prefill, dstate, global_row, head, slot,
};
use crate::speculative::DraftProposer;

/// The token a serial decode leaves at `PROMPT` is proposed from `NEXT`.
const NEXT: usize = PROMPT + 1;

#[test]
fn a_decode_keeps_its_capture_in_the_sequences_own_row() {
    for variant in ALL {
        let gpu = MockGpuBackend::new();
        let head = head(variant);
        let mut state = Some(after_prefill(&head, &gpu));
        let global = global_row(&gpu);
        dstate(state.as_mut().unwrap()).own_capture = true;
        let copies = gpu.d2d_count();
        keep_own_capture(state.as_deref_mut(), Some(global), NEXT, &gpu).unwrap();
        let d = dstate(state.as_mut().unwrap());
        assert!(!d.own_capture, "{variant:?}: the decode overwrote the row");
        // Legacy has no row of its own: no GPU work, nothing armed.
        let keeps = variant != FirstAppend::Legacy;
        assert_eq!(gpu.d2d_count(), copies + keeps as usize, "{variant:?}");
        assert_eq!(d.own_row_at, keeps.then_some(NEXT), "{variant:?}");
        if let Some(row) = d.own_row {
            let mut kept = vec![0u8; SLOT];
            gpu.copy_d2h(row, &mut kept).unwrap();
            assert_eq!(kept, vec![0xAB; SLOT], "{variant:?}");
        }
    }
}

/// The per-sequence decode loop runs every sequence through the one shared
/// capture row and nothing commits in between. At the next propose a
/// non-legacy sequence appends its own token; legacy appends the last writer.
#[test]
fn sequences_decoding_in_turn_each_append_their_own_row() {
    for variant in ALL {
        let gpu = MockGpuBackend::new();
        let head = head(variant);
        let global = gpu.alloc(SLOT).unwrap();
        let mut states = [0, 1].map(|_| Some(after_prefill(&head, &gpu)));
        for (i, state) in states.iter_mut().enumerate() {
            gpu.memset(global, 0xA0 + i as u8, SLOT).unwrap();
            keep_own_capture(state.as_deref_mut(), Some(global), NEXT, &gpu).unwrap();
        }
        for (i, state) in states.iter_mut().enumerate() {
            let d = dstate(state.as_mut().unwrap());
            head.append_decode_ctx(d, Some(global), NEXT, &gpu, 0)
                .unwrap();
            let expected = match variant {
                FirstAppend::Legacy => 0xA1,
                _ => 0xA0 + i as u8,
            };
            assert_eq!(
                (d.ctx_len, d.ctx_positions.clone()),
                (NEXT, vec![0, 1, 2]),
                "{variant:?} seq {i}"
            );
            assert_eq!(
                slot(&gpu, d, PROMPT),
                vec![expected; SLOT],
                "{variant:?} seq {i}"
            );
        }
    }
}

/// A decode whose propose never ran (the scheduler bailed out in between)
/// and whose shared row another sequence has overwritten since: the next
/// propose still appends this sequence's own token, exactly once.
#[test]
fn a_decode_without_its_propose_stays_request_local() {
    for variant in NON_LEGACY {
        let gpu = MockGpuBackend::new();
        let head = head(variant);
        let mut state = Some(after_prefill(&head, &gpu));
        let global = gpu.alloc(SLOT).unwrap();
        gpu.memset(global, 0x11, SLOT).unwrap();
        keep_own_capture(state.as_deref_mut(), Some(global), NEXT, &gpu).unwrap();
        gpu.memset(global, 0xAB, SLOT).unwrap();
        let d = dstate(state.as_mut().unwrap());
        head.append_decode_ctx(d, Some(global), NEXT, &gpu, 0)
            .unwrap();
        assert_eq!(slot(&gpu, d, PROMPT), vec![0x11; SLOT], "{variant:?}");
        assert_eq!(d.ctx_positions, vec![0, 1, 2], "{variant:?}");
        head.append_decode_ctx(d, Some(global), NEXT + 1, &gpu, 0)
            .unwrap();
        assert_eq!(d.ctx_len, NEXT, "{variant:?}: consumed");
    }
}

/// Decode steps that capture nothing (a multi-sequence batch) move the
/// sequence past its kept row: no variant but legacy appends at the propose
/// that follows, and nothing is read from the shared row.
#[test]
fn a_stretch_without_captures_appends_nothing() {
    for variant in NON_LEGACY {
        let gpu = MockGpuBackend::new();
        let head = head(variant);
        let mut state = Some(after_prefill(&head, &gpu));
        let global = global_row(&gpu);
        keep_own_capture(state.as_deref_mut(), Some(global), NEXT, &gpu).unwrap();
        let d = dstate(state.as_mut().unwrap());
        let copies = gpu.d2d_count() + gpu.memset_count();
        head.append_decode_ctx(d, Some(global), NEXT + 2, &gpu, 0)
            .unwrap();
        assert_eq!((d.ctx_len, d.own_row_at), (PROMPT, None), "{variant:?}");
        assert_eq!(gpu.d2d_count() + gpu.memset_count(), copies, "{variant:?}");
    }
}

/// Right after this sequence's own verify (unified ctx off, so no commit
/// covered it) the caller's row is its verify capture: appended once.
#[test]
fn own_verify_capture_is_appended_once_from_the_callers_row() {
    for variant in NON_LEGACY {
        let gpu = MockGpuBackend::new();
        let head = head(variant);
        let mut state = after_prefill(&head, &gpu);
        let global = global_row(&gpu);
        note_own_capture(state.as_mut());
        let d = dstate(&mut state);
        head.append_decode_ctx(d, Some(global), NEXT, &gpu, 0)
            .unwrap();
        assert_eq!(
            (d.ctx_len, d.ctx_positions.clone()),
            (NEXT, vec![0, 1, 2]),
            "{variant:?}"
        );
        assert_eq!(slot(&gpu, d, PROMPT), vec![0xAB; SLOT], "{variant:?}");
        assert!(!d.own_capture, "{variant:?}: consumed");
        head.append_decode_ctx(d, Some(global), NEXT + 1, &gpu, 0)
            .unwrap();
        assert_eq!(d.ctx_len, NEXT, "{variant:?}");
    }
}

/// A sequence without a drafter state (a worker rank, another proposer).
#[test]
fn keep_own_capture_ignores_absent_state() {
    let gpu = MockGpuBackend::new();
    let global = global_row(&gpu);
    let copies = gpu.d2d_count();
    keep_own_capture(None, Some(global), NEXT, &gpu).unwrap();
    assert_eq!(gpu.d2d_count(), copies);
}

/// The row of a prefill pass that holds its last position: the last row,
/// counted from this rank's first row under a sequence-parallel chunk.
#[test]
fn own_capture_row_is_the_last_row_this_rank_holds() {
    assert_eq!(own_capture_row(24, 0), Some(23));
    // Sequence-parallel, 8192 rows: rank 0 holds the upper half at row 0.
    assert_eq!(own_capture_row(8192, 4096), Some(4095));
    assert_eq!(own_capture_row(1, 0), Some(0));
    assert_eq!(own_capture_row(4096, 4096), None);
    assert_eq!(own_capture_row(0, 0), None);
}

/// Context bookkeeping at the end of a cold prefill with a 4096-slot window:
/// the last `ctx_len` positions are stamped and the first append is armed at
/// the prompt end. It has room only for prompts shorter than the window.
#[test]
fn prefill_seeds_the_window_and_arms_the_first_append() {
    const MAX_CTX: usize = 4096;
    let gpu = MockGpuBackend::new();
    let mut state = head(FirstAppend::Legacy).alloc_state(&gpu).unwrap();
    let d = dstate(&mut state);
    d.max_ctx_len = MAX_CTX;
    for (prompt, ctx_len) in [
        (50, 50),
        (500, 500),
        (1500, 1500),
        (3000, 3000),
        (17000, MAX_CTX),
    ] {
        d.seed_prefill_ctx(prompt, prompt, prompt);
        let stamps: Vec<i32> = ((prompt - ctx_len) as i32..prompt as i32).collect();
        assert_eq!(
            (d.ctx_len, &d.ctx_positions),
            (ctx_len, &stamps),
            "{prompt}"
        );
        assert_eq!(d.first_append_at, Some(prompt), "{prompt}");
        assert_eq!(d.ctx_len < d.max_ctx_len, prompt < MAX_CTX, "{prompt}");
    }
    // A pass that ends before the window starts has captured nothing yet.
    d.seed_prefill_ctx(17000, 8192, 8192);
    assert_eq!((d.ctx_len, d.ctx_positions.len()), (0, 0));
}
