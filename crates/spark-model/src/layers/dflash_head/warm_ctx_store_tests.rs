// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_WARM_CTX=restore`: the store's budget, one window per
//! prefill, and which entry an equal key keeps (mock backend, the prefill
//! simulation of `warm_ctx_tests`).

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::WarmCtx;
use super::first_append_tests::dstate;
use super::warm_ctx_store::{StoreRefs, WarmCtxStore, store_bytes, store_key};
use super::warm_ctx_tests::{Turn, W, exact, prefill, store, tokens, window};

#[test]
fn the_reservation_is_slots_times_a_full_window() {
    // GLM-5.3-Flash + DFlash2: 5 capture layers x 4096 x BF16, 2048 rows.
    let row = 5 * 4096 * 2;
    assert_eq!(store_bytes(8, 2048, row), Some(640 << 20));
    assert_eq!(store_bytes(1, 2048, row), Some(80 << 20));
    assert_eq!(store_bytes(0, 2048, row), Some(0));
    assert_eq!(store_bytes(usize::MAX, 2048, row), None);
    // Nothing is reserved unless this rank owns the drafter context and the
    // mode is `restore` (the test environment sets no ATLAS_DFLASH_WARM_CTX).
    let gpu = MockGpuBackend::new();
    for (layers, rank0) in [(5, true), (0, true), (5, false)] {
        let s = WarmCtxStore::for_build(&gpu, layers, 4096, Some(2048), rank0).unwrap();
        let key = store_key(0, &[1, 2, 3]);
        let saved = s.save(
            &gpu,
            key,
            StoreRefs::default(),
            gpu.alloc(64).unwrap(),
            0,
            1,
            row,
            0,
        );
        assert_eq!(saved.unwrap(), None, "{layers} {rank0}");
    }
    assert!(
        WarmCtxStore::reserve(&gpu, 0, 2048, row)
            .unwrap()
            .is_empty()
    );
}

/// Every checkpoint of a prefill lands in the one window: here a branch
/// point inside the window and the tail cut. Both keys restore the rows the
/// window holds (from 44, where the prompt's window starts).
#[test]
fn a_prefill_writes_one_window_for_all_its_checkpoints() {
    let gpu = MockGpuBackend::new();
    let store = store(&gpu, 4);
    let toks = tokens(64, 20);
    let turn = Turn {
        branch_at: Some(50),
        ..Turn::cold(60)
    };
    prefill(&gpu, &store, WarmCtx::Restore, &toks, turn);
    assert_eq!(store.len(), 1);
    for (skip, prompt) in [(50, 55), (turn.cut(), 58)] {
        let next = Turn {
            skip,
            ..Turn::cold(prompt)
        };
        let mut state = prefill(&gpu, &store, WarmCtx::Restore, &toks, next);
        assert_eq!(dstate(&mut state).ctx_floor, (prompt - W).max(44), "{skip}");
        assert!(exact(&window(&gpu, &mut state)), "{skip}");
    }
}

/// Conversations taking turns in round robin: each turn restores the
/// previous turn's window and stores its own in that slot, so `n` slots
/// serve `n` conversations without a miss.
#[test]
fn n_slots_serve_n_conversations() {
    const N: u32 = 4;
    let gpu = MockGpuBackend::new();
    let store = store(&gpu, N as usize);
    // Prompts stay inside one chunk past 48, so every turn's tail cut is the
    // checkpoint the next turn restores.
    let mut prompt = [60usize; N as usize];
    for c in 0..N {
        prefill(
            &gpu,
            &store,
            WarmCtx::Restore,
            &tokens(200, c),
            Turn::cold(60),
        );
    }
    for round in 0..3 {
        for c in 0..N {
            let i = c as usize;
            let next = Turn {
                skip: Turn::cold(prompt[i]).cut(),
                ..Turn::cold(prompt[i] + 6)
            };
            let mut state = prefill(&gpu, &store, WarmCtx::Restore, &tokens(200, c), next);
            let floor = dstate(&mut state).ctx_floor;
            assert_eq!(floor, next.prompt - W, "round {round} conversation {c}");
            prompt[i] = next.prompt;
        }
        assert_eq!(store.len(), N as usize);
    }
}

/// A seed-fallback prefill that reaches a checkpoint already stored with
/// more rows keeps the longer entry.
#[test]
fn an_equal_key_keeps_the_entry_with_more_rows() {
    let gpu = MockGpuBackend::new();
    let store = store(&gpu, 4);
    let toks = tokens(64, 21);
    // Window [24, 40): the cut 32 stores [24, 32).
    prefill(&gpu, &store, WarmCtx::Restore, &toks, Turn::cold(40));
    // Restored at 28 with nothing stored there: seeded from 28, it reaches
    // the same cut with [28, 32).
    let seeded = Turn {
        skip: 28,
        ..Turn::cold(40)
    };
    let mut state = prefill(&gpu, &store, WarmCtx::Restore, &toks, seeded);
    assert_eq!(dstate(&mut state).ctx_floor, 28);
    let next = Turn {
        skip: 32,
        ..Turn::cold(41)
    };
    let mut state = prefill(&gpu, &store, WarmCtx::Restore, &toks, next);
    assert_eq!(dstate(&mut state).ctx_floor, 41 - W);
    assert!(exact(&window(&gpu, &mut state)));
}

/// Store copies are ordered per window; a backend without events syncs.
#[test]
fn every_store_copy_is_ordered() {
    let gpu = MockGpuBackend::new();
    let store = store(&gpu, 2);
    let syncs = gpu.sync_count();
    prefill(
        &gpu,
        &store,
        WarmCtx::Restore,
        &tokens(64, 22),
        Turn::cold(40),
    );
    assert!(
        gpu.sync_count() > syncs,
        "the mock has no events: the save syncs"
    );
}
