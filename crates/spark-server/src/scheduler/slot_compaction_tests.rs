// SPDX-License-Identifier: AGPL-3.0-only

//! Slot compaction must never migrate a sequence onto a slot somebody owns.
//!
//! Regression for the prefill-overlap fault: `compact_survivors_into_range`
//! derived its targets from the ACTIVE list only, so a slot held by a still-
//! PREFILLING sequence looked free. A short request active on slot 1 was
//! compacted onto slot 0 while a long prompt was prefilling there; both then
//! shared SSM state and both released slot 0, leaving it on the pool free
//! list twice for the life of the process.
//!
//! `StubModel::compact_sequence` (in `lifecycle_tests.rs`) mirrors the pool
//! contract: a slot listed in `HELD_SLOTS` cannot be claimed (`Ok(false)`).
//! The pool half of the contract is covered in `spark-model`
//! (`ssm_pool::slot_guard_tests`).

use super::lifecycle_tests::{FREE_CALLS, HELD_SLOTS, MAX_SEQ_LEN, StubModel, test_seq};
use super::mod_helpers::{compact_survivors_into_range, retire_finished_sequences};
use super::types::ActiveSeq;

/// A live (unfinished) sequence on SSM slot `slot`.
fn live_on(slot: usize) -> ActiveSeq {
    let (mut a, _rx) = test_seq(vec![1], 10, None, 100);
    a.finished = false;
    a.seq.slot_idx = slot;
    a
}

fn slots(active: &[ActiveSeq]) -> Vec<usize> {
    active.iter().map(|a| a.seq.slot_idx).collect()
}

fn hold(held: &[usize]) {
    HELD_SLOTS.with(|h| *h.borrow_mut() = held.to_vec());
}

/// THE FAULT: L prefilling on slot 0 (not in `active`), S active on slot 1.
/// The per-tick retire pass must leave S on slot 1.
#[test]
fn retire_does_not_compact_onto_a_prefilling_sequences_slot() {
    hold(&[0]);
    let mut active = vec![live_on(1)];
    retire_finished_sequences(&StubModel::default(), &mut active, MAX_SEQ_LEN);
    assert_eq!(slots(&active), vec![1], "S must not be moved onto L's slot");
}

/// Unchanged behaviour when the target really is free.
#[test]
fn out_of_range_survivor_still_compacts_onto_a_free_slot() {
    hold(&[]);
    let mut active = vec![live_on(3)];
    compact_survivors_into_range(&StubModel::default(), &mut active);
    assert_eq!(slots(&active), vec![0]);
}

/// A refused target is dropped and the next candidate tried; a survivor with
/// no claimable target stays put instead of overwriting anyone.
#[test]
fn refused_target_falls_through_to_the_next_free_one() {
    hold(&[1]); // prefilling sequence on slot 1
    let mut active = vec![live_on(5), live_on(6)];
    compact_survivors_into_range(&StubModel::default(), &mut active);
    assert_eq!(slots(&active), vec![0, 6]);
}

/// Retiring a finished sequence frees its slot; the survivor may take it, but
/// still never the prefilling sequence's slot.
#[test]
fn retire_compacts_into_the_retired_slot_but_not_the_held_one() {
    hold(&[0]); // L prefilling on slot 0
    let (done, _rx) = test_seq(vec![1], 10, None, 100); // finished
    let mut done = done;
    done.seq.slot_idx = 1;
    let mut active = vec![done, live_on(2), live_on(3)];
    FREE_CALLS.with(|c| c.set(0));
    retire_finished_sequences(&StubModel::default(), &mut active, MAX_SEQ_LEN);
    assert_eq!(FREE_CALLS.with(|c| c.get()), 1, "the finished seq is freed");
    // n = 2 → candidates {0, 1}: slot 1 was just vacated, slot 0 is held.
    assert_eq!(slots(&active), vec![1, 3]);
}
