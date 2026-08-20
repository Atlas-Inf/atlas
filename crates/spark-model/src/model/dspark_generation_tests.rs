// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::atomic::{AtomicU64, Ordering};

use super::dspark_generation::next_dspark_generation;
use crate::layers::dflash_head::SequenceGeneration;
use crate::traits::SequenceState;

#[test]
fn generations_are_nonzero_and_monotonic() {
    let counter = AtomicU64::new(0);
    assert_eq!(next_dspark_generation(&counter).unwrap(), 1);
    assert_eq!(next_dspark_generation(&counter).unwrap(), 2);
}

#[test]
fn generation_exhaustion_does_not_wrap_to_zero() {
    let counter = AtomicU64::new(u64::MAX);
    assert!(next_dspark_generation(&counter).is_err());
    assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
}

#[test]
fn stable_owner_survives_compaction_and_detach_sentinel() {
    let owner = SequenceGeneration::new(7, 11).unwrap();
    let mut seq = SequenceState::host_only(7);
    seq.dspark_owner = Some(owner);

    seq.slot_idx = 3;
    assert_eq!(seq.expected_dspark_owner().unwrap(), owner);
    seq.slot_idx = usize::MAX;
    assert_eq!(seq.expected_dspark_owner().unwrap(), owner);
}

#[test]
fn dflash_hidden_save_slot_follows_immutable_owner_after_slot_migration() {
    let owner = SequenceGeneration::new(7, 11).unwrap();
    let mut seq = SequenceState::host_only(7);
    seq.dspark_owner = Some(owner);

    seq.slot_idx = 3;
    assert_eq!(seq.dflash_hidden_save_slot().unwrap(), owner.slot());
    seq.slot_idx = usize::MAX;
    assert_eq!(seq.dflash_hidden_save_slot().unwrap(), owner.slot());
}

#[test]
fn direct_serial_boundary_rejects_a_state_without_caller_owner() {
    let seq = SequenceState::host_only(0);
    assert!(seq.expected_dspark_owner().is_err());
}
