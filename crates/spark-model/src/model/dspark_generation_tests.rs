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
fn dflash_hidden_save_slot_follows_the_live_slot_after_migration() {
    // Admitted on slot 7, compacted to slot 3; the next admission is handed
    // slot 7. Their regions must differ, so the region is the live slot.
    let mut migrated = SequenceState::host_only(7);
    migrated.dspark_owner = Some(SequenceGeneration::new(7, 11).unwrap());
    migrated.slot_idx = 3;
    let mut admitted = SequenceState::host_only(7);
    admitted.dspark_owner = Some(SequenceGeneration::new(7, 12).unwrap());

    assert_eq!(migrated.dflash_hidden_save_slot().unwrap(), 3);
    assert_eq!(admitted.dflash_hidden_save_slot().unwrap(), 7);
    // The owner identity itself does not move.
    assert_eq!(migrated.expected_dspark_owner().unwrap().slot(), 7);
}

#[test]
fn dflash_hidden_save_slot_rejects_a_detached_or_unowned_sequence() {
    let mut seq = SequenceState::host_only(7);
    assert!(seq.dflash_hidden_save_slot().is_err());
    seq.dspark_owner = Some(SequenceGeneration::new(7, 11).unwrap());
    seq.slot_idx = usize::MAX;
    assert!(seq.dflash_hidden_save_slot().is_err());
}

#[test]
fn direct_serial_boundary_rejects_a_state_without_caller_owner() {
    let seq = SequenceState::host_only(0);
    assert!(seq.expected_dspark_owner().is_err());
}
