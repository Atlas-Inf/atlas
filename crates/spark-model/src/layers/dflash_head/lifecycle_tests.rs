// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::{HashMap, HashSet};

use super::lifecycle::*;

fn owner(slot: usize, generation: u64) -> SequenceGeneration {
    SequenceGeneration::new(slot, generation).unwrap()
}

#[test]
fn zero_generation_stride_capacity_and_excess_rows_fail_closed() {
    assert!(SequenceGeneration::new(0, 0).is_err());
    assert!(CaptureDescriptor::bind(owner(0, 1), 10, 1, 4, 0).is_err());
    assert!(CaptureDescriptor::bind(owner(0, 1), 10, 1, 0, 16).is_err());
    assert!(CaptureDescriptor::bind(owner(0, 1), 10, 5, 4, 16).is_err());
}

#[test]
fn descriptor_retains_exact_owner_position_rows_stride_and_range() {
    let o = owner(7, 11);
    let d = CaptureDescriptor::bind(o, 123, 3, 4, 64).unwrap();
    assert_eq!(d.owner(), o);
    assert_eq!((o.slot(), o.generation()), (7, 11));
    assert_eq!(d.absolute_position(), 123);
    assert_eq!(d.valid_rows(), 3);
    assert_eq!(d.row_capacity(), 4);
    assert_eq!(d.row_stride_bytes(), 64);
    assert_eq!(d.status(), CaptureStatus::Live);
    assert_eq!(d.row_range(o, 2).unwrap(), 128..192);
    assert!(d.row_range(o, 3).is_err());
}

#[test]
fn batch_reorder_never_changes_owner_identity() {
    let a = CaptureDescriptor::bind(owner(2, 8), 50, 1, 4, 32).unwrap();
    let b = CaptureDescriptor::bind(owner(5, 3), 90, 1, 4, 32).unwrap();
    let batch_one = [&a, &b];
    let batch_two = [&b, &a];
    assert_eq!(batch_one[0].owner(), batch_two[1].owner());
    assert_eq!(batch_one[1].owner(), batch_two[0].owner());
    assert_ne!(batch_one[0].owner(), batch_one[1].owner());
}

#[test]
fn width_churn_advances_monotonically_without_rebinding_owner() {
    let o = owner(3, 9);
    let mut d = CaptureDescriptor::bind(o, 100, 1, 4, 128).unwrap();
    d.advance(o, 101, 4, 128).unwrap();
    assert_eq!(
        (d.owner(), d.absolute_position(), d.valid_rows()),
        (o, 101, 4)
    );
    d.advance(o, 102, 1, 128).unwrap();
    assert_eq!(
        (d.owner(), d.absolute_position(), d.valid_rows()),
        (o, 102, 1)
    );
    assert!(d.advance(o, 101, 1, 128).is_err());
    assert!(d.advance(o, 103, 1, 64).is_err());
}

#[test]
fn stale_generation_cannot_access_advance_or_retire_new_owner() {
    let old = owner(4, 20);
    let live = owner(4, 21);
    let mut d = CaptureDescriptor::bind(live, 200, 2, 4, 64).unwrap();
    assert!(d.validate_access(old).is_err());
    assert!(d.row_range(old, 0).is_err());
    assert!(d.advance(old, 201, 2, 64).is_err());
    assert!(d.retire(old).is_err());
    assert_eq!(d.status(), CaptureStatus::Live);
}

#[test]
fn swapped_sequence_state_owner_is_rejected_even_when_slot_is_live() {
    let original = owner(4, 20);
    let swapped = owner(5, 20);
    let descriptor = CaptureDescriptor::bind(original, 200, 2, 4, 64).unwrap();
    assert!(descriptor.validate_access(swapped).is_err());
}

#[test]
fn retirement_is_same_owner_idempotent_and_blocks_every_access() {
    let o = owner(1, 2);
    let mut d = CaptureDescriptor::bind(o, 30, 2, 4, 16).unwrap();
    d.retire(o).unwrap();
    d.retire(o).unwrap();
    assert_eq!(d.status(), CaptureStatus::Retired);
    assert_eq!(d.valid_rows(), 0);
    assert!(d.validate_access(o).is_err());
    assert!(d.advance(o, 31, 1, 16).is_err());
}

#[test]
fn pointer_reuse_cannot_reuse_graph_identity_across_generation() {
    let old = DflashGraphIdentity::new(owner(8, 40), 0x100, 0x200, 0x300, 2).unwrap();
    let new = DflashGraphIdentity::new(owner(8, 41), 0x100, 0x200, 0x300, 2).unwrap();
    assert_ne!(old, new);
    let mut set = HashSet::new();
    set.insert(old);
    set.insert(new);
    assert_eq!(set.len(), 2);
    assert_eq!(new.owner(), owner(8, 41));
    assert_eq!(
        (
            new.block_table_ptr(),
            new.ctx_ptr(),
            new.markov_ptr(),
            new.lane()
        ),
        (0x100, 0x200, 0x300, 2)
    );
    assert!(DflashGraphIdentity::new(owner(8, 42), 0, 0x200, 0x300, 2).is_err());
    assert!(DflashGraphIdentity::new(owner(8, 42), 0x100, 0x200, 0x300, usize::MAX).is_err());
}

#[test]
fn stale_completion_cannot_select_new_generation_state() {
    let old = owner(9, 100);
    let new = owner(9, 101);
    let descriptor = CaptureDescriptor::bind(new, 501, 4, 4, 256).unwrap();
    let mut live = HashMap::new();
    live.insert(new, descriptor);
    assert!(!live.contains_key(&old));
    assert!(live[&new].validate_access(old).is_err());
    assert!(live[&new].validate_access(new).is_ok());
}

#[test]
fn row_offset_overflow_is_rejected() {
    let o = owner(1, 1);
    let d = CaptureDescriptor::bind(o, 0, 2, 2, usize::MAX).unwrap();
    assert!(d.row_range(o, 1).is_err());
}

#[test]
fn graph_retirement_removes_only_the_exact_generation_owner() {
    let old = owner(2, 10);
    let new = owner(2, 11);
    let mut graphs = HashMap::from([
        (DflashGraphIdentity::new(old, 1, 2, 3, 0).unwrap(), "old-a"),
        (DflashGraphIdentity::new(old, 4, 5, 6, 1).unwrap(), "old-b"),
        (DflashGraphIdentity::new(new, 1, 2, 3, 0).unwrap(), "new"),
    ]);
    let mut retired = take_owned_graphs(&mut graphs, old);
    retired.sort_unstable();
    assert_eq!(retired, vec!["old-a", "old-b"]);
    assert_eq!(graphs.len(), 1);
    assert_eq!(graphs.values().copied().collect::<Vec<_>>(), vec!["new"]);
    assert!(take_owned_graphs(&mut graphs, old).is_empty());
}
