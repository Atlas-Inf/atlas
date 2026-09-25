// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the capacity-reuse aux-blob accessors (`has_aux`, `take_aux`,
//! `with_aux`) and `free`'s clear-in-place behaviour. Separate `#[path]`
//! module per the ≤500 LoC cap idiom.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

/// Small Marconi-only pool (no decode-rollback region), same shape as
/// `ssm_snapshot_reap_tests::pool`.
fn pool(gpu: &dyn GpuBackend, slots: usize, layers: usize) -> SsmSnapshotPool {
    SsmSnapshotPool::new(
        slots, /*h_bytes*/ 32, /*conv_bytes*/ 16, layers, /*decode_ring*/ 0,
        /*decode_max_seqs*/ 0, /*hidden_bytes*/ 8, gpu,
    )
    .unwrap()
}

/// A freed slot's aux entry stays in the map with every blob cleared —
/// `has_aux` reports false, and `take_aux` hands the save path back the
/// cleared-but-still-allocated Vecs for refilling.
#[test]
fn free_clears_blobs_in_place_and_take_aux_returns_capacity() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 1);
    p.set_aux(0, vec![(3, vec![7u8; 64]), (9, vec![1u8; 32])]);
    assert!(p.has_aux(0));

    p.free(0);
    assert!(!p.has_aux(0), "freed slot must not report restorable aux");
    assert!(p.with_aux(0, |_| Ok(())).is_none());

    let taken = p.take_aux(0);
    assert_eq!(taken.len(), 2);
    assert!(taken.iter().all(|(_, b)| b.is_empty()));
    assert!(
        taken.iter().all(|(_, b)| b.capacity() > 0),
        "take_aux must return the freed slot's buffers with capacity intact"
    );
    assert!(!p.has_aux(0));
}

/// `take_aux` on a slot with no entry yields an empty set; `set_aux` +
/// `with_aux` round-trips the blobs without cloning them out.
#[test]
fn take_set_with_aux_round_trip() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 1);
    assert!(p.take_aux(5).is_empty());

    p.set_aux(1, vec![(2, vec![5u8; 16])]);
    let seen = p
        .with_aux(1, |blobs| {
            assert_eq!(blobs.len(), 1);
            assert_eq!(blobs[0].0, 2);
            assert_eq!(blobs[0].1, vec![5u8; 16]);
            Ok(42usize)
        })
        .expect("slot has aux")
        .unwrap();
    assert_eq!(seen, 42);
    // Entry still present after the read (with_aux does not consume).
    assert!(p.has_aux(1));
}
