// SPDX-License-Identifier: AGPL-3.0-only

use super::snapshot_pool_fits;

const GIB: usize = 1 << 30;

/// Pools comfortably inside the guard: Ok.
#[test]
fn fits_when_the_pools_leave_the_guard() {
    assert!(snapshot_pool_fits(5 * GIB, 10 * GIB, 100 * GIB, 2 * GIB).is_ok());
}

/// Job-100c's shape: a ~81 GB decode-rollback ring against ~40 GB free.
/// The error names the ring, the free space, and every remedy.
#[test]
fn a_ring_too_big_fails_with_the_remedies() {
    let err = snapshot_pool_fits(2 * GIB, 81 * GIB, 43 * GIB, 2 * GIB).unwrap_err();
    let msg = format!("{err}");
    for needle in [
        "decode-rollback ring",
        "free",
        "--speculative",
        "--dflash",
        "--max-batch-size",
        "ATLAS_SSM_DECODE_RING=0",
        "ATLAS_DISABLE_WATCHDOGS=1",
    ] {
        assert!(msg.contains(needle), "message lacks {needle}: {msg}");
    }
    // And the Marconi/ring split is there for the operator to reason with.
    assert!(msg.contains("Marconi"), "{msg}");
}

/// Saturating inputs must not panic on the arithmetic itself.
#[test]
fn huge_inputs_do_not_panic() {
    let _ = snapshot_pool_fits(usize::MAX / 2, usize::MAX / 4, usize::MAX, 2 * GIB);
    let _ = snapshot_pool_fits(usize::MAX, usize::MAX, 0, 0);
}

/// Exactly at the boundary: need == free - guard fits; one byte over does not.
#[test]
fn the_guard_byte_is_the_boundary() {
    let need = 10 * GIB;
    let free = need + 2 * GIB;
    assert!(snapshot_pool_fits(need - 1, 1, free, 2 * GIB).is_ok());
    assert!(snapshot_pool_fits(need, 1, free, 2 * GIB).is_err());
}
