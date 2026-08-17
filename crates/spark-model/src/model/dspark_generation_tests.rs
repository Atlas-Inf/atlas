// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::atomic::{AtomicU64, Ordering};

use super::dspark_generation::next_dspark_generation;

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
