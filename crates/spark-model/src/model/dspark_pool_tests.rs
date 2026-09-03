// SPDX-License-Identifier: AGPL-3.0-only

use super::dspark_pool::dflash_verify_rows;

#[test]
fn lightning_k3_allocates_exactly_four_verify_rows() {
    assert_eq!(dflash_verify_rows(true, 3).unwrap(), 4);
}

#[test]
fn generic_k15_allocates_sixteen_not_legacy_seventeen() {
    assert_eq!(dflash_verify_rows(true, 15).unwrap(), 16);
}

#[test]
fn inactive_is_zero_and_overflow_fails() {
    assert_eq!(dflash_verify_rows(false, usize::MAX).unwrap(), 0);
    assert!(dflash_verify_rows(true, usize::MAX).is_err());
}
