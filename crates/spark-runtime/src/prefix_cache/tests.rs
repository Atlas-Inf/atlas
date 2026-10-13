// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the prefix-cache result types and the no-op cache. Split out of
//! `prefix_cache.rs` (500-LoC cap).

use super::*;

#[test]
fn test_prefix_match_empty() {
    let m = PrefixMatch::empty();
    assert!(m.is_empty());
    assert_eq!(m.matched_tokens, 0);
    assert!(m.matched_blocks.is_empty());
}

#[test]
fn test_no_prefix_caching_is_noop() {
    let cache = NoPrefixCaching;
    let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let block_table = vec![0, 1];
    let disk_block_ids: Vec<u32> = vec![];

    let m = cache.lookup(&tokens, 4, 0, 0);
    assert!(m.is_empty());

    // These should not panic
    let new_acq = cache.insert(&tokens, &block_table, &disk_block_ids, 4, 0, 0);
    assert!(new_acq.disk_block_ids.is_empty());
    assert!(new_acq.blocks.is_empty());
    cache.release(&tokens, 4, 0);

    let evicted = cache.evict(10);
    assert!(evicted.is_empty());

    assert_eq!(cache.evict_snapshot_lru(), None);
    assert_eq!(cache.snapshot_count(), 0);

    assert_eq!(cache.stats(), (0, 0));
}
