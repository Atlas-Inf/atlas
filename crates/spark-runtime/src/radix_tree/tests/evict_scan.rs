// SPDX-License-Identifier: AGPL-3.0-only

//! Differential tests for the delete-path eviction rewrite
//! (`inner/evict.rs`). The new victim selection is a one-shot min-heap over
//! `(last_access, arena index)`; the old code re-scanned the whole arena once
//! per freed block. The ORACLE here is `RadixTreeInner::evict_linear_scan`, a
//! literal transcription of the pre-rewrite loop kept beside the private
//! arena fields it needs — NOT a second call into the code under test.
//!
//! Each case builds two byte-identical trees, drains one with the heap and
//! one with the oracle, and asserts the freed physical AND disk vectors, plus
//! a full structural snapshot of the surviving trees, agree exactly.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

const BS: usize = 16;

/// xorshift64* — deterministic, dependency-free RNG for reproducible draws.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// Run `num_blocks` through the heap (`evict`) on one tree and the literal
/// scan on an identical copy, asserting the freed vectors AND surviving
/// structure match. `build` must construct the same tree twice.
fn assert_heap_matches_scan<F>(build: F, num_blocks: usize, label: &str)
where
    F: Fn(&RadixTree),
{
    let heap_tree = RadixTree::new();
    let scan_tree = RadixTree::new();
    build(&heap_tree);
    build(&scan_tree);

    let heap = heap_tree.evict(num_blocks);
    let heap_struct = heap_tree.inner.lock().structure_snapshot();

    let (scan_phys, scan_disk) = scan_tree.inner.lock().evict_linear_scan(num_blocks);
    let scan_struct = scan_tree.inner.lock().structure_snapshot();

    assert_eq!(
        heap.physical, scan_phys,
        "{label}: freed physical order diverged (batch={num_blocks})"
    );
    // The public wrapper filters `u32::MAX`; compare against the oracle's raw
    // disk vector with the same filter so both speak the caller's language.
    let scan_disk_filtered: Vec<u32> = scan_disk.into_iter().filter(|&d| d != u32::MAX).collect();
    assert_eq!(
        heap.disk_block_ids, scan_disk_filtered,
        "{label}: freed disk order diverged (batch={num_blocks})"
    );
    assert_eq!(
        heap_struct, scan_struct,
        "{label}: surviving structure diverged (batch={num_blocks})"
    );
}

/// `chains` conversations of `depth` full blocks each, sharing one 0-block
/// first block so branches share a root child. A random subset is re-`lookup`ed
/// so some LRU-eligible nodes carry a live ref and must be skipped.
fn build_random(tree: &RadixTree, rng: &mut Rng) {
    let n_chains = 2 + (rng.next() % 6) as usize;
    let mut next_block: u32 = 1000;
    for _ in 0..n_chains {
        let depth = 1 + (rng.next() % 5) as usize;
        let mut tokens: Vec<u32> = (0..BS as u32).collect(); // shared first block
        tokens.extend((0..BS * depth).map(|_| (rng.next() % 97) as u32));
        let blocks: Vec<u32> = (0..depth + 1)
            .map(|_| {
                next_block += 1;
                next_block
            })
            .collect();
        tree.insert(&tokens, &blocks, &[], BS, 0, 0);
        tree.release(&tokens, BS, 0);
        if rng.next().is_multiple_of(3) {
            let _ = tree.lookup(&tokens, BS, 0, 0); // pin live, no release
        }
    }
}

/// Randomized differential across branching, live refs, ties and several
/// batch sizes. Every seed/batch pair runs heap vs. literal scan.
#[test]
fn differential_heap_matches_literal_scan() {
    for seed in 1..48u64 {
        for &batch in &[1usize, 2, 3, 5, 8, 23, usize::MAX] {
            assert_heap_matches_scan(
                |tree| {
                    let mut rng = Rng::new(seed);
                    build_random(tree, &mut rng);
                },
                batch,
                &format!("seed {seed}"),
            );
        }
    }
}

/// Adapters: two disjoint roots mean the arena interleaves two subtrees; the
/// heap must still match the scan on both.
#[test]
fn differential_heap_matches_scan_across_adapters() {
    for seed in 1..24u64 {
        for &batch in &[1usize, 4, 9, usize::MAX] {
            assert_heap_matches_scan(
                |tree| {
                    let mut rng = Rng::new(seed);
                    for adapter in 0..3u64 {
                        let depth = 1 + (rng.next() % 4) as usize;
                        let mut tokens: Vec<u32> = (0..BS as u32).collect();
                        tokens.extend((0..BS * depth).map(|_| (rng.next() % 89) as u32));
                        let blocks: Vec<u32> = (0..depth + 1)
                            .map(|i| 2000 + adapter as u32 * 100 + i as u32)
                            .collect();
                        tree.insert(&tokens, &blocks, &[], BS, 0, adapter);
                        tree.release(&tokens, BS, adapter);
                    }
                },
                batch,
                &format!("adapter seed {seed}"),
            );
        }
    }
}

/// Free-node reuse: a full drain followed by a fresh insert recycles arena
/// slots. The reused arena then drains by the same order under both schemes.
#[test]
fn differential_heap_matches_scan_with_free_node_reuse() {
    for &batch in &[1usize, 2, 6, usize::MAX] {
        assert_heap_matches_scan(
            |tree| {
                let a: Vec<u32> = (0..BS as u32).collect();
                tree.insert(&a, &[10], &[], BS, 0, 0);
                tree.release(&a, BS, 0);
                let _ = tree.evict(1); // free the slot
                let b: Vec<u32> = (500..500 + BS as u32).collect();
                tree.insert(&b, &[20, 30], &[], BS, 0, 0);
                tree.release(&b, BS, 0);
                let c: Vec<u32> = (900..900 + BS as u32).collect();
                tree.insert(&c, &[40], &[], BS, 0, 0);
                tree.release(&c, BS, 0);
            },
            batch,
            "free-node reuse",
        );
    }
}

/// Equal `last_access` stamps on COMPETING LEAVES (not a single chain): the
/// test explicitly assigns one shared stamp to independent leaf children of
/// the same parent, so the batch must break the tie by the smallest arena
/// index under both schemes. Each extra drain order is re-compared against the
/// oracle.
#[test]
fn equal_stamps_on_competing_leaves_match_scan() {
    let build = |tree: &RadixTree| {
        let shared: Vec<u32> = (0..BS as u32).collect();
        let mut left = shared.clone();
        left.extend(100..100 + BS as u32);
        let mut right = shared.clone();
        right.extend(200..200 + BS as u32);
        let mut mid = shared.clone();
        mid.extend(300..300 + BS as u32);
        tree.insert(&left, &[10, 20], &[], BS, 0, 0);
        tree.insert(&mid, &[10, 30], &[], BS, 0, 0);
        tree.insert(&right, &[10, 40], &[], BS, 0, 0);
        tree.release(&left, BS, 0);
        tree.release(&mid, BS, 0);
        tree.release(&right, BS, 0);
        // Force an actual equal-stamp tie; independent inserts otherwise get
        // distinct operation stamps. This is test-only access to the private
        // arena and mirrors the production tie rule exactly.
        tree.inner.lock().force_test_access(7);
    };
    // Parent 10 plus three siblings; drain them all in one batch.
    assert_heap_matches_scan(build, 4, "competing equal-stamp leaves");
    // And two at a time, so the tie-break must hold mid-drain too.
    assert_heap_matches_scan(build, 3, "competing equal-stamp leaves");
}

/// `evict(0)` and a fully-live tree both no-op identically under both schemes.
#[test]
fn zero_and_all_live_match_scan() {
    assert_heap_matches_scan(
        |tree| {
            let t: Vec<u32> = (0..BS as u32).collect();
            tree.insert(&t, &[7], &[], BS, 0, 0);
            let _held = tree.lookup(&t, BS, 0, 0); // pinned, no release
        },
        5,
        "all-live",
    );
    assert_heap_matches_scan(|_tree| {}, 0, "empty");
}

/// Representative CPU benchmark for the O(N + K·log N) claim. Ignored by
/// default: it is a manual measurement, not a correctness gate. `cargo test
/// -p spark-runtime --lib evict_scan -- --ignored --nocapture`.
///
/// Builds ~80_000 radix nodes and runs one `evict(1800)` under BOTH the heap
/// and the literal whole-arena scan, asserting the outputs are identical
/// element-wise and printing wall times. No threshold is asserted: the point
/// is to SEE the times, not to fail a flaky machine.
#[test]
#[ignore = "manual CPU benchmark; run with --ignored --nocapture"]
fn bench_evict_large_tree_vs_literal_scan() {
    const CONV: u32 = 20_000;
    const DEPTH: usize = 4;
    const RECLAIM: usize = 1800;

    let build = |tree: &RadixTree| {
        for conv in 0..CONV {
            let base = conv * 10_000;
            let tokens: Vec<u32> = (0..BS * DEPTH).map(|i| base + i as u32).collect();
            let blocks: Vec<u32> = (0..DEPTH as u32).map(|b| conv * 8 + b + 1).collect();
            tree.insert(&tokens, &blocks, &[], BS, 0, 0);
            tree.release(&tokens, BS, 0);
        }
    };

    let heap_tree = RadixTree::new();
    let scan_tree = RadixTree::new();
    build(&heap_tree);
    build(&scan_tree);
    let nodes = heap_tree.stats().0;
    assert!(nodes >= 80_000, "expected >= 80k nodes, built {nodes}");

    let t0 = std::time::Instant::now();
    let heap = heap_tree.evict(RECLAIM);
    let heap_time = t0.elapsed();

    let t1 = std::time::Instant::now();
    let (scan_phys, _scan_disk) = scan_tree.inner.lock().evict_linear_scan(RECLAIM);
    let scan_time = t1.elapsed();

    // Exact-output assertion: the benchmark doubles as a large differential.
    assert_eq!(heap.physical, scan_phys, "large-tree victim order diverged");
    assert_eq!(heap.len(), RECLAIM, "short reclaim: {}", heap.len());

    println!(
        "evict({RECLAIM}) over {nodes} nodes: heap {heap_time:?} vs literal scan {scan_time:?} \
         ({:.1}x)",
        scan_time.as_secs_f64() / heap_time.as_secs_f64()
    );
}
