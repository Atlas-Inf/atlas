// SPDX-License-Identifier: AGPL-3.0-only

//! The restore gate, and records kept across a restore
//! (`ATLAS_GLM_NVME_KEEP`), over the fixtures of `kv_nvme_tests.rs`.

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::prefix_cache::{NvmePrefixTier, PrefixCache};

use super::tests::{
    BS, POOL, Path, cache_request, dump, fill, glm_kv_on, pressure, tree_with_tier,
};
use super::{restore_prefix, restore_reads_cached_rows};

/// What the restore gate reads back: everything when the prefill will read
/// its cached prefix, nothing when it is about to rewrite all of it.
#[test]
fn a_restore_pages_in_only_what_the_prefill_reads() {
    let reads = |anchor, floor| restore_reads_cached_rows(anchor, 4096, 256, false, floor);
    assert!(
        reads(1024, false),
        "a usable anchor: [anchor, match) is read"
    );
    assert!(
        !reads(0, false),
        "no anchor: a recompute rewrites the prefix"
    );
    assert!(!reads(255, false), "below the Marconi / fault-in minimum");
    assert!(!reads(4096, false), "the exact-hit anchor is bypassed");
    assert!(restore_reads_cached_rows(4096, 4096, 256, true, false));
    assert!(
        reads(0, true),
        "ATLAS_GLM_PC_WRITE_FLOOR keeps a recompute's rows"
    );
}

/// `ATLAS_GLM_NVME_KEEP`: a restored conversation that is evicted again costs
/// no write, and comes back from the record it already had.
#[test]
fn a_kept_record_is_not_rewritten_when_its_block_is_evicted_again() {
    for path in [Path::Sync, Path::Fast] {
        let gpu = MockGpuBackend::new();
        let mut kv = glm_kv_on(&gpu, None, path);
        let tree = tree_with_tier(64);
        tree.set_keep_restored(true);
        let t: Vec<u32> = (0..3 * BS as u32).collect();
        let want = cache_request(&mut kv, &tree, &gpu, &t);
        pressure(&mut kv, &tree, &gpu);
        let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
        assert_eq!(r.restored, 3);
        assert_eq!(kv.nvme_io_stats().spilled_blocks, 3);

        pressure(&mut kv, &tree, &gpu);
        assert!(tree.lookup(&t, BS, 0, 0).is_empty(), "evicted again");
        assert_eq!(kv.nvme_io_stats().spilled_blocks, 3, "nothing written");
        let s = tree.nvme_stats();
        assert_eq!((s.spills, s.clean_evictions), (3, 3));

        let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
        assert_eq!((r.restored, r.failed), (3, false));
        let m = tree.lookup(&t, BS, 0, 0);
        for (i, &b) in m.matched_blocks.iter().enumerate() {
            assert_eq!(dump(&kv, &gpu, b), want[i], "block {i} bytes");
            assert_eq!(kv.ref_count(b), 1);
        }
        tree.release(&t, BS, 0);
        assert_eq!(kv.num_free_blocks(), POOL - 3);
    }
}

/// `ATLAS_GLM_NVME_KEEP`: a block a prefill recomputed in place after its
/// restore gives up its record (`nvme_forget_rewritten`), so the next restore
/// returns what the block held — not the older record.
#[test]
fn a_rewritten_block_is_written_again_although_records_are_kept() {
    let gpu = MockGpuBackend::new();
    let mut kv = glm_kv_on(&gpu, None, Path::Fast);
    let tree = tree_with_tier(64);
    tree.set_keep_restored(true);
    let t: Vec<u32> = (0..3 * BS as u32).collect();
    let mut want = cache_request(&mut kv, &tree, &gpu, &t);
    pressure(&mut kv, &tree, &gpu);
    restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
    // A prefill resumes at block 1 and recomputes blocks 1 and 2.
    let m = tree.lookup(&t, BS, 0, 0);
    for (i, &b) in m.matched_blocks.iter().enumerate().skip(1) {
        fill(&kv, &gpu, b, 0x90 + i as u8);
        want[i] = dump(&kv, &gpu, b);
    }
    tree.forget_kept(&t, BS, 0, 1..3);
    tree.release(&t, BS, 0);
    pressure(&mut kv, &tree, &gpu);
    let s = tree.nvme_stats();
    assert_eq!(
        (s.spills, s.clean_evictions),
        (5, 1),
        "blocks 1-2 rewritten"
    );
    let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
    assert_eq!((r.restored, r.failed), (3, false));
    let m = tree.lookup(&t, BS, 0, 0);
    for (i, &b) in m.matched_blocks.iter().enumerate() {
        assert_eq!(dump(&kv, &gpu, b), want[i], "block {i} bytes");
    }
}
