// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the NVMe-backed n-gram row cache. Split out of
//! `ngram_cache.rs` to keep that file under the 500-LoC cap.

use super::*;

#[test]
fn aligned_scratch_is_4k_aligned_and_two_blocks() {
    let mut b = AlignedBlock::new();
    let s = b.blocks(2);
    assert_eq!(s.len(), BLOCK * 2);
    assert_eq!(s.as_ptr() as usize % BLOCK, 0);
}

#[test]
fn row_wider_than_a_block_is_refused() {
    // A row larger than one block could span three with an unaligned base.
    let msg = match NgramRowCache::open(Path::new("/nonexistent"), None, 10, BLOCK + 8, 4) {
        Ok(_) => panic!("expected refusal for oversize row_stride"),
        Err(e) => format!("{e:#}"),
    };
    assert!(msg.contains("O_DIRECT block"), "{msg}");
}

/// The seam arithmetic: with a base offset that is only 8-byte aligned
/// (what a safetensors shard gives), rows land at every phase relative to
/// the 4 KiB block, and the covering span must stay within two blocks.
#[test]
fn straddling_rows_are_covered_by_two_blocks() {
    for base in [0u64, 8, 1234568, 4095, 4097] {
        for stride in [256usize, 512, 4096] {
            for id in [0u64, 1, 7, 8, 1023] {
                let byte = base + id * stride as u64;
                let block_off = byte - (byte % BLOCK as u64);
                let within = (byte - block_off) as usize;
                let n = if within + stride > BLOCK { 2 } else { 1 };
                assert!(
                    within + stride <= n * BLOCK,
                    "base={base} stride={stride} id={id} within={within} n={n}"
                );
            }
        }
    }
}

/// **The multi-file regression.** A segmented table's shards are not
/// necessarily in one file — RadixArk's NVFP4 conversion of
/// Qwen3.8-Flash-Next spreads 128 PLE shards over 10
/// `model-plefp8-*.safetensors`, interleaved: shards 0 and 1 in the first
/// file, shard 2 in the fourth. The loader used to REFUSE that outright
/// ("shard 2 lives in a different file from shard 0"), which is what a
/// first real load hit.
///
/// Pinned as a mapping test rather than a read test because opening a
/// cache needs a pinned GPU-addressable arena; the mapping is the part
/// that was wrong.
#[test]
fn shards_may_span_several_files() {
    let a = std::path::PathBuf::from("/ckpt/model-plefp8-00000.safetensors");
    let b = std::path::PathBuf::from("/ckpt/model-plefp8-00003.safetensors");
    // The real interleaving: 0,1 -> a; 2 -> b; 3 -> a again.
    let shards = vec![
        (a.clone(), 100),
        (a.clone(), 200),
        (b.clone(), 300),
        (a.clone(), 400),
    ];
    let (paths, shard_file) = plan_shard_files(&shards);
    assert_eq!(paths.len(), 2, "two distinct files, deduped");
    assert_eq!(paths[0], a.as_path());
    assert_eq!(paths[1], b.as_path());
    // Shard 2 must resolve to the SECOND file. Under the old single-file
    // assumption every shard pointed at `a` and shard 2 read whatever
    // happened to sit at offset 300 of the wrong file.
    assert_eq!(shard_file, vec![0, 0, 1, 0]);
}

/// The single-file case must stay exactly one descriptor — the LongCat
/// tables are one contiguous tensor and must not regress into N opens.
#[test]
fn one_file_stays_one_descriptor() {
    let a = std::path::PathBuf::from("/ckpt/only.safetensors");
    let shards: Vec<_> = (0..128u64).map(|i| (a.clone(), i * 4096)).collect();
    let (paths, shard_file) = plan_shard_files(&shards);
    assert_eq!(paths.len(), 1);
    assert!(shard_file.iter().all(|f| *f == 0));
    assert_eq!(shard_file.len(), 128);
}
