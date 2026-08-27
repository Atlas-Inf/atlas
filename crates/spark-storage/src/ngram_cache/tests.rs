// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the NVMe-backed n-gram row cache. Split out of
//! `ngram_cache.rs` to keep that file under the 500-LoC cap.

use super::segmented::plan_shard_files;
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

/// **Byte-identical rows across a MULTI-FILE table, read through the real
/// O_DIRECT path.**
///
/// `shards_may_span_several_files` pins the shard -> file MAPPING and needs no
/// GPU. This pins the thing the mapping exists for: that a resolved slot holds
/// the bytes of the row it claims, when those rows come from DIFFERENT backing
/// files and the read crosses shard and file boundaries.
///
/// Adopted from @maplepaladin73's independent GB10 bring-up on PR #16, whose
/// local patch carried this coverage and mine did not. The two of us fixed the
/// same multi-file defect the same way; this is the arm that would have caught
/// a per-shard file index that was merely PLAUSIBLE — off by one, or correct
/// for shard 0 and silently wrong for the rest.
///
/// Ignored by default: the arena is pinned, GPU-addressable memory, so it
/// needs a live CUDA context.
///
///     cargo test -p spark-storage --features cuda multi_file_rows -- --ignored
#[test]
#[ignore]
fn multi_file_rows_are_byte_identical() {
    use std::io::Write;

    const STRIDE: usize = 160; // FP8 row: head_dim 160 x 1 byte
    const ROWS_PER: u64 = 64;
    const PAD: usize = 8192; // keep every row's 4 KiB block inside the file

    // Byte pattern for (shard, local row) — distinct per row AND per offset
    // within the row, so a read that lands on the right row but the wrong
    // offset still fails.
    fn pattern(shard: usize, local: u64) -> Vec<u8> {
        (0..STRIDE)
            .map(|i| {
                (shard as u8)
                    .wrapping_mul(37)
                    .wrapping_add((local as u8).wrapping_mul(11))
                    .wrapping_add(i as u8)
            })
            .collect()
    }

    let dir = std::env::temp_dir().join(format!("ngram_mf_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path_a = dir.join("shards_a.bin");
    let path_b = dir.join("shards_b.bin");

    // The REAL interleave RadixArk produces: shards 0,1,3 in one file and
    // shard 2 in another, so shard index and file index disagree.
    let layout = [(0usize, &path_a), (1, &path_a), (2, &path_b), (3, &path_a)];

    let mut offsets: Vec<(std::path::PathBuf, u64)> = Vec::new();
    for file in [&path_a, &path_b] {
        let mut fh = std::fs::File::create(file).expect("create");
        // A non-4KiB-aligned base, which is what a safetensors data offset is.
        fh.write_all(&vec![0u8; 8]).expect("lead");
        let mut at = 8u64;
        for (shard, f) in layout.iter() {
            if *f != file {
                continue;
            }
            offsets.push(((*file).clone(), at));
            for local in 0..ROWS_PER {
                fh.write_all(&pattern(*shard, local)).expect("row");
            }
            at += ROWS_PER * STRIDE as u64;
        }
        fh.write_all(&vec![0u8; PAD]).expect("pad");
    }
    // `offsets` was filled per file, so put it back in SHARD order — the order
    // `open_segmented` indexes by.
    let mut shards: Vec<(std::path::PathBuf, u64)> = vec![Default::default(); layout.len()];
    let mut k = 0;
    for file in [&path_a, &path_b] {
        for (shard, f) in layout.iter() {
            if *f == file {
                shards[*shard] = offsets[k].clone();
                k += 1;
            }
        }
    }

    let mut cache = NgramRowCache::open_segmented(&shards, ROWS_PER, None, STRIDE, 256)
        .expect("segmented cache over two files");

    // Walk rows that cross both shard and FILE boundaries: the last row of
    // shard 1 (file A), the first and last of shard 2 (file B), the first of
    // shard 3 (file A again).
    let probes: Vec<u64> = vec![
        0,
        ROWS_PER - 1,
        ROWS_PER,
        2 * ROWS_PER - 1,
        2 * ROWS_PER,
        3 * ROWS_PER - 1,
        3 * ROWS_PER,
        4 * ROWS_PER - 1,
    ];
    let mut slots = Vec::new();
    cache.resolve(&probes, &mut slots).expect("resolve");
    assert_eq!(slots.len(), probes.len());

    for (probe, slot) in probes.iter().zip(&slots) {
        let shard = (*probe / ROWS_PER) as usize;
        let local = *probe % ROWS_PER;
        let got = cache.slot_bytes(*slot).expect("slot bytes");
        assert_eq!(
            got,
            &pattern(shard, local)[..],
            "row {probe} (shard {shard} local {local}, file {}) came back with \
             the wrong bytes — the shard -> file mapping or the offset is wrong",
            shards[shard].0.display()
        );
    }

    // Every distinct file opened once, not one descriptor per shard.
    let (_, shard_file) = super::segmented::plan_shard_files(&shards);
    assert_eq!(
        shard_file,
        vec![0, 0, 1, 0],
        "shard 2 must be the second file"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
