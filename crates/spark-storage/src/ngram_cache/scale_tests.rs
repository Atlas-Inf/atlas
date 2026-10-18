// SPDX-License-Identifier: AGPL-3.0-only

//! Segmented block-scale coverage for `NgramRowCache` — the NVFP4 PLE
/// table's sibling `shard_i.weight_scale` rows, faulted through the same
/// slot as the weight row. Split out of `tests.rs` for the 500-LoC cap.
use super::*;

/// Pinned arena needs a live CUDA context (see ).
#[cfg(feature = "cuda")]
fn cuda_ctx() -> crate::cuda_min::CudaCtx {
    crate::cuda_min::CudaCtx::new(0).expect("CUDA context")
}
#[cfg(not(feature = "cuda"))]
fn cuda_ctx() {}

/// Segmented block scales ride the same cache: every fault lands the row AND
/// its scale row, across several files, seam-straddling included; a second
/// attach is refused; evicted rows refault correctly.
#[test]
#[ignore]
fn segmented_block_scales_fault_evict_and_refuse_second_attach() {
    let _ctx = cuda_ctx();
    use std::io::Write;

    const STRIDE: usize = 160;
    const SSTRIDE: usize = 20; // dim/group style scale row
    const ROWS_PER: u64 = 64;
    const SLOTS: usize = 100; // < total rows, so eviction happens

    let dir = std::env::temp_dir().join(format!("ngram_ss_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let wa = dir.join("w_a.bin");
    let wb = dir.join("w_b.bin");
    let sa = dir.join("s_a.bin");
    let sb = dir.join("s_b.bin");

    let wpat = |shard: usize, local: u64| -> Vec<u8> {
        (0..STRIDE)
            .map(|i| (shard as u8 * 31) ^ (local as u8) ^ i as u8)
            .collect()
    };
    let spat = |shard: usize, local: u64| -> Vec<u8> {
        (0..SSTRIDE)
            .map(|i| (shard as u8 * 17) ^ !(local as u8) ^ i as u8)
            .collect()
    };

    // weight shards: 0,1 in wa / 2,3 in wb; scale shards: 0,1 in sa / 2,3 in sb
    let wlayout = [(0usize, &wa), (1, &wa), (2, &wb), (3, &wb)];
    let slayout = [(0usize, &sa), (1, &sa), (2, &sb), (3, &sb)];
    let write = |layout: &[(usize, &std::path::PathBuf)],
                 pat: &dyn Fn(usize, u64) -> Vec<u8>|
     -> Vec<(std::path::PathBuf, u64)> {
        let mut offsets: Vec<(std::path::PathBuf, u64)> = Vec::new();
        for file in [wa.clone(), wb.clone(), sa.clone(), sb.clone()] {
            if !layout.iter().any(|(_, f)| **f == file) {
                continue;
            }
            let mut fh = std::fs::File::create(&file).expect("create");
            fh.write_all(&[0u8; 24]).expect("lead"); // non-4KiB-aligned base
            let mut at = 24u64;
            for (shard, f) in layout.iter() {
                if *f != &file {
                    continue;
                }
                while offsets.len() <= *shard {
                    offsets.push((file.clone(), 0));
                }
                offsets[*shard] = (file.clone(), at);
                for local in 0..ROWS_PER {
                    fh.write_all(&pat(*shard, local)).expect("row");
                }
                at += ROWS_PER * STRIDE as u64; // file padded generously
            }
            fh.write_all(&[0u8; 8192]).expect("pad");
        }
        offsets
    };
    // scale rows are shorter — write them with their own stride
    let swrite = |layout: &[(usize, &std::path::PathBuf)]| -> Vec<(std::path::PathBuf, u64)> {
        let mut offsets: Vec<(std::path::PathBuf, u64)> = vec![];
        for file in [sa.clone(), sb.clone()] {
            let mut fh = std::fs::File::create(&file).expect("create");
            fh.write_all(&[0u8; 24]).expect("lead");
            let mut at = 24u64;
            for (shard, f) in layout.iter() {
                if *f != &file {
                    continue;
                }
                while offsets.len() <= *shard {
                    offsets.push((file.clone(), 0));
                }
                offsets[*shard] = (file.clone(), at);
                for local in 0..ROWS_PER {
                    fh.write_all(&spat(*shard, local)).expect("srow");
                }
                at += ROWS_PER * SSTRIDE as u64;
            }
            fh.write_all(&[0u8; 8192]).expect("pad");
        }
        offsets
    };
    let wshards = write(&wlayout, &wpat);
    let sshards = swrite(&slayout);

    let mut cache = NgramRowCache::open_segmented(&wshards, ROWS_PER, None, STRIDE, SLOTS)
        .expect("segmented cache");
    cache
        .attach_segmented_block_scales(&sshards, SSTRIDE)
        .expect("attach segmented scales");
    assert!(cache.block_scale_dev_va().expect("va").is_some());
    assert!(
        cache
            .attach_segmented_block_scales(&sshards, SSTRIDE)
            .is_err(),
        "second attach must be refused"
    );

    // Resolve more DISTINCT rows than slots across batches -> forces
    // evictions; every faulted slot must hold the matching row AND scale row.
    let ids: Vec<u64> = (0..256u64).collect();
    for chunk in ids.chunks(64) {
        let mut slots = Vec::new();
        cache.resolve(chunk, &mut slots).expect("resolve");
        cache.end_batch();
    }
    let (h, m, ev) = cache.stats();
    assert!(
        m > 0 && ev > 0,
        "expected misses and evictions: h{h} m{m} ev{ev}"
    );

    // Refault a slice — all must land identical bytes.
    let probe: Vec<u64> = (0..64u64).map(|i| i * 3 % 256).collect();
    let mut slots = Vec::new();
    cache.resolve(&probe, &mut slots).expect("resolve2");
    let sbase = cache
        .scales
        .as_ref()
        .expect("scales")
        .arena
        .slot_host_ptr(0, 0)
        .expect("scale arena") as *const u8;
    for (i, &id) in probe.iter().enumerate() {
        let slot = slots[i] as usize;
        let shard = (id / ROWS_PER) as usize;
        let local = id % ROWS_PER;
        assert_eq!(
            cache.slot_bytes(slot as u32).expect("row"),
            &wpat(shard, local)[..],
            "row {id}"
        );
        let srow = unsafe { std::slice::from_raw_parts(sbase.add(slot * SSTRIDE), SSTRIDE) };
        assert_eq!(srow, &spat(shard, local)[..], "scale row {id}");
    }
    cache.end_batch();
    std::fs::remove_dir_all(&dir).ok();
}
