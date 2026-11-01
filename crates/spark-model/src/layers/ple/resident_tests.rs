// SPDX-License-Identifier: AGPL-3.0-only
//
//! `NgramTable::ResidentNvfp4` gather parity: a synthetic SEGMENTED NVFP4
//! table interleaved over two backing files, gathered through the real
//! `NgramRowCache` (resolve -> slot) and through the resident buffers
//! `ngram_cache::resident::load_segments_to_device` fills (slot = row id).
//! The two gathers must be byte-identical, INCLUDING THE LAST ROW — a
//! shard-boundary bug would corrupt only rows near the end.
//!
//! GPU + filesystem test:
//!   cargo test -p spark-model --release resident_gather -- --ignored

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ple::ngram_nvfp4::Nvfp4RowGeom;

/// One xorshift byte so row bytes differ everywhere (the gather must not
/// pass on a degenerate all-zero table).
fn synth_table(rows_total: usize, row_bytes: usize) -> Vec<u8> {
    let mut s = 0x243f6a8885a308d3u64;
    (0..rows_total * row_bytes)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

/// Write `interleaved` shards: shard i lives at `offsets[i]` inside
/// `paths[i % paths.len()]` — shards alternate between the two files, so a
/// plan that only opens file 0 loses every odd shard.
fn write_shards(
    dir: &std::path::Path,
    tag: &str,
    shards: usize,
    bytes_per_shard: usize,
    seed: u64,
) -> (
    Vec<std::path::PathBuf>,
    Vec<(std::path::PathBuf, u64)>,
    Vec<u8>,
) {
    let paths: Vec<std::path::PathBuf> = (0..2)
        .map(|i| dir.join(format!("res_{tag}_{i}.bin")))
        .collect();
    let mut offsets = Vec::with_capacity(shards);
    let mut blob: Vec<Vec<u8>> = vec![Vec::new(); paths.len()];
    let mut table = Vec::with_capacity(shards * bytes_per_shard);
    // Fixed-size shard headers so different shards land at different
    // offsets in each file (the interleave matters, not the packing).
    let header = 4096usize;
    let mut s = seed;
    for i in 0..shards {
        let p = i % paths.len();
        let new_len = blob[p].len().next_multiple_of(header) + header;
        blob[p].resize(new_len, 0xAB);
        offsets.push((paths[p].clone(), blob[p].len() as u64));
        let rows = (0..bytes_per_shard)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect::<Vec<u8>>();
        blob[p].extend_from_slice(&rows);
        table.extend_from_slice(&rows);
    }
    for (p, bytes) in paths.iter().zip(blob) {
        std::fs::write(p, bytes).unwrap();
    }
    (paths, offsets, table)
}

/// The resident and cached gathers are compared over ALL rows plus a
/// second pass (re-fault) — slots deliberately fewer than rows.
#[test]
#[ignore = "GPU + filesystem"]
fn resident_gather_matches_cached_gather() {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    // dim 256, group 32 -> 128 B packed row, 8 B scale row; 8 rows/shard.
    let (dim, group) = (256usize, 32usize);
    let (row_b, scale_b) = (dim / 2, dim / group);
    let geom = Nvfp4RowGeom {
        dim,
        group,
        global: 1.0,
    };
    let _ = synth_table(0, 0);
    let dir = std::env::temp_dir().join(format!("atlas_ple_res_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (paths, wshards, wtable) = write_shards(&dir, "w", 4, 8 * row_b, 0xC0FFEE);
    let (spaths, sshards, stable) = write_shards(&dir, "s", 4, 8 * scale_b, 0xBADA55);
    let rows_total = (wshards.len() * 8) as u64;

    // Resident fill through the production streaming loader.
    let rows_dev = g.alloc(wtable.len()).unwrap();
    let scales_dev = g.alloc(stable.len()).unwrap();
    spark_storage::ngram_cache::resident::load_segments_to_device(
        &wshards,
        (8 * row_b) as u64,
        rows_dev,
        g,
        stream,
    )
    .unwrap();
    spark_storage::ngram_cache::resident::load_segments_to_device(
        &sshards,
        (8 * scale_b) as u64,
        scales_dev,
        g,
        stream,
    )
    .unwrap();

    // Cached gather: the same shard lists through the real row cache.
    let slots = 12; // < 32 rows: forces eviction + re-fault.
    let mut cache =
        spark_storage::NgramRowCache::open_segmented(&wshards, 8, None, row_b, slots).unwrap();
    cache
        .attach_segmented_block_scales(&sshards, scale_b)
        .unwrap();
    let cached_table = DevicePtr(cache.table_dev_va().unwrap());
    let cached_scales = DevicePtr(cache.block_scale_dev_va().unwrap().unwrap());

    let k = g
        .kernel("embed_from_argmax", "batched_embed_nvfp4")
        .unwrap();
    // Every row once, then the LAST half again (re-fault across the last
    // row's shard boundary).
    let ids: Vec<u64> = (0..rows_total).chain(rows_total / 2..rows_total).collect();
    let m = ids.len();
    let ids_dev = g.alloc(m * 4).unwrap();
    let out_res = g.alloc(m * dim * 2).unwrap();
    let out_cache = g.alloc(m * dim * 2).unwrap();

    // Resident leg: slot = row id.
    let idb: Vec<u8> = ids.iter().flat_map(|v| (*v as u32).to_le_bytes()).collect();
    g.copy_h2d_async(&idb, ids_dev, stream).unwrap();
    crate::layers::ops::batched_embed_nvfp4(
        g,
        k,
        ids_dev,
        rows_dev,
        scales_dev,
        geom.global,
        out_res,
        m as u32,
        dim as u32,
        group as u32,
        row_b as u32,
        scale_b as u32,
        stream,
    )
    .unwrap();

    // Cached leg: batched resolve -> slot gather, honoring the pin
    // contract (batch <= slots).
    let mut slot_ids = Vec::new();
    let batch = slots;
    for (bi, chunk) in ids.chunks(batch).enumerate() {
        cache.resolve(chunk, &mut slot_ids).unwrap();
        assert_eq!(slot_ids.len(), chunk.len());
        let sb: Vec<u8> = slot_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d_async(&sb, ids_dev, stream).unwrap();
        crate::layers::ops::batched_embed_nvfp4(
            g,
            k,
            ids_dev,
            cached_table,
            cached_scales,
            geom.global,
            out_cache.offset(bi * batch * dim * 2),
            chunk.len() as u32,
            dim as u32,
            group as u32,
            row_b as u32,
            scale_b as u32,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        cache.end_batch();
    }

    let mut a = vec![0u8; m * dim * 2];
    let mut b = vec![0u8; m * dim * 2];
    g.copy_d2h(out_res, &mut a).unwrap();
    g.copy_d2h(out_cache, &mut b).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let _ = paths;
    let _ = spaths;
    let (hits, misses, evictions) = cache.stats();
    assert!(misses >= rows_total, "cached leg must fault every row once");
    assert_eq!(
        a, b,
        "resident gather diverged from the cached path \
         (rows={rows_total}, hits={hits} misses={misses} evictions={evictions})"
    );
    println!(
        "PLE resident gather: BYTE-IDENTICAL over {m} lookups, {rows_total} rows \
         incl. the last row (hits={hits} misses={misses} evictions={evictions})"
    );
}
