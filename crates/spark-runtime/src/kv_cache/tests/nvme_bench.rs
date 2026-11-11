// SPDX-License-Identifier: AGPL-3.0-only

//! Disk microbench for the NVMe prefix tier: the REAL spill / restore code on
//! a real O_DIRECT file, with the mock GPU standing in for the KV pools (so
//! it measures disk I/O, checksums and the worker pipeline — not the GPU copy
//! engine, whose shapes are measured separately; see
//! `docs/glm-nvme-prefix-cache.md` §9).
//!
//! Ignored by default. On a host with an NVMe-backed checkout:
//!
//! ```sh
//! cargo test --release -p spark-runtime --lib nvme_disk_bench -- --ignored --nocapture
//! ```
//!
//! `ATLAS_NVME_BENCH_DIR` picks the directory (default `target/nvme-bench`
//! under the workspace), `ATLAS_NVME_BENCH_CONVS` the conversation count
//! (default 4 × 1,582 blocks ≈ 674 MB on disk per path, 2.7 GB of mock pools).

use std::time::Instant;

use super::*;
use crate::prefix_cache::{DiskRef, SpillOrder};

/// One GLM-5.3 conversation of ~25K tokens, as measured on the pair.
const CONV_BLOCKS: usize = 1582;
/// Blocks evicted per allocation miss (`PagedKvCache::evict_batch`).
const EVICT: usize = 32;

/// GLM-5.3 per-rank geometry: 11 sparse-MLA layers, FP8-G128 latent with V
/// aliasing K, pooled BF16 index — a 106,496 B record.
fn glm53(gpu: &MockGpuBackend, blocks: usize) -> PagedKvCache {
    let cfg = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: 11,
        dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let mut c = PagedKvCache::new_with_v_alias(cfg, blocks, gpu, true).unwrap();
    c.attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
        .unwrap();
    assert_eq!(c.nvme_record_bytes(), 106_496);
    c
}

fn attach(c: &mut PagedKvCache, gpu: &MockGpuBackend, dir: &std::path::Path, fast: bool) {
    let path = dir.join(format!("bench.{}.{fast}.swap", std::process::id()));
    let record = c.nvme_record_bytes();
    let _ = std::fs::remove_file(&path);
    if fast {
        let f = atlas_tier::SharedRecordFile::create(&path, record).unwrap();
        // As the factory does: the budget is allocated up front.
        f.reserve((3 * c.num_blocks() * record) as u64).unwrap();
        c.attach_nvme_fast(std::sync::Arc::new(f), gpu).unwrap();
    } else {
        let f = atlas_tier::DirectSwapFile::create(&path, record).unwrap();
        c.attach_nvme_spill(Box::new(f), gpu).unwrap();
    }
    std::fs::remove_file(&path).unwrap();
}

/// Distinct bytes per block in every layer's K region (index left zero).
fn fill(c: &PagedKvCache, gpu: &MockGpuBackend, block: u32) {
    let len = c.k_block_stride_bytes_for_layer(0);
    let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ (u64::from(block) << 20);
    let bytes: Vec<u8> = (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect();
    for l in 0..c.num_layers() {
        gpu.copy_h2d(&bytes, c.k_cache_ptr(l, block)).unwrap();
    }
}

fn k0(c: &PagedKvCache, gpu: &MockGpuBackend, block: u32) -> Vec<u8> {
    let mut out = vec![0u8; c.k_block_stride_bytes_for_layer(0)];
    gpu.copy_d2h(c.k_cache_ptr(0, block), &mut out).unwrap();
    out
}

fn tag(slot: u32) -> u64 {
    0xC0DE_0000 + u64::from(slot)
}

struct Timing {
    /// Time inside `nvme_write` — what the evicting request waits for.
    spill_ms: f64,
    /// Until every record is on disk.
    durable_ms: f64,
    restore_ms: Vec<f64>,
}

/// Spill `convs` conversations (leaf first, `EVICT` blocks per call), then
/// restore each root first. `slot_of` maps a spill ordinal to its record slot:
/// the identity is a fresh tier (contiguous runs), a permutation is a
/// recycled one (every record its own request). `epoch` makes the tags of
/// each pass distinct.
fn spill_then_restore(
    c: &mut PagedKvCache,
    gpu: &MockGpuBackend,
    convs: usize,
    epoch: u64,
    slot_of: impl Fn(u32) -> u32,
) -> Timing {
    let tag = |ordinal: u32| tag(ordinal) + (epoch << 40);
    let n = convs * CONV_BLOCKS;
    let mut orders = Vec::with_capacity(n);
    for conv in 0..convs {
        for i in 0..CONV_BLOCKS {
            let block = (conv * CONV_BLOCKS + (CONV_BLOCKS - 1 - i)) as u32;
            orders.push(SpillOrder {
                block,
                slot: slot_of(orders.len() as u32),
                tag: tag(orders.len() as u32),
            });
        }
    }
    let t0 = Instant::now();
    for batch in orders.chunks(EVICT) {
        assert!(c.nvme_write(batch, gpu, 0).is_empty());
    }
    let spill_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert_eq!(c.nvme_read(&[], &mut [], gpu, 0), (0, false)); // write-behind drained
    let durable_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert!(c.nvme_take_failed().is_empty());

    let mut restore_ms = Vec::new();
    for conv in 0..convs {
        // Path order = reverse spill order; targets are the pool's second half.
        let disk: Vec<DiskRef> = (0..CONV_BLOCKS)
            .rev()
            .map(|i| {
                let ordinal = (conv * CONV_BLOCKS + i) as u32;
                DiskRef {
                    slot: slot_of(ordinal),
                    tag: tag(ordinal),
                }
            })
            .collect();
        let first = (n + conv * CONV_BLOCKS) as u32;
        let mut blocks: Vec<u32> = (first..first + CONV_BLOCKS as u32).collect();
        let t = Instant::now();
        assert_eq!(
            c.nvme_read(&disk, &mut blocks, gpu, 0),
            (CONV_BLOCKS, false)
        );
        restore_ms.push(t.elapsed().as_secs_f64() * 1e3);
        for i in [0, CONV_BLOCKS / 2, CONV_BLOCKS - 1] {
            let source = (conv * CONV_BLOCKS + i) as u32;
            assert_eq!(k0(c, gpu, blocks[i]), k0(c, gpu, source), "block {i}");
        }
    }
    Timing {
        spill_ms,
        durable_ms,
        restore_ms,
    }
}

/// The pair scenario: the pool is full, so a restore first evicts as many
/// victims as it restores. Returns the milliseconds the restore log line
/// would report for conversation 0 (already on disk), with conversation 1
/// (resident, never spilled) as the victims.
fn restore_with_evictions(c: &mut PagedKvCache, gpu: &MockGpuBackend, base_slot: u32) -> f64 {
    let victims: Vec<SpillOrder> = (0..CONV_BLOCKS as u32)
        .map(|i| SpillOrder {
            block: (2 * CONV_BLOCKS as u32 - 1) - i,
            slot: base_slot + i,
            tag: tag(base_slot + i),
        })
        .collect();
    let disk: Vec<DiskRef> = (0..CONV_BLOCKS as u32)
        .rev()
        .map(|i| DiskRef {
            slot: i,
            tag: tag(i),
        })
        .collect();
    let mut blocks: Vec<u32> = (0..CONV_BLOCKS as u32).collect();
    let t = Instant::now();
    for batch in victims.chunks(EVICT) {
        assert!(c.nvme_write(batch, gpu, 0).is_empty());
    }
    assert_eq!(
        c.nvme_read(&disk, &mut blocks, gpu, 0),
        (CONV_BLOCKS, false)
    );
    t.elapsed().as_secs_f64() * 1e3
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

#[test]
#[ignore = "disk microbench: run explicitly, in release, on an NVMe-backed directory"]
fn nvme_disk_bench() {
    let dir = std::env::var_os("ATLAS_NVME_BENCH_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            // <target>/<profile>/deps/<test binary>: honours CARGO_TARGET_DIR.
            let exe = std::env::current_exe().unwrap();
            exe.ancestors().nth(3).unwrap().join("nvme-bench")
        });
    std::fs::create_dir_all(&dir).unwrap();
    let convs: usize = std::env::var("ATLAS_NVME_BENCH_CONVS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let n = convs * CONV_BLOCKS;
    let conv_mb = (CONV_BLOCKS * 106_496) as f64 / 1e6;
    println!(
        "NVMe prefix tier disk microbench: {convs} conversations × {CONV_BLOCKS} blocks \
         ({conv_mb:.0} MB each) in {}",
        dir.display()
    );
    // A recycled tier hands out slots in no useful order: a stride coprime
    // to the slot count visits every slot once, no two in a row.
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let stride = (n / 3..n).find(|&k| gcd(k, n) == 1).unwrap() as u64;
    for fast in [false, true] {
        let path = if fast { "fast" } else { "sync" };
        let gpu = MockGpuBackend::new();
        let mut c = glm53(&gpu, 2 * n);
        // The restore targets too: the mock pools are lazily committed, and
        // first-touch page faults are not the tier's cost.
        for b in 0..2 * n as u32 {
            fill(&c, &gpu, b);
        }
        attach(&mut c, &gpu, &dir, fast);
        let report = |layout: &str, t: &Timing| {
            let restore = median(&t.restore_ms);
            println!(
                "{path:>4} {layout:>10}: spill {:7.1} ms on the evicting thread ({:6.0} MB/s), \
                 durable after {:7.1} ms; restore {restore:6.1} ms per conversation \
                 ({:6.0} MB/s, {:5.1} us/block)",
                t.spill_ms,
                conv_mb * convs as f64 / t.spill_ms * 1e3,
                t.durable_ms,
                conv_mb / restore * 1e3,
                restore * 1e3 / CONV_BLOCKS as f64,
            );
        };
        // Pass 1 — a fresh tier: slots in spill order, the file growing.
        report(
            "contiguous",
            &spill_then_restore(&mut c, &gpu, convs, 0, |i| i),
        );
        if convs >= 2 {
            // Conversation 0 is on disk in slots 0..; restore it while
            // evicting conversation 1 into fresh slots.
            let ms = restore_with_evictions(&mut c, &gpu, n as u32);
            println!(
                "{path:>4}  full pool: restore + evict {CONV_BLOCKS} victims in {ms:6.1} ms \
                 ({:6.0} MB/s restored)",
                conv_mb / ms * 1e3
            );
        }
        // Pass 2 — the same slots recycled in scattered order (overwrites).
        let scattered = |i: u32| (u64::from(i) * stride % n as u64) as u32;
        report(
            "fragmented",
            &spill_then_restore(&mut c, &gpu, convs, 1, scattered),
        );
    }
    let _ = std::fs::remove_dir(&dir);
}
