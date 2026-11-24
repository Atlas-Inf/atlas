// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::{
    gpu::mock::{MockGpuBackend, MockLaunch},
    weights::{DeferredTensor, WeightTensor},
};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};
/// `fp8_table` selects the checkpoint contract: fp8 tables ship with a bf16
/// `weight_scale_inv` row per table row; bf16 tables ship alone (no scale
/// tensor) and `engram_fp8_embedding` is false.
fn fixture(
    gpu: &MockGpuBackend,
    rank: usize,
    world: usize,
    fp8_table: bool,
) -> (ModelConfig, WeightStore) {
    let mut c = atlas_core::config::parse_config(include_str!(
        "../../../../../atlas-core/src/config/fixtures/hy4/config.json"
    ))
    .unwrap();
    c.tp_rank = rank;
    c.tp_world_size = world;
    let hy4 = c.hy4.as_mut().unwrap();
    hy4.engram_fp8_embedding = fp8_table;
    let sizes = vec![vec![7; 8]; 2];
    hy4.engram_layer_head_vocab_sizes.insert(2, sizes);
    let p = "model.layers.1.engram_layer";
    let h = c.hidden_size;
    let mut m = HashMap::new();
    let mut add = |leaf: &str, shape: Vec<usize>, dtype: WeightDtype| {
        let ptr = gpu
            .alloc(shape.iter().product::<usize>() * dtype.byte_size())
            .unwrap();
        m.insert(format!("{p}.{leaf}"), WeightTensor { ptr, shape, dtype });
        ptr
    };
    let ptr = add("lookup_table", vec![120818], WeightDtype::Int32);
    gpu.copy_h2d(
        &(0..120818i32)
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>(),
        ptr,
    )
    .unwrap();
    add(
        "multi_head_embedding.embedding.weight",
        vec![112, 96],
        if fp8_table {
            WeightDtype::FP8E4M3
        } else {
            WeightDtype::BF16
        },
    );
    if fp8_table {
        add(
            "multi_head_embedding.embedding.weight_scale_inv",
            vec![112],
            WeightDtype::BF16,
        );
    }
    for i in 0..16 {
        add(
            &format!("lookup_per_head_norms.{i}.weight"),
            vec![96],
            WeightDtype::BF16,
        );
    }
    for i in 0..4 {
        for name in ["norm_key", "norm_query"] {
            add(&format!("{name}.{i}.weight"), vec![h], WeightDtype::BF16);
        }
        add(
            &format!("unified_per_hc_scales.{i}"),
            vec![h],
            WeightDtype::BF16,
        );
    }
    for i in 0..5 {
        let stem = if i == 4 {
            "value_proj".into()
        } else {
            format!("key_projs.{i}")
        };
        add(&format!("{stem}.weight"), vec![h, 1536], WeightDtype::BF16);
        add(&format!("{stem}.bias"), vec![h], WeightDtype::BF16);
    }
    add("unified_pre_gamma", vec![h], WeightDtype::BF16);
    (c, WeightStore::from_map(m))
}
struct Comm {
    rank: usize,
    calls: AtomicUsize,
    bytes: usize,
}
impl spark_comm::CommBackend for Comm {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_reduce(&self, _p: u64, _b: usize) -> Result<()> {
        anyhow::bail!("must use stream-ordered reduction")
    }
    fn all_reduce_async(&self, _p: u64, b: usize, _s: u64) -> Result<()> {
        assert_eq!(b, self.bytes);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn all_gather(&self, _s: u64, _r: u64, _b: usize) -> Result<()> {
        anyhow::bail!("unexpected gather")
    }
    fn reduce_scatter(&self, _s: u64, _r: u64, _b: usize) -> Result<()> {
        anyhow::bail!("unexpected scatter")
    }
    fn broadcast(&self, _p: u64, _b: usize, _r: usize) -> Result<()> {
        anyhow::bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        Ok(())
    }
    fn send_to(&self, _p: u64, _b: usize, _r: usize, _s: u64) -> Result<()> {
        anyhow::bail!("unexpected send")
    }
    fn recv_from(&self, _p: u64, _b: usize, _r: usize, _s: u64) -> Result<()> {
        anyhow::bail!("unexpected receive")
    }
}
#[test]
fn tp2_dispatch_owns_whole_heads_and_reduces_each_key_and_value_once() {
    let gpu = MockGpuBackend::new();
    let (c, store) = fixture(&gpu, 1, 2, true);
    let before = gpu.alloc_count();
    {
        let block = Hy4Engram::load(&c, &store, 1, &gpu).unwrap();
        assert_eq!(block.heads, 8..16);
        assert_eq!(block.rows, 56..112);
        assert_eq!(block.layout.map.len(), c.vocab_size);
        let mut scratch = block.scratch().unwrap();
        let hidden = gpu.alloc(c.hidden_size * 4 * 2).unwrap();
        let comm = Comm {
            rank: 1,
            calls: AtomicUsize::new(0),
            bytes: 3072 * 2,
        };
        assert!(
            block
                .forward(hidden, &[], 1, &mut scratch, None, 0)
                .is_err()
        );
        block
            .forward(hidden, &[10, 11], 12, &mut scratch, Some(&comm), 0)
            .unwrap();
        assert_eq!(comm.calls.load(Ordering::Relaxed), 5);
        assert_eq!(
            gpu.launches_snapshot()
                .iter()
                .map(|l| l.args)
                .collect::<Vec<_>>(),
            [12, 9, 9, 9, 9, 9, 9, 4, 4, 8]
        );
        gpu.free(hidden).unwrap();
    }
    assert_eq!(gpu.alloc_count(), before);
}
#[test]
fn missing_or_malformed_table_contract_is_rejected_without_derived_leaks() {
    let gpu = MockGpuBackend::new();
    let (c, store) = fixture(&gpu, 0, 1, true);
    let before = gpu.alloc_count();
    assert!(Hy4Engram::load(&c, &store, 0, &gpu).is_err());
    assert!(Hy4Engram::load(&c, &WeightStore::empty(), 1, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), before);
}

#[test]
fn batched_keys_respect_n_by_h_collective_capacity_and_reject_empty_batches() {
    let gpu = MockGpuBackend::new();
    let (c, store) = fixture(&gpu, 0, 2, true);
    let block = Hy4Engram::load(&c, &store, 1, &gpu).unwrap();
    let mut scratch = block.scratch().unwrap();
    let history = vec![1; 57];
    let hidden = gpu.alloc(history.len() * c.hidden_size * 4 * 2).unwrap();
    let comm = Comm {
        rank: 0,
        calls: AtomicUsize::new(0),
        bytes: history.len() * c.hidden_size * 2,
    };
    assert!(
        block
            .forward_batch(hidden, &[], 0, &mut scratch, Some(&comm), 0)
            .is_err()
    );
    assert!(
        block
            .forward_batch(hidden, &history, 58, &mut scratch, Some(&comm), 0)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
    block
        .forward_batch(hidden, &history, 0, &mut scratch, Some(&comm), 0)
        .unwrap();
    assert_eq!(comm.calls.load(Ordering::Relaxed), 5);
    let launches = gpu.launch_count();
    assert!(
        block
            .forward_batch(
                hidden,
                &vec![1; crate::layers::hy4::batch::PREFILL_ROWS + 1],
                0,
                &mut scratch,
                Some(&comm),
                0,
            )
            .is_err()
    );
    assert_eq!(gpu.launch_count(), launches);
    gpu.free(hidden).unwrap();
}

/// `engram_fp8_embedding=false` must load a BF16 table slice (2 bytes per
/// element, so the row byte window doubles), skip the weight_scale_inv
/// slice entirely, and route the per-row lookup launch to the bf16 twin —
/// the lookup launch drops the scales pointer arg (8 args vs the fp8 9).
#[test]
fn bf16_table_load_skips_scale_slice_and_routes_bf16_lookup() {
    let gpu = MockGpuBackend::new();
    let (c, store) = fixture(&gpu, 1, 2, false);
    let before = gpu.alloc_count();
    {
        let block = Hy4Engram::load(&c, &store, 1, &gpu).unwrap();
        assert!(!block.fp8_table);
        assert!(block.scales.is_null());
        assert_eq!(block.rows, 56..112);
        let base = store
            .get("model.layers.1.engram_layer.multi_head_embedding.embedding.weight")
            .unwrap()
            .ptr;
        assert_eq!(block.table.0, base.0 + 56 * 96 * 2);
        let mut scratch = block.scratch().unwrap();
        let hidden = gpu.alloc(c.hidden_size * 4 * 2).unwrap();
        let comm = Comm {
            rank: 1,
            calls: AtomicUsize::new(0),
            bytes: 3072 * 2,
        };
        block
            .forward(hidden, &[10, 11], 12, &mut scratch, Some(&comm), 0)
            .unwrap();
        assert_eq!(comm.calls.load(Ordering::Relaxed), 5);
        assert_eq!(
            gpu.launches_snapshot()
                .iter()
                .map(|l| l.args)
                .collect::<Vec<_>>(),
            [12, 8, 9, 9, 9, 9, 9, 4, 4, 8]
        );
        gpu.free(hidden).unwrap();
    }
    assert_eq!(gpu.alloc_count(), before);
}

/// A checkpoint spliced with fp8 tables while the config asks for bf16 must
/// be rejected by the BF16 table contract, without derived allocations.
#[test]
fn bf16_flag_rejects_fp8_table_contract_without_leaks() {
    let gpu = MockGpuBackend::new();
    let (mut c, store) = fixture(&gpu, 0, 1, true);
    let before = gpu.alloc_count();
    c.hy4.as_mut().unwrap().engram_fp8_embedding = false;
    assert!(Hy4Engram::load(&c, &store, 1, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), before);
}

/// A standalone shard for a deferred engram table: `0xAA` guard head, a
/// `[112, 96]` table of distinct bytes (`11048` guard bytes before its first
/// element), `0x55` tail. The load must read its rank window out of this
/// file — device-resident when NVMap is off, mmap'd when it is on.
fn defer_table(
    store: &mut WeightStore,
    tag: &str,
    fp8_table: bool,
) -> (std::path::PathBuf, Vec<u8>) {
    let dim = 96;
    let width = if fp8_table { 1 } else { 2 };
    let dir = std::env::temp_dir().join(format!(
        "atlas-hy4-engram-nvmap-{tag}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut bytes = vec![0xAAu8; 8192];
    let table: Vec<u8> = (0..112 * dim * width).map(|i| (i % 251) as u8).collect();
    bytes.extend_from_slice(&table);
    bytes.extend_from_slice(&[0x55u8; 16]);
    std::fs::write(dir.join("s.bin"), &bytes).unwrap();
    store.defer(
        "model.layers.1.engram_layer.multi_head_embedding.embedding.weight".into(),
        DeferredTensor {
            path: dir.join("s.bin"),
            offset: 8192,
            shape: vec![112, dim],
            dtype: if fp8_table {
                WeightDtype::FP8E4M3
            } else {
                WeightDtype::BF16
            },
        },
    );
    (dir, table)
}

/// The NVMap kernel base points into a process mapping, not a device
/// allocation: `bytes` read through the host pointer.
fn host_window(ptr: DevicePtr, bytes: usize) -> Vec<u8> {
    // SAFETY: `Hy4Engram::table_nvmap` holds a read-only mapping covering
    // [0, bytes) behind `ptr` for the block's lifetime (nvmap module).
    unsafe { std::slice::from_raw_parts(ptr.0 as *const u8, bytes) }.to_vec()
}

/// Deferred bf16 tables take the NVMe-mmap path by default: the mapped
/// slice must be the doubled byte window — base at `rows.start*dim*2`,
/// length `rows.len()*dim*2`, guard bytes excluded on both ends — and none
/// of it may back a device allocation (GB10 faults cold rows in from NVMe).
#[test]
fn nvmap_bf16_table_maps_doubled_byte_window_without_device_alloc() {
    let gpu = MockGpuBackend::new();
    let (c, mut store) = fixture(&gpu, 1, 2, false);
    let (dir, table) = defer_table(&mut store, "window", false);
    let dim = 96;
    let block = Hy4Engram::load(&c, &store, 1, &gpu).unwrap();
    assert!(block.scales.is_null());
    assert_eq!(block.rows, 56..112);
    assert!(gpu.read_alloc(block.table).is_none());
    assert_eq!(
        host_window(block.table, 56 * dim * 2),
        table[56 * dim * 2..].to_vec()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `ATLAS_HY4_ENGRAM_NVMAP=0` — forced here via the knob's test seam; the
/// env strings themselves are unit-tested in nvmap — must fall back to the
/// old device-resident bf16 slice: exactly one MORE device allocation than
/// the mmap path (the table copy), holding the same doubled byte window.
#[test]
fn nvmap_disabled_loads_resident_bf16_slice_with_one_extra_alloc() {
    let dim = 96;
    let resident = {
        let _knob = nvmap::test_override(false);
        let gpu = MockGpuBackend::new();
        let (c, mut store) = fixture(&gpu, 1, 2, false);
        let (dir, table) = defer_table(&mut store, "resident", false);
        let before = gpu.alloc_count();
        let block = Hy4Engram::load(&c, &store, 1, &gpu).unwrap();
        assert_eq!(
            gpu.read_alloc(block.table),
            Some(table[56 * dim * 2..].to_vec())
        );
        std::fs::remove_dir_all(&dir).unwrap();
        (gpu.alloc_count() - before, table)
    };
    let mapped = {
        let gpu = MockGpuBackend::new();
        let (c, mut store) = fixture(&gpu, 1, 2, false);
        let (dir, table) = defer_table(&mut store, "mapped", false);
        let before = gpu.alloc_count();
        let block = Hy4Engram::load(&c, &store, 1, &gpu).unwrap();
        assert!(gpu.read_alloc(block.table).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
        (gpu.alloc_count() - before, table)
    };
    assert_eq!(mapped.1, resident.1);
    assert_eq!(resident.0, mapped.0 + 1);
}

/// The knob, even forced on, never moves fp8 tables off the device-resident
/// path: an fp8 deferred slice still lands in a device allocation.
#[test]
fn nvmap_knob_never_applies_to_fp8_tables() {
    let gpu = MockGpuBackend::new();
    let (c, mut store) = fixture(&gpu, 1, 2, true);
    let (dir, table) = defer_table(&mut store, "fp8-gate", true);
    let _knob = nvmap::test_override(true);
    let dim = 96;
    let block = Hy4Engram::load(&c, &store, 1, &gpu).unwrap();
    assert_eq!(
        gpu.read_alloc(block.table),
        Some(table[56 * dim..].to_vec())
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Launch-shape parity between the two table formats: a bf16 engram must
/// produce exactly the fp8 per-row launch sequence (same kernels, grids and
/// blocks — the mock executes nothing, so this asserts routing/geometry,
/// not numerics) except the lookup arg list, which drops the scales pointer.
#[test]
fn bf16_table_forward_matches_fp8_launches_except_scale_arg() {
    fn run(fp8_table: bool) -> Vec<MockLaunch> {
        let gpu = MockGpuBackend::new();
        let (c, store) = fixture(&gpu, 0, 1, fp8_table);
        let block = Hy4Engram::load(&c, &store, 1, &gpu).unwrap();
        assert_eq!(block.fp8_table, fp8_table);
        assert_eq!(block.scales.is_null(), !fp8_table);
        let mut scratch = block.scratch().unwrap();
        let hidden = gpu.alloc(c.hidden_size * 4 * 2).unwrap();
        block
            .forward(hidden, &[10, 11], 12, &mut scratch, None, 0)
            .unwrap();
        gpu.free(hidden).unwrap();
        gpu.launches_snapshot()
    }
    let fp8 = run(true);
    let bf16 = run(false);
    assert_eq!(fp8.len(), 10);
    assert_eq!(fp8.len(), bf16.len());
    for (f, b) in fp8.iter().zip(bf16.iter()) {
        assert_eq!(f.grid, b.grid);
        assert_eq!(f.block, b.block);
    }
    let diff: Vec<usize> = (0..fp8.len())
        .filter(|&i| fp8[i].args != bf16[i].args)
        .collect();
    assert_eq!(diff, [1]);
    assert_eq!(fp8[1].args - bf16[1].args, 1);
}

/// Same parity for the batched prefill path: the two table formats launch
/// the same kernel sequence (the mock executes nothing, so this asserts
/// routing/geometry, not numerics), all collective shapes match, and only
/// the lookup_batch arg list differs — the scales pointer is absent.
#[test]
fn bf16_table_batch_matches_fp8_launches_except_scale_arg() {
    fn run(fp8_table: bool) -> (usize, Vec<MockLaunch>) {
        let gpu = MockGpuBackend::new();
        let (c, store) = fixture(&gpu, 0, 2, fp8_table);
        let block = Hy4Engram::load(&c, &store, 1, &gpu).unwrap();
        assert_eq!(block.fp8_table, fp8_table);
        let mut scratch = block.scratch().unwrap();
        let history = vec![1; 57];
        let hidden = gpu.alloc(history.len() * c.hidden_size * 4 * 2).unwrap();
        let comm = Comm {
            rank: 0,
            calls: AtomicUsize::new(0),
            bytes: history.len() * c.hidden_size * 2,
        };
        block
            .forward_batch(hidden, &history, 0, &mut scratch, Some(&comm), 0)
            .unwrap();
        gpu.free(hidden).unwrap();
        (comm.calls.load(Ordering::Relaxed), gpu.launches_snapshot())
    }
    let (fp8_calls, fp8) = run(true);
    let (bf16_calls, bf16) = run(false);
    assert_eq!(fp8_calls, 5);
    assert_eq!(fp8_calls, bf16_calls);
    assert_eq!(fp8.len(), bf16.len());
    for (f, b) in fp8.iter().zip(bf16.iter()) {
        assert_eq!(f.grid, b.grid);
        assert_eq!(f.block, b.block);
    }
    let diff: Vec<usize> = (0..fp8.len())
        .filter(|&i| fp8[i].args != bf16[i].args)
        .collect();
    assert_eq!(diff, [1]);
    assert_eq!(fp8[1].args - bf16[1].args, 1);
}
