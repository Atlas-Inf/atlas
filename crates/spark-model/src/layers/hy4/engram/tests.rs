// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::{gpu::mock::MockGpuBackend, weights::WeightTensor};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};
fn fixture(gpu: &MockGpuBackend, rank: usize, world: usize) -> (ModelConfig, WeightStore) {
    let mut c = atlas_core::config::parse_config(include_str!(
        "../../../../../atlas-core/src/config/fixtures/hy4/config.json"
    ))
    .unwrap();
    c.tp_rank = rank;
    c.tp_world_size = world;
    let sizes = vec![vec![7; 8]; 2];
    c.hy4
        .as_mut()
        .unwrap()
        .engram_layer_head_vocab_sizes
        .insert(2, sizes);
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
        WeightDtype::FP8E4M3,
    );
    add(
        "multi_head_embedding.embedding.weight_scale_inv",
        vec![112],
        WeightDtype::BF16,
    );
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
    let (c, store) = fixture(&gpu, 1, 2);
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
    let (c, store) = fixture(&gpu, 0, 1);
    let before = gpu.alloc_count();
    assert!(Hy4Engram::load(&c, &store, 0, &gpu).is_err());
    assert!(Hy4Engram::load(&c, &WeightStore::empty(), 1, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), before);
}

#[test]
fn batched_keys_respect_n_by_h_collective_capacity_and_reject_empty_batches() {
    let gpu = MockGpuBackend::new();
    let (c, store) = fixture(&gpu, 0, 2);
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
