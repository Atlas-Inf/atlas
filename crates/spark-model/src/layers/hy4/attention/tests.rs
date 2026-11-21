// SPDX-License-Identifier: AGPL-3.0-only

use super::{AttentionCache, Hy4Attention};
use anyhow::Result;
use atlas_core::config::{Hy4Attention as Kind, ModelConfig, parse_config};
use spark_comm::CommBackend;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend},
    weights::{WeightDtype, WeightStore, WeightTensor},
};
use std::{collections::HashMap, sync::Mutex};

#[path = "snapshot_tests.rs"]
mod snapshots;

#[path = "prefix_tests.rs"]
mod prefix;

fn config(rank: usize, world: usize) -> ModelConfig {
    let mut c = parse_config(include_str!(
        "../../../../../atlas-core/src/config/fixtures/hy4/config.json"
    ))
    .unwrap();
    c.hidden_size = 16;
    c.q_lora_rank = 16;
    c.num_attention_heads = 2 / world;
    c.max_position_embeddings = 2048;
    c.tp_rank = rank;
    c.tp_world_size = world;
    let h = c.hy4.as_mut().unwrap();
    h.swa_num_attention_heads = 4;
    h.compressed_indexer_num_attention_heads = 2;
    c
}
fn add(
    map: &mut HashMap<String, WeightTensor>,
    gpu: &MockGpuBackend,
    name: &str,
    shape: &[usize],
    dtype: WeightDtype,
) {
    let size = shape.iter().product::<usize>() * dtype.byte_size();
    let ptr = gpu.alloc(size).unwrap();
    if [
        "q_a_proj.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_b_proj.weight",
        "linear_gate.weight",
        "o_proj.weight",
    ]
    .iter()
    .any(|s| name.ends_with(s))
    {
        let bytes: Vec<_> = (0..size / 2)
            .flat_map(|n| ((n % 257) as u16).to_le_bytes())
            .collect();
        gpu.copy_h2d(&bytes, ptr).unwrap();
    }
    map.insert(
        name.into(),
        WeightTensor {
            ptr,
            shape: shape.to_vec(),
            dtype,
        },
    );
}
fn store(gpu: &MockGpuBackend, c: &ModelConfig, layer: usize) -> WeightStore {
    let mut m = HashMap::new();
    let hy = c.hy4.as_ref().unwrap();
    let swa = hy.layer_types[layer] == Kind::SlidingAttention;
    let heads = if swa {
        hy.swa_num_attention_heads
    } else {
        c.num_attention_heads * c.tp_world_size
    };
    let p = format!("model.layers.{layer}.self_attn");
    for (name, shape) in [
        ("q_a_proj.weight", vec![16, 16]),
        ("q_a_layernorm.weight", vec![16]),
        ("q_b_proj.weight", vec![heads * 256, 16]),
        ("kv_a_proj_with_mqa.weight", vec![576, 16]),
        ("kv_a_layernorm.weight", vec![512]),
        ("kv_b_proj.weight", vec![heads * 448, 512]),
        ("linear_gate.weight", vec![heads * 256, 16]),
        ("o_proj.weight", vec![16, heads * 256]),
        ("post_proj_norm.weight", vec![16]),
    ] {
        add(
            &mut m,
            gpu,
            &format!("{p}.{name}"),
            &shape,
            WeightDtype::BF16,
        );
    }
    add(
        &mut m,
        gpu,
        &format!("{p}.learnable_sink_param"),
        &[heads],
        WeightDtype::FP32,
    );
    if !swa {
        for (name, shape, dtype) in [
            ("query_projection.weight", vec![256, 16], WeightDtype::BF16),
            (
                "token_key_projection.weight",
                vec![128, 16],
                WeightDtype::BF16,
            ),
            ("token_key_norm.weight", vec![128], WeightDtype::BF16),
            (
                "head_weight_projection.weight",
                vec![2, 16],
                WeightDtype::BF16,
            ),
            (
                "key_value_projection.weight",
                vec![128, 128],
                WeightDtype::FP32,
            ),
            (
                "key_gate_projection.weight",
                vec![128, 128],
                WeightDtype::FP32,
            ),
            ("key_norm.weight", vec![128], WeightDtype::BF16),
            ("intra_group_ape", vec![4, 128], WeightDtype::FP32),
        ] {
            add(
                &mut m,
                gpu,
                &format!("{p}.compressed_indexer.{name}"),
                &shape,
                dtype,
            );
        }
    }
    WeightStore::from_map(m)
}
struct Comm<'a> {
    gpu: &'a MockGpuBackend,
    rank: usize,
    calls: Mutex<Vec<(usize, u64, usize)>>,
    exchanges: Mutex<Vec<(usize, u64, usize)>>,
}
impl CommBackend for Comm<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("blocking reduce")
    }
    fn all_reduce_async(&self, _: u64, bytes: usize, stream: u64) -> Result<()> {
        self.calls
            .lock()
            .unwrap()
            .push((bytes, stream, self.gpu.launch_count()));
        Ok(())
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("gather")
    }
    fn exchange_async(&self, _: u64, _: u64, bytes: usize, add: bool, stream: u64) -> Result<bool> {
        assert!(!add);
        self.exchanges
            .lock()
            .unwrap()
            .push((bytes, stream, self.gpu.launch_count()));
        Ok(true)
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        anyhow::bail!("broadcast")
    }
    fn barrier(&self) -> Result<()> {
        anyhow::bail!("barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("recv")
    }
}
#[test]
fn tp2_shards_q_gate_sink_output_and_transposes_only_local_kv_heads() {
    for rank in 0..2 {
        let gpu = MockGpuBackend::new();
        let c = config(rank, 2);
        let store = store(&gpu, &c, 0);
        let initial = gpu.alloc_count();
        {
            let a = Hy4Attention::load(&store, 0, &c, &gpu).unwrap();
            assert_eq!(a.heads, 1);
            assert_ne!(
                a.qb,
                store
                    .get("model.layers.0.self_attn.q_b_proj.weight")
                    .unwrap()
                    .ptr
                    .offset(rank * 256 * 16 * 2)
            );
            let mut qb = vec![0; 256 * 16 * 2];
            gpu.copy_d2h(a.qb, &mut qb).unwrap();
            let at = (7 * 16 + 11) * 2;
            assert_eq!(
                u16::from_le_bytes(qb[at..at + 2].try_into().unwrap()) as usize,
                (rank * 256 * 16 + 7 * 16 + 11) % 257
            );
            let mut qa = vec![0; 8 * 16 * 2];
            gpu.copy_d2h(a.qa, &mut qa).unwrap();
            assert_eq!(
                u16::from_le_bytes(qa[..2].try_into().unwrap()) as usize,
                rank * 8 * 16 % 257
            );
            assert_eq!(store.reclaimed_count(), 6);
            assert!(a.scales.is_empty());
            assert_eq!(
                a.sink,
                store
                    .get("model.layers.0.self_attn.learnable_sink_param")
                    .unwrap()
                    .ptr
                    .offset(rank * 4)
            );
            let mut key = vec![0; 512 * 192 * 2];
            gpu.copy_d2h(a.wk, &mut key).unwrap();
            let at = (13 * 192 + 5) * 2;
            assert_eq!(
                u16::from_le_bytes(key[at..at + 2].try_into().unwrap()) as usize,
                (rank * 448 * 512 + 5 * 512 + 13) % 257
            );
            let mut o = vec![0; 16 * 256 * 2];
            gpu.copy_d2h(a.output, &mut o).unwrap();
            let at = (7 * 256 + 11) * 2;
            assert_eq!(
                u16::from_le_bytes(o[at..at + 2].try_into().unwrap()) as usize,
                (7 * 512 + rank * 256 + 11) % 257
            );
        }
        assert_eq!(gpu.alloc_count(), initial - 6);
        assert_eq!(gpu.invalid_free_count(), 0);
    }
}
#[test]
fn swa_has_no_pool_reduces_before_post_norm_and_includes_sink_for_prefill_and_decode() {
    let gpu = MockGpuBackend::new();
    let c = config(0, 2);
    let store = store(&gpu, &c, 2);
    let a = Hy4Attention::load(&store, 2, &c, &gpu).unwrap();
    assert_eq!(a.heads, 2);
    assert!(a.index.is_none());
    assert_eq!(a.rope.rope_theta, 10000.);
    let mut cache = AttentionCache::new(&a, 1024, 64).unwrap();
    assert_eq!(cache.layout.table.len(), 9);
    assert_eq!(cache.pool_bytes, 0);
    let comm = Comm {
        gpu: &gpu,
        rank: 0,
        calls: Mutex::new(vec![]),
        exchanges: Mutex::new(vec![]),
    };
    cache
        .forward(
            DevicePtr(1),
            DevicePtr(2),
            &[[0; 3], [1; 3]],
            Some(&comm),
            17,
        )
        .unwrap();
    assert_eq!(gpu.launch_count(), 16);
    assert_eq!(*comm.calls.lock().unwrap(), [(64, 17, 15)]);
    assert_eq!(*comm.exchanges.lock().unwrap(), [(1184, 17, 2)]);
    let args: Vec<_> = gpu.launches_snapshot().iter().map(|l| l.args).collect();
    assert_eq!(
        args,
        [15, 15, 5, 6, 15, 15, 15, 9, 22, 22, 13, 5, 15, 3, 15, 6]
    );
    assert_eq!(cache.committed_len(), 2);
    assert!(
        cache
            .forward(DevicePtr(1), DevicePtr(2), &[[2; 3]], None, 17)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 16);
}
#[test]
fn dsa_partial_group_survives_chunk_and_prefix_fork_and_uses_all_score_stages() {
    let gpu = MockGpuBackend::new();
    let c = config(0, 1);
    let store = store(&gpu, &c, 0);
    let a = Hy4Attention::load(&store, 0, &c, &gpu).unwrap();
    assert_eq!(a.rope.rope_theta, 10000000.);
    let mut cache = AttentionCache::new(&a, 8, 64).unwrap();
    cache
        .forward(
            DevicePtr(1),
            DevicePtr(2),
            &[[0; 3], [1; 3], [2; 3]],
            None,
            9,
        )
        .unwrap();
    assert_eq!(gpu.launch_count(), 26);
    let copies = gpu.d2d_count();
    let mut fork = cache.fork(9).unwrap();
    assert_eq!(fork.committed_len(), 3);
    assert_eq!(gpu.d2d_count(), copies + 3);
    assert_ne!(fork.latent, cache.latent);
    assert_ne!(fork.pool, cache.pool);
    assert_ne!(fork.tail, cache.tail);
    fork.forward(DevicePtr(1), DevicePtr(2), &[[3; 3], [4; 3]], None, 9)
        .unwrap();
    assert_eq!(fork.committed_len(), 5);
    assert_eq!(cache.committed_len(), 3);
    assert_eq!(gpu.launch_count(), 26 + 28);
    assert_eq!(fork.first_group_position, Some([4; 3]));
    assert!(fork.fork(0).is_err());
    cache
        .forward(DevicePtr(1), DevicePtr(2), &[[3; 3]], None, 9)
        .unwrap();
    assert_eq!(cache.committed_len(), 4);
}
#[test]
fn fp8_is_explicit_and_reclaims_sources_and_temporary_bf16_matrices() {
    let gpu = MockGpuBackend::new();
    let mut c = config(1, 2);
    assert!(!c.hy4.as_ref().unwrap().attention_projection_fp8);
    c.hy4.as_mut().unwrap().attention_projection_fp8 = true;
    let store = store(&gpu, &c, 0);
    let initial = gpu.alloc_count();
    {
        let a = Hy4Attention::load(&store, 0, &c, &gpu).unwrap();
        assert_eq!(a.scales.len(), 7);
        assert_eq!(gpu.alloc_count(), initial - 6 + 14);
        assert_eq!(store.reclaimed_count(), 6);
        assert_eq!(gpu.launch_count(), 7);
        let mut cache = AttentionCache::new(&a, 8, 4).unwrap();
        cache
            .forward(DevicePtr::NULL, DevicePtr::NULL, &[], None, 0)
            .unwrap();
        assert_eq!(gpu.launch_count(), 7);
    }
    assert_eq!(gpu.alloc_count(), initial - 6);
    assert_eq!(gpu.invalid_free_count(), 0);
    c.q_lora_rank = 17;
    assert!(Hy4Attention::load(&store, 0, &c, &gpu).is_err());
}
#[test]
fn empty_and_malformed_batches_do_not_advance_or_launch_and_oom_cleans_partial_allocations() {
    let gpu = MockGpuBackend::new();
    let c = config(0, 1);
    let store = store(&gpu, &c, 0);
    let a = Hy4Attention::load(&store, 0, &c, &gpu).unwrap();
    let count = gpu.alloc_count();
    gpu.fail_alloc_after(6);
    assert!(AttentionCache::new(&a, 8, 64).is_err());
    assert_eq!(gpu.alloc_count(), count);
    let mut cache = AttentionCache::new(&a, 8, 64).unwrap();
    cache
        .forward(DevicePtr::NULL, DevicePtr::NULL, &[], None, 0)
        .unwrap();
    for positions in [vec![[-1; 3]], vec![[2048; 3]], vec![[0; 3]; 9]] {
        assert!(
            cache
                .forward(DevicePtr(1), DevicePtr(2), &positions, None, 0)
                .is_err()
        );
    }
    assert!(
        cache
            .forward(DevicePtr(1), DevicePtr(1), &[[0; 3]], None, 0)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
    assert_eq!(cache.committed_len(), 0);
    let allocated = gpu.alloc_count();
    gpu.fail_next_free();
    assert!(cache.release().is_err());
    assert_eq!(gpu.alloc_count(), allocated);
    cache.release().unwrap();
    cache.release().unwrap();
    assert!(
        cache
            .forward(DevicePtr(1), DevicePtr(2), &[[0; 3]], None, 0)
            .is_err()
    );
    assert_eq!(gpu.alloc_count(), count);
    assert!(AttentionCache::new(&a, 0, 64).is_err());
    assert!(AttentionCache::new(&a, 8, 0).is_err());
    assert!(Hy4Attention::load(&WeightStore::empty(), 0, &c, &gpu).is_err());
}
