// SPDX-License-Identifier: AGPL-3.0-only

use super::{Hy4FfnScratch, ffn::Hy4Ffn, hc::IndependentHc, weights::descriptors};
use anyhow::Result;
use atlas_core::config::{Hy4Mlp, ModelConfig, parse_config};
use spark_comm::CommBackend;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend},
    weights::{WeightDtype, WeightStore, WeightTensor},
};
use std::{collections::HashMap, sync::Mutex};

fn config(rank: usize, world: usize) -> ModelConfig {
    let mut c = parse_config(include_str!(
        "../../../../atlas-core/src/config/fixtures/hy4/config.json"
    ))
    .unwrap();
    c.hidden_size = 32;
    c.intermediate_size = 64;
    c.moe_intermediate_size = 16;
    c.shared_expert_intermediate_size = 32;
    c.num_experts = 4;
    c.num_experts_per_tok = 2;
    c.ep_rank = rank;
    c.tp_rank = rank;
    c.ep_world_size = world;
    c.tp_world_size = world;
    let hy = c.hy4.as_mut().unwrap();
    hy.routed_expert_hidden_size = 16;
    hy.mlp_layer_types = vec![Hy4Mlp::Dense, Hy4Mlp::Sparse];
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
    if dtype == WeightDtype::FP32 && shape.is_empty() {
        gpu.copy_h2d(&1f32.to_le_bytes(), ptr).unwrap();
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
fn mlp(map: &mut HashMap<String, WeightTensor>, gpu: &MockGpuBackend, p: &str, h: usize, i: usize) {
    for (key, n, k) in [("gate_proj", i, h), ("up_proj", i, h), ("down_proj", h, i)] {
        for (suffix, shape, dtype) in [
            ("weight", vec![n, k / 2], WeightDtype::UInt8),
            ("weight_scale", vec![n, k / 16], WeightDtype::FP8E4M3),
            ("input_scale", vec![], WeightDtype::FP32),
            ("weight_scale_2", vec![], WeightDtype::FP32),
        ] {
            add(map, gpu, &format!("{p}.{key}.{suffix}"), &shape, dtype);
        }
    }
}
fn store(gpu: &MockGpuBackend, c: &ModelConfig, dense: bool) -> WeightStore {
    let mut map = HashMap::new();
    let p = if dense {
        "model.layers.0.mlp"
    } else {
        "model.layers.1.mlp"
    };
    if dense {
        mlp(&mut map, gpu, p, 32, 64);
    } else {
        add(
            &mut map,
            gpu,
            &format!("{p}.gate.weight"),
            &[4, 32],
            WeightDtype::BF16,
        );
        add(
            &mut map,
            gpu,
            &format!("{p}.gate.e_score_correction_bias"),
            &[4],
            WeightDtype::FP32,
        );
        add(
            &mut map,
            gpu,
            &format!("{p}.routed_expert_down_proj.weight"),
            &[16, 32],
            WeightDtype::BF16,
        );
        add(
            &mut map,
            gpu,
            &format!("{p}.routed_expert_up_proj.weight"),
            &[32, 16],
            WeightDtype::BF16,
        );
        for e in 0..4 {
            if c.is_local_expert(e) {
                mlp(&mut map, gpu, &format!("{p}.experts.{e}"), 16, 16);
            }
        }
        mlp(&mut map, gpu, &format!("{p}.shared_experts"), 32, 32);
    }
    add(
        &mut map,
        gpu,
        &format!(
            "{p}.{}.weight",
            if dense {
                "post_fc2_norm"
            } else {
                "combine_output_norm"
            }
        ),
        &[32],
        WeightDtype::BF16,
    );
    WeightStore::from_map(map)
}

struct Comm<'a> {
    gpu: &'a MockGpuBackend,
    rank: usize,
    calls: Mutex<Vec<(usize, u64, usize)>>,
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
fn ep2_reads_only_local_experts_reduces_routed_before_shared_and_norm() {
    for rank in 0..2 {
        let gpu = MockGpuBackend::new();
        let c = config(rank, 2);
        let store = store(&gpu, &c, false);
        let initial = gpu.alloc_count();
        let mut block = Hy4Ffn::load(&store, 1, &c, &gpu).unwrap();
        assert_eq!(gpu.alloc_count(), initial + 1);
        let mut bytes = vec![0; 5 * 72];
        gpu.copy_d2h(block.table, &mut bytes).unwrap();
        for e in 0..4 {
            let ptr = u64::from_le_bytes(bytes[e * 72..e * 72 + 8].try_into().unwrap());
            assert_eq!(ptr != 0, c.is_local_expert(e));
        }
        let mut scratch = Hy4FfnScratch::new(&block, 3).unwrap();
        let x = gpu.alloc(3 * 32 * 2).unwrap();
        let out = gpu.alloc(3 * 32 * 2).unwrap();
        let comm = Comm {
            gpu: &gpu,
            rank,
            calls: Mutex::new(vec![]),
        };
        block
            .forward(x, out, 3, &mut scratch, Some(&comm), 17)
            .unwrap();
        assert_eq!(*comm.calls.lock().unwrap(), [(3 * 32 * 2, 17, 6)]);
        assert_eq!(gpu.launch_count(), 9);
        let launches = gpu.launches_snapshot();
        assert_eq!(launches[0].grid, [1, 3, 1]);
        assert_eq!(launches[0].block, [256, 1, 1]);
        assert_eq!(launches[1].block, [512, 1, 1]);
        assert_eq!(launches[3].grid, [2, 3, 2]);
        assert_eq!(launches[3].block, [256, 1, 1]);
        assert_eq!(launches[4].grid, [16, 3, 1]);
        assert_eq!(launches[4].block, [256, 1, 1]);
        assert_eq!(
            gpu.launches_snapshot()
                .iter()
                .map(|l| l.args)
                .collect::<Vec<_>>(),
            [7, 8, 7, 9, 9, 7, 9, 9, 6]
        );
        assert!(block.forward(x, out, 3, &mut scratch, None, 17).is_err());
        assert_eq!(gpu.launch_count(), 9);
        scratch.release().unwrap();
        scratch.release().unwrap();
        block.release().unwrap();
        block.release().unwrap();
        assert_eq!(gpu.invalid_free_count(), 0);
    }
}

#[test]
fn dense_zero_uses_unclamped_full_width_mlp_then_post_fc_norm_without_collective() {
    let gpu = MockGpuBackend::new();
    let c = config(0, 2);
    let store = store(&gpu, &c, true);
    let block = Hy4Ffn::load(&store, 0, &c, &gpu).unwrap();
    let mut scratch = Hy4FfnScratch::new(&block, 1).unwrap();
    let comm = Comm {
        gpu: &gpu,
        rank: 0,
        calls: Mutex::new(vec![]),
    };
    block
        .forward(DevicePtr(1), DevicePtr(2), 1, &mut scratch, Some(&comm), 0)
        .unwrap();
    assert!(comm.calls.lock().unwrap().is_empty());
    assert_eq!(gpu.launch_count(), 3);
    let launches = gpu.launches_snapshot();
    assert_eq!(launches[0].grid, [8, 1, 1]);
    assert_eq!(launches[0].block, [256, 1, 1]);
    assert_eq!(launches[1].grid, [32, 1, 1]);
    assert_eq!(launches[1].block, [256, 1, 1]);
}

#[test]
fn empty_rows_are_noop_and_malformed_buffers_topologies_and_weights_fail() {
    let gpu = MockGpuBackend::new();
    let mut c = config(0, 1);
    let store = store(&gpu, &c, false);
    let block = Hy4Ffn::load(&store, 1, &c, &gpu).unwrap();
    let mut scratch = Hy4FfnScratch::new(&block, 0).unwrap();
    block
        .forward(DevicePtr::NULL, DevicePtr::NULL, 0, &mut scratch, None, 0)
        .unwrap();
    assert_eq!(gpu.launch_count(), 0);
    assert!(
        block
            .forward(DevicePtr(1), DevicePtr(2), 1, &mut scratch, None, 0)
            .is_err()
    );
    assert!(Hy4FfnScratch::new(&block, 65536).is_err());
    let mut scratch = Hy4FfnScratch::new(&block, 1).unwrap();
    assert!(
        block
            .forward(DevicePtr(1), DevicePtr(1), 1, &mut scratch, None, 0)
            .is_err()
    );
    assert!(
        block
            .forward(DevicePtr::NULL, DevicePtr(2), 1, &mut scratch, None, 0)
            .is_err()
    );
    c.expert_tp = true;
    assert!(Hy4Ffn::load(&store, 1, &c, &gpu).is_err());
    c.expert_tp = false;
    c.tp_world_size = 2;
    assert!(Hy4Ffn::load(&store, 1, &c, &gpu).is_err());
    assert!(Hy4Ffn::load(&WeightStore::empty(), 1, &config(0, 1), &gpu).is_err());
    assert!(Hy4Ffn::load(&store, 9, &config(0, 1), &gpu).is_err());
}

#[test]
fn hc_loads_fp32_gates_bf16_highway_and_validates_shapes() {
    let gpu = MockGpuBackend::new();
    let c = config(0, 1);
    let mut map = HashMap::new();
    for (name, shape) in [
        ("hc_fn", vec![8, 128]),
        ("hc_scale", vec![2]),
        ("hc_base", vec![8]),
    ] {
        add(
            &mut map,
            &gpu,
            &format!("site.{name}"),
            &shape,
            WeightDtype::FP32,
        );
    }
    let store = WeightStore::from_map(map);
    let hc = IndependentHc::load(&store, "site", false, &c, &gpu).unwrap();
    hc.collapse(
        &gpu,
        DevicePtr::NULL,
        DevicePtr::NULL,
        DevicePtr::NULL,
        0,
        0,
    )
    .unwrap();
    assert_eq!(gpu.launch_count(), 0);
    assert!(
        hc.collapse(&gpu, DevicePtr::NULL, DevicePtr(1), DevicePtr(2), 1, 0)
            .is_err()
    );
    hc.collapse(&gpu, DevicePtr(1), DevicePtr(2), DevicePtr(3), 2, 0)
        .unwrap();
    hc.scatter(
        &gpu,
        [DevicePtr(1), DevicePtr(2), DevicePtr(3), DevicePtr(2)],
        2,
        0,
    )
    .unwrap();
    assert_eq!(gpu.launch_count(), 2);
    assert!(IndependentHc::load(&store, "site", true, &c, &gpu).is_err());
}

#[test]
fn descriptor_abi_and_scratch_cleanup_do_not_own_checkpoint_tensors() {
    let gpu = MockGpuBackend::new();
    let c = config(0, 1);
    let store = store(&gpu, &c, false);
    let count = gpu.alloc_count();
    {
        let block = Hy4Ffn::load(&store, 1, &c, &gpu).unwrap();
        let _scratch = Hy4FfnScratch::new(&block, 2).unwrap();
        assert_eq!(gpu.alloc_count(), count + 2);
    }
    assert_eq!(gpu.alloc_count(), count);
    assert_eq!(
        descriptors(&[crate::weight_map::ExpertWeight::null()]),
        vec![0; 72]
    );
}

#[test]
fn head_promotes_bf16_parameters_and_releases_only_its_copies() {
    let gpu = MockGpuBackend::new();
    let c = config(0, 1);
    let mut map = HashMap::new();
    for (name, shape) in [
        ("hc_head_fn", vec![4, 128]),
        ("hc_head_scale", vec![1]),
        ("hc_head_base", vec![4]),
    ] {
        add(
            &mut map,
            &gpu,
            &format!("head.{name}"),
            &shape,
            WeightDtype::BF16,
        );
    }
    let store = WeightStore::from_map(map);
    let initial = gpu.alloc_count();
    {
        let head = IndependentHc::load(&store, "head", true, &c, &gpu).unwrap();
        assert_eq!(gpu.alloc_count(), initial + 3);
        head.collapse(&gpu, DevicePtr(1), DevicePtr(2), DevicePtr::NULL, 1, 0)
            .unwrap();
        assert_eq!(gpu.launches_snapshot()[0].args, 8);
        assert!(head.scatter(&gpu, [DevicePtr(1); 4], 1, 0).is_err());
    }
    assert_eq!(gpu.alloc_count(), initial);
}
