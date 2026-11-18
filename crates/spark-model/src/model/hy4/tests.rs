// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::traits::Model;
use spark_runtime::{
    gpu::mock::MockGpuBackend,
    weights::{WeightDtype as D, WeightTensor},
};

fn config() -> ModelConfig {
    use atlas_core::config::{Hy4Attention as A, Hy4Mlp};
    let mut c = atlas_core::config::parse_config(include_str!(
        "../../../../atlas-core/src/config/fixtures/hy4/config.json"
    ))
    .unwrap();
    c.hidden_size = 32;
    c.intermediate_size = 64;
    c.num_hidden_layers = 1;
    c.vocab_size = 64;
    c.q_lora_rank = 16;
    c.num_attention_heads = 2;
    c.max_position_embeddings = 32;
    c.num_experts = 4;
    c.ep_world_size = 1;
    c.tp_world_size = 1;
    c.expert_tp = false;
    c.num_experts_per_tok = 2;
    c.moe_intermediate_size = 16;
    c.shared_expert_intermediate_size = 32;
    let h = c.hy4.as_mut().unwrap();
    h.layer_types = vec![A::SlidingAttention];
    h.swa_pattern = vec![1];
    h.mlp_layer_types = vec![Hy4Mlp::Dense];
    h.swa_num_attention_heads = 4;
    h.routed_expert_hidden_size = 16;
    h.engram_layer_ids.clear();
    h.engram_layer_head_vocab_sizes.clear();
    c
}
fn add(
    map: &mut HashMap<String, WeightTensor>,
    gpu: &MockGpuBackend,
    name: &str,
    shape: &[usize],
    dtype: D,
) {
    let bytes = shape.iter().product::<usize>() * dtype.byte_size();
    let ptr = gpu.alloc(bytes).unwrap();
    if shape.is_empty() {
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
fn store(gpu: &MockGpuBackend) -> WeightStore {
    let mut m = HashMap::new();
    for (name, shape) in [
        ("q_a_proj.weight", vec![16, 32]),
        ("q_a_layernorm.weight", vec![16]),
        ("q_b_proj.weight", vec![1024, 16]),
        ("kv_a_proj_with_mqa.weight", vec![576, 32]),
        ("kv_a_layernorm.weight", vec![512]),
        ("kv_b_proj.weight", vec![1792, 512]),
        ("linear_gate.weight", vec![1024, 32]),
        ("o_proj.weight", vec![32, 1024]),
        ("post_proj_norm.weight", vec![32]),
    ] {
        add(
            &mut m,
            gpu,
            &format!("model.layers.0.self_attn.{name}"),
            &shape,
            D::BF16,
        );
    }
    add(
        &mut m,
        gpu,
        "model.layers.0.self_attn.learnable_sink_param",
        &[4],
        D::FP32,
    );
    for site in ["hc_attn_layer", "hc_mlp_layer"] {
        for (name, shape) in [
            ("hc_fn", vec![8, 128]),
            ("hc_scale", vec![2]),
            ("hc_base", vec![8]),
        ] {
            add(
                &mut m,
                gpu,
                &format!("model.layers.0.{site}.hc_pre.{name}"),
                &shape,
                D::FP32,
            );
        }
    }
    for (name, n, k) in [
        ("gate_proj", 64, 32),
        ("up_proj", 64, 32),
        ("down_proj", 32, 64),
    ] {
        for (suffix, shape, dtype) in [
            ("weight", vec![n, k / 2], D::UInt8),
            ("weight_scale", vec![n, k / 16], D::FP8E4M3),
            ("input_scale", vec![], D::FP32),
            ("weight_scale_2", vec![], D::FP32),
        ] {
            add(
                &mut m,
                gpu,
                &format!("model.layers.0.mlp.{name}.{suffix}"),
                &shape,
                dtype,
            );
        }
    }
    for name in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.post_fc2_norm.weight",
    ] {
        add(
            &mut m,
            gpu,
            &format!("model.layers.0.{name}"),
            &[32],
            D::BF16,
        );
    }
    for (name, shape, dtype) in [
        ("model.embed_tokens.weight", vec![64, 32], D::BF16),
        ("model.norm.weight", vec![32], D::BF16),
        ("lm_head.weight", vec![64, 32], D::BF16),
        ("model.hc_head.hc_head_fn", vec![4, 128], D::FP32),
        ("model.hc_head.hc_head_scale", vec![1], D::FP32),
        ("model.hc_head.hc_head_base", vec![4], D::FP32),
    ] {
        add(&mut m, gpu, name, &shape, dtype);
    }
    WeightStore::from_map(m)
}
fn model() -> Hy4Model {
    let gpu = Box::new(MockGpuBackend::new());
    let weights = store(&gpu);
    Hy4Model::new(config(), weights, gpu, None, 8, 4, 2).unwrap()
}
#[test]
fn eager_dispatch_preserves_chunk_decode_and_slot_isolation() {
    let model = model();
    let stream = model.default_stream();
    let mut a = model.alloc_sequence().unwrap();
    let mut b = model.alloc_sequence().unwrap();
    assert!(model.alloc_sequence().is_err());
    assert!(
        model
            .prefill_chunk(&[1, 2], &mut a, 0, 1, false, stream)
            .unwrap()
            .is_null()
    );
    assert_eq!(
        model
            .prefill_chunk(&[1, 2], &mut a, 1, 1, true, stream)
            .unwrap(),
        model.logits_buffer_ptr()
    );
    model.prefill(&[9], &mut b, stream).unwrap();
    model
        .decode_batch(&[3, 10], &mut [&mut a, &mut b], stream)
        .unwrap();
    let r = model.borrow_dependent();
    let requests = r.requests.lock();
    assert_eq!(requests[&a.slot_idx].borrow_dependent().history, [1, 2, 3]);
    assert_eq!(requests[&b.slot_idx].borrow_dependent().history, [9, 10]);
    drop(requests);
    model.free_sequence(&mut a).unwrap();
    model.free_sequence(&mut a).unwrap();
    let replacement = model.alloc_sequence().unwrap();
    assert_eq!(
        model.borrow_dependent().requests.lock()[&replacement.slot_idx]
            .borrow_dependent()
            .history
            .len(),
        0
    );
}
#[test]
fn empty_invalid_and_discontinuous_inputs_do_not_advance_state() {
    let model = model();
    let mut seq = model.alloc_sequence().unwrap();
    let stream = model.default_stream();
    assert!(model.prefill(&[], &mut seq, stream).is_err());
    assert!(
        model
            .prefill_chunk(&[1], &mut seq, usize::MAX, 1, true, stream)
            .is_err()
    );
    assert!(model.decode(64, &mut seq, stream).is_err());
    assert!(
        model
            .prefill_chunk(&[1, 2], &mut seq, 1, 1, true, stream)
            .is_err()
    );
    assert_eq!(seq.seq_len, 0);
    model.prefill(&[1], &mut seq, stream).unwrap();
    assert!(model.decode_verify(&[], &mut seq, stream).is_err());
    assert_eq!(seq.seq_len, 1);
}
#[test]
fn bad_topology_and_missing_checkpoint_weights_fail_before_forward() {
    let mut c = config();
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    assert!(
        Hy4Model::new(
            c,
            WeightStore::empty(),
            Box::new(MockGpuBackend::new()),
            None,
            8,
            4,
            1
        )
        .is_err()
    );
    assert!(
        Hy4Model::new(
            config(),
            WeightStore::empty(),
            Box::new(MockGpuBackend::new()),
            None,
            8,
            4,
            1
        )
        .is_err()
    );
    assert!(
        Hy4Model::new(
            config(),
            WeightStore::empty(),
            Box::new(MockGpuBackend::new()),
            None,
            0,
            4,
            1
        )
        .is_err()
    );
}
#[test]
fn teardown_drops_request_and_weight_owners_once() {
    let mut model = model();
    model.alloc_sequence().unwrap();
    model.teardown().unwrap();
    assert!(model.cell.is_none());
    model.teardown().unwrap();
}
struct StopComm(Mutex<Vec<usize>>, usize);
impl CommBackend for StopComm {
    fn rank(&self) -> usize {
        self.1
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_reduce(&self, _p: u64, _b: usize) -> Result<()> {
        anyhow::bail!("unexpected reduction")
    }
    fn all_gather(&self, _s: u64, _r: u64, _b: usize) -> Result<()> {
        anyhow::bail!("unexpected gather")
    }
    fn reduce_scatter(&self, _s: u64, _r: u64, _b: usize) -> Result<()> {
        anyhow::bail!("unexpected scatter")
    }
    fn broadcast(&self, _p: u64, b: usize, r: usize) -> Result<()> {
        assert_eq!(r, 0);
        self.0.lock().push(b);
        Ok(())
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
fn scheduler_stop_reaches_worker_but_legacy_prefill_headers_do_not() {
    let gpu = MockGpuBackend::new();
    let weights = store(&gpu);
    let mut c = config();
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    let comm = Arc::new(StopComm(Mutex::new(Vec::new()), 0));
    let model = Hy4Model::new(c, weights, Box::new(gpu), Some(comm.clone()), 32, 2, 2).unwrap();
    model.ep_broadcast_cmd_for_seq(0, 0xFFFFFFF0).unwrap();
    model.ep_broadcast_cmd(4).unwrap();
    assert!(comm.0.lock().is_empty());
    model
        .ep_broadcast_cmd_for_seq(0, super::wire::STOP)
        .unwrap();
    assert_eq!(*comm.0.lock(), [4, 12]);
}

#[test]
fn worker_bootstrap_slots_do_not_send_commands_and_accept_head_allocation() {
    let gpu = MockGpuBackend::new();
    let weights = store(&gpu);
    let mut c = config();
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    c.tp_rank = 1;
    c.ep_rank = 1;
    let comm = Arc::new(StopComm(Mutex::new(Vec::new()), 1));
    let model = Hy4Model::new(c, weights, Box::new(gpu), Some(comm.clone()), 32, 2, 2).unwrap();
    let mut a = model.alloc_sequence().unwrap();
    let mut b = model.alloc_sequence().unwrap();
    assert!(model.alloc_sequence().is_err());
    assert_eq!([a.slot_idx, b.slot_idx], [0, 1]);
    assert!(comm.0.lock().is_empty());
    let write_alloc = || {
        model
            .borrow_owner()
            .gpu
            .copy_h2d(
                &[super::wire::ALLOC, 0, 0, 0]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
                model.borrow_dependent().command,
            )
            .unwrap();
    };
    write_alloc();
    assert!(model.ep_worker_step(&mut []).unwrap());
    assert_eq!(model.borrow_dependent().requests.lock().len(), 2);
    model
        .borrow_dependent()
        .requests
        .lock()
        .get_mut(&0)
        .unwrap()
        .with_dependent_mut(|_, data| data.history.push(1));
    write_alloc();
    assert!(model.ep_worker_step(&mut []).is_err());
    let before = comm.0.lock().len();
    model.free_sequence(&mut a).unwrap();
    model.free_sequence(&mut b).unwrap();
    model.free_sequence(&mut b).unwrap();
    assert!(model.borrow_dependent().requests.lock().is_empty());
    assert_eq!(comm.0.lock().len(), before);
}
