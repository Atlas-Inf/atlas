// SPDX-License-Identifier: AGPL-3.0-only
//! Real allocations/builders/transaction; no forged Ready or ownership receipts.
use super::recording::Gpu;
use super::*;
use crate::layers::moe::MoeLayer;
use crate::weight_loader::glm5::retirement::RetirementLog;
use crate::weight_map::{DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype as D, WeightStore, WeightTensor};
use std::collections::HashMap;
#[path = "gate_up_btile_publication_fault_tests.rs"]
mod publication_fault_tests;

fn tensor(
    gpu: &Gpu,
    map: &mut HashMap<String, WeightTensor>,
    name: String,
    shape: &[usize],
    dtype: D,
) -> DevicePtr {
    let bytes = shape.iter().product::<usize>() * dtype.byte_size();
    let ptr = gpu.alloc(bytes).unwrap();
    if dtype == D::FP32 && bytes == 4 {
        gpu.copy_h2d(&1.25f32.to_le_bytes(), ptr).unwrap();
    }
    map.insert(
        name,
        WeightTensor {
            ptr,
            shape: shape.into(),
            dtype,
        },
    );
    ptr
}
fn projection(
    gpu: &Gpu,
    map: &mut HashMap<String, WeightTensor>,
    prefix: String,
    n: usize,
    k: usize,
) -> QuantizedWeight {
    let weight = tensor(gpu, map, format!("{prefix}.weight"), &[n, k / 2], D::UInt8);
    let weight_scale = tensor(
        gpu,
        map,
        format!("{prefix}.weight_scale"),
        &[n, k / 16],
        D::FP8E4M3,
    );
    tensor(gpu, map, format!("{prefix}.weight_scale_2"), &[1], D::FP32);
    QuantizedWeight {
        weight,
        weight_scale,
        weight_scale_2: 1.25,
        ..QuantizedWeight::null()
    }
}
pub(super) fn setup(
    gpu: &Gpu,
    rank: usize,
) -> (WeightStore, atlas_core::config::ModelConfig, MoeLayer) {
    setup_with(gpu, rank, false)
}
/// [`setup`], or with `expert_tp` every expert local at the rank's slice of
/// the routed intermediate width.
pub(super) fn setup_with(
    gpu: &Gpu,
    rank: usize,
    expert_tp: bool,
) -> (WeightStore, atlas_core::config::ModelConfig, MoeLayer) {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.moe_intermediate_size = 2048;
    config.shared_expert_intermediate_size = 2048;
    config.num_experts = 288;
    config.num_experts_per_tok = 8;
    config.num_hidden_layers = 42;
    config.tp_world_size = 2;
    config.ep_world_size = 2;
    config.tp_rank = rank;
    config.ep_rank = rank;
    config.adapter_max_rank = 0;
    config.scoring_func = "sigmoid".into();
    config.expert_tp = expert_tp;
    let inter = config.routed_inter_local();
    let mut map = HashMap::new();
    let mut weights = MoeWeights::empty(288);
    let lp = config.layer_prefix(0);
    for (e, expert) in weights.experts.iter_mut().enumerate() {
        if !config.is_local_expert(e) {
            *expert = ExpertWeight::null();
            continue;
        }
        *expert = ExpertWeight {
            gate_proj: projection(
                gpu,
                &mut map,
                format!("{lp}.mlp.experts.{e}.gate_proj"),
                inter,
                4096,
            ),
            up_proj: projection(
                gpu,
                &mut map,
                format!("{lp}.mlp.experts.{e}.up_proj"),
                inter,
                4096,
            ),
            down_proj: projection(
                gpu,
                &mut map,
                format!("{lp}.mlp.experts.{e}.down_proj"),
                4096,
                inter,
            ),
        };
    }
    weights.shared_expert = ExpertWeight {
        gate_proj: projection(
            gpu,
            &mut map,
            format!("{lp}.mlp.shared_experts.gate_proj"),
            2048,
            4096,
        ),
        up_proj: projection(
            gpu,
            &mut map,
            format!("{lp}.mlp.shared_experts.up_proj"),
            2048,
            4096,
        ),
        down_proj: projection(
            gpu,
            &mut map,
            format!("{lp}.mlp.shared_experts.down_proj"),
            4096,
            2048,
        ),
    };
    weights.gate = DenseWeight {
        weight: tensor(
            gpu,
            &mut map,
            format!("{lp}.mlp.gate.weight"),
            &[288, 4096],
            D::BF16,
        ),
    };
    weights.correction_bias = Some(DenseWeight {
        weight: tensor(
            gpu,
            &mut map,
            format!("{lp}.mlp.gate.e_score_correction_bias"),
            &[288],
            D::FP32,
        ),
    });
    let mut layer = MoeLayer::new(weights, 288, None, gpu, &config).unwrap();
    layer.unified_layout = false;
    layer.hybrid_layout = false;
    layer.nvfp4_prequant_moe = true;
    (WeightStore::from_map(map), config, layer)
}
#[test]
fn actual_session_publication_owns_tables_retires_down_and_revokes_native_gu_once() {
    for rank in [0, 1] {
        let gpu = Gpu::new();
        let (mut store, config, mut layer) = setup(&gpu, rank);
        let originals = store.len();
        let log = RetirementLog::new(&store, &gpu).unwrap();
        let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
        gpu.clear();
        session.prepare(&mut layer, &log, &config, 0).unwrap();
        assert!(layer.btile_storage.is_published());
        assert!(layer.gate_ptrs.packed_ptrs.is_null() && layer.up_ptrs.packed_ptrs.is_null());
        assert!(layer.gate_ptrs_t.is_none() && layer.up_ptrs_t.is_none());
        assert!(layer.down_ptrs_t.is_some());
        assert!(
            layer
                .weights
                .experts
                .iter()
                .all(|e| e.gate_proj.is_null() && e.up_proj.is_null() && e.down_proj.is_null())
        );
        gpu.clear();
        assert!(session.prepare(&mut layer, &log, &config, 0).is_err());
        assert!(
            gpu.trace().is_empty(),
            "second publication must refuse before work"
        );
        session.close().unwrap();
        log.finish().rebuild(&mut store, &gpu).unwrap();
        assert_eq!(
            store.len(),
            originals - 144 * 2,
            "only routed native down packed/scales retire"
        );
        assert!(store.names().any(|name| name.ends_with("gate_proj.weight")));
        assert!(
            store
                .names()
                .any(|name| name.ends_with("down_proj.weight_scale_2"))
        );
    }
}
