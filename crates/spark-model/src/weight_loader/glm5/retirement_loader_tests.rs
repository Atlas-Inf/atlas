// SPDX-License-Identifier: AGPL-3.0-only
//! Drive the real GLM loader sites, including legacy-versus-tracked free order.
use super::tests::recording::Gpu;
use super::*;
use crate::weight_map::{DenseWeight, QuantizeCtx};
use atlas_core::config::ModelConfig;

fn insert(
    map: &mut HashMap<String, WeightTensor>,
    gpu: &dyn GpuBackend,
    name: &str,
    shape: &[usize],
    dtype: WeightDtype,
) {
    map.insert(
        name.into(),
        WeightTensor {
            ptr: gpu
                .alloc(shape.iter().product::<usize>() * dtype.byte_size())
                .unwrap(),
            shape: shape.into(),
            dtype,
        },
    );
}

fn kda_store(gpu: &dyn GpuBackend, tp: usize) -> WeightStore {
    let mut map = HashMap::new();
    for (name, shape) in [
        ("q_proj", vec![128 * tp, 128]),
        ("k_proj", vec![128 * tp, 128]),
        ("v_proj", vec![128 * tp, 128]),
        ("o_proj", vec![128, 128 * tp]),
        ("b_proj", vec![tp, 128]),
        ("f_b_proj", vec![128 * tp, 128]),
        ("g_b_proj", vec![128 * tp, 128]),
        ("f_a_proj", vec![128, 128]),
        ("g_a_proj", vec![128, 128]),
        ("o_norm", vec![128]),
        ("q_conv1d", vec![128 * tp, 4]),
        ("k_conv1d", vec![128 * tp, 4]),
        ("v_conv1d", vec![128 * tp, 4]),
    ] {
        insert(
            &mut map,
            gpu,
            &format!("l.self_attn.{name}.weight"),
            &shape,
            WeightDtype::BF16,
        );
    }
    insert(&mut map, gpu, "l.self_attn.A_log", &[tp], WeightDtype::FP32);
    insert(
        &mut map,
        gpu,
        "l.self_attn.dt_bias",
        &[128 * tp],
        WeightDtype::FP32,
    );
    WeightStore::from_map(map)
}

#[test]
fn actual_kda_free_sites_track_exact_sources_with_unchanged_legacy_events() {
    for (tp, rank) in [(1, 0), (2, 0), (2, 1)] {
        let mut traces = Vec::new();
        for tracked in [false, true] {
            let gpu = Gpu::default();
            let mut store = kda_store(&gpu, tp);
            let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
            cfg.hidden_size = 128;
            cfg.linear_num_key_heads = 1;
            cfg.linear_num_value_heads = 1;
            cfg.linear_key_head_dim = 128;
            cfg.linear_value_head_dim = 128;
            cfg.linear_conv_kernel_dim = 4;
            cfg.tp_world_size = tp;
            cfg.tp_rank = rank;
            let log = RetirementLog::new(&store, &gpu).unwrap();
            let weights = super::super::components::load_kda_weights(
                &store,
                "l",
                &cfg,
                &gpu,
                QuantizeCtx {
                    absmax_k: spark_runtime::gpu::KernelHandle(1),
                    quantize_k: spark_runtime::gpu::KernelHandle(1),
                    stream: 0,
                },
                tracked.then_some(&log),
            )
            .unwrap();
            assert!(!weights.q_proj.nvfp4.weight.is_null());
            traces.push(gpu.frees.lock().iter().map(|p| p.0).collect::<Vec<_>>());
            if tracked {
                log.finish().rebuild(&mut store, &gpu).unwrap();
                let expected = if tp == 1 { 8 } else { 3 };
                assert_eq!(store.len(), expected);
                for name in ["f_a_proj", "g_a_proj", "o_norm"] {
                    assert!(store.contains(&format!("l.self_attn.{name}.weight")));
                }
            }
        }
        assert_eq!(
            traces[0], traces[1],
            "TP{tp} rank{rank} actual free sequence changed"
        );
    }
}

#[test]
fn actual_mla_source_sites_retire_only_tp_shards_not_replicated_body() {
    for (tp, rank) in [(1, 0), (2, 0), (2, 1)] {
        let gpu = Gpu::default();
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.hidden_size = 128;
        cfg.num_attention_heads = 1;
        cfg.num_key_value_heads = 1;
        cfg.q_lora_rank = 16;
        cfg.kv_lora_rank = 16;
        cfg.qk_nope_head_dim = 16;
        cfg.qk_rope_head_dim = 0;
        cfg.v_head_dim = 16;
        cfg.head_dim = 16;
        cfg.tp_world_size = tp;
        cfg.tp_rank = rank;
        let mut map = HashMap::new();
        for (name, shape) in [
            ("q_a_proj.weight", vec![16, 128]),
            ("q_b_proj.weight", vec![16 * tp, 16]),
            ("kv_a_proj_with_mqa.weight", vec![16, 128]),
            ("kv_b_proj.weight", vec![32 * tp, 16]),
            ("o_proj.weight", vec![128, 16 * tp]),
            ("q_a_layernorm.weight", vec![16]),
            ("kv_a_layernorm.weight", vec![16]),
        ] {
            insert(
                &mut map,
                &gpu,
                &format!("l.self_attn.{name}"),
                &shape,
                WeightDtype::BF16,
            );
        }
        for name in [
            "wq_b.weight",
            "wk.weight",
            "weights_proj.weight",
            "index_kpool_compress_gate",
            "index_kpool_compress_ape",
            "k_norm.weight",
            "k_norm.bias",
        ] {
            insert(
                &mut map,
                &gpu,
                &format!("l.self_attn.indexer.{name}"),
                &[16],
                WeightDtype::BF16,
            );
        }
        let mut store = WeightStore::from_map(map);
        let log = RetirementLog::new(&store, &gpu).unwrap();
        let norm = DenseWeight {
            weight: gpu.alloc(256).unwrap(),
        };
        let _layer = super::super::layers::load_mla_layer(
            &store,
            "l",
            0,
            0,
            norm,
            norm,
            crate::layers::FfnComponent::None,
            None,
            &cfg,
            &gpu,
            spark_runtime::kv_cache::KvCacheDtype::Bf16,
            false,
            Some(&log),
        )
        .unwrap();
        log.finish().rebuild(&mut store, &gpu).unwrap();
        for name in ["q_b_proj", "kv_b_proj", "o_proj"] {
            assert_eq!(
                store.contains(&format!("l.self_attn.{name}.weight")),
                tp == 1
            );
        }
        assert!(store.contains("l.self_attn.q_a_proj.weight"));
        assert!(store.contains("l.self_attn.kv_a_proj_with_mqa.weight"));
    }
}

#[test]
fn actual_kda_each_source_or_derived_free_failure_is_spent_without_retry() {
    let run = |gpu: &Gpu, store: &WeightStore, log: &RetirementLog<'_>| {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.hidden_size = 128;
        cfg.linear_num_key_heads = 1;
        cfg.linear_num_value_heads = 1;
        cfg.linear_key_head_dim = 128;
        cfg.linear_value_head_dim = 128;
        cfg.linear_conv_kernel_dim = 4;
        cfg.tp_world_size = 2;
        cfg.tp_rank = 0;
        super::super::components::load_kda_weights(
            store,
            "l",
            &cfg,
            gpu,
            QuantizeCtx {
                absmax_k: spark_runtime::gpu::KernelHandle(1),
                quantize_k: spark_runtime::gpu::KernelHandle(1),
                stream: 0,
            },
            Some(log),
        )
    };
    let baseline = Gpu::default();
    let store = kda_store(&baseline, 2);
    let log = RetirementLog::new(&store, &baseline).unwrap();
    run(&baseline, &store, &log).unwrap();
    let free_points = baseline.frees.lock().clone();
    assert_eq!(
        free_points.len(),
        19,
        "all sharded/full/vector and hot/conv frees"
    );
    for failed in free_points {
        let gpu = Gpu::default();
        let mut store = kda_store(&gpu, 2);
        let log = RetirementLog::new(&store, &gpu).unwrap();
        *gpu.fail.lock() = Some(failed);
        assert!(run(&gpu, &store, &log).is_err());
        assert!(log.finish().cleanup(&mut store, &gpu).is_err());
        let attempts = gpu.frees.lock();
        assert_eq!(attempts.iter().filter(|&&p| p == failed).count(), 1);
        assert_eq!(
            attempts
                .iter()
                .map(|p| p.0)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            attempts.len()
        );
        assert!(store.is_empty());
    }
}
