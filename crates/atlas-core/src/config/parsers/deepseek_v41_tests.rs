// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::config::parse_config;

/// deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json, revision
/// 1bae0e1088993b9b82de08f59de5be5a34e42170, unedited.
const PUBLISHED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_data/deepseek_v41_flash_1017_config.json"
));

fn edited(
    edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) -> Result<ModelConfig> {
    let mut raw: serde_json::Value = serde_json::from_str(PUBLISHED).unwrap();
    edit(raw["text_config"].as_object_mut().unwrap());
    parse_config(&raw.to_string())
}

#[test]
fn the_published_config_parses_as_its_own_family() {
    let c = parse_config(PUBLISHED).expect("parse 1017");
    assert_eq!(c.model_type, "deepseek_v41");
    assert_eq!(
        (
            c.hidden_size,
            c.num_hidden_layers,
            c.num_attention_heads,
            c.num_key_value_heads
        ),
        (5120, 40, 64, 1)
    );
    assert_eq!(
        (c.head_dim, c.qk_rope_head_dim, c.qk_nope_head_dim),
        (512, 64, 448)
    );
    assert_eq!((c.q_lora_rank, c.o_lora_rank, c.o_groups), (1280, 1024, 8));
    assert_eq!(
        (
            c.num_experts,
            c.num_experts_per_tok,
            c.moe_intermediate_size
        ),
        (384, 6, 2304)
    );
    assert_eq!(c.shared_expert_intermediate_size, 2304);
    assert_eq!(c.scoring_func, "sqrtsoftplus");
    assert!(c.use_routing_bias);
    assert_eq!(c.routed_scaling_factor, 1.5);
    assert_eq!(c.num_hash_layers, 0, "V4.1 has no hash-routed layers");
    assert_eq!((c.hc_mult, c.hc_sinkhorn_iters), (4, 20));
    assert_eq!(c.sliding_window, 128);
    assert_eq!(
        c.compress_ratios.len(),
        43,
        "40 layers and three DSpark stages"
    );
    assert_eq!(
        (c.index_n_heads, c.index_head_dim, c.index_topk),
        (32, 128, 512)
    );
    assert_eq!(c.candidate_source_layer_id, Some(20));
    assert_eq!((c.candidate_topk_blocks, c.candidate_block_size), (2048, 8));
    assert_eq!(c.num_mtp_modules, 3);
    assert_eq!((c.dspark_block_size, c.dspark_noise_token_id), (5, 128799));
    assert_eq!(c.dspark_target_layer_ids, [37, 38, 39]);
    assert_eq!(
        (c.dspark_n_routed_experts, c.dspark_num_experts_per_tok),
        (128, 3)
    );
    assert_eq!(c.max_position_embeddings, 1 << 20);
    assert_eq!(
        (c.yarn_factor, c.yarn_original_max_position_embeddings),
        (16.0, 65536)
    );
    assert_eq!(
        c.rope_theta, 160000.0,
        "compressed layers rotate at compress_rope_theta"
    );
    assert_eq!(c.eos_token_id, 1);
    let q = c.quantization_config.as_ref().expect("quantization_config");
    assert_eq!(q.quant_method, "fp8");
    assert_eq!(
        q.weight_block_size,
        [32, 32],
        "V4.1 FP8 blocks are 32x32, not V4's 128"
    );
}

#[test]
fn the_architecture_alone_selects_the_family() {
    let mut raw: serde_json::Value = serde_json::from_str(PUBLISHED).unwrap();
    raw.as_object_mut().unwrap().remove("model_type");
    assert_eq!(
        parse_config(&raw.to_string()).unwrap().model_type,
        "deepseek_v41"
    );
}

#[test]
fn compressed_kv_and_top_k_come_from_the_latest_source() {
    let c = parse_config(PUBLISHED).unwrap();
    let kv: Vec<Option<usize>> = [0, 1, 2, 7, 8, 13, 14, 19, 20, 39]
        .map(|l| c.kv_source_layer(l))
        .into();
    assert_eq!(
        kv,
        [
            None,
            None,
            Some(2),
            Some(2),
            Some(8),
            Some(8),
            Some(14),
            Some(14),
            Some(20),
            Some(20)
        ]
    );
    let index: Vec<Option<usize>> = [1, 2, 19, 20, 23, 24, 35, 36, 39]
        .map(|l| c.index_source_layer(l))
        .into();
    assert_eq!(
        index,
        [
            None,
            Some(2),
            Some(14),
            Some(20),
            Some(20),
            Some(24),
            Some(32),
            Some(36),
            Some(36)
        ]
    );
}

#[test]
fn the_engram_geometry_validates() {
    let c = parse_config(PUBLISHED).unwrap();
    let e = c
        .deepseek_v41_engram(2)
        .unwrap()
        .expect("1017 has Engram layers");
    assert_eq!(e.layer_ids, [1, 14]);
    assert_eq!((e.n_cols(), e.head_dim), (24, 256));
}

#[test]
fn a_consumer_whose_source_compresses_differently_is_refused() {
    // layer 20 is the only 1:1 source; without it, layers 20..39 would read
    // layer 14's 2:1 cache
    let err = edited(|t| {
        t.insert("kv_source_layer_ids".into(), serde_json::json!([2, 8, 14]));
    })
    .unwrap_err();
    assert!(format!("{err:#}").contains("layer 20"), "{err:#}");
}

#[test]
fn a_dspark_stage_with_compressed_attention_is_refused() {
    let err = edited(|t| {
        let ratios = t["compress_ratios"].as_array_mut().unwrap();
        ratios[41] = serde_json::json!(1);
    })
    .unwrap_err();
    assert!(format!("{err:#}").contains("DSpark"), "{err:#}");
}

#[test]
fn a_candidate_source_that_runs_no_indexer_is_refused() {
    let err = edited(|t| {
        t.insert("candidate_source_layer_id".into(), serde_json::json!(21));
    })
    .unwrap_err();
    assert!(format!("{err:#}").contains("candidate"), "{err:#}");
}

#[test]
fn engram_tables_that_disagree_with_the_hash_are_refused() {
    let err = edited(|t| {
        t.insert(
            "engram_num_embeddings".into(),
            serde_json::json!([384006168, 384016681]),
        );
    })
    .unwrap_err();
    assert!(format!("{err:#}").contains("Engram layer 14"), "{err:#}");
}

#[test]
fn a_config_without_text_config_is_refused() {
    let err = parse_config(
        r#"{"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"]}"#,
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("text_config"), "{err:#}");
}

#[test]
fn v4_compressing_layers_still_own_their_kv() {
    // DeepSeek-V4 declares no sources: each compressing layer is its own, and
    // which of them index is not a sharing question.
    let c = ModelConfig {
        compress_ratios: vec![0, 4, 128],
        ..ModelConfig::qwen3_next_80b_nvfp4()
    };
    assert_eq!(
        [0, 1, 2].map(|l| c.kv_source_layer(l)),
        [None, Some(1), Some(2)]
    );
    assert_eq!(
        [0, 1, 2].map(|l| c.index_source_layer(l)),
        [None, None, None]
    );
}
