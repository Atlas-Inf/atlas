// SPDX-License-Identifier: AGPL-3.0-only

//! Name-mapping tests against REAL tensor names taken verbatim from
//! `primitive-ai/Ling-3.2-air-NVFP4/model.safetensors.index.json`
//! (~/atlas-campaign/context/ling-nvfp4-index.json, 253,044 tensors).

use super::*;

/// The 13 `attention.*` tensors of dec layer 0 (a KDA layer; verbatim from
/// the index), including the two bare params.
const LAYER_0_KDA: &[&str] = &[
    "model.layers.0.attention.A_log",
    "model.layers.0.attention.dt_bias",
    "model.layers.0.attention.b_proj.weight",
    "model.layers.0.attention.f_proj.weight",
    "model.layers.0.attention.g_proj.weight",
    "model.layers.0.attention.k_conv1d.weight",
    "model.layers.0.attention.k_proj.weight",
    "model.layers.0.attention.o_norm.weight",
    "model.layers.0.attention.o_proj.weight",
    "model.layers.0.attention.q_conv1d.weight",
    "model.layers.0.attention.q_proj.weight",
    "model.layers.0.attention.v_conv1d.weight",
    "model.layers.0.attention.v_proj.weight",
];

/// The 6 `attention.*` tensors of dec layer 5 (a gated-MLA layer; verbatim).
const LAYER_5_MLA: &[&str] = &[
    "model.layers.5.attention.dense.weight",
    "model.layers.5.attention.g_proj.weight",
    "model.layers.5.attention.kv_a_layernorm.weight",
    "model.layers.5.attention.kv_a_proj_with_mqa.weight",
    "model.layers.5.attention.kv_b_proj.weight",
    "model.layers.5.attention.q_proj.weight",
];

/// All 12 compression members of `model.layers.2.mlp.experts.0` (verbatim).
const LAYER_2_EXPERT_0: &[&str] = &[
    "model.layers.2.mlp.experts.0.down_proj.input_global_scale",
    "model.layers.2.mlp.experts.0.down_proj.weight_global_scale",
    "model.layers.2.mlp.experts.0.down_proj.weight_packed",
    "model.layers.2.mlp.experts.0.down_proj.weight_scale",
    "model.layers.2.mlp.experts.0.gate_proj.input_global_scale",
    "model.layers.2.mlp.experts.0.gate_proj.weight_global_scale",
    "model.layers.2.mlp.experts.0.gate_proj.weight_packed",
    "model.layers.2.mlp.experts.0.gate_proj.weight_scale",
    "model.layers.2.mlp.experts.0.up_proj.input_global_scale",
    "model.layers.2.mlp.experts.0.up_proj.weight_global_scale",
    "model.layers.2.mlp.experts.0.up_proj.weight_packed",
    "model.layers.2.mlp.experts.0.up_proj.weight_scale",
];

/// Model-level and skip-namespace anchors (verbatim).
const MODEL_LEVEL: &[&str] = &[
    "model.word_embeddings.weight",
    "model.norm.weight",
    "lm_head.weight",
    "linear_proj.0.bias",
    "linear_proj.2.weight",
    "model.visual.blocks.0.attn.qkv.weight",
];

fn sorted(mut names: Vec<String>) -> Vec<String> {
    names.sort();
    names
}

#[test]
fn kda_attention_tensors_match_the_index() {
    assert_eq!(
        sorted(kda_attention_tensors("model.layers.0")),
        sorted(
            LAYER_0_KDA
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        )
    );
}

#[test]
fn mla_attention_tensors_match_the_index() {
    assert_eq!(
        sorted(mla_attention_tensors("model.layers.5")),
        sorted(
            LAYER_5_MLA
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        )
    );
}

#[test]
fn expert_tensor_names_match_the_index() {
    assert_eq!(
        sorted(expert_tensor_names("model.layers.2", 0)),
        sorted(
            LAYER_2_EXPERT_0
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        )
    );
    // Four members per projection, three projections: the W4A4 checkpoint
    // carries `input_global_scale` alongside the glm5 fp4l triple.
    assert_eq!(EXPERT_PART_SUFFIXES.len(), 4);
    assert_eq!(EXPERT_PROJ_STEMS.len(), 3);
}

#[test]
fn layer_norm_and_mlps() {
    assert_eq!(
        layer_norm_tensors("model.layers.0"),
        [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight"
        ]
        .map(String::from)
        .to_vec()
    );
    assert_eq!(
        dense_mlp_tensors("model.layers.0"),
        [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight"
        ]
        .map(String::from)
        .to_vec()
    );
}

#[test]
fn router_and_shared_expert_anchors() {
    assert_eq!(
        router_tensors("model.layers.2"),
        [
            "model.layers.2.mlp.gate.weight",
            "model.layers.2.mlp.gate.expert_bias"
        ]
        .map(String::from)
        .to_vec()
    );
    assert_eq!(
        shared_expert_tensors("model.layers.2"),
        [
            "model.layers.2.mlp.shared_experts.gate_proj.weight",
            "model.layers.2.mlp.shared_experts.up_proj.weight",
            "model.layers.2.mlp.shared_experts.down_proj.weight"
        ]
        .map(String::from)
        .to_vec()
    );
}

#[test]
fn model_level_anchors_match_the_index() {
    assert_eq!(WORD_EMBEDDINGS, MODEL_LEVEL[0]);
    assert_eq!(FINAL_NORM, MODEL_LEVEL[1]);
    assert_eq!(LM_HEAD, MODEL_LEVEL[2]);
    assert!(MODEL_LEVEL[3].starts_with(SKIP_PROJECTOR));
    assert!(MODEL_LEVEL[5].starts_with(SKIP_TOWER));
}

#[test]
fn a_kda_layer_and_mla_layer_disagree_everywhere_but_g_and_q_proj() {
    // The tensors usable as discriminators (signal doc in plan.rs): A_log /
    // dt_bias mark KDA, kv_a_proj_with_mqa marks MLA. g_proj / q_proj exist
    // in BOTH and must not be treated as kind evidence.
    let kda = sorted(kda_attention_tensors("model.layers.0"));
    let mla = sorted(mla_attention_tensors("model.layers.1"));
    let shared = ["g_proj.weight", "q_proj.weight"];
    let kda_stem = |n: &String| {
        n.strip_prefix("model.layers.0.attention.")
            .unwrap()
            .to_string()
    };
    let mla_stem = |n: &String| {
        n.strip_prefix("model.layers.1.attention.")
            .unwrap()
            .to_string()
    };
    let kda_has = |stem: &str| kda.iter().any(|n| kda_stem(n) == stem);
    let mla_has = |stem: &str| mla.iter().any(|n| mla_stem(n) == stem);
    for stem in shared {
        assert!(kda_has(stem) && mla_has(stem), "{stem} must exist in both");
    }
    assert!(kda_has("A_log") && !mla_has("A_log"));
    assert!(mla_has("kv_a_proj_with_mqa.weight"));
}
