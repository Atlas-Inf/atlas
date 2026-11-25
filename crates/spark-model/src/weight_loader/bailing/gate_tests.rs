// SPDX-License-Identifier: AGPL-3.0-only

//! Router-gating and SwiGLU-limit parsing against synthetic text_configs,
//! with the Ling-3.2-air constants as the reference shape. Full-checkpoint
//! parsing of the REAL config (both lists verbatim) lives in
//! `config_tests.rs`.

use super::*;
use serde_json::{Value, json};

/// The Ling router shape: sigmoid, group-limited top-8 over 512 experts in
/// 8 groups of 64, FP32 router, expert bias on.
fn ling() -> Value {
    json!({
        "num_experts": 512,
        "num_experts_per_tok": 8,
        "n_group": 8,
        "topk_group": 4,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "score_function": "sigmoid",
        "router_dtype": "fp32",
        "moe_router_enable_expert_bias": true,
        "num_hidden_layers": 42,
        "expert_swiglu_limit_list": routed_list(),
        "share_expert_swiglu_limit_list": shared_list(),
    })
}

fn routed_list() -> Vec<usize> {
    [vec![0; 35], vec![4; 7]].concat()
}

fn shared_list() -> Vec<usize> {
    [vec![0; 34], vec![5; 6], vec![7; 2]].concat()
}

#[test]
fn ling_router_constants() {
    let gating = RouterGating::from_text_config(&ling()).unwrap();
    assert_eq!(gating.num_experts, 512);
    assert_eq!(gating.top_k, 8);
    assert_eq!(gating.n_group, 8);
    assert_eq!(gating.topk_group, 4);
    assert_eq!(gating.experts_per_group(), 64);
    assert_eq!(gating.routed_scaling_factor, 2.5);
    assert!(gating.norm_topk_prob && gating.expert_bias && gating.router_fp32);
}

#[test]
fn swiglu_reference_semantics_are_a_clamp_not_softmax() {
    let limits = SwigluLimits::from_text_config(&ling()).unwrap();
    // Routed: clamp 4 exactly on layers 35..=41.
    for idx in [0, 1, 34, 42] {
        assert_eq!(limits.routed_limit(idx), None, "layer {idx}");
    }
    for idx in 35..=41 {
        assert_eq!(limits.routed_limit(idx), Some(4), "layer {idx}");
    }
    // Shared: 5 on 34..=39, 7 on 40..=41.
    assert_eq!(limits.shared_limit(33), None);
    for idx in 34..=39 {
        assert_eq!(limits.shared_limit(idx), Some(5), "layer {idx}");
    }
    for idx in 40..=41 {
        assert_eq!(limits.shared_limit(idx), Some(7), "layer {idx}");
    }
}

#[test]
fn a_missing_limit_list_means_no_layer_clamps() {
    let mut text_config = ling();
    text_config
        .as_object_mut()
        .unwrap()
        .remove("expert_swiglu_limit_list")
        .unwrap();
    text_config
        .as_object_mut()
        .unwrap()
        .remove("share_expert_swiglu_limit_list")
        .unwrap();
    let limits = SwigluLimits::from_text_config(&text_config).unwrap();
    for idx in 0..42 {
        assert_eq!(limits.routed_limit(idx), None);
        assert_eq!(limits.shared_limit(idx), None);
    }
}

#[test]
fn limit_lists_must_match_num_hidden_layers() {
    let mut text_config = ling();
    text_config["expert_swiglu_limit_list"] = json!([0, 0, 4]);
    let err = SwigluLimits::from_text_config(&text_config).unwrap_err();
    assert!(
        err.to_string().contains("expert_swiglu_limit_list"),
        "{err}"
    );
    assert!(err.to_string().contains("42"), "{err}");
}

#[test]
fn out_of_range_entries_are_nonpositive_mean_no_clamp_like_the_reference() {
    let mut text_config = ling();
    // HF `_get_swiglu_limit` treats a zero entry as None; so does the parse.
    text_config["expert_swiglu_limit_list"] = json!(vec![0; 42]);
    let limits = SwigluLimits::from_text_config(&text_config).unwrap();
    assert_eq!(limits.routed_limit(35), None);
}

#[test]
fn softmax_routing_is_refused() {
    let mut text_config = ling();
    text_config["score_function"] = json!("softmax");
    let err = RouterGating::from_text_config(&text_config).unwrap_err();
    assert!(err.to_string().contains("sigmoid"), "{err}");
}

#[test]
fn grouping_must_divide_the_expert_axis() {
    let mut text_config = ling();
    text_config["n_group"] = json!(7);
    let err = RouterGating::from_text_config(&text_config).unwrap_err();
    assert!(err.to_string().contains("n_group"), "{err}");
}

#[test]
fn topk_group_must_stay_within_the_groups() {
    let mut text_config = ling();
    text_config["topk_group"] = json!(9);
    let err = RouterGating::from_text_config(&text_config).unwrap_err();
    assert!(err.to_string().contains("topk_group"), "{err}");
}

#[test]
fn non_fp32_router_is_refused() {
    let mut text_config = ling();
    text_config["router_dtype"] = json!("bf16");
    let err = RouterGating::from_text_config(&text_config).unwrap_err();
    assert!(err.to_string().contains("fp32"), "{err}");
}
