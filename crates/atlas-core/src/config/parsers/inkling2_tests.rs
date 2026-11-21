// SPDX-License-Identifier: AGPL-3.0-only

use crate::config::{LayerType, parse_config};

/// thinkingmachines/Inkling-2-Mini `config.json`, byte-identical to the
/// 2026-11-20 release.
const MINI: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_data/inkling2_mini_config.json"
));

#[test]
fn mini_unrolls_the_loop_into_48_visits() {
    let config = parse_config(MINI).expect("parse inkling2_loop");
    let ink = config.inkling2.as_ref().expect("inkling2 params");
    assert_eq!(config.model_type, "inkling2_loop");
    assert_eq!(ink.physical_layers, 24);
    // prelude 0..6, core 6..18 three times, coda 18..24
    assert_eq!(config.num_hidden_layers, 48);
    assert_eq!(config.layer_types, vec![LayerType::FullAttention; 48]);
    let physical: Vec<usize> = (0..48).map(|v| ink.visit_physical(v)).collect();
    let expected: Vec<usize> = (0..6)
        .chain(6..18)
        .chain(6..18)
        .chain(6..18)
        .chain(18..24)
        .collect();
    assert_eq!(physical, expected);
    assert_eq!(ink.visit_pass(5), 0);
    assert_eq!(ink.visit_pass(6), 0);
    assert_eq!(ink.visit_pass(18), 1);
    assert_eq!(ink.visit_pass(41), 2);
    assert_eq!(ink.visit_pass(42), 0);
}

#[test]
fn mini_sliding_and_full_visits_follow_their_physical_layer() {
    let config = parse_config(MINI).unwrap();
    let ink = config.inkling2.as_ref().unwrap();
    assert_eq!(config.sliding_window, 1024);
    // physical full layers 5, 11, 17, 23 land on visits 5; 11, 23, 35; 17, 29, 41; 47
    let full: Vec<usize> = (0..48)
        .filter(|v| !config.sliding_attention_layers.contains(v))
        .collect();
    assert_eq!(full, [5, 11, 17, 23, 29, 35, 41, 47]);
    assert_eq!(config.sliding_attention_layers.len(), 40);
    assert!(
        config
            .sliding_attention_layers
            .iter()
            .all(|&v| ink.local_physical.contains(&ink.visit_physical(v)))
    );
}

#[test]
fn mini_maps_attention_moe_and_logit_fields() {
    let config = parse_config(MINI).unwrap();
    assert_eq!(config.hidden_size, 2304);
    assert_eq!(config.num_attention_heads, 24);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 201024);
    assert_eq!(config.eos_token_id, 200006);
    assert_eq!(config.max_position_embeddings, 1_048_576);
    // no RoPE: position comes from the relative bias and the short convs
    assert_eq!(config.rotary_dim(), 0);
    assert_eq!(config.num_experts, 128);
    assert_eq!(config.num_experts_per_tok, 8);
    assert_eq!(config.moe_intermediate_size, 1152);
    assert_eq!(config.shared_expert_intermediate_size, 0);
    assert_eq!(config.scoring_func, "sigmoid");
    assert!(config.use_routing_bias);
    assert!(config.norm_topk_prob);
    assert_eq!(config.routed_scaling_factor, 8.0);
    assert_eq!(config.final_logit_softcapping, 30.0);
    assert!(config.attn_gated);
    assert_eq!(config.num_ssm_layers(), 0);
    // checkpoint keys are per physical layer
    assert_eq!(config.layer_prefix(17), "model.llm.layers.17");
}

#[test]
fn mini_keeps_the_inkling_specific_numerics() {
    let config = parse_config(MINI).unwrap();
    let ink = config.inkling2.as_ref().unwrap();
    assert_eq!(ink.unpadded_vocab_size, 200058);
    assert_eq!((ink.d_rel, ink.rel_extent), (16, 1024));
    assert_eq!(ink.log_scaling_n_floor, 8192);
    assert_eq!(ink.log_scaling_alpha, 0.1);
    assert_eq!(ink.qk_l2_clip_threshold, 22.627_417);
    assert_eq!(ink.rel_l2_clip_threshold, 8.0);
    assert_eq!(ink.sconv_kernel_size, 4);
    assert_eq!(ink.attn_res_block_size, 6);
    assert_eq!((ink.mlp_gate_softcap, ink.mlp_up_softcap), (4.0, 25.0));
    assert_eq!((ink.core_start, ink.core_end, ink.num_passes), (6, 18, 3));
    assert_eq!(ink.loop_gate_rank, 64);
}

#[test]
fn mini_wires_the_dspark_drafter() {
    let config = parse_config(MINI).unwrap();
    assert_eq!(config.dspark_block_size, 8);
    assert_eq!(config.dspark_markov_rank, 256);
    // drafter reads the inputs of visits 5, 23, 41 and the final state (48)
    assert_eq!(config.dspark_target_layer_ids, [5, 23, 41, 48]);
}

fn with_text(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut raw: serde_json::Value = serde_json::from_str(MINI).unwrap();
    edit(raw.get_mut("text_config").unwrap());
    raw.to_string()
}

#[test]
fn rejects_layouts_the_port_does_not_implement() {
    type Edit = fn(&mut serde_json::Value);
    let cases: [(&str, Edit); 5] = [
        ("variable loop", |t| {
            t["loop_config"]["is_varloop"] = true.into()
        }),
        ("unaligned core", |t| {
            t["loop_config"]["core_start"] = 5.into()
        }),
        ("one pass", |t| t["loop_config"]["num_passes"] = 1.into()),
        ("shared experts", |t| t["n_shared_experts"] = 1.into()),
        ("window != rel extent", |t| {
            t["sliding_window_size"] = 512.into()
        }),
    ];
    for (name, edit) in cases {
        let json = with_text(edit);
        assert!(parse_config(&json).is_err(), "{name} must be refused");
    }
}
