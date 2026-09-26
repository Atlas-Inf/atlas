// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn stems(names: &[&str]) -> Vec<String> {
    dense_stems(names.iter().copied())
}

#[test]
fn dense_stems_keeps_dense_linears_sorted() {
    assert_eq!(
        stems(&[
            "model.layers.1.self_attn.o_proj.trellis",
            "lm_head.trellis",
            "model.layers.0.linear_attn.in_proj_qkv.trellis",
        ]),
        vec![
            "lm_head",
            "model.layers.0.linear_attn.in_proj_qkv",
            "model.layers.1.self_attn.o_proj",
        ]
    );
}

#[test]
fn dense_stems_drops_experts_and_non_trellis_names() {
    assert!(
        stems(&[
            "model.layers.3.mlp.experts.7.gate_proj.trellis",
            "model.layers.3.mlp.shared_expert.down_proj.trellis",
            "model.layers.3.mlp.gate_proj.weight",
            "model.layers.0.input_layernorm.weight",
        ])
        .is_empty()
    );
}

#[test]
fn dense_stems_drops_ngram_embedding() {
    assert!(stems(&["lp.ple.ple_embedding.ngram_embedding.trellis"]).is_empty());
}

#[test]
fn keeps_packed_with_gdn_stems_only_under_native() {
    let gdn = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.35.linear_attn.in_proj_z",
        "model.language_model.layers.7.linear_attn.out_proj",
    ];
    for stem in gdn {
        assert!(keeps_packed_with(stem, true), "{stem}");
        assert!(!keeps_packed_with(stem, false), "{stem}");
    }
    // Attention and expert stems never keep their packing.
    assert!(!keeps_packed_with(
        "model.language_model.layers.3.self_attn.q_proj",
        true
    ));
    assert!(!keeps_packed_with(
        "model.language_model.layers.3.mlp.experts.7.gate_proj",
        true
    ));
}

#[test]
fn dense_stems_keeps_vision_and_mtp_linears() {
    assert_eq!(
        stems(&[
            "mtp.fc.trellis",
            "visual.blocks.0.attn.qkv.trellis",
            "mtp.layers.0.self_attn.q_proj.trellis",
        ]),
        vec![
            "mtp.fc",
            "mtp.layers.0.self_attn.q_proj",
            "visual.blocks.0.attn.qkv",
        ]
    );
}
