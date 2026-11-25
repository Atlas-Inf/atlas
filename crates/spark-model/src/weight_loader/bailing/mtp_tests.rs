// SPDX-License-Identifier: AGPL-3.0-only

//! MTP layer name tests against the REAL `model.layers.42.*` namespace.

use super::*;

#[test]
fn mtp_tensors_match_the_index() {
    // Verbatim from the checkpoint (layer 42, all non-expert tensors):
    // enorm / hnorm / eh_proj / final_layernorm are the four MTP-only
    // additions on top of the standard layer invariants.
    assert_eq!(
        mtp_tensors("model.layers.42"),
        [
            "model.layers.42.enorm.weight",
            "model.layers.42.hnorm.weight",
            "model.layers.42.eh_proj.weight",
            "model.layers.42.final_layernorm.weight"
        ]
        .map(String::from)
        .to_vec()
    );
}

#[test]
fn mtp_attention_is_gated_mla() {
    assert_eq!(mtp_attention_kind(), AttentionKind::GatedMla);
}
