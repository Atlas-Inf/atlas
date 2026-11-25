// SPDX-License-Identifier: AGPL-3.0-only

//! NEXTN MTP layer name mapping (Ling-3.2 ships ONE module at
//! `model.layers.42.*`, `num_nextn_predict_layers = 1`).
//!
//! `BailingMoeV3MTPLayer` (modeling file, ~line 1442) is a full transformer
//! layer the reference runs on `eh_proj(cat([enorm(emb), hnorm(hidden)]))`:
//! gated-MLA attention (`mtp_use_kda = false`), its own 512-expert MoE with
//! shared expert, and a final layernorm the main stack does not carry.
//! Distinctive tensor: `eh_proj.weight` (`[hidden, 2 * hidden]`).

use super::config::AttentionKind;

/// The MTP-only tensors of layer prefix `lp` (4 names). The layer's shared
/// `input_layernorm` / `post_attention_layernorm`, attention and MoE tensors
/// come from [`super::names`] like any decoder layer.
pub(crate) fn mtp_tensors(lp: &str) -> Vec<String> {
    [
        format!("{lp}.enorm.weight"),
        format!("{lp}.hnorm.weight"),
        format!("{lp}.eh_proj.weight"),
        format!("{lp}.final_layernorm.weight"),
    ]
    .into_iter()
    .collect()
}

/// The attention architecture of every Bailing MTP layer. Fixed `GatedMla`
/// because the loader refuses `mtp_use_kda = true` checkpoints
/// ([`super::config::BailingConfig::validate`]); the return type keeps the
/// call sites uniform with decoder layers.
pub(crate) fn mtp_attention_kind() -> AttentionKind {
    AttentionKind::GatedMla
}

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;
