// SPDX-License-Identifier: AGPL-3.0-only

//! Safetensors name mapping for the Ling-3.2 (Bailing-Moe-V3-VL, NVFP4)
//! checkpoint.
//!
//! Ground truth is the shipped `model.safetensors.index.json` of
//! `primitive-ai/Ling-3.2-air-NVFP4` (253,044 tensors) cross-checked against
//! `modeling_bailing_moe_v3_vl.py`. The key namespaces:
//!
//! | module                     | tensors (per decoder layer `N`)                      |
//! |----------------------------|-------------------------------------------------------|
//! | embedding / final norm     | `model.word_embeddings.weight`, `model.norm.weight`    |
//! | LM head                    | `lm_head.weight`                                      |
//! | KDA attention (35 layers)  | `attention.{q,k,v,g,b,f,o}_proj.weight`, `attention.A_log`, `attention.dt_bias`, `attention.{q,k,v}_conv1d.weight`, `attention.o_norm.weight` |
//! | gated MLA (7 layers + MTP)  | `attention.{q_proj,kv_a_proj_with_mqa,kv_b_proj,kv_a_layernorm,g_proj,dense}.weight` |
//! | dense MLP (layers 0..2)     | `mlp.{gate,up,down}_proj.weight`                      |
//! | MoE (layers 2..=42)        | `mlp.gate.weight`, `mlp.gate.expert_bias` (FP32), `mlp.shared_experts.{gate,up,down}_proj.weight` (BF16), `mlp.experts.<e>.{gate,up,down}_proj.{weight_packed,weight_scale,weight_global_scale,input_global_scale}` |
//! | MTP (layer 42)             | `eh_proj.weight`, `enorm.weight`, `hnorm.weight`, `final_layernorm.weight` + MLA attention + MoE |
//! | vision tower + projector   | `model.visual.*`, `linear_proj.{0,2}.{weight,bias}` (skipped) |
//!
//! KDA vs MLA disambiguation is by tensor signature, not index arithmetic: a
//! KDA layer carries `attention.A_log` / `attention.dt_bias`; a gated-MLA layer
//! carries `attention.kv_a_proj_with_mqa`. `g_proj` exists in both and is NOT
//! a discriminator.

/// Per-projection stems of a KDA (`BailingMoeV3KimiDeltaAttention`) layer.
pub(super) const KDA_PROJ_STEMS: &[&str] = &[
    "q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "f_proj", "o_proj",
];

/// Short-convolution filters and the gated output norm of a KDA layer.
pub(super) const KDA_CONV_STEMS: &[&str] = &["q_conv1d", "k_conv1d", "v_conv1d", "o_norm"];

/// Bare parameters (no `.weight` suffix) of a KDA layer. FP32 on disk.
pub(super) const KDA_PARAM_STEMS: &[&str] = &["A_log", "dt_bias"];

/// Per-projection stems of a gated-MLA (`BailingMoeV3MultiLatentAttention`)
/// layer. `dense` is the output projection (`o_proj` in other families).
pub(super) const MLA_PROJ_STEMS: &[&str] = &[
    "q_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "kv_a_layernorm",
    "g_proj",
    "dense",
];

/// Projection stems of a dense `BailingMoeV3MLP` / shared expert.
pub(super) const MLP_PROJ_STEMS: &[&str] = &["gate_proj", "up_proj", "down_proj"];

/// Compression pieces of one NVFP4 routed-expert projection
/// (compressed-tensors `nvfp4-pack-quantized`, group size 16).
pub(super) const EXPERT_PART_SUFFIXES: &[&str] = &[
    "weight_packed",
    "weight_scale",
    "weight_global_scale",
    "input_global_scale",
];

/// The three projections of one routed or shared expert.
pub(super) const EXPERT_PROJ_STEMS: &[&str] = &["gate_proj", "up_proj", "down_proj"];

/// Model-level tensor names (the text decoder, not the ViT or projector).
pub(super) const WORD_EMBEDDINGS: &str = "model.word_embeddings.weight";
pub(super) const FINAL_NORM: &str = "model.norm.weight";
pub(super) const LM_HEAD: &str = "lm_head.weight";

/// Namespace prefixes this phase-1 (text-only) loader skips at upload.
/// See [`super::vision`] for the extension seam.
pub(super) const SKIP_TOWER: &str = "model.visual.";
pub(super) const SKIP_PROJECTOR: &str = "linear_proj.";

/// Every tensor name a KDA layer's `attention` module must contribute
/// (13 names).
pub(super) fn kda_attention_tensors(lp: &str) -> Vec<String> {
    KDA_PROJ_STEMS
        .iter()
        .chain(KDA_CONV_STEMS)
        .map(|s| format!("{lp}.attention.{s}.weight"))
        .chain(
            KDA_PARAM_STEMS
                .iter()
                .map(|s| format!("{lp}.attention.{s}")),
        )
        .collect()
}

/// Every tensor name a gated-MLA layer's `attention` module must contribute
/// (6 names).
pub(super) fn mla_attention_tensors(lp: &str) -> Vec<String> {
    MLA_PROJ_STEMS
        .iter()
        .map(|s| format!("{lp}.attention.{s}.weight"))
        .collect()
}

/// The two layernorm weights every decoder and MTP layer carries.
pub(super) fn layer_norm_tensors(lp: &str) -> Vec<String> {
    [
        format!("{lp}.input_layernorm.weight"),
        format!("{lp}.post_attention_layernorm.weight"),
    ]
    .into_iter()
    .collect()
}

/// Every tensor name a dense `BailingMoeV3MLP` must contribute (3 names).
pub(super) fn dense_mlp_tensors(lp: &str) -> Vec<String> {
    MLP_PROJ_STEMS
        .iter()
        .map(|s| format!("{lp}.mlp.{s}.weight"))
        .collect()
}

/// Router tensors of one `BailingMoeV3SparseMoeBlock` (2 names): the FP32
/// gate weight and the FP32 per-expert routing bias.
pub(super) fn router_tensors(lp: &str) -> Vec<String> {
    [
        format!("{lp}.mlp.gate.weight"),
        format!("{lp}.mlp.gate.expert_bias"),
    ]
    .into_iter()
    .collect()
}

/// The shared expert's three BF16 projections.
pub(super) fn shared_expert_tensors(lp: &str) -> Vec<String> {
    MLP_PROJ_STEMS
        .iter()
        .map(|s| format!("{lp}.mlp.shared_experts.{s}.weight"))
        .collect()
}

/// The 12 compression members of one routed expert `e` of layer prefix `lp`:
/// three projections, each `{weight_packed, weight_scale,
/// weight_global_scale, input_global_scale}`.
pub(super) fn expert_tensor_names(lp: &str, expert: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(EXPERT_PROJ_STEMS.len() * EXPERT_PART_SUFFIXES.len());
    for proj in EXPERT_PROJ_STEMS {
        for part in EXPERT_PART_SUFFIXES {
            names.push(format!("{lp}.mlp.experts.{expert}.{proj}.{part}"));
        }
    }
    names
}

#[cfg(test)]
#[path = "names_tests.rs"]
mod tests;
