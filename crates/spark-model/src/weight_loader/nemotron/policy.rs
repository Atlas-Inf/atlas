// SPDX-License-Identifier: AGPL-3.0-only

//! Loader-owned load policies for the Nemotron-H family — which tensor
//! namespaces are dead weight and whether the fast loader may pack shards.

use atlas_core::config::ModelConfig;

/// `nemotron_h_omni` (Nemotron 3.5 Super VL, text-only serve): the parser
/// sets `mtp_num_hidden_layers = 0` and `vision = None`, so the omni
/// wrapper's vision/MTP towers are dead weight — on the NVFP4 drop that is
/// ~3.45 GB resident (measured from safetensors headers, 2026-10-08:
/// `language_model.mtp.` 1.77 GB, `vision_model.` 1.31 GB, `mlp1.` 0.38 GB).
///
/// Every prefix must end in `.` — `vision_model.` must not match a
/// hypothetical `vision_model_extra.*`.
pub(super) fn dead_namespaces(config: &ModelConfig) -> Vec<String> {
    if config.model_type != "nemotron_h_omni" {
        return Vec::new();
    }
    // The vision tower stays dead only while it isn't served: when
    // `config.vision` carries omni params, `vision_model.*`, `mlp1.*` and
    // `vision_projector.*` are loadable weights, not dead namespaces.
    let vision_served = config.vision.as_ref().is_some_and(|v| v.omni.is_some());
    let mut dead = vec!["language_model.mtp."];
    if !vision_served {
        dead.extend(["vision_model.", "mlp1.", "vision_projector."]);
    }
    dead.iter().map(|s| s.to_string()).collect()
}

/// Single-GPU omni only: the TP sharding path in `nemotron.rs` frees
/// store-owned quantization tensors (`gpu.free(q.weight)` etc.), and EP
/// needs its own audit before packing. At tp=ep=1 no load-path `gpu.free`
/// reaches a store-owned pointer, and teardown's whole-store drain is
/// covered by the backend's slab-interior no-op.
pub(super) fn slab_pack_weights(config: &ModelConfig) -> bool {
    config.model_type == "nemotron_h_omni" && config.tp_world_size <= 1 && config.ep_world_size <= 1
}

/// Native FP8 attention lever: `nemotron_h_omni` by MODEL TYPE (survives the
/// text-only `config.vision = None` drop on targets without a vision encoder
/// — see `ModelConfig::is_nemotron_h_omni`). `ATLAS_NEMOTRON_NATIVE_FP8_ATTN`
/// =0 opts out, =1 opts in on non-omni; unknown values fall back to default.
pub(super) fn native_fp8_attn(config: &ModelConfig, env: Option<&str>) -> bool {
    match env {
        Some("0") => false,
        Some("1") => true,
        _ => config.is_nemotron_h_omni(),
    }
}
