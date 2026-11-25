// SPDX-License-Identifier: AGPL-3.0-only

//! NEXTN MTP layer name mapping (Ling-3.2 ships ONE module at
//! `model.layers.42.*`, `num_nextn_predict_layers = 1`).
//!
//! `BailingMoeV3MTPLayer` (modeling file, ~line 1442) is a full transformer
//! layer the reference runs on `eh_proj(cat([enorm(emb), hnorm(hidden)]))`:
//! gated-MLA attention (`mtp_use_kda = false`), its own 512-expert MoE with
//! shared expert, and a final layernorm the main stack does not carry.
//! Distinctive tensor: `eh_proj.weight` (`[hidden, 2 * hidden]`).
//!
//! [`load_ling_mtp_module`] (the campaign's MTP-wiring slice) loads that
//! whole module as a family-driver draft body plus its four MTP-only
//! tensors, exactly the way `glm5/mtp.rs` loads GLM-5's predictor and
//! `deepseek_v4.rs` loads V4's: the body is the same gated-MLA layer code
//! the target's 7 MLA layers run, constructed at `model.layers.42` with its
//! own shared device scratch and paged KV pool (owned by
//! `layers::ling_mtp::LingMtpHead`).

use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::model::bailing_c1::{FfnKind, LingMlaLayer, LingScratch, LingSpec};
use crate::weight_map::{DenseWeight, dense};

use super::config::AttentionKind;
use super::layers;

/// The loaded draft module: layer-42's full body (gated-MLA attention +
/// its own 512-expert MoE + shared expert) plus the four NEXTN-only tensors
/// ([`mtp_tensors`]).
pub struct LingMtpModule {
    pub(crate) body: LingMlaLayer,
    pub enorm: DenseWeight,
    pub hnorm: DenseWeight,
    /// `eh_proj.weight` `[hidden, 2 * hidden]` — reads the `[2, hidden]`
    /// pair `ling_mtp_eh_norm` writes.
    pub eh_proj: DenseWeight,
    /// The MTP layer's own `final_layernorm` (the main stack's final norm is
    /// NOT shared here: the reference applies this one after the body).
    pub final_norm: DenseWeight,
}

/// Row cap of the draft module's own scratch pool (the target's is sized by
/// `ATLAS_LING_PREFILL_ROWS`; the drafter prefill chunks the prompt by this
/// many rows and the draft forward is one row). 128 keeps the drafter's
/// private scratch to a few tens of MB while a 2K-token prompt takes ~15
/// chunked kv-only passes.
pub(crate) const MTP_SCRATCH_ROWS: usize = 128;

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

/// Load the single NEXTN draft module when the checkpoint ships one
/// (`num_nextn_predict_layers = 1` → `config.num_mtp_modules == 1`).
/// `Ok(None)` = this checkpoint has no draft layer (speculative decode off).
///
/// Deliberately self-contained like `glm5::load_glm5_mtp_module`: it resolves
/// its own kernel handles (cheap id lookups) and one small shared scratch
/// for the draft body, sized by [`MTP_SCRATCH_ROWS`] — the target's scratch
/// stays with the target.
pub fn load_ling_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<LingMtpModule>> {
    if config.num_mtp_modules == 0 {
        return Ok(None);
    }
    ensure!(
        config.model_type == "bailing_moe_v3_vl",
        "ling: MTP module load on a non-bailing model_type {:?}",
        config.model_type
    );
    ensure!(
        config.num_mtp_modules == 1 && config.mtp_num_hidden_layers == 1,
        "ling: the GB10 draft path serves exactly one NEXTN module, config \
         has {} (mtp_num_hidden_layers {})",
        config.num_mtp_modules,
        config.mtp_num_hidden_layers
    );
    let spec = LingSpec::from_config(config)?;
    let idx = config.num_hidden_layers;
    let lp = format!("model.layers.{idx}");
    for name in mtp_tensors(&lp) {
        ensure!(
            store.contains(&name),
            "ling: NEXTN module advertised but {name} missing from the store"
        );
    }
    let kernels = Arc::new(crate::model::bailing_c1::LingKernels::resolve(gpu)?);
    let max_blocks = LingScratch::max_blocks(config.max_position_embeddings, 8);
    let inter = spec.dense_inter.max(spec.moe_inter.max(spec.shared_inter));
    let scratch = LingScratch::arc(
        gpu,
        MTP_SCRATCH_ROWS.min(spec.prefill_rows),
        spec.hidden,
        spec.experts,
        spec.top_k,
        inter,
        max_blocks,
    )?;
    // The SwiGLU clamp lists the config carries are length
    // num_hidden_layers (42) — the MTP layer has no entry, i.e. no clamp,
    // like every other list-shortfall in the family plan.
    let ffn = layers::load_ffn(
        store,
        &lp,
        crate::model::bailing_c1::LayerSpec {
            attention: LayerType::FullAttention,
            ffn: FfnKind::Moe,
            routed_swiglu_limit: 0,
            shared_swiglu_limit: 0,
        },
        &spec,
        gpu,
        &kernels,
        &scratch,
    )
    .context("ling: MTP FFN load")?;
    let body = LingMlaLayer::new(
        dense(store, &format!("{lp}.input_layernorm.weight"))?,
        dense(store, &format!("{lp}.post_attention_layernorm.weight"))?,
        layers::load_mla_weights(store, &lp, gpu)?,
        ffn,
        Arc::clone(&kernels),
        Arc::clone(&scratch),
        0,
        config,
    )?;
    let module = LingMtpModule {
        body,
        enorm: dense(store, &format!("{lp}.enorm.weight"))?,
        hnorm: dense(store, &format!("{lp}.hnorm.weight"))?,
        eh_proj: dense(store, &format!("{lp}.eh_proj.weight"))?,
        final_norm: dense(store, &format!("{lp}.final_layernorm.weight"))?,
    };
    tracing::info!(
        layer = idx,
        "ling: NEXTN draft module loaded (gated-MLA body + own 512E MoE + \
         eh_proj [{}, {}])",
        config.hidden_size,
        2 * config.hidden_size
    );
    Ok(Some(module))
}

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;
