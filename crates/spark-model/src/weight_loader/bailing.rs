// SPDX-License-Identifier: AGPL-3.0-only

//! Bailing-Moe-V3-VL weight loader — Ling-3.2-air (NVFP4).
//!
//! The config parse ([`config`]), the exact safetensors name mapping
//! ([`names`] + [`plan`], validated once per load), the MTP layer map
//! ([`mtp`]) and the text-only vision skip ([`vision`]) come from PR #115.
//! The decode-path wiring landed with the family driver campaign slice:
//! [`load_layers`] now builds the layer stack via [`layers`] on top of
//! `crate::model::bailing_c1` (decode-only, text-only, conservative;
//! see that module for the step sequence, module-to-kernel table and the
//! deliberate seams: MTP drafting, KDA slot-pool decode, graphs, vision).
//!
//! NVFP4 expert loading reuses the GLM-5 precedent
//! (`weight_loader/glm5/fp4l.rs` + `weight_map` primitives): routed experts
//! are compressed-tensors `nvfp4-pack-quantized`
//! (`{weight_packed, weight_scale, weight_global_scale,
//! input_global_scale}` per projection) and load through
//! `weight_map::{detect_nvfp4_variant, quantized_any}` exactly as
//! `glm5/ffn.rs` loads `mlp.experts.<e>.{gate,up,down}_proj`. The W4A4
//! activation-side `input_global_scale` members are consumed by the MoE
//! decode kernels (off-disk sidecar wiring lands with them). The dequant
//! kernels live in `kernels/gb10/common/dequant_nvfp4_bf16.cu` and are NOT
//! re-implemented here.

mod config;
mod gate;
mod layers;
mod mtp;
mod names;
mod plan;
mod vision;

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

/// The gated-MLA layers write BF16 pool rows this phase; refuse a
/// quantized-KV pool loudly at load instead of corrupting the cache.
fn ensure_kv_bf16(idx: usize, dtype: KvCacheDtype) -> Result<()> {
    let _ = idx;
    ensure!(
        dtype == KvCacheDtype::Bf16,
        "ling: this phase writes BF16 paged K rows; pass --kv-dtype bf16"
    );
    Ok(())
}

use crate::layer::TransformerLayer;
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{DenseWeight, MtpWeights, dense};

/// Bailing-Moe-V3-VL (`model_type = bailing_moe_v3_vl`), dispatched from
/// [`crate::factory::loader_for_config`].
pub struct BailingWeightLoader;

impl ModelWeightLoader for BailingWeightLoader {
    /// Not yet: no TP slice wiring exists for the Ling layer shapes this
    /// phase (the GLM-5 loader's `expert_tp`/`tp.rs` are the pattern to port
    /// when TP lands). Honest `false` fails fast at startup on
    /// `--tp-size > 1` instead of silently loading replicated weights.
    fn supports_tp(&self) -> bool {
        false
    }

    /// The ViT tower (`model.visual.*`) and the projector
    /// (`linear_proj.*`, held by `BailingMoeV3VLForConditionalGeneration`
    /// OUTSIDE `model.`) are dead weight in the text-only phase — skipped
    /// at upload. See [`vision`] for the extension seam.
    fn dead_namespaces(&self, _config: &ModelConfig) -> Vec<String> {
        vision::dead_namespaces()
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let plan = plan::build(store, config)?;
        tracing::info!("bailing: checkpoint plan validated ({})", plan.summary());
        // Text-only, conservative: the gated-MLA layers write BF16 pool
        // rows; an FP8/KV-quantized pool is out of this phase.
        for (idx, dtype) in layer_kv_dtypes.iter().enumerate() {
            ensure_kv_bf16(idx, *dtype)?;
        }
        layers::load_all(store, config, gpu, layer_kv_dtypes, &plan)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, names::WORD_EMBEDDINGS)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, names::FINAL_NORM)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, names::LM_HEAD)
    }

    /// The Ling NEXTN layer (`model.layers.42.*`) does not fit the
    /// Qwen-shaped [`MtpWeights`] (gated-MLA attention + its own 512E MoE +
    /// `eh_proj`), same situation as GLM-5's dedicated `Glm5MtpModule`.
    /// The draft-side MTP path lands with the `ling_mtp_eh.cu` kernels;
    /// [`mtp`] carries the name mapping for it.
    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
