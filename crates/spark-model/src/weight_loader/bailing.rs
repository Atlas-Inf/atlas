// SPDX-License-Identifier: AGPL-3.0-only

//! Bailing-Moe-V3-VL weight loader — Ling-3.2-air (NVFP4).
//!
//! This PR is the config parse ([`config`]), the exact safetensors name
//! mapping ([`names`] + [`plan`], validated once per load), the MTP layer
//! map ([`mtp`]) and the text-only vision skip ([`vision`]). Decode layers
//! themselves are NOT constructed here yet: the Ling KDA/gated-MLA/MoE
//! kernels are being landed by the sibling kernels PRs
//! (`kernels/gb10/ling-3.2-air/nvfp4/{kda,gated_mla,moe_decode,ling_mtp_eh}.cu`),
//! and constructing [`crate::layer::TransformerLayer`]s for them is the
//! integration step. Until then [`ModelWeightLoader::load_layers`] builds
//! and validates the whole-checkpoint plan and fails with a loud pointer at
//! that wiring, rather than silently misloading.
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
mod mtp;
mod names;
mod plan;
mod vision;

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

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
        _gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let plan = plan::build(store, config)?;
        tracing::info!("bailing: checkpoint plan validated ({})", plan.summary());
        bail!(
            "bailing: Ling-3.2-air decode kernels are not wired into the layer \
             factory yet ({}); the checkpoint plan validated — see \
             weight_loader/bailing.rs for the integration seam",
            plan.summary()
        );
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
