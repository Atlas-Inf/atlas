// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 Flash loader (hybrid KDA/MLA + mHC + NVFP4 MoE).

mod components;
mod dense;
mod expert_tp;
mod layers;
mod mla_twins;
pub(crate) mod retirement;
mod tp;
mod vision;

#[cfg(test)]
mod nvidia_dense_tests;

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense, dense_auto};

pub struct Glm5WeightLoader;

impl ModelWeightLoader for Glm5WeightLoader {
    fn supports_tp(&self) -> bool {
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        layers::load_all(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(
            store,
            &format!("{}.embed_tokens.weight", config.weight_prefix),
            gpu,
        )
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, &format!("{}.norm.weight", config.weight_prefix), gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        crate::model::glm_k3_head::initialize_from_weight(
            gpu,
            config,
            store
                .get("lm_head.weight")
                .context("GLM-5: missing lm_head.weight")?,
        )?;
        dense(store, "lm_head.weight").context("GLM-5: missing lm_head.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }

    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        vision::load(store, config, gpu)
    }
}
