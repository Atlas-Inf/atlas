// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 weight loader: the 1017 release as published, nothing
//! requantized. The layers are `layers::deepseek_v41`.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense, dense_auto};

pub struct DeepSeekV41WeightLoader;

impl ModelWeightLoader for DeepSeekV41WeightLoader {
    /// Attention runs whole on every rank; the routed experts are what is
    /// split, so more than one Spark is expert-parallel (`--ep-size`).
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        crate::layers::deepseek_v41::layer::load_layers(store, config, gpu)
    }

    /// The vision tower is not served yet, the draft stages only where
    /// asked for (`ATLAS_DSV41_DRAFT`), and a debug cut of the model
    /// (`ATLAS_DSV41_DEBUG_LAYERS`) reads only its first layers.
    fn dead_namespaces(&self, config: &ModelConfig) -> Vec<String> {
        let mut dead: Vec<String> = ["vision.", "aligner.", "image_"].map(String::from).to_vec();
        if !crate::layers::deepseek_v41::draft_seq::requested() {
            dead.push("mtp.".to_string());
        }
        dead.extend((config.num_hidden_layers..128).map(|layer| format!("layers.{layer}.")));
        dead
    }

    /// The layers keep their attention cache themselves, per sequence
    /// (`layers::deepseek_v41::SeqState`), and nothing reads the paged pool,
    /// so it is given the narrowest key there is.
    fn kv_layer_dims(&self, config: &ModelConfig) -> Vec<(usize, usize)> {
        vec![(1, 2); config.num_hidden_layers]
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "embed.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // The release's norm weights are the scale itself, and the model
        // applies them with `rms_norm_vanilla`.
        dense_auto(store, "norm.weight", gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "head.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
