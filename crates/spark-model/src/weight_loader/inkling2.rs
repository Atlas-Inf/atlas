// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking Machines Inkling 2 Mini (`inkling2_loop`) weight loader.
//!
//! Strict: the checkpoint's formats are kept (see `plan`). The engine runs 48
//! visits over 24 physical layers; visits of the same physical layer will share
//! one set of weight handles, each with its own KV and conv state.
//!
//! Work in progress: the layout is checked at load, but the layers (attention
//! with relative bias and per-token sinks, block attention residuals, short
//! convs, softcapped SwiGLU MoE, the loop's message gates) are not built yet,
//! so `load_layers` refuses instead of serving a wrong model.

mod plan;

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense};

pub struct Inkling2WeightLoader;

/// The (dtype, shape) of every tensor the store holds.
fn header(store: &WeightStore) -> Result<plan::Header> {
    store
        .names()
        .map(|n| {
            let t = store.get(n)?;
            Ok((n.to_string(), (t.dtype, t.shape.clone())))
        })
        .collect()
}

impl ModelWeightLoader for Inkling2WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let plan = plan::plan(&header(store)?, config)?;
        tracing::info!(
            "Inkling2 checkpoint layout ok: {} physical layers, {} visits",
            plan.layers.len(),
            config.num_hidden_layers
        );
        bail!("inkling2_loop layers are not implemented yet (SparkInkling port in progress)")
    }

    /// The vision tower is not ported yet; text serving leaves it on disk.
    fn dead_namespaces(&self, _config: &ModelConfig) -> Vec<String> {
        vec!["model.visual.".to_string()]
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.llm.embed.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.llm.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.llm.unembed.weight")
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
