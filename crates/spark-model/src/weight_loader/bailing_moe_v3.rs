// SPDX-License-Identifier: AGPL-3.0-only

//! Ling 3.2 air (`bailing_moe_v3_vl`) checkpoint boundary.
//!
//! The parser and layout checker are intentionally landed before the CUDA
//! forward path.  Selecting this loader must therefore fail explicitly; it
//! must never enter a Qwen/DeepSeek compatibility loader by model-type alias.

mod plan;

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights};

pub struct BailingMoeV3WeightLoader;

fn header(store: &WeightStore) -> plan::Header {
    store.names().map(str::to_owned).collect()
}

fn not_implemented<T>() -> Result<T> {
    bail!("bailing_moe_v3_vl forward is not implemented; refusing to run the Ling 3.2 checkpoint")
}

impl ModelWeightLoader for BailingMoeV3WeightLoader {
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
        // Check the complete key layout before the explicit capability error.
        // This keeps a malformed Ling checkpoint distinguishable from the
        // current lack of a forward implementation.
        let _ = plan::plan(&header(store), config)?;
        not_implemented()
    }

    fn dead_namespaces(&self, _config: &ModelConfig) -> Vec<String> {
        vec!["model.visual.".to_string()]
    }

    fn load_embedding(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        not_implemented()
    }

    fn load_final_norm(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        not_implemented()
    }

    fn load_lm_head(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        not_implemented()
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        not_implemented()
    }
}
