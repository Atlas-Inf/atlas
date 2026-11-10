// SPDX-License-Identifier: AGPL-3.0-only

//! Ricursive Ripple weight loader (target model; the DFlash2 drafter loads
//! through `dflash_loader`). Strict: the checkpoint's formats are kept —
//! ModelOpt NVFP4 MLPs, BF16 attention, embedding and lm_head.

mod load_layers;

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::Qwen3AttentionLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense};

pub struct RicursiveWeightLoader;

impl ModelWeightLoader for RicursiveWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(store, config, gpu, layer_kv_dtypes)
    }

    /// The value-residual source (`x0`) and blend scratch, `[max_tokens, h]`
    /// BF16 each, shared by all layers.
    fn attach_layer_buffers(
        &self,
        layers: &mut [Box<dyn TransformerLayer>],
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        max_tokens: usize,
    ) -> Result<()> {
        let bytes = max_tokens * config.hidden_size * 2;
        let x0 = gpu.alloc(bytes).context("allocate value-residual x0")?;
        let blended = gpu.alloc(bytes).context("allocate value-residual blend")?;
        for (i, layer) in layers.iter_mut().enumerate() {
            let attn = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<Qwen3AttentionLayer>())
                .with_context(|| format!("ricursive layer {i} is not an attention layer"))?;
            let mut vr = attn
                .value_residual
                .with_context(|| format!("ricursive layer {i} has no value residual"))?;
            ensure!(vr.capture == (i == 0), "only layer 0 captures x0");
            vr.x0 = x0;
            vr.blended = blended;
            vr.max_rows = max_tokens;
            attn.set_value_residual(vr)?;
        }
        Ok(())
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.embed_tokens.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
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
