// SPDX-License-Identifier: AGPL-3.0-only

//! Ling 3.2 air (`bailing_moe_v3_vl`) checkpoint boundary.
//!
//! Both supported checkpoints use the same text and vision geometry. The
//! original routed FP8 tensors remain in their immutable safetensors shards.

mod plan;

use crate::layers::ling_mla::{LingMlaConfig, alloc_rope_inv_freq};
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::ling_paging::LingPagingReader;
use spark_runtime::weights::{WeightDtype, WeightStore};
use std::sync::{Arc, Mutex};

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::VisionTower;
use crate::weight_map::{DenseWeight, MtpWeights};

pub struct BailingMoeV3WeightLoader;

fn header(store: &WeightStore) -> plan::Header {
    store
        .names()
        .chain(store.deferred_names())
        .map(str::to_owned)
        .collect()
}

fn bf16(store: &WeightStore, name: &str, shape: &[usize]) -> Result<DenseWeight> {
    let tensor = store.get(name)?;
    ensure!(
        tensor.dtype == WeightDtype::BF16 && tensor.shape == shape,
        "{name} must be BF16 with shape {shape:?}"
    );
    Ok(DenseWeight { weight: tensor.ptr })
}

impl ModelWeightLoader for BailingMoeV3WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn slab_pack_weights(&self, config: &ModelConfig) -> bool {
        // The A16 pointer tables borrow the checkpoint storage throughout
        // inference, including the small per-expert group/global scales.
        config
            .quantization_config
            .as_ref()
            .is_some_and(|q| q.quant_method == "compressed-tensors")
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let _ = plan::plan(&header(store), config)?;
        ensure!(
            layer_kv_dtypes
                .iter()
                .all(|dtype| *dtype == KvCacheDtype::Bf16),
            "Ling attention requires a BF16 KV cache"
        );
        let reader = Arc::new(Mutex::new(LingPagingReader::new(
            if config.mtp_num_hidden_layers == 0
                || config
                    .quantization_config
                    .as_ref()
                    .is_some_and(|q| q.quant_method == "compressed-tensors")
            {
                50
            } else {
                13
            },
        )?));
        let frequencies = alloc_rope_inv_freq(gpu, LingMlaConfig::default())?;
        crate::layers::ling_decoder::load_ling_layers(store, config, gpu, reader, frequencies)
    }

    fn kv_layer_dims(&self, config: &ModelConfig) -> Vec<(usize, usize)> {
        vec![
            (
                config.num_attention_heads,
                config.qk_nope_head_dim + config.qk_rope_head_dim / config.num_attention_heads
            );
            config.num_attention_layers()
        ]
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bf16(
            store,
            "model.word_embeddings.weight",
            &[config.vocab_size, config.hidden_size],
        )
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bf16(store, "model.norm.weight", &[config.hidden_size])
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bf16(
            store,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )
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
    ) -> Result<Option<VisionTower>> {
        crate::layers::ling_vision::load_ling_vision(store, config, gpu)
    }
}
