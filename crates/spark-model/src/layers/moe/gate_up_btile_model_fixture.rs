// SPDX-License-Identifier: AGPL-3.0-only
//! Test-only construction through real publication, concrete FFN and model.
use super::*;
use crate::{
    layer::TransformerLayer,
    layers::{FfnComponent, qwen3_attention::Qwen3AttentionLayer},
    model::TransformerModel,
    weight_map::{AttentionWeights, DenseWeight, QuantizedWeight},
};
use spark_runtime::{
    buffers::BufferArena,
    gpu::DevicePtr,
    kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache},
};
use std::sync::{Arc, Mutex};

pub(crate) struct Fixture {
    pub model: TransformerModel,
    pub originals: Vec<DevicePtr>,
    pub history: History,
}
pub(crate) struct History(Arc<Mutex<Vec<recording::Event>>>);
impl History {
    pub(crate) fn frees(&self, ptr: Option<DevicePtr>) -> usize {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|e| matches!(e,recording::Event::Free(p) if ptr.is_none_or(|v|v==*p)))
            .count()
    }
    pub(crate) fn clear(&self) {
        self.0.lock().unwrap().clear();
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.0.lock().unwrap().is_empty()
    }
}
pub(crate) fn model_fixture() -> Fixture {
    let gpu = Box::new(recording::Gpu::new());
    let events = gpu.events.clone();
    let (mut store, mut cfg, mut moe) = resident_tests::setup(&gpu, 0);
    let originals = store.names().map(|n| store.get(n).unwrap().ptr).collect();
    let dense = DenseWeight {
        weight: moe.weights.gate.weight,
    };
    let log =
        crate::weight_loader::glm5::retirement::RetirementLog::new(&store, gpu.as_ref()).unwrap();
    let mut session = load::BTileLoadSession::new(gpu.as_ref(), &cfg, 77).unwrap();
    session.prepare(&mut moe, &log, &cfg, 0).unwrap();
    session.close().unwrap();
    log.finish().rebuild(&mut store, gpu.as_ref()).unwrap();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![atlas_core::config::LayerType::FullAttention];
    cfg.linear_num_key_heads = 0;
    cfg.linear_num_value_heads = 0;
    cfg.num_mtp_modules = 0;
    cfg.vocab_size = 8;
    cfg.kv_lora_rank = 512;
    cfg.qk_rope_head_dim = 0;
    let attn = AttentionWeights {
        q_proj: dense,
        k_proj: dense,
        v_proj: dense,
        o_proj: QuantizedWeight::null(),
        q_norm: dense,
        k_norm: dense,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let concrete = Qwen3AttentionLayer::new_ungated(
        dense,
        attn,
        dense,
        FfnComponent::Moe(moe),
        0,
        None,
        None,
        None,
        gpu.as_ref(),
        KvCacheDtype::Bf16,
        0,
        &cfg,
    )
    .unwrap();
    let mut layers: Vec<Box<dyn TransformerLayer>> = vec![Box::new(concrete)];
    let buffers = BufferArena::new(&cfg, 5, 64, 16, 1, gpu.as_ref()).unwrap();
    crate::layers::moe::bind_resident_btile_arenas(
        &cfg,
        &store,
        gpu.as_ref(),
        &mut layers,
        &buffers,
    )
    .unwrap();
    let kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 128,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        4,
        gpu.as_ref(),
    )
    .unwrap();
    let ssm_pools = crate::model::ssm_pools::SsmPools::new(
        &cfg,
        1,
        false,
        false,
        true,
        false,
        4,
        1,
        gpu.as_ref(),
    )
    .unwrap();
    let mut model = TransformerModel::new(
        cfg,
        dense,
        dense,
        dense,
        None,
        None,
        None,
        layers,
        buffers,
        kv,
        vec![],
        gpu,
        64,
        1,
        crate::layers::MtpQuantization::Bf16,
        false,
        Box::new(spark_runtime::prefix_cache::NoPrefixCaching),
        8,
        None,
        false,
        None,
        1,
        16,
        ssm_pools,
    )
    .unwrap();
    model.adopt_weight_store(store);
    Fixture {
        model,
        originals,
        history: History(events),
    }
}
