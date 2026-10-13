// SPDX-License-Identifier: AGPL-3.0-only
//! Real constructor, store adoption and public teardown after the owned seam.
use super::tests::recording::Gpu;
use super::*;
use crate::{model::TransformerModel, traits::Model, weight_map::DenseWeight};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::{
    buffers::BufferArena,
    kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache},
};

#[test]
fn actual_model_adopts_only_filtered_store_and_teardown_frees_each_owner_once() {
    let gpu = Box::new(Gpu::default());
    let frees = gpu.frees.clone();
    let gate = gpu.alloc(8192).unwrap();
    let down = gpu.alloc(128).unwrap();
    let mut store = WeightStore::from_map(HashMap::from([
        (
            "gate".into(),
            WeightTensor {
                ptr: gate,
                shape: vec![4096],
                dtype: WeightDtype::BF16,
            },
        ),
        (
            "down".into(),
            WeightTensor {
                ptr: down,
                shape: vec![128],
                dtype: WeightDtype::UInt8,
            },
        ),
    ]));
    let log = RetirementLog::new(&store, gpu.as_ref()).unwrap();
    log.release_checkpoint(&store, "down", down, gpu.as_ref())
        .unwrap();
    // Complete the owned-map handoff before the constructor consumes gpu.
    log.finish().rebuild(&mut store, gpu.as_ref()).unwrap();
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.model_type = "qwen2".into();
    cfg.hidden_size = 128;
    cfg.vocab_size = 8;
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![LayerType::FullAttention];
    cfg.num_attention_heads = 1;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 128;
    cfg.intermediate_size = 128;
    cfg.moe_intermediate_size = 128;
    cfg.shared_expert_intermediate_size = 128;
    cfg.num_experts = 2;
    cfg.num_experts_per_tok = 1;
    cfg.linear_num_key_heads = 0;
    cfg.linear_num_value_heads = 0;
    cfg.num_mtp_modules = 0;
    let buffers = BufferArena::new(&cfg, 5, 64, 16, 1, gpu.as_ref()).unwrap();
    let arena_norm = buffers.norm_output();
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
    let dense = DenseWeight { weight: gate };
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
        vec![],
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
    model.teardown().unwrap();
    {
        let recorded = frees.lock();
        assert!(
            recorded.iter().position(|&p| p == arena_norm).unwrap()
                < recorded.iter().position(|&p| p == gate).unwrap(),
            "arena released before adopted checkpoint store"
        );
    }
    for ptr in [gate, down] {
        assert_eq!(frees.lock().iter().filter(|&&p| p == ptr).count(), 1);
    }
    let count = frees.lock().len();
    model.teardown().unwrap();
    assert_eq!(
        frees.lock().len(),
        count,
        "actual repeated teardown is idempotent"
    );
}
