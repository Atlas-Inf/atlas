// SPDX-License-Identifier: AGPL-3.0-only
//! Actual layer dispatch; recorded kernels do not establish CUDA numerics.
use super::*;
use crate::layer::ForwardContext;
use crate::layers::{
    FfnComponent,
    qwen3_attention::{GlmIndexerWeights, MlaWeights},
};
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{GpuBackend, KernelHandle, mock::MockGpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, SparseIndexCacheConfig};
#[path = "mla_long_context_test_gpu.rs"]
mod gpu;
use gpu::TestGpu;
#[path = "mla_owner_batch_tests.rs"]
mod owner_batch_tests;
#[path = "mla_query_dispatch_tests.rs"]
mod query_dispatch_tests;
#[path = "mla_shard_tests.rs"]
mod shard_tests;
#[path = "mla_split_context_tests.rs"]
mod split_tests;
#[path = "mla_write_floor_tests.rs"]
mod write_floor_tests;
fn fixture(run: impl FnOnce(&TestGpu, &atlas_core::config::ModelConfig, &Qwen3AttentionLayer)) {
    fixture_with(KvCacheDtype::Bf16, |gpu, config, layer| {
        run(gpu, config, layer)
    })
}
/// [`fixture`] on a `kv_dtype` cache, with the layer open to edits.
fn fixture_with(
    kv_dtype: KvCacheDtype,
    run: impl FnOnce(&TestGpu, &atlas_core::config::ModelConfig, &mut Qwen3AttentionLayer),
) {
    let gpu = TestGpu::default();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.num_hidden_layers = 1;
    config.layer_types = vec![atlas_core::config::LayerType::FullAttention];
    config.num_attention_heads = 32;
    config.num_key_value_heads = 32;
    config.head_dim = 256;
    config.q_lora_rank = 1536;
    config.kv_lora_rank = 512;
    config.qk_nope_head_dim = 256;
    config.qk_rope_head_dim = 0;
    config.v_head_dim = 256;
    config.index_n_heads = 32;
    config.index_head_dim = 128;
    config.index_topk = 2048;
    config.index_kpool = 4;
    config.max_position_embeddings = 32768;
    config.tp_world_size = 2;
    config.ep_world_size = 2;
    // The mock never dereferences weight data, but the query/O plans inspect
    // the resident matrix spans for alias safety. Give each fake weight a
    // credible, aligned logical range without allocating hundreds of MB of
    // backing storage in the test GPU.
    let mut next_weight = 0x1_0000_0000u64;
    let mut dense = |bytes: usize| {
        let bytes = bytes.max(2);
        let ptr = DevicePtr(next_weight);
        next_weight += ((bytes + 255) & !255) as u64;
        DenseWeight { weight: ptr }
    };
    let absent = DenseWeight {
        weight: DevicePtr::NULL,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        dense(8192),
        AttentionWeights {
            q_proj: dense(32 * 256 * 4096 * 2),
            k_proj: dense(32 * 256 * 4096 * 2),
            v_proj: dense(32 * 256 * 4096 * 2),
            o_proj: QuantizedWeight::null(),
            q_norm: dense(512),
            k_norm: dense(512),
            q_norm_full: None,
            k_norm_full: None,
            k_scale: 1.0,
            v_scale: 1.0,
        },
        dense(8192),
        FfnComponent::None,
        0,
        None,
        None,
        None,
        &gpu,
        kv_dtype,
        0,
        &config,
    )
    .unwrap();
    layer.set_mla_weights(MlaWeights {
        wq_a: dense(1536 * 4096 * 2),
        wq_a_nvfp4: None,
        wq_a_fp8: None,
        wq_b: dense(32 * 256 * 1536 * 2),
        wq_b_nvfp4: None,
        wq_b_fp8: None,
        q_a_norm: dense(1536 * 2),
        wkv_a: dense(512 * 4096 * 2),
        wkv_a_nvfp4: None,
        wkv_a_fp8: None,
        wkv_b: dense(32 * 512 * 512 * 2),
        kv_a_norm: dense(512 * 2),
        wkv_a_rope: absent,
        wkv_a_merged: dense(512 * 4096 * 2),
        // Comparison mode validates the real operand range, not a two-byte stub.
        wo: dense(4096 * 32 * 256 * 2),
        wo_nvfp4: None,
        wo_a: absent,
        wo_a_nvfp4: None,
        wo_a_fp8: None,
        wo_b: absent,
        wo_b_nvfp4: None,
        wo_b_fp8: None,
        w_uk_t: dense(32 * 256 * 512 * 2),
        w_uv: dense(32 * 512 * 256 * 2),
        wq_b_rope: absent,
        w_qk_absorbed: absent,
        w_uk_block_diag: absent,
        w_uv_block_diag: absent,
        yarn_inv_freq: DevicePtr::NULL,
        main_inv_freq: DevicePtr::NULL,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        o_lora_rank: 0,
        nope: 256,
        rope: 0,
        v_dim: 256,
        glm_indexer: Some(GlmIndexerWeights {
            wq_b: dense(2),
            wk: dense(2),
            weights_proj: dense(2),
            kpool_gate: dense(2),
            kpool_ape: dense(2),
            k_norm_weight: dense(2),
            k_norm_bias: dense(2),
        }),
        compressor: None,
        attn_sink: DevicePtr::NULL,
    });

    layer.dense_gemv_k = KernelHandle(806);
    layer.dense_gemv_batchm_k = KernelHandle(807);
    layer.mla_batched_gemv_k = KernelHandle(808);
    layer.mla_cache_assemble_k = KernelHandle(820);
    layer.reshape_cache_k = KernelHandle(821);
    layer.glm_index_layernorm_k = KernelHandle(800);
    layer.glm_index_tail_write_k = KernelHandle(801);
    layer.glm_index_kpool_finalize_k = KernelHandle(802);
    layer.glm_index_logits_decode_k = KernelHandle(812);
    layer.glm_index_topk_expand_k = KernelHandle(803);
    layer.glm_sparse_attn_decode_k = KernelHandle(804);
    run(&gpu, &config, &mut layer);
}

#[path = "mla_causal_dispatch_tests.rs"]
mod causal_dispatch_tests;
