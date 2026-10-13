// SPDX-License-Identifier: AGPL-3.0-only

//! BF16 and GLM `fp8_g128` latent arms of `write_kv_cache` (GLM latent
//! round-trip and K-only latent write).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_kv_cache_bf16_latent(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        stream: u64,
    ) -> Result<()> {
        // A latent shard writes only the rows whose block this rank stores,
        // at its local slots (V aliases K).
        let (k_pool, v_pool, slot) = if kv_cache.latent_shard().is_some() {
            let pool = kv_cache.latent_pool_ptr(self.attn_layer_idx);
            let local = self.glm_shard_local_slots(kv_cache, gpu, slot, num_tokens, stream)?;
            (pool, pool, local)
        } else {
            (
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                slot,
            )
        };
        match self.kv_dtype {
            KvCacheDtype::Bf16 => {
                ops::reshape_and_cache(
                    gpu,
                    self.reshape_cache_k,
                    k,
                    v,
                    k_pool,
                    v_pool,
                    slot,
                    num_tokens,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    key_stride,
                    value_stride,
                    kv_cache.cache_stride() as u64,
                    stream,
                )?;
                // GLM's V aliases K, so rounding the K side covers both.
                if self.glm_latent_qdq_k.0 != 0 && num_kv_heads == 1 && head_dim == 512 {
                    ops::glm_latent_qdq_fp8g128(
                        gpu,
                        self.glm_latent_qdq_k,
                        k_pool,
                        slot,
                        num_tokens,
                        stream,
                    )?;
                }
                Ok(())
            }
            // GLM latent: V aliases K, so only the K side is written.
            KvCacheDtype::Fp8G128 => {
                anyhow::ensure!(
                    num_kv_heads == 1 && head_dim == 512,
                    "fp8_g128 KV cache stores the GLM NoPE-512 latent only"
                );
                ops::glm_latent_cache_write_fp8g128(
                    gpu,
                    self.reshape_cache_k,
                    k,
                    k_pool,
                    slot,
                    num_tokens,
                    block_size,
                    key_stride,
                    stream,
                )
            }
            _ => unreachable!(),
        }
    }
}
