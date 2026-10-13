// SPDX-License-Identifier: AGPL-3.0-only

//! Validated GLM NoPE cache geometry and current-layout byte accounting.

use anyhow::{Result, ensure};
use spark_runtime::kv_cache::{KvCacheConfig, SparseIndexCacheConfig, TailSlotPlan};

/// Only constructible for the GLM shape supported by the existing kernels.
/// This describes geometry, not a new KV dtype or allocation policy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GlmMlaShape {
    _validated: (),
}

impl GlmMlaShape {
    const LATENT_WIDTH: usize = 512;
    const INDEX_WIDTH: usize = 128;
    const TOKENS_PER_POOL: usize = 4;

    pub(crate) fn new(kv_lora_rank: usize, rope_dim: usize) -> Result<Self> {
        ensure!(
            kv_lora_rank == Self::LATENT_WIDTH && rope_dim == 0,
            "GLM MLA cache requires latent rank512 and zero RoPE"
        );
        Ok(Self { _validated: () })
    }

    pub(crate) fn for_model(model: &str, rank: usize, rope: usize) -> Result<Option<Self>> {
        (model == "glm5_next")
            .then(|| Self::new(rank, rope))
            .transpose()
    }

    pub(crate) fn num_kv_heads(self) -> usize {
        1
    }
    pub(crate) fn head_dim(self) -> usize {
        Self::LATENT_WIDTH
    }

    /// Requirement of the BF16 attention handle, independently of the dtype
    /// selected for allocation. Existing dtype dispatch/eligibility is unchanged.
    pub(crate) fn bf16_decode_module(self) -> &'static str {
        "paged_decode_attn_512"
    }

    pub(crate) fn bf16_index(self, pool: usize, head_dim: usize) -> Result<SparseIndexCacheConfig> {
        self.validate_index(pool, head_dim)?;
        Ok(SparseIndexCacheConfig::bf16(
            Self::TOKENS_PER_POOL,
            Self::INDEX_WIDTH,
        ))
    }

    fn validate_index(self, pool: usize, head_dim: usize) -> Result<()> {
        ensure!(
            pool == Self::TOKENS_PER_POOL && head_dim == Self::INDEX_WIDTH,
            "GLM semantic index requires four-token pools and width128"
        );
        Ok(())
    }
}

/// Checked accounting for the CURRENT ownership: distinct K/V allocations,
/// plus pooled keys/scales and a full-block raw key/gate tail when attached.
/// No pointers are owned or aliased by this plan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GlmCachePlan {
    token_block_size: usize,
    block_bytes_all_layers: usize,
    /// Pool-size-independent bytes (slot-mapped index tails).
    fixed_bytes: usize,
}

impl GlmCachePlan {
    pub(crate) fn new(
        shape: GlmMlaShape,
        config: &KvCacheConfig,
        index: Option<SparseIndexCacheConfig>,
    ) -> Result<Self> {
        ensure!(
            config.block_size > 0 && u32::try_from(config.block_size).is_ok(),
            "GLM cache token block size must fit positive u32"
        );
        ensure!(
            config.num_layers > 0,
            "GLM cache needs at least one attention layer"
        );
        ensure!(
            config.num_kv_heads == shape.num_kv_heads() && config.head_dim == shape.head_dim(),
            "GLM cache allocation must match one-head NoPE512 geometry"
        );
        // Validate an upper bound BEFORE invoking the runtime's existing
        // unchecked dtype formulas, including its per-layer summation. BF16
        // is the widest current K/V format; narrower/mixed formats stay intact.
        config
            .block_size
            .checked_mul(shape.head_dim())
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_mul(config.num_layers))
            .ok_or_else(|| anyhow::anyhow!("GLM K/V block byte count overflow"))?;
        for layer in 0..config.num_layers {
            ensure!(
                config.dims_for_layer(layer) == (shape.num_kv_heads(), shape.head_dim()),
                "GLM cache layer{layer} geometry disagrees with NoPE512"
            );
        }
        let index_bytes = if let Some(index) = index {
            shape.validate_index(index.tokens_per_pool, index.head_dim)?;
            // Width and pool size are now fixed and the block fits u32; on
            // smaller hosts check the largest index component before its API.
            config
                .block_size
                .checked_mul(index.head_dim)
                .and_then(|n| n.checked_mul(8))
                .ok_or_else(|| anyhow::anyhow!("GLM semantic-index block byte count overflow"))?;
            index
                .block_bytes(config.block_size)?
                .checked_mul(config.num_layers)
                .ok_or_else(|| anyhow::anyhow!("GLM semantic-index layer byte count overflow"))?
        } else {
            0
        };
        let total = config
            .block_bytes_kv_all_layers()
            .checked_add(index_bytes)
            .ok_or_else(|| anyhow::anyhow!("GLM combined cache block byte count overflow"))?;
        Ok(Self {
            token_block_size: config.block_size,
            block_bytes_all_layers: total,
            fixed_bytes: 0,
        })
    }

    /// The plan when index tails are lent per `plan`
    /// (`PagedKvCache::attach_sparse_index_with_tail_slots`): each block
    /// keeps only a `u32` map entry, and the slot pool is a fixed cost.
    pub(crate) fn slotted_tails(
        self,
        config: &KvCacheConfig,
        index: SparseIndexCacheConfig,
        plan: TailSlotPlan,
    ) -> Self {
        let tail = index.tail_block_bytes(config.block_size) * config.num_layers;
        Self {
            block_bytes_all_layers: self.block_bytes_all_layers - tail + std::mem::size_of::<u32>(),
            fixed_bytes: self.fixed_bytes + plan.capacity() * tail,
            ..self
        }
    }

    /// The plan when every layer's V side aliases its K storage
    /// (`PagedKvCache::new_with_v_alias`): the V bytes are not allocated.
    pub(crate) fn aliased_v(self, config: &KvCacheConfig) -> Self {
        let v_bytes: usize = (0..config.num_layers)
            .map(|layer| config.v_block_bytes_for_layer(layer))
            .sum();
        Self {
            block_bytes_all_layers: self.block_bytes_all_layers - v_bytes,
            ..self
        }
    }

    pub(crate) fn block_bytes_all_layers(self) -> usize {
        self.block_bytes_all_layers
    }

    pub(crate) fn num_blocks_for_budget(self, available_bytes: usize) -> usize {
        available_bytes.saturating_sub(self.fixed_bytes) / self.block_bytes_all_layers
    }

    pub(crate) fn bytes_for_blocks(self, blocks: usize) -> Result<usize> {
        ensure!(
            blocks > 0 && u32::try_from(blocks).is_ok(),
            "GLM physical block count must fit positive u32"
        );
        let token_slots = blocks
            .checked_mul(self.token_block_size)
            .ok_or_else(|| anyhow::anyhow!("GLM physical slot capacity overflow"))?;
        ensure!(
            i64::try_from(token_slots).is_ok(),
            "GLM physical slots exceed i64 metadata"
        );
        blocks
            .checked_mul(self.block_bytes_all_layers)
            .and_then(|n| n.checked_add(self.fixed_bytes))
            .ok_or_else(|| anyhow::anyhow!("GLM cache allocation byte count overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::kv_cache::{
        KvCacheConfig, KvCacheDtype, PagedKvCache, SparseIndexCacheConfig,
    };

    fn config(layers: usize, dtype: KvCacheDtype) -> KvCacheConfig {
        KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 512,
            num_layers: layers,
            dtype,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        }
    }

    #[test]
    fn slotted_tails_move_tail_bytes_from_blocks_to_a_fixed_pool() {
        let shape = GlmMlaShape::new(512, 0).unwrap();
        let cfg = config(11, KvCacheDtype::Bf16);
        let index = shape.bf16_index(4, 128).unwrap();
        let plan = GlmCachePlan::new(shape, &cfg, Some(index))
            .unwrap()
            .aliased_v(&cfg);
        let slots = TailSlotPlan {
            lag_blocks: 258,
            sequences: 5,
        };
        let slotted = plan.slotted_tails(&cfg, index, slots);
        // 8 KiB raw key+gate tail per layer leaves each block; a u32 map
        // entry joins it; 5 × 260 lent tails become fixed.
        assert_eq!(
            slotted.block_bytes_all_layers(),
            plan.block_bytes_all_layers() - 11 * 8192 + 4
        );
        assert_eq!(
            slotted.bytes_for_blocks(1).unwrap(),
            slotted.block_bytes_all_layers() + 1300 * 11 * 8192
        );
        assert_eq!(slotted.num_blocks_for_budget(1300 * 11 * 8192), 0);
        assert_eq!(
            slotted.num_blocks_for_budget(1300 * 11 * 8192 + 3 * slotted.block_bytes_all_layers()),
            3
        );
    }

    #[test]
    fn aliased_v_drops_exactly_the_v_bytes() {
        let shape = GlmMlaShape::new(512, 0).unwrap();
        let cfg = config(11, KvCacheDtype::Bf16);
        let plan = GlmCachePlan::new(shape, &cfg, None).unwrap();
        // BF16 NoPE512, 16-token blocks: 16 KiB per side per layer.
        assert_eq!(plan.block_bytes_all_layers(), 11 * 2 * 16384);
        assert_eq!(plan.aliased_v(&cfg).block_bytes_all_layers(), 11 * 16384);
    }

    #[test]
    fn shape_is_the_shared_source_for_binding_and_allocation() {
        let shape = GlmMlaShape::for_model("glm5_next", 512, 0)
            .unwrap()
            .unwrap();
        assert_eq!((shape.num_kv_heads(), shape.head_dim()), (1, 512));
        assert_eq!(shape.bf16_decode_module(), "paged_decode_attn_512");
        for (rank, rope) in [(0, 0), (256, 0), (512, 64), (576, 0), (usize::MAX, 1)] {
            assert!(GlmMlaShape::for_model("glm5_next", rank, rope).is_err());
        }
        assert!(
            GlmMlaShape::for_model("deepseek_v4", 512, 64)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn accounting_preserves_bf16_fp8_and_mixed_dtype_layouts() {
        let shape = GlmMlaShape::new(512, 0).unwrap();
        let index_cfg = SparseIndexCacheConfig::bf16(4, 128);
        let index = Some(index_cfg);
        for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Fp8] {
            let cfg = config(11, dtype);
            let plan = GlmCachePlan::new(shape, &cfg, index).unwrap();
            let legacy = cfg.block_bytes_kv_all_layers() + index_cfg.block_bytes(16).unwrap() * 11;
            assert_eq!(plan.block_bytes_all_layers(), legacy);
            assert_eq!(plan.num_blocks_for_budget(legacy * 7 + legacy - 1), 7);
            assert_eq!(plan.bytes_for_blocks(2).unwrap(), legacy * 2);
            if dtype == KvCacheDtype::Bf16 {
                assert_eq!(legacy, 461824);
            }
        }
        let mut cfg = config(2, KvCacheDtype::Fp8);
        cfg.layer_dtypes = vec![KvCacheDtype::Bf16];
        assert_eq!(
            GlmCachePlan::new(shape, &cfg, index)
                .unwrap()
                .block_bytes_all_layers(),
            67584
        );
        cfg.layer_dims = vec![(1, 512)];
        assert_eq!(
            GlmCachePlan::new(shape, &cfg, index)
                .unwrap()
                .block_bytes_all_layers(),
            67584
        );
        let scaled_index = SparseIndexCacheConfig {
            dtype: spark_runtime::kv_cache::SparseIndexCacheDtype::Fp8E4m3Scaled,
            ..index_cfg
        };
        let cfg = config(1, KvCacheDtype::Fp8);
        let plan = GlmCachePlan::new(shape, &cfg, Some(scaled_index)).unwrap();
        assert_eq!(
            plan.block_bytes_all_layers(),
            cfg.block_bytes_kv_all_layers() + scaled_index.block_bytes(cfg.block_size).unwrap()
        );
        // MTP's optional semantic index stays optional and its one-layer BF16
        // K/V allocations retain the same size.
        assert_eq!(
            GlmCachePlan::new(shape, &config(1, KvCacheDtype::Bf16), None)
                .unwrap()
                .block_bytes_all_layers(),
            32768
        );
    }

    #[test]
    fn inconsistent_geometry_and_overflow_fail_before_allocation() {
        let shape = GlmMlaShape::new(512, 0).unwrap();
        let index = Some(SparseIndexCacheConfig::bf16(4, 128));
        for (heads, dim, block, layers) in [
            (2, 512, 16, 1),
            (1, 576, 16, 1),
            (1, 512, 0, 1),
            (1, 512, 15, 1),
            (1, 512, 16, 0),
            (1, 512, usize::MAX - 3, 1),
            (1, 512, 16, usize::MAX),
        ] {
            let mut cfg = config(layers, KvCacheDtype::Bf16);
            cfg.num_kv_heads = heads;
            cfg.head_dim = dim;
            cfg.block_size = block;
            assert!(GlmCachePlan::new(shape, &cfg, index).is_err());
        }
        let mut cfg = config(2, KvCacheDtype::Bf16);
        cfg.layer_dims = vec![(1, 512), (1, 576)];
        assert!(GlmCachePlan::new(shape, &cfg, index).is_err());
        for index in [
            SparseIndexCacheConfig::bf16(0, 128),
            SparseIndexCacheConfig::bf16(4, 0),
            SparseIndexCacheConfig::bf16(8, 128),
            SparseIndexCacheConfig::bf16(4, 64),
        ] {
            assert!(GlmCachePlan::new(shape, &config(1, KvCacheDtype::Bf16), Some(index)).is_err());
        }
        let plan = GlmCachePlan::new(shape, &config(1, KvCacheDtype::Bf16), index).unwrap();
        assert_eq!(plan.num_blocks_for_budget(0), 0);
        assert!(plan.bytes_for_blocks(0).is_err());
        assert!(plan.bytes_for_blocks(usize::MAX).is_err());
        if let Ok(too_many) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(plan.bytes_for_blocks(too_many).is_err());
        }
    }

    #[test]
    fn existing_allocator_keeps_separate_pools_and_full_block_tails() {
        let gpu = MockGpuBackend::new();
        let cfg = config(1, KvCacheDtype::Bf16);
        let index = SparseIndexCacheConfig::bf16(4, 128);
        let plan = GlmCachePlan::new(GlmMlaShape::new(512, 0).unwrap(), &cfg, Some(index)).unwrap();
        assert_eq!(plan.bytes_for_blocks(2).unwrap(), 83968);
        let mut cache = PagedKvCache::new(cfg, 2, &gpu).unwrap();
        cache.attach_sparse_index(index, &gpu).unwrap();
        assert_ne!(cache.k_pool_ptr(0), cache.v_pool_ptr(0));
        assert_ne!(
            cache.sparse_index_pool_ptr(0),
            cache.sparse_index_tail_pool_ptr(0)
        );
        assert_eq!(cache.sparse_index_block_stride_bytes(0), 1024);
        assert_eq!(cache.sparse_index_tail_block_stride_bytes(0), 8192);
        assert_eq!(gpu.alloc_count(), 4);
        let allocated: usize = [
            cache.k_pool_ptr(0),
            cache.v_pool_ptr(0),
            cache.sparse_index_pool_ptr(0),
            cache.sparse_index_tail_pool_ptr(0),
        ]
        .into_iter()
        .map(|ptr| gpu.read_alloc(ptr).unwrap().len())
        .sum();
        assert_eq!(allocated, plan.bytes_for_blocks(2).unwrap());
    }
}
