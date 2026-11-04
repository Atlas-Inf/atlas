// SPDX-License-Identifier: AGPL-3.0-only
//! Checked private-pool sizing. Active contexts share one eager work arena.
use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferSizes;
use spark_runtime::kv_cache::{KvCacheConfig, SparseIndexCacheConfig};

pub(super) fn block_capacity(
    max_seq_len: usize,
    max_sequences: usize,
    block_size: usize,
) -> Result<(usize, usize)> {
    ensure!(
        max_seq_len > 0 && max_sequences > 0 && block_size > 0,
        "native MTP context, sequence and block capacities must be nonzero"
    );
    ensure!(
        max_seq_len <= u32::MAX as usize,
        "native MTP context exceeds u32 metadata"
    );
    let per_sequence = max_seq_len.div_ceil(block_size);
    // A bounded extra tail block per admitted sequence permits snapshot cloning
    // at an otherwise-full partial block. Cached complete pages are refcounted.
    let total = per_sequence
        .checked_add(1)
        .and_then(|n| n.checked_mul(max_sequences))
        .context("native MTP private block capacity overflows")?;
    ensure!(
        total <= u32::MAX as usize,
        "native MTP private block IDs exceed u32"
    );
    Ok((per_sequence, total))
}

pub(super) fn aggregate_blocks(target_blocks: usize, tail_blocks: usize) -> Result<usize> {
    ensure!(
        target_blocks > 0,
        "native MTP requires a nonempty target pool"
    );
    let total = target_blocks
        .checked_add(tail_blocks)
        .context("native MTP aggregate block count overflows")?;
    ensure!(
        total <= u32::MAX as usize,
        "native MTP block IDs exceed u32"
    );
    Ok(total)
}

pub(super) fn workspace_config(config: &ModelConfig) -> ModelConfig {
    let mut allocation = config.clone();
    // The private predictor never runs full attention prefill: its prompt and
    // repair path writes KV only, and single-row decode uses expert_down_out
    // for index logits. Avoid the unrelated 2048-row QSA prefill scratch.
    // Actual forwards always receive the original, unchanged draft config.
    allocation.index_compress_ratio = 0;
    allocation
}

pub(super) fn private_pool_bytes(
    config: &KvCacheConfig,
    index: SparseIndexCacheConfig,
    blocks: usize,
) -> Result<usize> {
    // One ordinary MLA layer, K/V aliased, BF16 latent and BF16 sparse index.
    let latent = config
        .block_size
        .checked_mul(config.num_kv_heads)
        .and_then(|n| n.checked_mul(config.head_dim))
        .and_then(|n| n.checked_mul(2))
        .context("native MTP latent block bytes overflow")?;
    latent
        .checked_add(index.block_bytes(config.block_size)?)
        .and_then(|n| n.checked_mul(blocks))
        .context("native MTP private pool bytes overflow")
}

pub(super) fn validate_workspace(
    config: &ModelConfig,
    sizes: &BufferSizes,
    max_seq_len: usize,
    block_size: usize,
) -> Result<()> {
    ensure!(
        config.index_kpool > 0,
        "native MTP sparse pool width is zero"
    );
    let index_logits = max_seq_len
        .div_ceil(config.index_kpool)
        .checked_mul(4)
        .context("native MTP index scratch bytes overflow")?;
    ensure!(
        sizes.expert_down_out >= index_logits,
        "native MTP index logits exceed private scratch"
    );
    let metadata = max_seq_len
        .div_ceil(block_size)
        .checked_mul(4)
        .and_then(|n| n.checked_add(crate::layers::mtp_meta::MTP_META_OFFSET + 256))
        .context("native MTP block table metadata bytes overflow")?;
    ensure!(
        sizes.scratch >= metadata,
        "native MTP block table exceeds private scratch"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aggregate_pool_tracks_target_admission_not_four_maximum_contexts() {
        assert_eq!(aggregate_blocks(10_000, 4).unwrap(), 10_004);
        assert_eq!(aggregate_blocks(65_536, 4).unwrap(), 65_540);
        assert!(aggregate_blocks(0, 4).is_err());
        assert!(aggregate_blocks(u32::MAX as usize, 1).is_err());
        assert!(aggregate_blocks(usize::MAX, 4).is_err());
    }
    #[test]
    fn pool_is_bounded_for_four_independent_million_token_contexts() {
        assert_eq!(block_capacity(1_048_576, 4, 16).unwrap(), (65_536, 262_148));
        assert_eq!(block_capacity(17, 4, 16).unwrap(), (2, 12));
        for args in [
            (0, 4, 16),
            (100, 0, 16),
            (100, 4, 0),
            (usize::MAX, 4, 16),
            (100, usize::MAX, 16),
        ] {
            assert!(block_capacity(args.0, args.1, args.2).is_err());
        }
    }
    #[test]
    fn pool_accounting_counts_aliased_latents_and_index_tails_once() {
        let config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 512,
            num_layers: 1,
            dtype: spark_runtime::kv_cache::KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let index = SparseIndexCacheConfig::bf16(4, 128);
        assert_eq!(private_pool_bytes(&config, index, 1).unwrap(), 25_600);
        assert_eq!(
            private_pool_bytes(&config, index, 262_148).unwrap(),
            6_710_988_800
        );
        assert!(private_pool_bytes(&config, index, usize::MAX).is_err());
    }
    #[test]
    fn workspace_clone_does_not_change_runtime_index_geometry() {
        let mut runtime = ModelConfig::qwen3_next_80b_nvfp4();
        runtime.index_compress_ratio = 4;
        runtime.index_kpool = 4;
        let allocation = workspace_config(&runtime);
        assert_eq!(runtime.index_compress_ratio, 4);
        assert_eq!(allocation.index_compress_ratio, 0);
        assert_eq!(allocation.index_kpool, 4);
        let sizes = BufferSizes::from_config(&allocation, 128, 1_048_576, 16, 1);
        assert_eq!(sizes.qsa_select_scratch, 256);
        validate_workspace(&runtime, &sizes, 1_048_576, 16).unwrap();
    }
}
