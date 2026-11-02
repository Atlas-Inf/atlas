// SPDX-License-Identifier: AGPL-3.0-only

//! Storage geometry for an auxiliary sparse-attention semantic index.

use anyhow::{Result, bail};

/// A semantic-index cache shares the main KV cache's physical block IDs, but
/// stores one index key per `tokens_per_pool` source tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseIndexCacheConfig {
    pub tokens_per_pool: usize,
    pub head_dim: usize,
    pub dtype: SparseIndexCacheDtype,
}

impl SparseIndexCacheConfig {
    pub const fn bf16(tokens_per_pool: usize, head_dim: usize) -> Self {
        Self {
            tokens_per_pool,
            head_dim,
            dtype: SparseIndexCacheDtype::Bf16,
        }
    }

    /// Total pooled values, optional scales, and the four-token staging tail
    /// carried by one physical KV block.
    pub fn block_bytes(self, kv_block_size: usize) -> Result<usize> {
        if self.tokens_per_pool == 0 || self.head_dim == 0 {
            bail!("sparse index cache dimensions must be non-zero");
        }
        if !kv_block_size.is_multiple_of(self.tokens_per_pool) {
            bail!(
                "KV block size {kv_block_size} must be divisible by sparse index pool size {}",
                self.tokens_per_pool
            );
        }
        Ok(self.values_block_bytes(kv_block_size)
            + self.scales_block_bytes(kv_block_size)
            + self.tail_block_bytes(kv_block_size))
    }

    pub(super) fn entries_per_block(self, kv_block_size: usize) -> usize {
        kv_block_size / self.tokens_per_pool
    }

    pub(super) fn values_block_bytes(self, kv_block_size: usize) -> usize {
        let element_bytes = match self.dtype {
            SparseIndexCacheDtype::Bf16 => 2,
            SparseIndexCacheDtype::Fp8E4m3Scaled => 1,
        };
        self.entries_per_block(kv_block_size) * self.head_dim * element_bytes
    }

    pub(super) fn scales_block_bytes(self, kv_block_size: usize) -> usize {
        match self.dtype {
            SparseIndexCacheDtype::Bf16 => 0,
            SparseIndexCacheDtype::Fp8E4m3Scaled => {
                self.entries_per_block(kv_block_size) * std::mem::size_of::<f32>()
            }
        }
    }

    /// Uncompressed keys and gates for the pool currently being assembled.
    /// Keeping this paged (rather than in per-layer scalar state) makes a
    /// prefill/decode seam and interleaved sequences obey identical ownership.
    pub fn tail_block_bytes(self, kv_block_size: usize) -> usize {
        kv_block_size * self.head_dim * 2 * std::mem::size_of::<u16>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseIndexCacheDtype {
    Bf16,
    /// One FP8-E4M3 vector plus one FP32 scale per pooled entry.
    Fp8E4m3Scaled,
}
