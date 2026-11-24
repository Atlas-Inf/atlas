// SPDX-License-Identifier: AGPL-3.0-only

//! [`WindowPlan`]: which layers keep window-bounded KV, how the pool is
//! sized, and the unpooled entries those layers leave in the primary list.

use super::super::{KvCacheConfig, LayerPool};
use crate::gpu::DevicePtr;

/// Which layers keep window-bounded KV, and the pool they share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowPlan {
    /// Attention layers whose KV lives in the window pool.
    pub layers: Vec<usize>,
    /// Keys a query reads back (its own included).
    pub window: usize,
    /// Local blocks, the null block included.
    pub blocks: usize,
    /// Snapshot windows kept pinned at most.
    pub pins: usize,
}

impl WindowPlan {
    /// Keys a tile loads at once: the prefill kernels start a row block's
    /// keys at the tile holding its first in-window key
    /// (`inkling2_flash_prefill.cu` `FP_BC`, `prefill_paged_compute.cuh` `BC`).
    pub const KEY_TILE: usize = 32;

    /// First logical block a forward whose first row sits at `first_pos`
    /// reads; no later forward of the sequence reads below it.
    pub fn first_block(&self, first_pos: usize, block_size: usize) -> usize {
        let first_key = (first_pos + 1).saturating_sub(self.window);
        first_key / Self::KEY_TILE * Self::KEY_TILE / block_size
    }

    /// Most blocks one sequence holds for a forward of `rows` rows.
    pub fn span_blocks(window: usize, block_size: usize, rows: usize) -> usize {
        (window + Self::KEY_TILE - 1 + rows).div_ceil(block_size) + 1
    }

    /// Local blocks for `seqs` live sequences of up to `verify_rows` rows a
    /// step, one forward of `batch_rows` new rows, and `pins` snapshots.
    pub fn size_blocks(
        window: usize,
        block_size: usize,
        seqs: usize,
        verify_rows: usize,
        batch_rows: usize,
        pins: usize,
    ) -> usize {
        1 + seqs * Self::span_blocks(window, block_size, verify_rows)
            + batch_rows.div_ceil(block_size)
            + seqs
            + pins * Self::span_blocks(window, block_size, 0)
    }

    /// K + V bytes of one local block over the window layers.
    pub fn local_block_bytes(&self, config: &KvCacheConfig) -> usize {
        self.layers
            .iter()
            .map(|&l| config.k_block_bytes_for_layer(l) + config.v_block_bytes_for_layer(l))
            .sum()
    }

    /// K + V bytes of one primary block over the other layers.
    pub fn primary_block_bytes(&self, config: &KvCacheConfig) -> usize {
        config.block_bytes_kv_all_layers() - self.local_block_bytes(config)
    }

    /// Bytes of the window pools and the map for `primary_blocks` blocks.
    pub fn pool_bytes(&self, config: &KvCacheConfig, primary_blocks: usize) -> usize {
        self.blocks * self.local_block_bytes(config) + primary_blocks * 4
    }
}

impl LayerPool {
    /// A window layer's entry in the primary pools: strides and dtype as a
    /// pooled layer has them, no pools (primary-indexed loops skip it).
    pub(in crate::kv_cache) fn window_layer(config: &KvCacheConfig, layer: usize) -> Self {
        Self {
            k_pool: DevicePtr::NULL,
            v_pool: DevicePtr::NULL,
            k_block_stride: config.k_block_bytes_for_layer(layer),
            v_block_stride: config.v_block_bytes_for_layer(layer),
            dtype: config.dtype_for_layer(layer),
            sparse_index_values: DevicePtr::NULL,
            sparse_index_scales: DevicePtr::NULL,
            sparse_index_tail: DevicePtr::NULL,
            sparse_index_values_block_stride: 0,
            sparse_index_scales_block_stride: 0,
            sparse_index_tail_block_stride: 0,
        }
    }
}
