// SPDX-License-Identifier: AGPL-3.0-only

//! `PagedKvCache` block zeroing: one memset per per-layer array per run of
//! contiguous blocks. Split from `paged_impl.rs` (500-LoC cap).

use super::PagedKvCache;
use crate::gpu::DevicePtr;

impl PagedKvCache {
    /// Zero all KV data in a block across all layers.
    /// Prevents stale KV data from previous sequences from leaking into
    /// new sequences via paged attention reads beyond the current seq_len.
    pub fn zero_block(
        &self,
        block_idx: u32,
        gpu: &dyn crate::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        self.zero_block_run(block_idx, 1, gpu, stream)
    }

    /// [`Self::zero_block`] for many blocks: one memset per per-layer array
    /// per run of contiguous block ids (a prefill chunk's fresh blocks come
    /// off the free list as a few runs, not one memset storm per block).
    pub fn zero_blocks(
        &self,
        blocks: &[u32],
        gpu: &dyn crate::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        let mut sorted = blocks.to_vec();
        sorted.sort_unstable();
        for run in sorted.chunk_by(|a, b| *b == *a + 1) {
            self.zero_block_run(run[0], run.len(), gpu, stream)?;
        }
        Ok(())
    }

    fn zero_block_run(
        &self,
        first: u32,
        count: usize,
        gpu: &dyn crate::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        // Slotted tails are indexed by slot, not block, and every row is
        // written before its pool is finalized: nothing to zero.
        let slotted = self.tail_slots.is_some();
        for layer in &self.layers {
            for (base, stride) in [
                (layer.k_pool, layer.k_block_stride),
                (layer.owned_v_pool(), layer.v_block_stride),
                (
                    layer.sparse_index_values,
                    layer.sparse_index_values_block_stride,
                ),
                (
                    layer.sparse_index_scales,
                    layer.sparse_index_scales_block_stride,
                ),
                (
                    if slotted {
                        DevicePtr::NULL
                    } else {
                        layer.sparse_index_tail
                    },
                    layer.sparse_index_tail_block_stride,
                ),
            ] {
                if base.is_null() || stride == 0 {
                    continue;
                }
                gpu.memset_async(
                    base.offset(first as usize * stride),
                    0,
                    count * stride,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
