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
        // A latent shard holds only its own blocks of the run, which sit at
        // consecutive local slots; every other pool is indexed by block id.
        let (latent_first, latent_count) = match &self.latent_shard {
            Some(shard) => shard.owned_run(first, count),
            None => (first as usize, count),
        };
        for layer in &self.layers {
            for (base, stride, first, count) in [
                (
                    layer.k_pool,
                    layer.k_block_stride,
                    latent_first,
                    latent_count,
                ),
                (
                    layer.owned_v_pool(),
                    layer.v_block_stride,
                    first as usize,
                    count,
                ),
                (
                    layer.sparse_index_values,
                    layer.sparse_index_values_block_stride,
                    first as usize,
                    count,
                ),
                (
                    layer.sparse_index_scales,
                    layer.sparse_index_scales_block_stride,
                    first as usize,
                    count,
                ),
                (
                    if slotted {
                        DevicePtr::NULL
                    } else {
                        layer.sparse_index_tail
                    },
                    layer.sparse_index_tail_block_stride,
                    first as usize,
                    count,
                ),
            ] {
                if base.is_null() || stride == 0 || count == 0 {
                    continue;
                }
                gpu.memset_async(base.offset(first * stride), 0, count * stride, stream)?;
            }
        }
        Ok(())
    }

    /// DIAGNOSTIC (ATLAS_KV_POISON): fill a freshly-allocated block with 0xFF
    /// (a NaN bit-pattern in both bf16 `0xFFFF` and fp8-e4m3 `0xFF`) instead of
    /// zero. Any KV region that decode/attention reads but prefill never wrote
    /// then yields deterministic NaN rather than plausible-but-wrong zeros.
    /// Used to falsify the "unwritten fresh tail block" hypothesis: if cache-ON
    /// output goes NaN under poison while cache-OFF stays clean, a fresh block
    /// is being read unwritten; if both stay clean, fresh KV is fully written
    /// and the run-to-run nondeterminism originates elsewhere (scratch/scan).
    pub fn poison_block(
        &self,
        block_idx: u32,
        gpu: &dyn crate::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        // A latent shard stores (and so poisons) only its own blocks.
        let Some(slot) = self.latent_slot(block_idx) else {
            return Ok(());
        };
        for layer in self.layers.iter().filter(|l| !l.k_pool.is_null()) {
            let k_offset = slot as usize * layer.k_block_stride;
            let v_offset = slot as usize * layer.v_block_stride;
            gpu.memset_async(
                layer.k_pool.offset(k_offset),
                0xFF,
                layer.k_block_stride,
                stream,
            )?;
            gpu.memset_async(
                layer.v_pool.offset(v_offset),
                0xFF,
                layer.v_block_stride,
                stream,
            )?;
        }
        Ok(())
    }
}
