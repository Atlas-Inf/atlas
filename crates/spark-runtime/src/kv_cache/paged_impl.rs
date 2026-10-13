// SPDX-License-Identifier: AGPL-3.0-only
//
// `PagedKvCache` impl block. Split from `kv_cache.rs` so the parent file
// keeps the public types small enough to read at a glance. The struct
// definition lives in the parent.

use anyhow::{Result, bail};

use super::block_trace::BlockTrace;
use super::free_blocks::FreeBlocks;
use super::{KvCacheConfig, KvCacheDtype, LayerPool, PagedKvCache};
use crate::gpu::{DevicePtr, GpuBackend};

impl PagedKvCache {
    /// Allocate the KV cache pool on the GPU.
    pub fn new(config: KvCacheConfig, num_blocks: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_v_alias(config, num_blocks, gpu, false)
    }

    /// [`Self::new`], optionally with every layer's V side aliasing its K
    /// storage: for caches whose writers store one latent on both sides
    /// (GLM-5 NoPE MLA), which halves the pool.
    pub fn new_with_v_alias(
        config: KvCacheConfig,
        num_blocks: usize,
        gpu: &dyn GpuBackend,
        v_aliases_k: bool,
    ) -> Result<Self> {
        Self::new_with_k_slots(config, num_blocks, num_blocks, gpu, v_aliases_k)
    }

    /// [`Self::new_with_v_alias`] with `k_slots` K-pool block slots per layer
    /// (`num_blocks` except under a latent shard, see `latent_shard.rs`).
    pub(super) fn new_with_k_slots(
        config: KvCacheConfig,
        num_blocks: usize,
        k_slots: usize,
        gpu: &dyn GpuBackend,
        v_aliases_k: bool,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut total_bytes: usize = 0;
        for i in 0..config.num_layers {
            // Per-side block_bytes: for symmetric dtypes both are equal; for
            // asymmetric (e.g. Bf16KTurbo3V) the K pool is allocated bf16-sized
            // and the V pool is allocated turbo3-sized — avoids the 4× V over-
            // allocation that would result from a single MAX-sized stride.
            let k_block_bytes = config.k_block_bytes_for_layer(i);
            let k_pool_bytes = k_slots * k_block_bytes;
            // A failed alloc frees the pools this call already claimed, so a
            // caller can release memory and retry (build_model's reserve
            // balloon fallback) without the first attempt's layers leaking.
            let release = |layers: &[LayerPool]| {
                for l in layers {
                    let _ = gpu.free(l.k_pool);
                    if l.v_pool != l.k_pool {
                        let _ = gpu.free(l.v_pool);
                    }
                }
            };
            let k_pool = match gpu.alloc(k_pool_bytes) {
                Ok(p) => p,
                Err(e) => {
                    release(&layers);
                    return Err(e);
                }
            };
            let (v_pool, v_block_bytes, v_pool_bytes) = if v_aliases_k {
                (k_pool, k_block_bytes, 0)
            } else {
                let v_block_bytes = config.v_block_bytes_for_layer(i);
                let v_pool_bytes = num_blocks * v_block_bytes;
                match gpu.alloc(v_pool_bytes) {
                    Ok(p) => (p, v_block_bytes, v_pool_bytes),
                    Err(e) => {
                        let _ = gpu.free(k_pool);
                        release(&layers);
                        return Err(e);
                    }
                }
            };
            total_bytes += k_pool_bytes + v_pool_bytes;
            layers.push(LayerPool {
                k_pool,
                v_pool,
                k_block_stride: k_block_bytes,
                v_block_stride: v_block_bytes,
                dtype: config.dtype_for_layer(i),
                sparse_index_values: DevicePtr::NULL,
                sparse_index_scales: DevicePtr::NULL,
                sparse_index_tail: DevicePtr::NULL,
                sparse_index_values_block_stride: 0,
                sparse_index_scales_block_stride: 0,
                sparse_index_tail_block_stride: 0,
            });
        }

        let free_blocks = FreeBlocks::new(num_blocks, 1);
        let block_ref_counts = vec![0u32; num_blocks];

        let has_mixed = !config.layer_dtypes.is_empty()
            && config.layer_dtypes.iter().any(|d| *d != config.dtype);
        if has_mixed {
            let hp_count = config
                .layer_dtypes
                .iter()
                .filter(|d| **d != config.dtype)
                .count();
            tracing::info!(
                "KV cache: {} blocks × {} layers ({} high-precision) = {:.1} GB total (mixed dtype)",
                num_blocks,
                config.num_layers,
                hp_count,
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        } else {
            tracing::info!(
                "KV cache: {} blocks × {} layers = {:.1} GB total{}",
                num_blocks,
                config.num_layers,
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                if v_aliases_k { " (V aliases K)" } else { "" },
            );
        }

        Ok(Self {
            layers,
            num_blocks,
            free_blocks,
            block_ref_counts,
            config,
            sparse_index_config: None,
            tail_slots: None,
            trace: BlockTrace::new(num_blocks),
            nvme: None,
            latent_shard: None,
        })
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
        for layer in &self.layers {
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

    /// Increment reference count on a block (for prefix cache sharing).
    #[track_caller]
    pub fn inc_ref(&mut self, block_idx: u32) {
        debug_assert!((block_idx as usize) < self.num_blocks);
        self.block_ref_counts[block_idx as usize] += 1;
        if self.trace.is_on() {
            let after = self.block_ref_counts[block_idx as usize];
            self.trace.record(
                block_idx as usize,
                "inc",
                after,
                std::panic::Location::caller(),
            );
        }
    }

    /// Decrement reference count. Returns true if block was freed (count hit 0).
    #[track_caller]
    pub fn dec_ref(&mut self, block_idx: u32) -> bool {
        let idx = block_idx as usize;
        debug_assert!(idx < self.num_blocks);
        // Saturating, not wrapping: `debug_assert` is compiled out in release and
        // the workspace sets no `overflow-checks`, so a 0-ref decrement would wrap
        // to u32::MAX and pin that block for the process lifetime — a silent,
        // unrecoverable pool leak. Log loudly and refuse instead; an over-release
        // is a refcount bug worth seeing, not worth crashing or corrupting for.
        debug_assert!(
            self.block_ref_counts[idx] > 0,
            "dec_ref on block with 0 refs"
        );
        if self.block_ref_counts[idx] == 0 {
            let caller = std::panic::Location::caller();
            tracing::error!(
                "dec_ref on block {block_idx} with 0 refs (from {caller}) — refcount bug \
                 (ignoring; would otherwise wrap to u32::MAX and pin the block){}",
                if self.trace.is_on() {
                    format!("\n  history: {}", self.trace.dump(idx))
                } else {
                    String::from(" [set ATLAS_KV_TRACE=1 for this block's ref history]")
                }
            );
            return false;
        }
        self.block_ref_counts[idx] -= 1;
        if self.trace.is_on() {
            let after = self.block_ref_counts[idx];
            self.trace
                .record(idx, "dec", after, std::panic::Location::caller());
        }
        if self.block_ref_counts[idx] == 0 {
            self.free_blocks.push(block_idx);
            self.release_tail_slot_if_freed(block_idx);
            true
        } else {
            false
        }
    }

    /// Free a previously allocated block (decrements ref, frees if count hits 0).
    #[track_caller]
    pub fn free_block(&mut self, block_idx: u32) {
        self.dec_ref(block_idx);
    }

    /// Free all blocks in a block table.
    #[track_caller]
    pub fn free_blocks(&mut self, block_table: &[u32]) {
        for &idx in block_table {
            self.free_block(idx);
            // A departing sequence never reads these raw index tails again. A
            // block that outlives it is one the prefix cache published: a
            // whole block of committed tokens, whose pools are finalized, and
            // which no holder appends to (a snapshot replay over it needs no
            // tail: the kernel skips the block). So the tail returns to the
            // pool now instead of when the cache finally evicts the block.
            if let Some(tails) = self.tail_slots.as_mut() {
                tails.release(idx);
            }
        }
    }

    /// Return a block to the free pool directly, bypassing ref counting.
    /// Used by eviction: the radix tree already removed its reference.
    #[track_caller]
    pub fn return_evicted_block(&mut self, block_idx: u32) {
        let idx = block_idx as usize;
        debug_assert!(idx < self.num_blocks);
        // Release exactly the prefix cache's OWN reference (the "+1" a cached
        // block carries per sequence.rs `inc_ref` when the radix node is
        // inserted), not every reference. Force-zeroing here freed blocks that
        // an active sequence still held in its block_table whenever the radix
        // ref-count and the KV pool ref-count had desynced (e.g. a warm prefill
        // that matched a prefix then failed to allocate its suffix, or the
        // sliding-window partial-cache path). alloc_block would then re-hand
        // that still-live block to a new prefill and zero_block would memset it
        // under a concurrent decode → aliased KV pointer → CUDA_ERROR_ILLEGAL_
        // ADDRESS (700). Decrementing keeps a still-referenced block alive.
        //
        // Push to the free list ONLY on a real 1->0 transition. A block already
        // at 0 refs is already ON the free list, so pushing again duplicates the
        // entry and two subsequent `alloc_block`s hand the SAME physical block to
        // two sequences: they interleave writes into each other's KV, and when
        // both later free it the second `dec_ref` underflows. That is reachable
        // whenever the cache returns a block it never took a ref on.
        if self.block_ref_counts[idx] == 0 {
            tracing::warn!(
                "return_evicted_block({block_idx}) with 0 refs (from {}) — the prefix cache \
                 returned a block it holds no reference on; ignoring (re-pushing it would \
                 duplicate a free-list entry and alias the block across sequences){}",
                std::panic::Location::caller(),
                if self.trace.is_on() {
                    format!("\n  history: {}", self.trace.dump(idx))
                } else {
                    String::new()
                }
            );
            return;
        }
        self.block_ref_counts[idx] -= 1;
        if self.trace.is_on() {
            let after = self.block_ref_counts[idx];
            self.trace
                .record(idx, "evict_return", after, std::panic::Location::caller());
        }
        if self.block_ref_counts[idx] == 0 {
            self.free_blocks.push(block_idx);
            self.release_tail_slot_if_freed(block_idx);
        }
    }

    /// Current reference count for a block.
    pub fn ref_count(&self, block_idx: u32) -> u32 {
        self.block_ref_counts[block_idx as usize]
    }

    /// Get K cache pointer for a layer and block.
    #[track_caller]
    pub fn k_cache_ptr(&self, layer_idx: usize, block_idx: u32) -> DevicePtr {
        self.assert_unsharded("k_cache_ptr");
        let layer = &self.layers[layer_idx];
        layer
            .k_pool
            .offset(block_idx as usize * layer.k_block_stride)
    }

    /// Get V cache pointer for a layer and block.
    #[track_caller]
    pub fn v_cache_ptr(&self, layer_idx: usize, block_idx: u32) -> DevicePtr {
        self.assert_unsharded("v_cache_ptr");
        let layer = &self.layers[layer_idx];
        layer
            .v_pool
            .offset(block_idx as usize * layer.v_block_stride)
    }

    /// Get the full K cache pool pointer for a layer (for paged decode kernel).
    /// Panics under a latent shard, whose pool holds only this rank's blocks
    /// (see [`Self::latent_pool_ptr`]).
    #[track_caller]
    pub fn k_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.assert_unsharded("k_pool_ptr");
        self.layers[layer_idx].k_pool
    }

    /// Get the full V cache pool pointer for a layer (panics when sharded).
    #[track_caller]
    pub fn v_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.assert_unsharded("v_pool_ptr");
        self.layers[layer_idx].v_pool
    }

    /// Cache stride in elements (for FP8/BF16 kernels that need explicit stride).
    /// Same for all layers (element count is dtype-independent).
    pub fn cache_stride(&self) -> usize {
        self.config.cache_stride_elements()
    }

    /// Block stride in bytes (for NVFP4 kernels), using the uniform dtype.
    pub fn block_stride_bytes(&self) -> usize {
        self.config.block_bytes()
    }

    /// Block stride in bytes for a specific attention layer.
    /// For symmetric dtypes returns the K stride (which equals V).
    /// For asymmetric dtypes returns the K-side stride; use
    /// `v_block_stride_bytes_for_layer` for the V-side stride explicitly.
    pub fn block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].k_block_stride
    }

    /// K-side block stride in bytes for a specific attention layer.
    /// Same as `block_stride_bytes_for_layer`; named for clarity in asym call sites.
    pub fn k_block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].k_block_stride
    }

    /// V-side block stride in bytes for a specific attention layer.
    /// Differs from K-side only for asymmetric KV cache dtypes.
    pub fn v_block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].v_block_stride
    }

    /// NVFP4 data section size in bytes per block (uniform).
    pub fn nvfp4_data_bytes(&self) -> usize {
        self.config.nvfp4_data_bytes()
    }

    /// Turbo4 data section bytes (same layout as NVFP4: 4-bit packed).
    pub fn turbo4_data_bytes(&self) -> usize {
        self.config.turbo4_data_bytes()
    }

    /// Turbo3 data section bytes (3-bit packed).
    pub fn turbo3_data_bytes(&self) -> usize {
        self.config.turbo3_data_bytes()
    }

    /// Turbo2 data section bytes (2-bit packed).
    pub fn turbo2_data_bytes(&self) -> usize {
        self.config.turbo2_data_bytes()
    }

    /// Turbo8 data section bytes (FP8 E4M3 per element).
    pub fn turbo8_data_bytes(&self) -> usize {
        self.config.turbo8_data_bytes()
    }

    /// Turbo4 scale section bytes (same layout as NVFP4: FP8 per-group).
    pub fn turbo4_scale_bytes(&self) -> usize {
        self.config.turbo4_scale_bytes()
    }

    /// Cache configuration (read-only). Used by attention layers to query
    /// the `--high-speed-swap` HBM-shrink cap (`cache_blocks_per_seq`).
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Effective KV cache dtype for a specific attention layer.
    pub fn dtype_for_layer(&self, layer_idx: usize) -> KvCacheDtype {
        self.layers[layer_idx].dtype
    }

    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn dtype(&self) -> KvCacheDtype {
        self.config.dtype
    }

    /// Number of attention layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Read K and V data for one block at one layer from GPU to host.
    ///
    /// Returns `(k_data, v_data)` sized to each side's block stride
    /// (which may differ for asymmetric dtypes).
    pub fn read_block(
        &self,
        layer_idx: usize,
        block_idx: u32,
        gpu: &dyn GpuBackend,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        self.ensure_unsharded("read_block")?;
        let k_stride = self.layers[layer_idx].k_block_stride;
        let v_stride = self.layers[layer_idx].v_block_stride;
        let k_ptr = self.k_cache_ptr(layer_idx, block_idx);
        let v_ptr = self.v_cache_ptr(layer_idx, block_idx);

        let mut k_data = vec![0u8; k_stride];
        let mut v_data = vec![0u8; v_stride];
        gpu.copy_d2h(k_ptr, &mut k_data)?;
        gpu.copy_d2h(v_ptr, &mut v_data)?;

        Ok((k_data, v_data))
    }

    /// Write K and V data for one block at one layer from host to GPU.
    pub fn write_block(
        &self,
        layer_idx: usize,
        block_idx: u32,
        k_data: &[u8],
        v_data: &[u8],
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.ensure_unsharded("write_block")?;
        let k_ptr = self.k_cache_ptr(layer_idx, block_idx);
        let v_ptr = self.v_cache_ptr(layer_idx, block_idx);
        gpu.copy_h2d(k_data, k_ptr)?;
        gpu.copy_h2d(v_data, v_ptr)?;
        Ok(())
    }

    /// Compute how many blocks can fit given available GPU memory.
    /// Accounts for mixed dtypes when layer_dtypes is set.
    pub fn compute_num_blocks(config: &KvCacheConfig, available_bytes: usize) -> Result<usize> {
        let bytes_per_block = config.block_bytes_kv_all_layers();
        if bytes_per_block == 0 {
            bail!("KV cache block size is zero");
        }
        Ok(available_bytes / bytes_per_block)
    }
}
