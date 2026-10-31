// SPDX-License-Identifier: AGPL-3.0-only

//! Allocation and sizing for the auxiliary paged semantic-index cache.

use anyhow::{Result, bail};

use super::{KvBuffer, KvCacheConfig, PagedKvCache, SparseIndexCacheConfig};
use crate::gpu::{DevicePtr, GpuBackend};

impl PagedKvCache {
    /// Attach one sparse semantic-index pool to every attention layer.
    /// Allocation is transactional: a partial failure frees every new buffer.
    pub fn attach_sparse_index(
        &mut self,
        spec: SparseIndexCacheConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.attach_sparse_index_with_tail_slots(spec, None, gpu)
    }

    /// [`Self::attach_sparse_index`] with raw tails lent per `tail_slots`
    /// (see `tail_slots.rs`) instead of one tail per physical block.
    pub fn attach_sparse_index_with_tail_slots(
        &mut self,
        spec: SparseIndexCacheConfig,
        tail_slots: Option<super::TailSlotPlan>,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if self.sparse_index_config.is_some() {
            bail!("sparse index cache is already attached");
        }
        spec.block_bytes(self.config.block_size)?;
        let tails = tail_slots
            .map(|plan| {
                super::tail_slots::TailSlots::new(self.num_blocks, plan, gpu, &self.placement)
            })
            .transpose()?;
        let tail_entries = tails.as_ref().map_or(self.num_blocks, |t| t.capacity());
        let values_stride = spec.values_block_bytes(self.config.block_size);
        let scales_stride = spec.scales_block_bytes(self.config.block_size);
        let tail_stride = spec.tail_block_bytes(self.config.block_size);
        let mut allocations: Vec<(DevicePtr, DevicePtr, DevicePtr)> =
            Vec::with_capacity(self.layers.len());
        let placement = &self.placement;
        for layer in 0..self.layers.len() {
            let values = match placement.alloc(
                gpu,
                KvBuffer::IndexValues(layer),
                self.num_blocks * values_stride,
            ) {
                Ok(ptr) => ptr,
                Err(error) => {
                    free_allocations(gpu, allocations);
                    return Err(error);
                }
            };
            let scales = if scales_stride == 0 {
                DevicePtr::NULL
            } else {
                match placement.alloc(
                    gpu,
                    KvBuffer::IndexScales(layer),
                    self.num_blocks * scales_stride,
                ) {
                    Ok(ptr) => ptr,
                    Err(error) => {
                        let _ = gpu.free(values);
                        free_allocations(gpu, allocations);
                        return Err(error);
                    }
                }
            };
            let tail = match placement.alloc(
                gpu,
                KvBuffer::IndexTail(layer),
                tail_entries * tail_stride,
            ) {
                Ok(ptr) => ptr,
                Err(error) => {
                    let _ = gpu.free(values);
                    if !scales.is_null() {
                        let _ = gpu.free(scales);
                    }
                    free_allocations(gpu, allocations);
                    if let Some(tails) = tails {
                        let _ = gpu.free(tails.map);
                    }
                    return Err(error);
                }
            };
            allocations.push((values, scales, tail));
        }
        for (layer, (values, scales, tail)) in self.layers.iter_mut().zip(allocations) {
            layer.sparse_index_values = values;
            layer.sparse_index_scales = scales;
            layer.sparse_index_tail = tail;
            layer.sparse_index_values_block_stride = values_stride;
            layer.sparse_index_scales_block_stride = scales_stride;
            layer.sparse_index_tail_block_stride = tail_stride;
        }
        self.sparse_index_config = Some(spec);
        self.tail_slots = tails;
        let block_bytes = spec.block_bytes(self.config.block_size)? - tail_stride;
        let total =
            self.layers.len() * (self.num_blocks * block_bytes + tail_entries * tail_stride);
        tracing::info!(
            "Sparse index cache: {} blocks × {} layers × {} bytes/block + {} tails ({}) \
             = {:.1} MiB",
            self.num_blocks,
            self.layers.len(),
            block_bytes,
            tail_entries,
            if tail_slots.is_some() {
                "slot-mapped"
            } else {
                "one per block"
            },
            total as f64 / (1024.0 * 1024.0),
        );
        Ok(())
    }

    pub fn sparse_index_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].sparse_index_values
    }

    pub fn sparse_index_scale_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].sparse_index_scales
    }

    pub fn sparse_index_tail_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].sparse_index_tail
    }

    pub fn sparse_index_block_stride_bytes(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].sparse_index_values_block_stride
    }

    pub fn sparse_index_tail_block_stride_bytes(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].sparse_index_tail_block_stride
    }

    pub fn sparse_index_config(&self) -> Option<SparseIndexCacheConfig> {
        self.sparse_index_config
    }

    /// Size the physical pool while accounting for an index attached to every
    /// attention layer.
    pub fn compute_num_blocks_with_sparse_index(
        config: &KvCacheConfig,
        index: SparseIndexCacheConfig,
        available_bytes: usize,
    ) -> Result<usize> {
        let bytes_per_block = config
            .block_bytes_kv_all_layers()
            .checked_add(index.block_bytes(config.block_size)? * config.num_layers)
            .ok_or_else(|| anyhow::anyhow!("KV + sparse index block size overflow"))?;
        if bytes_per_block == 0 {
            bail!("KV + sparse index cache block size is zero");
        }
        Ok(available_bytes / bytes_per_block)
    }
}

fn free_allocations(gpu: &dyn GpuBackend, allocations: Vec<(DevicePtr, DevicePtr, DevicePtr)>) {
    for (values, scales, tail) in allocations {
        let _ = gpu.free(values);
        if !scales.is_null() {
            let _ = gpu.free(scales);
        }
        let _ = gpu.free(tail);
    }
}
