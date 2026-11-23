// SPDX-License-Identifier: AGPL-3.0-only

use super::{loader::Hy4Attention, memory::Owned};
use anyhow::{Result, ensure};
use atlas_core::hy4_reference::paged::PagedLayout;
use spark_runtime::gpu::DevicePtr;

/// Per-request, per-layer pages plus one-row scratch. No cross-request aliases.
/// Prefix forks deep-copy the resident pages and partial group, not scratch.
pub struct AttentionCache<'layer, 'gpu> {
    pub(super) layer: &'layer Hy4Attention<'gpu>,
    _memory: Owned<'gpu>,
    pub(super) layout: PagedLayout,
    pub(super) table: DevicePtr,
    pub(super) latent: DevicePtr,
    pub(super) pool_table: DevicePtr,
    pub(super) pool: DevicePtr,
    pub(super) tail: DevicePtr,
    pub(super) latent_bytes: usize,
    pub(super) pool_bytes: usize,
    pub(super) max_tokens: usize,
    pub(super) len: usize,
    pub(super) first_group_position: Option<[i64; 3]>,
    pub(super) stream: Option<u64>,
    pub(super) poisoned: bool,
    pub(super) scratch: Scratch,
    pub(super) batch: Option<super::batch::BatchScratch<'gpu>>,
    pub(super) prefix: Option<super::prefix::Prefix<'gpu>>,
}
pub(super) struct Scratch {
    pub qkv: DevicePtr,
    pub qc: DevicePtr,
    pub kv: DevicePtr,
    pub q: DevicePtr,
    pub absorbed: DevicePtr,
    pub latent_out: DevicePtr,
    pub softmax_parts: DevicePtr,
    pub values: DevicePtr,
    pub gates: DevicePtr,
    pub partial: DevicePtr,
    pub key: DevicePtr,
    pub pooled: DevicePtr,
    pub query: DevicePtr,
    pub hw: DevicePtr,
    pub quant: DevicePtr,
    pub scales: DevicePtr,
    pub scores: DevicePtr,
    pub indices: DevicePtr,
}

impl<'layer, 'gpu> AttentionCache<'layer, 'gpu> {
    pub fn new(
        layer: &'layer Hy4Attention<'gpu>,
        max_tokens: usize,
        page_size: usize,
    ) -> Result<Self> {
        ensure!(
            layer.memory.is_live()
                && max_tokens > 0
                && max_tokens <= layer.max_context
                && page_size > 0
                && page_size <= 65536,
            "Hy4 cache capacity/page size"
        );
        let pages = match layer.window {
            Some(w) => PagedLayout::window_pages(w, page_size)?,
            None => max_tokens.div_ceil(page_size),
        };
        let layout = PagedLayout {
            page_size,
            physical_pages: pages,
            table: (0..pages).collect(),
            circular: layer.window.is_some(),
        };
        let mut memory = Owned::new(layer.memory.gpu);
        let table = memory.upload(
            &(0..pages as u32)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let latent_bytes = pages
            .checked_mul(page_size)
            .and_then(|n| n.checked_mul(576 * 2))
            .ok_or_else(|| anyhow::anyhow!("Hy4 cache byte overflow"))?;
        let latent = memory.alloc(latent_bytes)?;
        let (pool_table, pool, pool_bytes, tail) = if layer.index.is_some() {
            let pool_pages = max_tokens.div_ceil(4).div_ceil(64);
            let table = memory.upload(
                &(0..pool_pages as u32)
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )?;
            let bytes = pool_pages * 8448;
            (
                table,
                memory.alloc(bytes)?,
                bytes,
                memory.alloc(4 * 128 * 2)?,
            )
        } else {
            (DevicePtr::NULL, DevicePtr::NULL, 0, DevicePtr::NULL)
        };
        let mut bf = |elements| memory.alloc(elements * 2);
        let qc = bf(layer.qrank)?;
        let kv = bf(576)?;
        let q = bf(layer.heads * 256)?;
        let absorbed = bf(layer.heads * 576)?;
        let latent_out = bf(layer.heads * 512)?;
        let values = bf(layer.heads * 256)?;
        let gates = bf(layer.heads * 256)?;
        let partial = bf(layer.hidden)?;
        let softmax_parts = memory.alloc(
            layer.heads * layer.window.unwrap_or(2048).min(max_tokens).div_ceil(128) * 514 * 4,
        )?;
        let (key, pooled, query, hw, quant, scales, scores, indices) =
            if let Some(ix) = &layer.index {
                (
                    memory.alloc(128 * 2)?,
                    memory.alloc(128 * 2)?,
                    memory.alloc(ix.heads * 128 * 2)?,
                    memory.alloc(ix.heads * 2)?,
                    memory.alloc(ix.heads * 128)?,
                    memory.alloc(ix.heads * 4)?,
                    memory.alloc(max_tokens.div_ceil(4) * 4)?,
                    memory.alloc(2048 * 4)?,
                )
            } else {
                (
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                )
            };
        let scratch = Scratch {
            qkv: memory.alloc((layer.qrank + 576) * 2)?,
            qc,
            kv,
            q,
            absorbed,
            latent_out,
            softmax_parts,
            values,
            gates,
            partial,
            key,
            pooled,
            query,
            hw,
            quant,
            scales,
            scores,
            indices,
        };
        Ok(Self {
            layer,
            _memory: memory,
            layout,
            table,
            latent,
            pool_table,
            pool,
            tail,
            latent_bytes,
            pool_bytes,
            max_tokens,
            len: 0,
            first_group_position: None,
            stream: None,
            poisoned: false,
            scratch,
            batch: None,
            prefix: None,
        })
    }
    pub fn committed_len(&self) -> usize {
        self.len
    }
    /// Drain the bound compute stream first. A failed free can be retried.
    pub fn release(&mut self) -> Result<()> {
        self.poisoned = true;
        self.batch = None;
        self.prefix = None;
        self._memory.release()
    }

    /// Keep the snapshot on the same stream as its prefix. For speculative
    /// work, fork before the draft and discard rejected state; never rewind
    /// lengths over overwritten tail/window pages.
    pub fn fork(&self, stream: u64) -> Result<Self> {
        ensure!(
            !self.poisoned && self.stream.is_none_or(|s| s == stream),
            "Hy4 invalid cache fork/stream"
        );
        let mut copy = Self::new(self.layer, self.max_tokens, self.layout.page_size)?;
        copy.copy_state_from(self, stream)?;
        Ok(copy)
    }

    pub fn fork_committed(&self, stream: u64) -> Result<Self> {
        ensure!(
            !self.poisoned && self.stream.is_none_or(|s| s == stream),
            "Hy4 invalid cache fork/stream"
        );
        let mut copy = Self::new(self.layer, self.max_tokens, self.layout.page_size)?;
        copy.copy_committed_state_from(self, stream)?;
        Ok(copy)
    }

    pub fn copy_state_from(&mut self, source: &Self, stream: u64) -> Result<()> {
        self.copy_state_from_impl(source, stream, false)
    }

    pub fn copy_committed_state_from(&mut self, source: &Self, stream: u64) -> Result<()> {
        self.copy_state_from_impl(source, stream, true)
    }

    fn copy_state_from_impl(
        &mut self,
        source: &Self,
        stream: u64,
        committed_only: bool,
    ) -> Result<()> {
        ensure!(
            !source.poisoned
                && std::ptr::eq(self.layer, source.layer)
                && self.max_tokens == source.max_tokens
                && self.layout.page_size == source.layout.page_size
                && self.layout.table == source.layout.table
                && self.stream.is_none_or(|s| s == stream)
                && source.stream.is_none_or(|s| s == stream),
            "Hy4 invalid attention snapshot"
        );
        let gpu = self.layer.memory.gpu;
        if committed_only {
            let pages = source.len.div_ceil(self.layout.page_size);
            let latent_bytes = pages
                .checked_mul(self.layout.page_size)
                .and_then(|n| n.checked_mul(576 * 2))
                .ok_or_else(|| anyhow::anyhow!("Hy4 snapshot byte overflow"))?
                .min(self.latent_bytes);
            if latent_bytes > 0 {
                gpu.copy_d2d_async(source.latent, self.latent, latent_bytes, stream)?;
            }
            if self.pool_bytes > 0 {
                let pool_bytes = (source.len / 4)
                    .div_ceil(64)
                    .checked_mul(8448)
                    .ok_or_else(|| anyhow::anyhow!("Hy4 snapshot pool overflow"))?
                    .min(self.pool_bytes);
                if pool_bytes > 0 {
                    gpu.copy_d2d_async(source.pool, self.pool, pool_bytes, stream)?;
                }
                gpu.copy_d2d_async(source.tail, self.tail, 4 * 128 * 2, stream)?;
            }
        } else {
            gpu.copy_d2d_async(source.latent, self.latent, self.latent_bytes, stream)?;
            if self.pool_bytes > 0 {
                gpu.copy_d2d_async(source.pool, self.pool, self.pool_bytes, stream)?;
                gpu.copy_d2d_async(source.tail, self.tail, 4 * 128 * 2, stream)?;
            }
        }
        self.len = source.len;
        self.first_group_position = source.first_group_position;
        self.stream = Some(stream);
        self.poisoned = false;
        Ok(())
    }
}
