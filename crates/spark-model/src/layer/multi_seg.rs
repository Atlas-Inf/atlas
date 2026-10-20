// SPDX-License-Identifier: AGPL-3.0-only

//! One sequence's rows of a multi-sequence prefill pass
//! (`ATLAS_QWEN4EXP_PREFILL_MULTI`, `TransformerLayer::prefill_multi`).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::{ForwardContext, LayerState};

/// A sequence's rows `[row0, row0 + rows)` of the pass, at sequence positions
/// `[start, start + rows)`, with the per-sequence state a layer's recurrent and
/// attention parts need. `ctx` is the pass's context with THIS sequence's
/// attention metadata and token ids; the row-wise parts take the pass's own.
pub struct MultiSeg<'s, 'c> {
    pub row0: usize,
    pub rows: usize,
    pub start: usize,
    pub kv_write_start: usize,
    pub state: &'s mut dyn LayerState,
    pub block_table: &'s mut Vec<u32>,
    pub disk_block_ids: &'s mut Vec<u32>,
    pub disk_last_offloaded_per_layer: &'s mut Vec<u32>,
    pub ctx: &'s ForwardContext<'c>,
}

/// One layer's multi-sequence prefill pass: the sequences' rows `segs`, back
/// to back in `hidden` (`total` rows), with the pass's own context.
pub struct MultiPass<'s, 'c> {
    pub hidden: DevicePtr,
    pub total: usize,
    pub segs: Vec<MultiSeg<'s, 'c>>,
    pub kv_cache: &'s mut PagedKvCache,
    pub ctx: &'s ForwardContext<'c>,
    pub stream: u64,
}

/// The default `TransformerLayer::prefill_multi`.
pub(crate) fn unsupported(_pass: &mut MultiPass<'_, '_>) -> Result<()> {
    anyhow::bail!("this layer has no multi-sequence prefill")
}
