// SPDX-License-Identifier: AGPL-3.0-only

//! One sequence's DeepSeek-V4.1 attention cache on one layer, and what
//! source layers hand to the layers after them within a pass.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::cache_recipe::WINDOW;
use super::index::INDEX_DIM;
use super::ops::HEAD_DIM;
use crate::layer::LayerState;

/// A source layer's compressed keys for one sequence, their index keys, and
/// the group its compressor is filling.
pub struct CompressedKeys {
    /// `[capacity, 512]` BF16, one row a completed group.
    pub keys: DevicePtr,
    /// `[capacity, 128]` BF16: what the indexer scores each group by.
    pub index_keys: DevicePtr,
    pub capacity: usize,
    /// `[ratio, 512]` FP32: the key and gate projections of the group being
    /// filled.
    pub(super) kv_state: DevicePtr,
    pub(super) score_state: DevicePtr,
}

/// One sequence's attention cache on one layer.
pub struct SeqState {
    /// `[128, 512]` BF16 ring: slot `position % 128` holds that position's key.
    pub window: DevicePtr,
    pub compressed: Option<CompressedKeys>,
}

impl SeqState {
    /// `ratio` is the layer's compression ratio when it produces compressed
    /// keys itself, else 0. `max_positions` bounds the sequence.
    pub fn alloc(gpu: &dyn GpuBackend, ratio: usize, max_positions: usize) -> Result<Self> {
        let window = gpu.alloc(WINDOW * HEAD_DIM * 2)?;
        gpu.memset(window, 0, WINDOW * HEAD_DIM * 2)?;
        let compressed = match ratio {
            0 => None,
            ratio => {
                let capacity = max_positions / ratio;
                Some(CompressedKeys {
                    keys: gpu.alloc(capacity.max(1) * HEAD_DIM * 2)?,
                    index_keys: gpu.alloc(capacity.max(1) * INDEX_DIM * 2)?,
                    capacity,
                    kv_state: gpu.alloc(ratio * HEAD_DIM * 4)?,
                    score_state: gpu.alloc(ratio * HEAD_DIM * 4)?,
                })
            }
        };
        Ok(Self { window, compressed })
    }

    pub fn free(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.window)?;
        if let Some(c) = self.compressed.take() {
            for p in [c.keys, c.index_keys, c.kv_state, c.score_state] {
                gpu.free(p)?;
            }
        }
        Ok(())
    }
}

impl LayerState for SeqState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// What source layers hand down the stack for one row of a pass. Layers run
/// in order, so a source has written before its readers ask.
#[derive(Clone)]
pub(super) struct RowHandoff {
    /// The latest key source's compressed keys and index keys.
    pub(super) compressed: DevicePtr,
    pub(super) index_keys: DevicePtr,
    /// The compressed positions the latest index source chose for the row's
    /// query, in ascending order.
    pub(super) chosen: Vec<i32>,
    /// One flag per compressed position, from the candidate source: the
    /// positions the index sources after it may choose from.
    pub(super) candidates: Vec<bool>,
}

impl Default for RowHandoff {
    fn default() -> Self {
        Self {
            compressed: DevicePtr::NULL,
            index_keys: DevicePtr::NULL,
            chosen: Vec::new(),
            candidates: Vec::new(),
        }
    }
}

/// What source layers hand down the stack within one pass, per row.
#[derive(Default)]
pub struct Handoff {
    pub(super) rows: Vec<RowHandoff>,
}
