// SPDX-License-Identifier: AGPL-3.0-only

//! A layer the draft head reads (`draft_seq.rs`): what it hands the head of
//! each sequence it runs.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::block::{SeqState, Shape};
use super::super::draft_seq::{self, DraftSeq};

/// Which of the layers the head reads this is, and of how many, for a head
/// of `stages` stages.
pub(super) struct Target {
    pub(super) index: usize,
    pub(super) of: usize,
    pub(super) stages: usize,
    pub(super) mean: KernelHandle,
}

impl Target {
    /// A new sequence's side of the head, on the first of the layers.
    pub(super) fn sequence(
        &self,
        gpu: &dyn GpuBackend,
        hidden: usize,
    ) -> Result<Option<Arc<Mutex<DraftSeq>>>> {
        if self.index > 0 {
            return Ok(None);
        }
        let seq = DraftSeq::alloc(gpu, hidden, self.of, self.stages)?;
        Ok(Some(Arc::new(Mutex::new(seq))))
    }

    /// The layer is about to run `rows` tokens of `state`'s sequence from
    /// `position`: the head reads their stream (`[rows, copies, hidden]`
    /// BF16) as the layer's attention does. The first of the layers says
    /// whose sequence the others' rows are.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn take(
        &self,
        gpu: &dyn GpuBackend,
        shape: &Shape,
        streams: DevicePtr,
        rows: usize,
        state: &SeqState,
        position: usize,
        stream: u64,
    ) -> Result<()> {
        let tap = draft_seq::tap();
        let seq = match &state.draft {
            Some(seq) => {
                tap.set(seq);
                seq.clone()
            }
            None => tap
                .current()
                .context("deepseek_v41: a draft target layer ran before the first of them")?,
        };
        let (hidden, copies) = (shape.hidden, shape.hc.copies);
        seq.lock().expect("draft sequence").take(
            gpu, self.mean, self.index, streams, rows, position, hidden, copies, stream,
        )
    }
}
