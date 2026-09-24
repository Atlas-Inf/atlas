// SPDX-License-Identifier: AGPL-3.0-only

//! The public forward entry points — thin wrappers over
//! `forward_with_ids` that fix `num_tokens`/`fresh`/`ids_override` per call
//! shape (decode row, batched verify rows, generic prefill/decode forward).
//! Split out of `layer.rs` for the <=500 LoC cap.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::PleLayer;
use crate::layer::ForwardContext;
use crate::layers::ple::PleSeqState;

impl PleLayer {
    /// Inject into `highway` `[T, hc_mult*hidden]` FP32, in place.
    ///
    /// `fresh` starts a new sequence (prefill from position 0).
    /// One highway ROW with an explicit id — the multi-seq decode entry
    /// (`ctx.host_token_ids` holds the whole batch; the caller slices).
    pub fn forward_row(
        &self,
        st: &mut PleSeqState,
        highway_row: DevicePtr,
        ids: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway_row, 1, false, Some(ids), ctx, stream)
    }

    /// Multi-token forward against an EXPLICIT id slice — the batched verify
    /// path, where the rows of one sequence are a sub-slice of the batch's
    /// host ids rather than its prefix. `fresh` is false: a verify step never
    /// starts a sequence.
    pub fn forward_rows(
        &self,
        st: &mut PleSeqState,
        highway: DevicePtr,
        ids: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway, ids.len(), false, Some(ids), ctx, stream)
    }

    pub fn forward(
        &self,
        st: &mut PleSeqState,
        highway: DevicePtr,
        num_tokens: usize,
        fresh: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway, num_tokens, fresh, None, ctx, stream)
    }
}
