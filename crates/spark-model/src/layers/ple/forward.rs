// SPDX-License-Identifier: AGPL-3.0-only

//! The public forward entry points — thin wrappers over
//! `forward_with_ids` that fix `num_tokens`/`fresh`/`ids_override` per call
//! shape (one sequence's explicit rows, generic prefill/decode forward).
//! Split out of `layer.rs` for the <=500 LoC cap.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::PleLayer;
use crate::layer::ForwardContext;
use crate::layers::ple::PleSeqState;

impl PleLayer {
    /// Multi-token forward against an EXPLICIT id slice — one sequence of a
    /// batched verify or multi-sequence decode, whose rows are a sub-slice of
    /// the batch's host ids rather than its prefix ([`Self::forward_seqs`]).
    /// `fresh` is false: a verify step never starts a sequence.
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

    /// Inject into `highway` `[T, hc_mult*hidden]` FP32, in place.
    /// `fresh` starts a new sequence (prefill from position 0).
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
