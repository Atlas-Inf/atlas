// SPDX-License-Identifier: AGPL-3.0-only
//! Test-only bridge from model prefill to the real drafter append path.
use super::{BlockDiffusionDraftHead, DflashProposerState, FirstAppend};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(crate) fn head(variant: FirstAppend, hidden_size: usize) -> BlockDiffusionDraftHead {
    let mut head = super::free_state_tests::zero_head();
    head.target_layer_ids = vec![0];
    head.target_hidden_size = hidden_size;
    head.max_seq_len = 9;
    head.ctx_window = 9;
    head.query_causal = false;
    head.window_size = Some(8); // avoid shared accumulator pool in mock tests
    head.startup.diagnostics.first_append = variant;
    head.startup.diagnostics.no_decode_append = false;
    head
}

pub(crate) fn append(
    head: &BlockDiffusionDraftHead,
    state: &mut DflashProposerState,
    stack: DevicePtr,
    position: usize,
    gpu: &dyn GpuBackend,
) {
    head.append_decode_ctx(state, Some(stack), position, gpu, gpu.default_stream())
        .unwrap();
}
