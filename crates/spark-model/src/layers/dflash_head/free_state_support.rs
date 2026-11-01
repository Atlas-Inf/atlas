// SPDX-License-Identifier: AGPL-3.0-only

//! Test-only `DflashProposerState` builders shared by the dflash_head test
//! modules — moved out of `free_state_tests.rs` for the 500-LoC cap (no
//! behaviour change).

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::lifecycle::CaptureDescriptor;
use super::{DflashProposerState, SequenceGeneration};

/// A state as `alloc_state` returns it: accumulator allocated, every ctx
/// watermark at zero, no paged blocks. This is the shape `adopt_dflash_ctx`
/// sees in production (the call site guards on `ctx_len == 0`).
#[allow(dead_code)]
pub(super) fn fresh_state(gpu: &MockGpuBackend) -> Box<DflashProposerState> {
    fresh_state_with_owner(gpu, None)
}

/// Fresh state with a lifecycle bound, as `alloc_seq` leaves it before the
/// first ctx seed — `adopt_dflash_ctx` reads the owner off this to re-key
/// carried graphs.
#[allow(dead_code)]
pub(super) fn fresh_state_with_owner(
    gpu: &MockGpuBackend,
    own: Option<SequenceGeneration>,
) -> Box<DflashProposerState> {
    let mut s = fresh_state_inner(gpu);
    if let Some(own) = own {
        s.lifecycle = Some(CaptureDescriptor::bind(own, 0, 0, 4, 16).unwrap());
    }
    s
}

#[allow(dead_code)]
pub(super) fn fresh_state_inner(gpu: &MockGpuBackend) -> Box<DflashProposerState> {
    Box::new(DflashProposerState {
        block_table: Vec::new(),
        seq_len: 0,
        last_num_drafted: 0,
        last_draft_conf: Vec::new(),
        prefill_done: false,
        ctx_hidden_acc: gpu.alloc(4096).unwrap(),
        ctx_len: 0,
        last_num_accepted: 0,
        skip_next_decode_append: false,
        max_ctx_len: 1024,
        ctx_acc_rows: 4096,
        ctx_prefill_origin: None,
        ctx_prefill_base: 0,
        ctx_slot_bytes: 64,
        block_table_dev: None,
        ctx_count_drafter: 0,
        max_ctx_count_drafter: 0,
        ctx_committed: 0,
        ctx_positions: Vec::new(),
        end_floor: 0,
        first_append_at: None,
        own_capture: false,
        own_row_at: None,
        own_row: None,
        warm_ctx: super::WarmCtx::Legacy,
        ctx_floor: 0,
        ctx_run: None,
        ctx_store: super::warm_ctx_store::StoreRefs::default(),
        lane_id: 0,
        lifecycle: None,
    })
}
