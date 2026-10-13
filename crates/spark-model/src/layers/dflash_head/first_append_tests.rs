// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_FIRST_APPEND`: the parser, the pure source decision, and
//! the accumulator bookkeeping of the real append path on the mock backend.

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::first_append::{AppendSource, FirstAppend, Preceding, decode_append_source};
use super::free_state_tests::zero_head;
use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::speculative::{DraftProposer, ProposerState};

pub(super) const NON_LEGACY: [FirstAppend; 3] =
    [FirstAppend::None, FirstAppend::Own, FirstAppend::Zero];
pub(super) const ALL: [FirstAppend; 4] = [
    FirstAppend::Legacy,
    FirstAppend::None,
    FirstAppend::Own,
    FirstAppend::Zero,
];
const PRECEDING: [Preceding; 4] = [
    Preceding::Prefill,
    Preceding::Decode,
    Preceding::Verify,
    Preceding::Other,
];
const STACK: DevicePtr = DevicePtr(0x1000);
const OWN_ROW: DevicePtr = DevicePtr(0x2000);

#[test]
fn parser_defaults_to_legacy_and_accepts_the_four_variants() {
    assert_eq!(FirstAppend::parse(None).unwrap(), FirstAppend::Legacy);
    assert_eq!(FirstAppend::default(), FirstAppend::Legacy);
    for (raw, variant) in [
        ("legacy", FirstAppend::Legacy),
        ("none", FirstAppend::None),
        ("own", FirstAppend::Own),
        ("zero", FirstAppend::Zero),
    ] {
        assert_eq!(FirstAppend::parse(Some(raw)).unwrap(), variant, "{raw}");
    }
}

#[test]
fn parser_rejects_every_other_value() {
    for raw in ["", "0", "1", "Own", "none ", "off", "legacy,own"] {
        let error = FirstAppend::parse(Some(raw)).unwrap_err().to_string();
        assert!(
            error.contains("ATLAS_DFLASH_FIRST_APPEND"),
            "{raw:?}: {error}"
        );
        assert!(error.contains(&format!("{raw:?}")), "{raw:?}: {error}");
    }
}

/// Legacy is today's condition verbatim, whatever this sequence ran last and
/// whether or not it has a row of its own.
#[test]
fn legacy_source_is_exactly_the_old_branch() {
    for bits in 0u8..16 {
        let [suppressed, has_stack, room, has_own_row] = [0, 1, 2, 3].map(|b| bits & (1 << b) != 0);
        let old_branch = !suppressed && has_stack && room;
        for preceding in PRECEDING {
            assert_eq!(
                decode_append_source(
                    FirstAppend::Legacy,
                    suppressed,
                    has_stack.then_some(STACK),
                    room,
                    preceding,
                    has_own_row.then_some(OWN_ROW),
                ),
                old_branch.then_some(AppendSource::Row(STACK)),
                "bits={bits:04b} {preceding:?}"
            );
        }
    }
}

fn source(variant: FirstAppend, preceding: Preceding) -> Option<AppendSource> {
    decode_append_source(variant, false, Some(STACK), true, preceding, Some(OWN_ROW))
}

#[test]
fn first_propose_source_per_variant() {
    assert_eq!(source(FirstAppend::None, Preceding::Prefill), None);
    assert_eq!(
        source(FirstAppend::Own, Preceding::Prefill),
        Some(AppendSource::Row(OWN_ROW))
    );
    assert_eq!(
        source(FirstAppend::Zero, Preceding::Prefill),
        Some(AppendSource::Zero)
    );
}

/// A non-legacy variant reads the caller's (model-global) row only right
/// after this sequence's own verify; its own decode is appended from its own
/// row, and with neither there is nothing to append.
#[test]
fn non_legacy_reads_the_global_row_only_after_its_own_verify() {
    for variant in NON_LEGACY {
        for preceding in PRECEDING {
            assert_eq!(
                source(variant, preceding) == Some(AppendSource::Row(STACK)),
                preceding == Preceding::Verify,
                "{variant:?} {preceding:?}"
            );
        }
        assert_eq!(
            source(variant, Preceding::Decode),
            Some(AppendSource::Row(OWN_ROW)),
            "{variant:?}"
        );
        assert_eq!(source(variant, Preceding::Other), None, "{variant:?}");
        // Without a row of its own a sequence appends nothing after a decode.
        assert_eq!(
            decode_append_source(variant, false, Some(STACK), true, Preceding::Decode, None),
            None,
            "{variant:?}"
        );
    }
}

/// The legacy gates (commit already appended, no stack, full accumulator)
/// apply to every variant.
#[test]
fn legacy_gates_apply_to_every_variant() {
    for variant in NON_LEGACY {
        for preceding in PRECEDING {
            let source = |suppressed, stack, room| {
                decode_append_source(variant, suppressed, stack, room, preceding, Some(OWN_ROW))
            };
            let case = format!("{variant:?} {preceding:?}");
            assert_eq!(source(true, Some(STACK), true), None, "{case}");
            assert_eq!(source(false, None, true), None, "{case}");
            assert_eq!(source(false, Some(STACK), false), None, "{case}");
        }
    }
}

pub(super) const SLOT: usize = 16;
pub(super) const WINDOW: usize = 4;
pub(super) const PROMPT: usize = 2;

pub(super) fn head(variant: FirstAppend) -> BlockDiffusionDraftHead {
    let mut head = zero_head();
    head.target_layer_ids = vec![0, 1];
    head.target_hidden_size = SLOT / 4;
    head.max_seq_len = WINDOW;
    head.startup.diagnostics.first_append = variant;
    head.startup.diagnostics.no_decode_append = false;
    head
}

pub(super) fn dstate(state: &mut Box<dyn ProposerState>) -> &mut DflashProposerState {
    state.as_any_mut().downcast_mut().unwrap()
}

/// A sequence right after a `PROMPT`-token prefill: slots `0..PROMPT` stamped
/// with their positions and the first append armed at the prompt end. The
/// slot the append lands in starts dirty so a zero append is observable.
pub(super) fn after_prefill(
    head: &BlockDiffusionDraftHead,
    gpu: &MockGpuBackend,
) -> Box<dyn ProposerState> {
    let mut state = head.alloc_state(gpu).unwrap();
    let d = dstate(&mut state);
    d.ctx_len = PROMPT;
    d.ctx_positions = (0..PROMPT as i32).collect();
    d.first_append_at = Some(PROMPT);
    gpu.memset(d.ctx_hidden_acc.offset(PROMPT * SLOT), 0xEE, SLOT)
        .unwrap();
    if let Some(row) = d.own_row {
        gpu.memset(row, 0xC0, SLOT).unwrap();
    }
    state
}

/// The model-global capture row: another request's hidden stack.
pub(super) fn global_row(gpu: &MockGpuBackend) -> DevicePtr {
    let row = gpu.alloc(SLOT).unwrap();
    gpu.memset(row, 0xAB, SLOT).unwrap();
    row
}

pub(super) fn slot(gpu: &MockGpuBackend, d: &DflashProposerState, index: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; SLOT];
    gpu.copy_d2h(d.ctx_hidden_acc.offset(index * SLOT), &mut bytes)
        .unwrap();
    bytes
}

/// (ctx_len, ctx_positions, bytes of the slot after the prompt) after the
/// first propose of a sequence under `variant`.
fn first_propose(variant: FirstAppend) -> (usize, Vec<i32>, Vec<u8>) {
    let gpu = MockGpuBackend::new();
    let head = head(variant);
    let mut state = after_prefill(&head, &gpu);
    let global = global_row(&gpu);
    let d = dstate(&mut state);
    head.append_decode_ctx(d, Some(global), PROMPT, &gpu, 0)
        .unwrap();
    assert_eq!(d.first_append_at, None, "{variant:?}");
    assert_eq!(d.ctx_positions.len(), d.ctx_len, "{variant:?}");
    (d.ctx_len, d.ctx_positions.clone(), slot(&gpu, d, PROMPT))
}

#[test]
fn legacy_first_propose_appends_the_global_row() {
    let (ctx_len, positions, appended) = first_propose(FirstAppend::Legacy);
    assert_eq!((ctx_len, positions), (PROMPT + 1, vec![0, 1, 1]));
    assert_eq!(appended, vec![0xAB; SLOT]);
}

#[test]
fn none_first_propose_leaves_the_prefill_context_alone() {
    let (ctx_len, positions, untouched) = first_propose(FirstAppend::None);
    assert_eq!((ctx_len, positions), (PROMPT, vec![0, 1]));
    assert_eq!(untouched, vec![0xEE; SLOT]);
}

#[test]
fn own_first_propose_appends_the_sequences_own_row_with_the_legacy_stamp() {
    let (ctx_len, positions, appended) = first_propose(FirstAppend::Own);
    assert_eq!((ctx_len, positions), (PROMPT + 1, vec![0, 1, 1]));
    assert_eq!(appended, vec![0xC0; SLOT]);
}

#[test]
fn zero_first_propose_appends_a_zero_row_with_the_legacy_stamp() {
    let (ctx_len, positions, appended) = first_propose(FirstAppend::Zero);
    assert_eq!((ctx_len, positions), (PROMPT + 1, vec![0, 1, 1]));
    assert_eq!(appended, vec![0; SLOT]);
}

/// A propose that is not the one right after prefill (a serial stretch ran
/// in between) gets the first-append action in no variant; legacy still
/// appends the global row.
#[test]
fn only_the_propose_at_the_prefill_end_is_the_first() {
    for variant in ALL {
        let gpu = MockGpuBackend::new();
        let head = head(variant);
        let mut state = after_prefill(&head, &gpu);
        let global = global_row(&gpu);
        let d = dstate(&mut state);
        head.append_decode_ctx(d, Some(global), PROMPT + 1, &gpu, 0)
            .unwrap();
        assert_eq!(
            d.first_append_at, None,
            "{variant:?}: consumed by any propose"
        );
        let legacy = variant == FirstAppend::Legacy;
        assert_eq!(d.ctx_len, PROMPT + legacy as usize, "{variant:?}");
    }
}

/// A commit that already appended the capture suppresses the append and
/// every one-shot marker is consumed with it.
#[test]
fn a_commit_suppresses_the_append_and_consumes_the_markers() {
    for variant in ALL {
        let gpu = MockGpuBackend::new();
        let head = head(variant);
        let mut state = after_prefill(&head, &gpu);
        let global = global_row(&gpu);
        let d = dstate(&mut state);
        d.skip_next_decode_append = true;
        d.own_capture = true;
        d.own_row_at = Some(PROMPT);
        head.append_decode_ctx(d, Some(global), PROMPT, &gpu, 0)
            .unwrap();
        assert_eq!(
            (d.ctx_len, d.ctx_positions.len()),
            (PROMPT, PROMPT),
            "{variant:?}"
        );
        assert!(!d.skip_next_decode_append && !d.own_capture, "{variant:?}");
        assert_eq!(
            (d.first_append_at, d.own_row_at),
            (None, None),
            "{variant:?}"
        );
    }
}

/// A prompt that fills the window leaves no room: no variant appends.
#[test]
fn a_full_accumulator_appends_in_no_variant() {
    for variant in ALL {
        let gpu = MockGpuBackend::new();
        let head = head(variant);
        let mut state = after_prefill(&head, &gpu);
        let global = global_row(&gpu);
        let d = dstate(&mut state);
        d.ctx_len = WINDOW;
        d.ctx_positions = (0..WINDOW as i32).collect();
        d.first_append_at = Some(WINDOW);
        let copies = gpu.d2d_count() + gpu.memset_count();
        head.append_decode_ctx(d, Some(global), WINDOW, &gpu, 0)
            .unwrap();
        assert_eq!(
            (d.ctx_len, d.ctx_positions.len()),
            (WINDOW, WINDOW),
            "{variant:?}"
        );
        assert_eq!(gpu.d2d_count() + gpu.memset_count(), copies, "{variant:?}");
    }
}

/// Every non-legacy variant gets the spare row (its own decode captures land
/// there); legacy allocations are byte-for-byte what they were.
#[test]
fn alloc_state_reserves_the_own_row_for_non_legacy_only() {
    for variant in ALL {
        let gpu = MockGpuBackend::new();
        let mut state = head(variant).alloc_state(&gpu).unwrap();
        let d = dstate(&mut state);
        let own = variant != FirstAppend::Legacy;
        assert_eq!(d.max_ctx_len, WINDOW, "{variant:?}");
        assert_eq!(
            gpu.read_alloc(d.ctx_hidden_acc).unwrap().len(),
            (WINDOW + own as usize) * SLOT,
            "{variant:?}"
        );
        assert_eq!(
            d.own_row,
            own.then(|| d.ctx_hidden_acc.offset(WINDOW * SLOT)),
            "{variant:?}"
        );
        assert_eq!(
            (d.first_append_at, d.own_row_at, d.own_capture),
            (None, None, false),
            "{variant:?}"
        );
    }
}

/// `ATLAS_GLM_DRAFT_TP_CTX` announces a propose's context rows before it
/// runs: read from the untouched state, they are the rows the append leaves
/// past the committed watermark, whatever the markers and the room.
#[test]
fn the_pending_rows_are_what_the_propose_appends_past_the_watermark() {
    for variant in ALL {
        for bits in 0u8..32 {
            let gpu = MockGpuBackend::new();
            let head = head(variant);
            let mut state = after_prefill(&head, &gpu);
            let global = global_row(&gpu);
            let d = dstate(&mut state);
            let [skip, verify, decode, full, committed] =
                [0, 1, 2, 3, 4].map(|b| bits >> b & 1 != 0);
            d.skip_next_decode_append = skip;
            d.own_capture = verify;
            d.own_row_at = decode.then_some(PROMPT);
            if full {
                d.ctx_len = WINDOW;
                d.ctx_positions = (0..WINDOW as i32).collect();
            }
            d.ctx_committed = if committed { 1 } else { 0 };
            let pending = head.pending_ctx_rows(d, Some(global), PROMPT);
            head.append_decode_ctx(d, Some(global), PROMPT, &gpu, 0)
                .unwrap();
            assert_eq!(
                pending,
                d.ctx_len - d.ctx_committed,
                "{variant:?} bits={bits:05b}"
            );
        }
    }
}
