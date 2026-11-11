// SPDX-License-Identifier: AGPL-3.0-only

//! The mixed forward (a prefill chunk fused with the active sequences'
//! decode) against speculation, on the scripted model and the real continue
//! phase. The stub's `mixed_forward` is the trait default, a `decode_batch`
//! then the chunk, so `decode_calls` counts mixed steps.

use super::super::test_support::RespRx;
use super::queue_first_chunk::{Tick, chunks, request};
use super::*;

/// A 40-token prompt 16 tokens in, beside `n` decoding sequences.
fn mid_prefill<'a>(model: &'a PreemptStubModel, n: usize) -> (Tick<'a>, Vec<RespRx>) {
    let mut t = Tick::new(model, false, false);
    let (req, rx) = request(40);
    t.admit(vec![req]);
    assert_eq!(t.queue(), vec![(40, 16, None)]);
    let mut rxs = vec![rx];
    for slot in 1..=n {
        let (a, rx) = active_seq(slot, 3);
        t.active.push(a);
        rxs.push(rx);
    }
    (t, rxs)
}

/// A DFlash serve speculates at every width and its mixed forward leaves the
/// prefilling sequence without drafter context, so the chunk runs alone.
#[test]
fn a_dflash_serve_runs_the_chunk_alone_beside_two_decoders() {
    let model = PreemptStubModel::default();
    let (mut t, _rxs) = mid_prefill(&model, 2);
    t.step_spec(true, true);
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 0);
    assert_eq!(chunks(&model), vec![(0, 16), (16, 16)]);
    assert_eq!(t.queue(), vec![(40, 32, None)]);
    assert_eq!(t.active.len(), 2);
}

/// One decoder speculating on any lane takes its own step: no mixed step.
#[test]
fn a_lone_speculating_decoder_is_not_mixed() {
    let model = PreemptStubModel::default();
    let (mut t, _rxs) = mid_prefill(&model, 1);
    t.step_spec(true, false);
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 0);
    assert_eq!(chunks(&model), vec![(0, 16), (16, 16)]);
}

/// The single-sequence lanes decode plainly past one decoder, where the
/// mixed step still applies.
#[test]
fn two_decoders_on_a_single_sequence_lane_still_mix() {
    let model = PreemptStubModel::default();
    let (mut t, _rxs) = mid_prefill(&model, 2);
    t.step_spec(true, false);
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 1);
    assert_eq!(chunks(&model), vec![(0, 16), (16, 16)]);
}
