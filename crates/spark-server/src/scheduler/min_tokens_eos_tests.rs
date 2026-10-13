// SPDX-License-Identifier: AGPL-3.0-only

//! An end token below the `min_tokens` floor is handled identically by serial
//! decode (`process_decode_logits`) and by the MTP/verify emission
//! (`emit_token`, every K=2/3/4 verdict and the batched paths): discarded,
//! never recorded in `output_tokens`, so it never counts toward the floor.
//!
//! The MTP path used to record it before testing the floor. A run that ends
//! its answer and then keeps predicting the end token (fed back as the next
//! input, as serial decode feeds it) filled the floor with discarded end
//! tokens: depth-3 prose with `min_tokens=384` stopped after 184 visible
//! tokens, where serial decode and K=1 returned 384.

use crate::scheduler::{
    cancel_test_model as model, decode_logits_step::process_decode_logits, emit_step::emit_token,
    sched_ctx::SchedCtx, test_support::test_seq, types::ActiveSeq,
};
use spark_runtime::gpu::DevicePtr;
use std::time::Instant;

/// The fixture model's sampled winner; registered as the end token here.
pub(super) const END: u32 = 101;

/// A live, uncancelled sequence (the response receiver is leaked: a dropped
/// one reads as a client disconnect and finishes the sequence).
pub(super) fn seq(output: usize, min_tokens: usize) -> ActiveSeq {
    let (mut a, rx) = test_seq((0..output as u32).map(|i| 1000 + i).collect(), 64, None, 40);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = min_tokens;
    a.eos_tokens = vec![END];
    a.inside_thinking = false;
    a.think_ended = true;
    a.require_tool_call = false;
    a.tool_request = false;
    a.tools_present = false;
    a
}

/// One end token through serial decode or through the MTP emission.
pub(super) fn step(a: ActiveSeq, serial: bool, sched: &SchedCtx) -> ActiveSeq {
    let mut rows = vec![a];
    if serial {
        process_decode_logits(
            &model::TestModel {
                tokens: vec![END],
                host_logits: true,
                cancel_after_sampling: None,
                cancel_after_row_commit: None,
                verify: None,
            },
            &mut rows,
            DevicePtr::NULL,
            Instant::now(),
            None,
            None,
            None,
            None,
            None,
            false,
            sched,
        );
    } else {
        emit_token(&mut rows[0], END, None, sched);
    }
    rows.pop().unwrap()
}

#[test]
fn an_end_token_below_the_floor_is_discarded_alike_by_serial_and_mtp() {
    let sched = SchedCtx::for_test();
    for (output, floor) in [(3, 5), (4, 5), (5, 5), (0, 0), (3, 0)] {
        let (s, m) = (
            step(seq(output, floor), true, &sched),
            step(seq(output, floor), false, &sched),
        );
        let below = output < floor;
        assert_eq!(s.finished, !below, "serial at {output}/{floor}");
        assert_eq!(
            m.finished, s.finished,
            "mtp vs serial finish at {output}/{floor}"
        );
        assert_eq!(
            m.output_tokens.len(),
            s.output_tokens.len(),
            "at {output}/{floor}"
        );
        if below {
            assert_eq!(
                m.output_tokens.len(),
                output,
                "a discarded end token is not output"
            );
        }
    }
}

#[test]
fn repeated_end_tokens_never_fill_the_floor() {
    // The failing shape: the answer is over, the model keeps predicting the
    // end token. Each is discarded; the floor still needs real tokens.
    let sched = SchedCtx::for_test();
    let mut a = seq(3, 5);
    for _ in 0..4 {
        a = step(a, false, &sched);
        assert!(!a.finished, "a discarded end token reached the floor");
        assert_eq!(a.output_tokens.len(), 3);
    }
    for t in [7u32, 8] {
        emit_token(&mut a, t, None, &sched);
    }
    assert_eq!(a.output_tokens.len(), 5);
    a = step(a, false, &sched);
    assert!(a.finished, "the floor is met: the end token stops");
}

#[test]
fn a_role_boundary_below_the_floor_is_discarded_alike_by_serial_and_mtp() {
    // `<|im_start|>` is registered as an end token AND as the MTP path's
    // ChatML hard stop. After the answer's discarded `<|im_end|>` the model
    // writes the next role header; serial decode discards that `<|im_start|>`
    // below the floor, and the MTP path used to end the turn on it ("stop"
    // at 182 of min_tokens=384, depth-3 prose). An explicit floor wins there
    // too; above it the hard stop is unchanged.
    let mut sched = SchedCtx::for_test();
    sched.limits.im_start_hard_stop = Some(END);
    for (output, floor) in [(3, 5), (4, 5), (5, 5), (3, 0)] {
        let (s, m) = (
            step(seq(output, floor), true, &sched),
            step(seq(output, floor), false, &sched),
        );
        let below = output < floor;
        assert_eq!(m.finished, !below, "mtp at {output}/{floor}");
        assert_eq!(m.finished, s.finished, "mtp vs serial at {output}/{floor}");
        if below {
            assert_eq!(m.output_tokens.len(), output, "discarded, not output");
            assert_eq!(s.output_tokens.len(), output);
        }
    }
}
