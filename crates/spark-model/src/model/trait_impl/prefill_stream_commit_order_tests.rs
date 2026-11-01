// SPDX-License-Identifier: AGPL-3.0-only
//! The default-stream work that must be ordered after a verify commit still
//! in flight on the secondary stream. Split out of `prefill_stream_tests.rs`
//! (500-LoC cap).

use super::*;

/// The default stream waits on the secondary event, then `then` happens.
fn after_commit(order: &[Order], secondary_event: u64, then: &Order) -> bool {
    let wait = Order::Wait(DEFAULT, secondary_event);
    let at = |o| order.iter().position(|x| x == o);
    matches!((at(&wait), at(then)), (Some(w), Some(t)) if w < t)
}

/// A verify commit folds its accepted rows into the SSM slot on the secondary
/// stream. The head orders its next decode after it in the scheduler
/// (`sync_secondary`); no wire command carries that, so the worker orders
/// its own plain decode.
#[test]
fn the_worker_orders_a_plain_decode_after_the_verify_commit() {
    for ep in [1, 2] {
        let mut f = Fixture::new(2, ep, 1);
        let event = f.model.secondary_event;
        // Command word 0 is token 0: a plain decode in the addressed slot.
        let err = f.model.ep_worker_dispatch_cmd(0, &mut f.seq).unwrap_err();
        assert!(format!("{err:#}").contains("stream fixture"), "{err:#}");
        assert!(after_commit(&f.order(), event, &Order::Decode), "EP{ep}");
    }
}

/// A verify step that finishes a sequence commits on the secondary stream and
/// the sequence is freed (or its slot compacted) straight away: the zero and
/// the slot copy on the default stream must come after that commit.
#[test]
fn freeing_and_compacting_a_slot_come_after_the_verify_commit() {
    let f = Fixture::new(1, 1, 0);
    let event = f.model.secondary_event;
    let mut seq = f.model.alloc_sequence().unwrap();
    f.order();
    f.model.free_sequence(&mut seq).unwrap();
    // The zero is synced on the host right after it is issued.
    assert!(after_commit(&f.order(), event, &Order::Sync(DEFAULT)));

    let mut seq = f.model.alloc_sequence().unwrap();
    let to = seq.slot_idx + 1;
    f.order();
    f.model.compact_sequence(&mut seq, to).unwrap();
    // Likewise the slot copy.
    assert!(after_commit(&f.order(), event, &Order::Sync(DEFAULT)));
}
