// SPDX-License-Identifier: AGPL-3.0-only

use super::run_supervised;
use atlas_core::fault::FaultLatch;

#[test]
fn panic_with_str_payload_latches_the_fault() {
    let latch = FaultLatch::new();
    let panicked = run_supervised(&latch, || panic!("prefill CUDA stream gone"));
    assert!(panicked);
    let reason = latch.fault().expect("fault must be latched");
    assert!(reason.contains("scheduler thread panicked"));
    assert!(reason.contains("prefill CUDA stream gone"));
}

#[test]
fn panic_with_string_payload_latches_the_fault() {
    let latch = FaultLatch::new();
    let panicked = run_supervised(&latch, || {
        std::panic::panic_any(String::from("owned payload"));
    });
    assert!(panicked);
    let reason = latch.fault().expect("fault must be latched");
    assert!(reason.contains("scheduler thread panicked"));
    assert!(reason.contains("owned payload"));
}

#[test]
fn clean_body_returns_false_and_leaves_the_latch_clear() {
    let latch = FaultLatch::new();
    let panicked = run_supervised(&latch, || ());
    assert!(!panicked);
    assert!(!latch.is_faulted());
}

#[test]
fn second_fault_keeps_the_first_reason() {
    let latch = FaultLatch::new();
    latch.latch("original fault".to_string());
    let panicked = run_supervised(&latch, || panic!("later panic"));
    assert!(panicked);
    assert_eq!(latch.fault(), Some("original fault"));
}
