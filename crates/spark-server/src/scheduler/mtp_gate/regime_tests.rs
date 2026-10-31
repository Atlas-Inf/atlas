// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn widening_with_startup_cost(steady_verify_ms: u64) -> MtpGate {
    let mut gate = MtpGate::new(3);
    for _ in 0..WINDOW_STEPS {
        gate.record_verify_step(Duration::from_millis(50), 2, 1);
    }
    // Seven newly admitted sequences bootstrap at serial cost. This cost
    // belongs to MTP, but it is not a full steady-throughput window.
    gate.record_verify_step(Duration::from_millis(800), 8, 8);
    for _ in 0..80 {
        match gate.next_step() {
            GateStep::MeasureVerify => {
                gate.record_verify_step(Duration::from_millis(steady_verify_ms), 20, 8)
            }
            GateStep::MeasureDecode => gate.record_decode(Duration::from_millis(146), 8),
        }
    }
    gate
}

#[test]
fn widening_bootstrap_cost_does_not_disable_faster_steady_mtp() {
    let gate = widening_with_startup_cost(240);
    assert!(
        !gate.in_serial_mode(),
        "steady MTP delivers 83.3 tok/s versus serial 54.8; startup must not trap it in Serial: MTP={:?}, serial={:?}",
        gate.mtp_tps_debug(),
        gate.serial_tps_debug(),
    );
}

#[test]
fn widening_still_selects_serial_when_steady_mtp_is_slower() {
    let gate = widening_with_startup_cost(500);
    assert!(
        gate.in_serial_mode(),
        "40 tok/s MTP must lose to 54.8 tok/s serial"
    );
}
