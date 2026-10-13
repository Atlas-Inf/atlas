// SPDX-License-Identifier: AGPL-3.0-only

//! A step's pending flags are all-or-nothing: the forward follows them.

use super::*;

fn state(pending: bool) -> SsmLayerState {
    SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        kda_records: DevicePtr::NULL,
        conv_state_intermediates: Vec::new(),
        gdn_commit_qkv: DevicePtr(0x3000),
        gdn_commit_gb: DevicePtr(0x4000),
        gdn_commit_pending: pending,
        gdn_fuse_n: DevicePtr::NULL,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    }
}

fn pending(flags: &[bool], ks: &[usize]) -> Result<bool> {
    let mut own: Vec<SsmLayerState> = flags.iter().map(|&p| state(p)).collect();
    let mut states: Vec<&mut (dyn LayerState + 'static)> = own
        .iter_mut()
        .map(|s| s as &mut (dyn LayerState + 'static))
        .collect();
    let gdn = GdnStates::Multi {
        states: &mut states,
        ks,
        wy_tables: DevicePtr::NULL,
    };
    Qwen3SsmLayer::gdn_step_pending(&gdn, ks.iter().sum())
}

#[test]
fn a_step_is_pending_only_when_every_sequence_is() {
    assert!(!pending(&[false, false, false], &[4, 3, 1]).unwrap());
    assert!(pending(&[true, true, true], &[4, 3, 1]).unwrap());
    assert!(pending(&[true], &[8]).unwrap());
}

#[test]
fn a_mixed_step_is_an_error_not_a_fallback() {
    // One pending sequence beside a storing one: whichever kernel ran, one
    // of them would commit from the wrong state.
    assert!(pending(&[true, false], &[4, 4]).is_err());
    assert!(pending(&[false, true, false], &[2, 2, 2]).is_err());
}

#[test]
fn the_rows_must_add_up() {
    let mut own = [state(true), state(true)];
    let mut states: Vec<&mut (dyn LayerState + 'static)> = own
        .iter_mut()
        .map(|s| s as &mut (dyn LayerState + 'static))
        .collect();
    let gdn = GdnStates::Multi {
        states: &mut states,
        ks: &[4, 4],
        wy_tables: DevicePtr::NULL,
    };
    assert!(Qwen3SsmLayer::gdn_step_pending(&gdn, 7).is_err());
}

#[test]
fn a_single_sequence_step_reads_its_flag() {
    let mut s = state(true);
    let gdn = GdnStates::Single(&mut s);
    assert!(Qwen3SsmLayer::gdn_step_pending(&gdn, 4).unwrap());
}
