// SPDX-License-Identifier: AGPL-3.0-only

use super::dflash_ctx_serializes_with;
use crate::scheduler::levers::SchedLevers;

// The production levers (`from_env` with no ATLAS_* set) arm unified ctx,
// which serializes the MTP bootstrap unless the switch scopes it to DFlash.
#[test]
fn unified_ctx_serializes_only_when_unscoped() {
    let mut levers = SchedLevers::defaults();
    levers.dflash_unified_ctx = true;
    assert!(dflash_ctx_serializes_with(&levers, false));
    assert!(!dflash_ctx_serializes_with(&levers, true));
}

#[test]
fn serial_append_follows_the_same_rule() {
    let mut levers = SchedLevers::defaults();
    levers.dflash_serial_append = true;
    assert!(dflash_ctx_serializes_with(&levers, false));
    assert!(!dflash_ctx_serializes_with(&levers, true));
    assert!(!dflash_ctx_serializes_with(&SchedLevers::defaults(), false));
}
