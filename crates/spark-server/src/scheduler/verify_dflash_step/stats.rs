// SPDX-License-Identifier: AGPL-3.0-only

//! The drafter/acceptance telemetry one DFlash verify feeds (split out of
//! `verify_dflash_tail` so a strict verify carrying only a dead draft can
//! skip it).

use crate::scheduler::ActiveSeq;

pub(super) fn record(
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    (drafts, draft_conf): (&[u32], &[f32]),
    verified: &[u32],
    num_accepted: usize,
    dflash_verify_raw_argmax: bool,
) {
    crate::scheduler::mtp_accept_debug::record(
        1,
        drafts.len(),
        drafts.first() == verified.first(),
        num_accepted,
    );
    if !dflash_verify_raw_argmax {
        a.mtp_acct.record_depth_verify(
            drafts.len(),
            num_accepted,
            sched.levers.mtp_single_depth_adapt,
        );
    }

    // Adaptive speculation (ATLAS_DFLASH_ADAPTIVE=1): feed the rolling
    // accept window; may suspend this seq's speculation (see adaptive_spec).
    crate::scheduler::adaptive_spec::record_verify(a, num_accepted, sched);
    a.spec_adapt.survival.record(drafts.len(), num_accepted);
    crate::scheduler::dflash_conf_width::record(draft_conf, drafts.len(), num_accepted);
}
