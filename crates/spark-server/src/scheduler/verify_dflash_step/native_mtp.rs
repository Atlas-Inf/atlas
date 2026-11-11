// SPDX-License-Identifier: AGPL-3.0-only

//! Preserve target rows, then repair native GLM private MTP state.
use super::*;

pub(super) fn save_and_repair(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_accepted: usize,
    pre_verify_len: usize,
    bonus_token_idx: usize,
) -> bool {
    let native_mtp = model.native_mtp_enabled();
    // DFlash hidden is captured per-layer inside the verify graph
    // (verify_d.rs try_dflash_capture at position k-1), mirroring verify_b.rs.
    // No post-loop save needed; calling save_dflash_hidden_for_propose here
    // would overwrite the correct per-layer intermediates with a repeated
    // final-layer hidden, collapsing all 5 slots to the same value.
    if let Err(e) = model.save_hidden_for_mtp(bonus_token_idx, 0) {
        tracing::error!("save_hidden_for_mtp (dflash): {e:#}");
        if native_mtp {
            a.abort_on_engine_error(format_args!(
                "native GLM MTP bonus hidden staging failed: {e:#}"
            ));
            return false;
        }
        if spark_model::speculative::glm_repair_policy::enabled() {
            a.finished = true;
            return false;
        }
    }

    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
        if native_mtp {
            a.abort_on_engine_error(format_args!(
                "native GLM MTP private-state trim failed: {e:#}"
            ));
            return false;
        }
        if spark_model::speculative::glm_repair_policy::enabled() {
            a.finished = true;
            return false;
        }
    }

    // Native GLM's private rows were generated from predictor hidden states.
    // Keep the verified input pair, then rebuild every accepted draft row from
    // the corresponding canonical target norm row before any propose clobbers
    // the shared target buffers. The bonus row was copied above into the
    // separate MTP hidden-save buffer and is unaffected by these repairs.
    if native_mtp {
        for (target_row, &accepted_draft) in drafts.iter().take(num_accepted).enumerate() {
            if let Err(e) = model.repair_mtp_after_verify_row(
                accepted_draft,
                pre_verify_len + 1 + target_row,
                target_row,
                &mut a.seq,
                0,
            ) {
                a.abort_on_engine_error(format_args!(
                    "native GLM MTP canonical repair row {target_row} failed: {e:#}"
                ));
                return false;
            }
        }
    }

    true
}
