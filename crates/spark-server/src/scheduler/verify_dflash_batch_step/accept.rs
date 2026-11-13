// SPDX-License-Identifier: AGPL-3.0-only

//! The per-sequence DSpark accept of a batched verify (`step_verify_dflash_batched`).

use super::*;

/// Shared DSpark accept / emit / UNIFIED_CTX / re-propose. `seq` has already
/// been advanced by `drafts.len()+1` (same contract as `decode_verify_dflash`).
pub(in crate::scheduler) fn apply_dflash_accept(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    verified: &[u32],
    _num_drafts: usize,
    _dflash_verify_raw_argmax: bool,
) {
    let mut num_accepted = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            num_accepted += 1;
        } else {
            break;
        }
    }
    crate::scheduler::adaptive_spec::record_verify(a, num_accepted, sched);

    let tokens_len = drafts.len() + 1;
    let pre_verify_len = a.seq.seq_len.saturating_sub(tokens_len);
    let target_seq_len = pre_verify_len + num_accepted + 1;
    let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
    if to_drop > 0 {
        a.seq.seq_len = target_seq_len;
        let pop_n = to_drop.min(a.seq.tokens.len());
        for _ in 0..pop_n {
            a.seq.tokens.pop();
        }
    }

    if sched.levers.dflash_unified_ctx
        && let Err(e) = model.commit_ctx(&mut a.seq, num_accepted + 1, pre_verify_len)
    {
        tracing::error!("commit_ctx (kgamma batched): {e:#}");
    }

    // Committed before emission (see `verify_dflash_tail`): an emit that
    // finishes the sequence returns, and finish caches the committed state.
    let k_verify = drafts.len() + 1;
    let total_accepted = num_accepted + 1;
    if let Err(e) = model.commit_accepted_prefix(&mut a.seq, total_accepted, k_verify) {
        tracing::error!("commit_accepted_prefix (dflash batched): {e:#}");
        a.engine_error = Some(format!("{e:#}"));
        a.finished = true;
        return;
    }

    for i in 0..num_accepted {
        emit_span_row(a, drafts[i], None, sched, i, total_accepted);
        if a.finished {
            return;
        }
    }
    let bonus_idx = num_accepted;
    if bonus_idx < verified.len() {
        let bonus = verified[bonus_idx];
        emit_span_row(a, bonus, None, sched, bonus_idx, total_accepted);
        if a.finished {
            return;
        }
        a.last_token = bonus;
    }

    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            "dflash",
            if num_accepted == drafts.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();

    let bonus_token_idx = total_accepted.saturating_sub(1);
    if let Err(e) = model.save_hidden_for_mtp(bonus_token_idx, 0) {
        tracing::error!("save_hidden_for_mtp (dflash batched): {e:#}");
    }
    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state (dflash batched): {e:#}");
    }
    // Propose is deferred to step_verify_dflash_batched so every seq
    // uses stash hiddens + eager propose_batch (no shared-graph clobber).
}
