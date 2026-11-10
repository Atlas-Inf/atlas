// SPDX-License-Identifier: AGPL-3.0-only

//! Exact DFlash tree verify for a lone sequence (`ATLAS_DFLASH_TREE=<rows>`,
//! default off; `spark_model::speculative::draft_tree`).
//!
//! In place of the chain's γ rows the step verifies a best-first draft tree
//! of at most `<rows>` rows (2..=16) built from the drafter's own lattice
//! (`Model::dflash_draft_tree`), every row serial decode's logits for the
//! path ending at it (`Model::decode_verify_dflash_tree`). The accepted path
//! follows each row's pick into the child carrying it; its rows are moved
//! into path order (`Model::dflash_tree_gather_rows`) and picked through
//! serial decode's pipeline exactly as the chain's exact verify picks
//! (`verify_pick_prefix_with_pipeline`: thinking budget, watchdogs,
//! penalties), which stops at the first row whose pick is not the path's
//! next token. The commit then leaves KV, recurrent state, the drafter's
//! context and the sequence as a chain accepting that path would.
//!
//! Only where the chain would take the exact raw-argmax lane and nothing
//! scores or masks rows: a greedy request with no grammar and no logprobs,
//! on a model that serves the tree verify. Everything else, and every
//! batched (C>1) verify, keeps the chain.

use super::*;
use spark_model::speculative::draft_tree::{DraftTree, rows_lever};

/// The tree to verify for `a` this step, when the step may take one.
fn tree_for(
    model: &dyn Model,
    a: &ActiveSeq,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    raw: bool,
) -> Option<DraftTree> {
    let rows = rows_lever();
    if rows == 0
        || !exact_picks(raw, model.verify_bit_exact())
        || model.native_mtp_enabled()
        || a.grammar_state.is_some()
        || a.top_logprobs.is_some()
        || !(a.temperature == 0.0 || verify_ctx.sampling.force_temp_zero)
        || !model.dflash_tree_supported()
    {
        return None;
    }
    match model.dflash_draft_tree(rows) {
        // The lattice must be this sequence's: drafted after its last token.
        Ok(Some(t)) if t.rows() >= 2 && t.tokens[0] == a.last_token => Some(t),
        Ok(_) => None,
        Err(e) => {
            tracing::warn!("dflash_draft_tree: {e:#}; verifying the chain");
            None
        }
    }
}

/// Verify a lone sequence's step as a draft tree. Returns whether it ran
/// (`false`: nothing done, the caller verifies the chain).
pub(super) fn try_step_verify_tree(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    raw: bool,
) -> bool {
    let Some(tree) = tree_for(model, a, verify_ctx, raw) else {
        return false;
    };
    let _step_timer = crate::scheduler::mtp_timing::StepTimer::new(&sched.timing, a.seq.seq_len);
    if let Err(e) = model.sync_secondary() {
        a.abort_on_engine_error(format_args!("sync_secondary: {e:#}"));
        return true;
    }
    let step_timing = std::env::var("ATLAS_DFLASH_STEP_TIMING").ok().as_deref() == Some("1");
    let k = tree.rows();
    let t_verify = std::time::Instant::now();
    let argmax = match model.decode_verify_dflash_tree(&tree, &mut a.seq) {
        Ok(v) => v,
        Err(e) => {
            a.abort_on_engine_error(format_args!("decode_verify_dflash_tree: {e:#}"));
            return true;
        }
    };
    sched
        .timing
        .record(crate::scheduler::mtp_timing::Phase::VerifyForward, t_verify);
    let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;
    a.last_token_time = Instant::now();

    // The raw picks' path, its rows in order, then serial decode's picks.
    let raw_path = tree.walk(&argmax);
    if let Err(e) = model.dflash_tree_gather_rows(&raw_path) {
        a.abort_on_engine_error(format_args!("dflash_tree_gather_rows: {e:#}"));
        return true;
    }
    let path_argmax: Vec<u32> = raw_path.iter().map(|&r| argmax[r]).collect();
    let drafts: Vec<u32> = raw_path[1..].iter().map(|&r| tree.tokens[r]).collect();
    // Present the sequence as a chain verify of the path's rows leaves it:
    // the pick pipeline places row `i` at `seq_len - rows + i + 1` (its
    // ceilings, the min_tokens floor) and reads the span's tokens.
    let pre_verify_len = a.seq.seq_len.saturating_sub(k);
    let tokens_before = a.seq.tokens.len().saturating_sub(k);
    a.seq.tokens.truncate(tokens_before);
    a.seq
        .tokens
        .extend(raw_path.iter().map(|&r| tree.tokens[r]));
    a.seq.seq_len = pre_verify_len + raw_path.len();
    let verified = crate::scheduler::verify_pipeline_helper::verify_pick_prefix_with_pipeline(
        model,
        &path_argmax,
        &drafts,
        a,
        verify_ctx,
        0,
    );
    let num_accepted = drafts
        .iter()
        .zip(&verified)
        .take_while(|(d, v)| d == v)
        .count()
        .min(verified.len().saturating_sub(1));
    stats::record(a, sched, (&drafts, &[]), &verified, num_accepted, raw);

    if let Err(e) = model.commit_dflash_tree(&mut a.seq, &raw_path[..=num_accepted]) {
        a.abort_on_engine_error(format_args!("commit_dflash_tree: {e:#}"));
        return true;
    }
    if sched.levers.dflash_unified_ctx {
        if let Err(e) = model.commit_ctx(&mut a.seq, num_accepted + 1, pre_verify_len) {
            tracing::error!("commit_ctx (tree): {e:#}");
        }
    } else if std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1")
        && let Err(e) = model.dflash_eagle_kgamma_append(&mut a.seq, num_accepted, pre_verify_len)
    {
        tracing::error!("dflash_eagle_kgamma_append (tree): {e:#}");
    }

    let total_accepted = num_accepted + 1;
    for (i, &d) in drafts.iter().take(num_accepted).enumerate() {
        emit_span_row(a, d, None, sched, i, total_accepted);
        if a.finished {
            return true;
        }
    }
    if let Some(&bonus) = verified.get(num_accepted) {
        emit_span_row(a, bonus, None, sched, num_accepted, total_accepted);
        if a.finished {
            return true;
        }
        a.last_token = bonus;
    }
    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            "dflash_tree",
            if num_accepted == drafts.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();
    if !native_mtp::save_and_repair(
        model,
        a,
        &drafts,
        num_accepted,
        pre_verify_len,
        num_accepted,
    ) {
        return true;
    }
    propose_after_verify(
        model,
        a,
        sched,
        num_drafts,
        raw,
        false,
        num_accepted,
        (step_timing, verify_ms, k, num_accepted),
    );
    true
}
