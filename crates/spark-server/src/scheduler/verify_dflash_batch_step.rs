// SPDX-License-Identifier: AGPL-3.0-only

//! Batched DSpark verify: one `decode_verify_batched` over n sequences,
//! then the existing DSpark accept-prefix per sequence.
//!
//! Propose stays per-seq in this cut (Phase 4 batches it). Kill switch
//! `ATLAS_NO_DFLASH_BATCH_VERIFY` (presence) keeps the serial loop.

use super::*;
use spark_model::traits::SequenceState;
use std::time::Instant;

pub(super) fn dspark_batch_verify_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_NO_DFLASH_BATCH_VERIFY").is_ok())
}

/// Lightning product fail-closed for the batched DSpark propose paths:
/// an empty or erroneous (batched or per-seq fallback) proposal marks the
/// sequence with the client-visible truncation guard and finishes it.
/// Generic DFlash/MTP keeps the legacy log-and-continue behavior.
fn fail_closed_if_lightning(
    model: &dyn spark_model::traits::Model,
    a: &mut super::types::ActiveSeq,
    site: &'static str,
    _slot_idx: usize,
) {
    crate::scheduler::helpers::handle_dspark_batched_proposal_failure(model, a, site);
}

/// n>=2 DSpark sequences, each with `ks[i]-1` pending drafts (uniform K=3
/// → ks=4). Grammarless. Caller sorted / classified.
pub(super) fn step_verify_dflash_batched(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    ks: &[usize],
    propose_nd: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    let n = batch.len();
    debug_assert_eq!(ks.len(), n);
    debug_assert!(n >= 2);

    if let Err(e) = model.sync_secondary() {
        tracing::error!("dflash-batched sync_secondary: {e:#}");
        for a in batch.iter_mut() {
            a.finished = true;
        }
        return;
    }

    let mut off: Vec<usize> = Vec::with_capacity(n + 1);
    let mut acc = 0usize;
    for &k in ks {
        off.push(acc);
        acc += k;
    }
    off.push(acc);

    let mut drafts_per_seq: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut tokens: Vec<u32> = Vec::with_capacity(acc);
    for (i, a) in batch.iter_mut().enumerate() {
        let d = std::mem::take(&mut a.pending_drafts);
        a.pending_draft_conf.clear();
        debug_assert_eq!(d.len() + 1, ks[i]);
        tokens.push(a.last_token);
        tokens.extend_from_slice(&d);
        drafts_per_seq.push(d);
    }

    let t_verify = Instant::now();
    let results: Vec<u32> = if std::env::var("ATLAS_DFLASH_VERIFY_COMPUTE_SERIAL").as_deref()
        == Ok("1")
    {
        // Diagnostic boundary: retain the batched scheduler/accept/commit path
        // while replacing only the target forward with the proven per-sequence
        // K=4 verifier. This distinguishes compute drift from batched verdict
        // bookkeeping without changing drafts, row slices, or re-propose.
        let mut all = Vec::with_capacity(acc);
        for i in 0..n {
            match model.decode_verify_dflash(&tokens[off[i]..off[i + 1]], &mut batch[i].seq, 0) {
                Ok(mut r) => all.append(&mut r),
                Err(e) => {
                    tracing::error!("decode_verify_dflash serial diagnostic (i={i}): {e:#}");
                    for a in batch.iter_mut() {
                        a.finished = true;
                    }
                    return;
                }
            }
        }
        all
    } else {
        let mut seq_refs: Vec<&mut SequenceState> = batch.iter_mut().map(|a| &mut a.seq).collect();
        match model.decode_verify_batched(&tokens, ks, &mut seq_refs, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_verify_batched dflash (n={n} ks={ks:?}): {e:#}");
                for a in batch.iter_mut() {
                    a.finished = true;
                }
                return;
            }
        }
    };
    let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;
    tracing::info!("DFLASH BATCHED verify n={n} R={} {:.1}ms", acc, verify_ms);

    let mut verifieds: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut accepts: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let a = &mut batch[i];
        let drafts = &drafts_per_seq[i];
        let raw = &results[off[i]..off[i + 1]];
        let verified: Vec<u32> = if dflash_verify_raw_argmax && !sched.levers.dflash_masked_verify {
            raw.to_vec()
        } else {
            crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
                model, raw, a, verify_ctx, off[i],
            )
        };
        let mut num_accepted = 0usize;
        for j in 0..drafts.len() {
            if j + 1 >= verified.len() {
                break;
            }
            if drafts[j] == verified[j] {
                num_accepted += 1;
            } else {
                break;
            }
        }
        accepts.push(num_accepted);
        verifieds.push(verified);
    }
    tracing::info!(
        "DFLASH BATCHED verify n={n} R={} {:.1}ms accept={:?}",
        acc,
        verify_ms,
        accepts
    );
    let stash_rows: Vec<usize> = accepts
        .iter()
        .enumerate()
        .map(|(i, &acc_n)| off[i] + acc_n)
        .collect();
    if let Err(e) = model.stash_verify_hidden_rows(&stash_rows, 0) {
        tracing::error!("stash_verify_hidden_rows (dflash): {e:#}");
    }
    for i in 0..n {
        if let Err(e) = model.pack_dflash_save_seq(i, ks[i], 0) {
            tracing::error!("pack_dflash_save_seq({i}): {e:#}");
        }
        apply_dflash_accept(
            model,
            batch[i],
            sched,
            &drafts_per_seq[i],
            &verifieds[i],
            propose_nd,
            dflash_verify_raw_argmax,
        );
    }
    if let Err(e) = model.restore_dflash_save_front(ks[0], 0) {
        tracing::error!("restore_dflash_save_front: {e:#}");
        for a in batch.iter_mut() {
            a.finished = true;
        }
        return;
    }
    let t_propose = Instant::now();
    let pending: Vec<usize> = (0..n)
        .filter(|&i| !batch[i].finished && batch[i].pending_drafts.is_empty())
        .collect();
    if pending.len() >= 2 {
        let tokens: Vec<u32> = pending.iter().map(|&i| batch[i].last_token).collect();
        let positions: Vec<usize> = pending.iter().map(|&i| batch[i].seq.seq_len).collect();
        let stash_idx: Vec<usize> = pending.clone();
        let result = {
            let mut seq_refs: Vec<&mut SequenceState> = Vec::with_capacity(pending.len());
            let mut it = batch.iter_mut();
            let mut prev = 0usize;
            for (j, &i) in pending.iter().enumerate() {
                let step = if j == 0 { i } else { i - prev - 1 };
                seq_refs.push(&mut it.nth(step).expect("pending in batch").seq);
                prev = i;
            }
            model.run_mtp_propose_batched(
                &tokens,
                &positions,
                &stash_idx,
                propose_nd,
                &mut seq_refs,
                0,
                None,
            )
        };
        match result {
            Ok(Some(all)) => {
                for (j, &i) in pending.iter().enumerate() {
                    if !all[j].is_empty() {
                        batch[i].pending_drafts = all[j].clone();
                    } else {
                        let slot = batch[i].seq.slot_idx;
                        let target = &mut batch[i];
                        fail_closed_if_lightning(model, target, "returned empty drafts", slot);
                    }
                }
            }
            Ok(None) | Err(_) => {
                // Lightning product: the batched proposer declined or
                // errored. The per-sequence fallback below is a diagnostic
                // recovery path for generic DFlash; a product serve treats
                // the batch failure itself as an admission violation and
                // fails every pending sequence closed rather than
                // continuing on a degraded path.
                let product_fail_closed =
                    crate::scheduler::helpers::dspark_proposal_failure_fails_closed(
                        model.is_lightning_dspark_product(),
                    );
                for &i in &pending {
                    if product_fail_closed {
                        let slot = batch[i].seq.slot_idx;
                        let target = &mut batch[i];
                        fail_closed_if_lightning(
                            model,
                            target,
                            "batched proposer declined/errored",
                            slot,
                        );
                        continue;
                    }
                    if let Err(e) = model.save_hidden_for_mtp_from_stash(i, 0) {
                        tracing::error!("save_hidden_for_mtp_from_stash({i}): {e:#}");
                        continue;
                    }
                    match model.run_mtp_propose_multi(
                        batch[i].last_token,
                        batch[i].seq.seq_len,
                        propose_nd,
                        &mut batch[i].seq,
                        0,
                        None,
                    ) {
                        Ok(d) if !d.is_empty() => batch[i].pending_drafts = d,
                        Ok(_) => {
                            let slot = batch[i].seq.slot_idx;
                            let target = &mut batch[i];
                            fail_closed_if_lightning(
                                model,
                                target,
                                "fallback returned empty drafts",
                                slot,
                            )
                        }
                        Err(e) => {
                            tracing::error!("run_mtp_propose_multi fallback: {e:#}");
                            let slot = batch[i].seq.slot_idx;
                            let target = &mut batch[i];
                            fail_closed_if_lightning(model, target, "fallback errored", slot);
                        }
                    }
                }
            }
        }
    } else if let Some(&i) = pending.first() {
        if let Err(e) = model.save_hidden_for_mtp_from_stash(i, 0) {
            tracing::error!("save_hidden_for_mtp_from_stash({i}): {e:#}");
            // Lightning product: a stash failure means the drafter cannot
            // propose; fail closed instead of leaving the sequence to a
            // silent serial bootstrap on the next step.
            let slot = batch[i].seq.slot_idx;
            let target = &mut batch[i];
            fail_closed_if_lightning(model, target, "stash save failed (single)", slot);
        } else {
            match model.run_mtp_propose_multi(
                batch[i].last_token,
                batch[i].seq.seq_len,
                propose_nd,
                &mut batch[i].seq,
                0,
                None,
            ) {
                Ok(d) if !d.is_empty() => batch[i].pending_drafts = d,
                Ok(_) => {
                    let slot = batch[i].seq.slot_idx;
                    let target = &mut batch[i];
                    fail_closed_if_lightning(model, target, "single returned empty drafts", slot);
                }
                Err(e) => {
                    tracing::error!("run_mtp_propose_multi: {e:#}");
                    let slot = batch[i].seq.slot_idx;
                    let target = &mut batch[i];
                    fail_closed_if_lightning(model, target, "single errored", slot);
                }
            }
        }
    }
    tracing::info!(
        "DFLASH BATCHED propose n={n} pending={} {:.1}ms",
        pending.len(),
        t_propose.elapsed().as_secs_f64() * 1000.0
    );
}

/// Shared DSpark accept / emit / UNIFIED_CTX / re-propose. `seq` has already
/// been advanced by `drafts.len()+1` (same contract as `decode_verify_dflash`).
pub(super) fn apply_dflash_accept(
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

    for i in 0..num_accepted {
        emit_token(a, drafts[i], None, sched);
        if a.finished {
            return;
        }
    }
    let bonus_idx = num_accepted;
    if bonus_idx < verified.len() {
        let bonus = verified[bonus_idx];
        emit_token(a, bonus, None, sched);
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

    let k_verify = drafts.len() + 1;
    let total_accepted = num_accepted + 1;
    if let Err(e) = model.commit_accepted_prefix(&mut a.seq, total_accepted, k_verify) {
        tracing::error!("commit_accepted_prefix (dflash batched): {e:#}");
        a.finished = true;
        return;
    }
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
