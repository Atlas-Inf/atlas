// SPDX-License-Identifier: AGPL-3.0-only

//! Batched DSpark verify: one `decode_verify_batched` over n sequences,
//! then the existing DSpark accept-prefix per sequence.
//!
//! Propose stays per-seq in this cut (Phase 4 batches it). Kill switch
//! `ATLAS_NO_DFLASH_BATCH_VERIFY` (presence) keeps the serial loop.

use super::*;
use std::time::Instant;
use spark_model::traits::SequenceState;

pub(super) fn dspark_batch_verify_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_NO_DFLASH_BATCH_VERIFY").is_ok())
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
    let results: Vec<u32> = {
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
    tracing::info!(
        "DFLASH BATCHED verify n={n} R={} {:.1}ms",
        acc,
        verify_ms
    );

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
        apply_dflash_accept(
            model,
            a,
            sched,
            drafts,
            &verified,
            propose_nd,
            dflash_verify_raw_argmax,
        );
    }
}

/// Shared DSpark accept / emit / UNIFIED_CTX / re-propose. `seq` has already
/// been advanced by `drafts.len()+1` (same contract as `decode_verify_dflash`).
pub(super) fn apply_dflash_accept(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    verified: &[u32],
    num_drafts: usize,
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

    if sched.levers.dflash_unified_ctx {
        if let Err(e) = model.commit_ctx(&mut a.seq, num_accepted + 1, pre_verify_len) {
            tracing::error!("commit_ctx (kgamma batched): {e:#}");
        }
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

    let _gmask = mtp_grammar_mask_for(a);
    if crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            num_drafts,
            &mut a.seq,
            0,
            _gmask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => tracing::error!("run_mtp_propose_multi (dflash batched): {e:#}"),
        }
    }
}
