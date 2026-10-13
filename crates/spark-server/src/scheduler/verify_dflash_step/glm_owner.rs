// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched GLM verify: one target traversal, then each owner's tail.

use super::*;

/// One owner-group target traversal: `(rows, tokens, seqs) -> verified`.
type OwnerTraverse<'a> =
    dyn FnMut(usize, &[u32], &mut [&mut SequenceState]) -> anyhow::Result<Vec<u32>> + 'a;

/// Owner-batched GLM verify (repaired K3 or DFlash block): one target
/// traversal for every owner, then each owner's ordinary tail in order, each
/// preceded by restoring that owner's verify rows on both ranks.
pub fn step_verify_glm_long_batched(
    model: &dyn Model,
    group: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    step_verify_glm_long_with(
        model,
        group,
        sched,
        num_drafts,
        verify_ctx,
        dflash_verify_raw_argmax,
        &mut |rows, tokens, seqs| model.decode_verify_glm_long_owner_rows(rows, tokens, seqs),
    );
}

/// [`step_verify_glm_long_batched`] with the owners' target traversal
/// supplied: `traverse(rows, owner-major tokens, seqs)` must leave each owner
/// advanced by `rows` rows and its final rows staged for
/// `begin_glm_long_owner_tail`, returning the owner-major argmax IDs (as
/// `decode_verify_glm_long_owner_rows` does, or a prefill chunk carrying the
/// owners).
#[allow(clippy::too_many_arguments)]
pub fn step_verify_glm_long_with(
    model: &dyn Model,
    group: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
    traverse: &mut OwnerTraverse<'_>,
) {
    let fail_all = |group: &mut [&mut ActiveSeq]| group.iter_mut().for_each(|a| a.finished = true);
    let t_step = Instant::now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        fail_all(group);
        return;
    }
    let sync_ms = t_step.elapsed().as_secs_f64() * 1000.0;
    let (drafts, confs): (Vec<Vec<u32>>, Vec<Vec<f32>>) =
        group.iter_mut().map(|a| a.take_drafts()).unzip();
    let rows = drafts[0].len() + 1;
    let tokens: Vec<Vec<u32>> = group
        .iter()
        .zip(&drafts)
        .map(|(a, d)| {
            std::iter::once(a.last_token)
                .chain(d.iter().copied())
                .collect()
        })
        .collect();
    let flat: Vec<u32> = tokens.concat();
    let positions: Vec<usize> = group.iter().map(|a| a.seq.seq_len).collect();
    let step_timing = std::env::var("ATLAS_DFLASH_STEP_TIMING").ok().as_deref() == Some("1");
    let t_verify = std::time::Instant::now();
    let verified = {
        let mut seqs: Vec<&mut SequenceState> = group.iter_mut().map(|a| &mut a.seq).collect();
        traverse(rows, &flat, &mut seqs)
    };
    let verified = match verified {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("GLM owner traversal (n={} rows={rows}): {e:#}", group.len());
            fail_all(group);
            return;
        }
    };
    sched
        .timing
        .record(crate::scheduler::mtp_timing::Phase::VerifyForward, t_verify);
    let verify_ms = if step_timing {
        t_verify.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    // DFlash captures each owner's verify rows into its stable hidden-save
    // slot. A tail's commit_ctx reads the front slot, so each owner's region is
    // packed there first; slot 0 IS the front, so its owner runs last, after
    // the preserved front is restored.
    let n = group.len();
    let save_slots: Vec<Option<usize>> = group
        .iter()
        .map(|a| {
            dflash_verify_raw_argmax
                .then(|| a.seq.dflash_hidden_save_slot().ok())
                .flatten()
        })
        .collect();
    let front_owner = (0..n).find(|&o| save_slots[o] == Some(0));
    let order: Vec<usize> = (0..n)
        .filter(|&o| Some(o) != front_owner)
        .chain(front_owner)
        .collect();
    if front_owner.is_some()
        && let Err(e) = model.preserve_dflash_save_front(rows, 0)
    {
        tracing::error!("preserve_dflash_save_front: {e:#}");
        fail_all(group);
        return;
    }
    let mut deferred: Vec<(usize, usize)> = Vec::new();
    for (done, &owner) in order.iter().enumerate() {
        let a = &mut *group[owner];
        let _step_timer =
            crate::scheduler::mtp_timing::StepTimer::new(&sched.timing, positions[owner]);
        let begun = model
            .begin_glm_long_owner_tail(a.seq.slot_idx as u32, owner, &tokens[owner])
            .and_then(|()| front_save_slot(model, save_slots[owner], rows));
        if let Err(e) = begun {
            tracing::error!("begin_glm_long_owner_tail (owner={owner}): {e:#}");
            for &o in &order[done..] {
                group[o].finished = true;
            }
            return;
        }
        a.last_token_time = Instant::now();
        if let Some(next) = verify_dflash_tail(
            model,
            a,
            sched,
            &drafts[owner],
            &confs[owner],
            num_drafts,
            verify_ctx,
            dflash_verify_raw_argmax,
            &tokens[owner],
            verified[owner * rows..(owner + 1) * rows].to_vec(),
            step_timing,
            verify_ms,
            dflash_verify_raw_argmax,
            true,
        ) {
            deferred.push((owner, next));
        }
    }
    propose_owner_batch(
        model,
        group,
        &deferred,
        &save_slots,
        rows,
        sched,
        step_timing,
    );
    if step_timing {
        tracing::info!(
            "GLM OWNER STEP_TIMING: owners={} rows={rows} sync={sync_ms:.1}ms verify={verify_ms:.1}ms total={:.1}ms",
            group.len(),
            t_step.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Put `slot`'s hidden-save region at the front the DFlash commit and
/// per-sequence propose read (restoring the preserved front for slot 0).
fn front_save_slot(model: &dyn Model, slot: Option<usize>, rows: usize) -> anyhow::Result<()> {
    match slot {
        Some(0) => model.restore_dflash_save_front(rows, 0),
        Some(slot) => model.pack_dflash_save_seq(slot, rows, 0),
        None => Ok(()),
    }
}

/// One drafter pass for every owner whose tail deferred its re-propose
/// (`(owner, drafts)`), falling back to per-sequence proposals when the
/// proposer declines or the draft counts differ.
fn propose_owner_batch(
    model: &dyn Model,
    group: &mut [&mut ActiveSeq],
    deferred: &[(usize, usize)],
    save_slots: &[Option<usize>],
    rows: usize,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    step_timing: bool,
) {
    if deferred.is_empty() {
        return;
    }
    let t_propose = Instant::now();
    let num_drafts = deferred[0].1;
    let mut owners: Vec<usize> = deferred.iter().map(|&(o, _)| o).collect();
    owners.sort_unstable();
    let batched = if owners.len() >= 2 && deferred.iter().all(|&(_, d)| d == num_drafts) {
        let tokens: Vec<u32> = owners.iter().map(|&o| group[o].last_token).collect();
        let positions: Vec<usize> = owners.iter().map(|&o| group[o].seq.seq_len).collect();
        let stash = vec![0usize; owners.len()];
        let mut seqs: Vec<&mut SequenceState> = group
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| owners.binary_search(i).is_ok())
            .map(|(_, a)| &mut a.seq)
            .collect();
        model.run_mtp_propose_batched(
            &tokens, &positions, &stash, num_drafts, &mut seqs, 0, None, None,
        )
    } else {
        Ok(None)
    };
    let fail = |a: &mut ActiveSeq, outcome| {
        crate::scheduler::helpers::handle_dspark_repropose_failure(model, a, outcome);
        if spark_model::speculative::glm_repair_policy::enabled() {
            a.finished = true;
        }
    };
    match batched {
        Ok(Some(all)) if all.len() == owners.len() => {
            for (&o, d) in owners.iter().zip(all) {
                if d.is_empty() {
                    fail(group[o], crate::scheduler::helpers::ProposalOutcome::Empty);
                } else {
                    group[o].set_proposed_drafts(d);
                }
            }
        }
        other => {
            if let Err(e) = other {
                tracing::error!("run_mtp_propose_batched (owners={}): {e:#}", owners.len());
            }
            // Slot 0 last: its region is the (restored) front.
            let front = deferred.iter().position(|&(o, _)| save_slots[o] == Some(0));
            let order = (0..deferred.len())
                .filter(|&i| Some(i) != front)
                .chain(front);
            for i in order {
                let (o, drafts) = deferred[i];
                let a = &mut *group[o];
                let proposal = front_save_slot(model, save_slots[o], rows).and_then(|()| {
                    model.run_mtp_propose_multi(
                        a.last_token,
                        a.seq.seq_len,
                        drafts,
                        &mut a.seq,
                        0,
                        None,
                    )
                });
                match proposal {
                    Ok(d) if !d.is_empty() => a.set_proposed_drafts(d),
                    Ok(_) => fail(a, crate::scheduler::helpers::ProposalOutcome::Empty),
                    Err(e) => {
                        tracing::error!("run_mtp_propose_multi (owner {o}): {e:#}");
                        fail(a, crate::scheduler::helpers::ProposalOutcome::Error);
                    }
                }
            }
        }
    }
    sched
        .timing
        .record(crate::scheduler::mtp_timing::Phase::Propose, t_propose);
    if step_timing {
        tracing::info!(
            "GLM OWNER PROPOSE: owners={} {:.1}ms",
            deferred.len(),
            t_propose.elapsed().as_secs_f64() * 1000.0
        );
    }
}
