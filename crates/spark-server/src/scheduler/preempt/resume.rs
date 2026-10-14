// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;
use super::PREEMPT_IMMUNITY_TOKENS;

/// Resume a requeued victim by re-prefilling its token history.
///
/// Rebuilds KV and SSM state for `tokens` (prompt + processed output) on a
/// freshly allocated sequence; `a.last_token` — already streamed to the
/// client — stays the pending decode input, so nothing is re-sampled or
/// re-emitted and the stream continues exactly where it paused. The prefill
/// logits are discarded for the same reason. On error the client has
/// already been notified and freed — the caller just logs.
pub(in crate::scheduler) fn resume_preempted_seq(
    model: &dyn Model,
    p: PreemptedSeq,
) -> Result<ActiveSeq> {
    let PreemptedSeq { mut a, tokens } = p;
    let mut seq = match model.alloc_sequence() {
        Ok(s) => s,
        Err(e) => {
            send_error_to_sink(&mut a.sink, &format!("preempt-resume alloc failed: {e:#}"));
            return Err(e);
        }
    };
    seq.session_hash = a.session_hash;
    seq.disable_mtp = a.disable_mtp;
    seq.adapter_slot = a.seq.adapter_slot;
    // Task #24/#25 parity with the swap-in path: keep the STABLE adapter_id
    // stamped at the original prefill and re-acquire the slot ref released
    // by the preempt-time free. (Preempted seqs also hold the LoRA
    // quiescence gate in `run`, so the adapter cannot rotate in between.)
    seq.adapter_id = a.seq.adapter_id;
    seq.acquired_adapter_slot = model.acquire_adapter_slot(a.seq.adapter_slot);

    // EP: mirror the non-chunked prefill preamble so the worker mirrors the
    // re-prefill (no-ops on non-EP models).
    let prefill_result = (|| -> Result<()> {
        model.ep_broadcast_disable_mtp_for_seq(seq.slot_idx as u32, a.disable_mtp)?;
        model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF0)?;
        model.ep_broadcast_cmd(tokens.len() as u32)?;
        model.ep_broadcast_cmd(0)?;
        model.ep_broadcast_cmd(tokens.len() as u32)?;
        model.ep_broadcast_tokens(&tokens)?;
        model.prefill(&tokens, &mut seq, 0)?;
        Ok(())
    })();
    if let Err(e) = prefill_result {
        a.seq = seq;
        send_error(
            model,
            &mut a,
            &format!("preempt-resume re-prefill failed: {e:#}"),
        );
        return Err(e);
    }
    // NOTE: `seq.prompt_len` now spans the WHOLE history — deliberately left
    // as prefill stamped it, because it must describe what prefill actually
    // ref-bumped for `cache_sequence`'s prompt/generated split; restoring
    // the original prompt length would double-bump the pre-preemption
    // output blocks at finish and pin them forever. The min_tokens floor is
    // an absolute position, so it carries over to the new owner unchanged.
    a.seq.eos_ban.arm(&mut seq);
    a.seq = seq;
    // The rollback ring's GPU snapshots died with the old slot; start empty
    // (identical to the disk-swap resume path).
    a.ssm_rollback_ring = SsmDecodeRing::new(model.decode_rollback_ring_slots());
    a.preempt_immune_until_tokens = a.output_tokens.len() + PREEMPT_IMMUNITY_TOKENS;
    Ok(a)
}

/// Swapped sequences released their GPU ownership when spilled. Remove dead
/// buffered callers and their spill files before restoring any GPU state.
pub(in crate::scheduler) fn discard_disconnected_swapped(
    swapped: &mut Vec<SwappedSeq>,
    spill: &mut KvSpillManager,
) {
    swapped.retain(|s| {
        if !s.sink.receiver_closed() {
            return true;
        }
        if let Err(e) = spill.remove_file(s.swap_id) {
            if e.downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
            {
                return false;
            }
            tracing::warn!("discard disconnected spill: {e:#}");
            // Retain ownership so a later sweep can retry cleanup. The
            // swap-in selector excludes closed sinks even on cleanup failure.
            return true;
        }
        false
    });
}

/// Resume requeued victims while KV blocks and batch slots allow.
///
/// Cheapest-first (fewest blocks to re-prefill). Gated on the history
/// fitting with one block of growth headroom, reclaiming from the prefix
/// cache exactly like the swap-in loop — cached blocks never volunteer
/// themselves (see `reclaim_prefix_blocks` docs).
pub(in crate::scheduler) fn resume_preempted_seqs(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    preempted: &mut Vec<PreemptedSeq>,
    max_batch_size: usize,
    block_size: usize,
) {
    // GPU ownership was released at preemption. Drop disconnected callers
    // before reclaiming cache blocks or rebuilding their entire history.
    preempted.retain(|p| !p.a.sink.receiver_closed());
    while !preempted.is_empty() && active.len() < max_batch_size {
        let Some((idx, needed)) = preempted
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.tokens.len() / block_size.max(1) + 1))
            .min_by_key(|&(_, n)| n)
        else {
            return;
        };
        // +1 growth block of headroom: resuming into an exactly-full pool
        // would just re-preempt on the next decode step.
        let want = needed + 1;
        let total = model.num_total_blocks();
        if total > 0 && want > total {
            // Can never fit — erroring is honest; queueing forever is not.
            let mut p = preempted.remove(idx);
            send_error_to_sink(
                &mut p.a.sink,
                &format!(
                    "preempted sequence needs {want} KV blocks but the pool has {total}; \
                     cannot resume"
                ),
            );
            continue;
        }
        let mut free = model.num_free_blocks();
        while free < want {
            let got = model.reclaim_prefix_blocks(want - free);
            if got == 0 {
                break;
            }
            free = model.num_free_blocks();
        }
        if free < want {
            return;
        }
        let p = preempted.remove(idx);
        let n_tokens = p.tokens.len();
        match resume_preempted_seq(model, p) {
            Ok(a) => {
                tracing::info!(
                    "Preempt-resume: re-prefilled {n_tokens} tokens \
                     ({} generated so far), decode continues",
                    a.output_tokens.len(),
                );
                active.push(a);
            }
            Err(e) => tracing::error!("Preempt-resume failed: {e:#}"),
        }
    }
}
