// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence DFlash raw-argmax verify policy (behind `dflash_seq_uses_raw_argmax`).

use super::dflash_verify_uses_raw_argmax;

pub(super) fn dflash_seq_raw_argmax_policy(
    raw_requested: bool,
    masked_requested: bool,
    lightning_product: bool,
    argmax_only_head: bool,
    a: &mut crate::scheduler::types::ActiveSeq,
    argmax_ids: &[u32],
) -> bool {
    if argmax_only_head {
        return true;
    }
    // Per-row gate: the slow path advances `inside_tool_body` across the K
    // picks, so a mid-block `</tool_call>` changes later rows' params —
    // a block-start classification is wrong (#175, BFCL parallel_multiple_78).
    dflash_verify_uses_raw_argmax(raw_requested, masked_requested, lightning_product)
        && a.grammar_state.is_none()
        && !a.require_tool_call
        && crate::scheduler::fast_greedy::fast_gate_over_rows(a, argmax_ids)
            == crate::scheduler::fast_greedy::PenaltyGate::Neutral
}

#[cfg(test)]
mod tests {
    #[test]
    fn per_sequence_gate_stays_raw_only_for_neutral_greedy() {
        use super::dflash_seq_raw_argmax_policy as gate;
        let (mut a, _rx) = crate::scheduler::test_support::test_seq(vec![1, 2, 3], 10, None, 32);
        // Neutral penalties + no grammar + no tool-call EOS suppression:
        // the verify pick is provably the raw argmax — shortcut stays on.
        assert!(gate(true, false, false, false, &mut a, &[]));
        // The non-thinking preset ships presence_penalty=1.5 — a reduce-only
        // penalty that CAN flip the argmax against already-emitted tokens.
        a.presence_penalty = 1.5;
        assert!(!gate(true, false, false, false, &mut a, &[]));
        a.presence_penalty = 0.0;
        a.repetition_penalty = 1.05;
        assert!(!gate(true, false, false, false, &mut a, &[]));
        a.repetition_penalty = 1.0;
        a.dry_multiplier = 0.4;
        assert!(!gate(true, false, false, false, &mut a, &[]));
        a.dry_multiplier = 0.0;
        a.require_tool_call = true;
        assert!(!gate(true, false, false, false, &mut a, &[]));
        // Process-wide flag still dominates: raw off stays off per-seq.
        a.require_tool_call = false;
        assert!(!gate(false, false, false, false, &mut a, &[]));
    }

    #[test]
    fn argmax_only_verify_head_always_takes_the_raw_path() {
        use super::dflash_seq_raw_argmax_policy as gate;
        let (mut a, _rx) = crate::scheduler::test_support::test_seq(vec![1, 2, 3], 10, None, 32);
        // A head that leaves the logits buffer partly stale (GLM's TP2 vocab
        // split) cannot feed the pipeline, whatever the request asks for.
        a.presence_penalty = 1.5;
        a.require_tool_call = true;
        assert!(gate(true, false, false, true, &mut a, &[]));
        assert!(gate(false, true, false, true, &mut a, &[]));
        assert!(!gate(true, false, false, false, &mut a, &[]));
    }
}
