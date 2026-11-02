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

/// Whether this sequence's DFlash verify picks come from the argmax-only
/// split head's rows (`verify_pipeline_helper::split_rows`): the head can
/// sample (`verify_split_sampling`) and the pick is not the raw argmax, so a
/// sampled request is one even where `dflash_seq_uses_raw_argmax` keeps
/// other heads greedy. A strict structured-output greedy pick is the raw
/// argmax: the split head masks its rows (`strict_spec`).
pub fn dflash_seq_uses_split_rows(
    raw_requested: bool,
    masked_requested: bool,
    model: &dyn spark_model::traits::Model,
    a: &mut crate::scheduler::types::ActiveSeq,
    argmax_ids: &[u32],
) -> bool {
    model.verify_logits_argmax_only()
        && model.verify_split_sampling()
        && !(a.strict_grammar() && a.temperature == 0.0)
        && (a.temperature > 0.0
            || !dflash_seq_raw_argmax_policy(
                raw_requested,
                masked_requested,
                model.is_lightning_dspark_product(),
                false,
                a,
                argmax_ids,
            ))
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

    /// A split head that can sample picks from its rows every sequence the
    /// raw argmax would not serve, a sampled one included; other heads never.
    #[test]
    fn a_sampling_split_head_takes_every_non_raw_pick_from_its_rows() {
        use super::dflash_seq_uses_split_rows as split;
        use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel, VerifyScript};
        let model = |verify: Option<std::sync::Arc<VerifyScript>>| TestModel {
            tokens: vec![],
            host_logits: false,
            cancel_after_sampling: None,
            cancel_after_row_commit: None,
            verify,
        };
        let split_head = model(Some(Default::default()));
        let (mut a, _rx) = crate::scheduler::test_support::test_seq(vec![1, 2, 3], 10, None, 32);
        let script = |on: bool| {
            SCRIPT.with(|s| {
                *s.borrow_mut() = on.then(|| Script {
                    rows: vec![],
                    cursor: 0,
                })
            })
        };
        script(true);
        // Greedy and neutral: the raw argmax is the pick.
        assert!(!split(true, false, &split_head, &mut a, &[]));
        a.temperature = 0.7;
        assert!(split(true, false, &split_head, &mut a, &[]));
        a.temperature = 0.0;
        a.presence_penalty = 1.5;
        assert!(split(true, false, &split_head, &mut a, &[]));
        a.presence_penalty = 0.0;
        a.require_tool_call = true;
        assert!(split(true, false, &split_head, &mut a, &[]));
        a.require_tool_call = false;
        assert!(split(false, false, &split_head, &mut a, &[]));
        // A full-logits head, or a split head that cannot sample: never.
        a.temperature = 0.7;
        assert!(!split(true, false, &model(None), &mut a, &[]));
        script(false);
        assert!(!split(true, false, &split_head, &mut a, &[]));
    }
}
