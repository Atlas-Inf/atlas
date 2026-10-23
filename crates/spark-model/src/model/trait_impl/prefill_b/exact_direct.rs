// SPDX-License-Identifier: AGPL-3.0-only

//! Direct exact-hit resume (`ATLAS_EXACT_DIRECT=1`, with
//! `ATLAS_MARCONI_EXACT=1`): on an exact full-prompt leaf hit the whole prompt
//! is already restored (SSM + aux state at token N) and `finalize_last`
//! overwrites `normed` with the snapshot's stashed post-norm hidden, so the
//! 1-token layer re-run contributes nothing but a second advance of every
//! recurrent and aux state, which the fixup then has to undo. Skip it.

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// True when this chunk's layer forward is skipped: last chunk of an
    /// exact leaf hit whose snapshot carries the stashed hidden, opted in,
    /// and no prompt logprobs (those read every prompt row's hidden).
    pub(in crate::model) fn prefill_b_exact_direct(
        &self,
        seq: &SequenceState,
        prompt_len: usize,
        is_last_chunk: bool,
        proc_count: usize,
    ) -> bool {
        let direct = is_last_chunk
            && proc_count == 1
            && seq.collect_prompt_logprobs.is_none()
            && crate::model::mtp_carry::exact_direct()
            && seq
                .marconi_exact_snap
                .is_some_and(|s| self.ssm_snapshots.has_hidden(s));
        if direct {
            tracing::info!(
                "exact-hit direct resume: {prompt_len} prompt tokens restored, no forward"
            );
        }
        direct
    }
}
