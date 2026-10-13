// SPDX-License-Identifier: AGPL-3.0-only

//! EP worker command helpers: the `(seq_id, cmd)` idle first-word receive,
//! GLM whole-worker routing, the native-only fence, width-generic verify and
//! the verified-step verdict.
//!
//! EP worker protocol extensions (see `ep_worker_step_impl`):
//! - 0xFFFFFFF5: generic verify → K, K tokens, then num accepted drafts
//! - 0xFFFFFFF6: set request-local native-only fence → disabled (0/1)
//! - 0xFFFFFFF7: synchronize vision metadata and BF16 encoder rows
//! - 0xFFFFFFE1: distributed GLM MTP propose → token, position, drafts, hidden row
//! - 0xFFFFFFEB: prefill chunk carrying DFlash verify owners (`glm_fused_chunk`)
//! - 0xFFFFFFEC: serve this rank's half of a split DFlash propose (`draft_assist`)

use anyhow::Result;

use super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

#[cfg(test)]
#[path = "impl_a2/idle_command_tests.rs"]
mod idle_command_tests;

impl TransformerModel {
    /// Receive a `(seq_id, cmd)` pair from rank 0. Worker-side counterpart
    /// of [`Self::ep_broadcast_seq_and_cmd`].
    ///
    /// With `v2` enabled the returned `seq_id` is the slot the head wants
    /// the worker to dispatch the command into; with `v2` disabled the
    /// returned `seq_id` is always 0 (the legacy singleton slot).
    pub(super) fn ep_recv_seq_and_cmd(&self, v2: bool) -> Result<(u32, u32)> {
        // Only this outer first word can be waiting for a future command.
        // In v2 the following command word is already part of active traffic.
        let first = self.ep_receive_idle_word()?;
        if v2 {
            Ok((first, self.ep_broadcast_u32(0)?))
        } else {
            Ok((0, first))
        }
    }

    pub(super) fn ep_receive_idle_word(&self) -> Result<u32> {
        let comm = self
            .comm
            .as_ref()
            .expect("ep_receive_idle_word without comm");
        if self.ep_cmd_words_on_host(1) {
            return Ok(self.ep_cmd_words(&[0])?[0]);
        }
        let stream = self.gpu.default_stream();
        comm.receive_idle_command_word(self.ep_cmd_buf.0)?;
        self.gpu.synchronize(stream)?;
        let mut buf = [0u8; 4];
        self.gpu.copy_d2h(self.ep_cmd_buf, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Worker side of `0xFFFFFFF6`: set the request-local native-only fence.
    pub(super) fn ep_worker_set_native_fence(&self, seq: &mut SequenceState) -> Result<()> {
        let disabled = self.ep_broadcast_u32(0)?;
        anyhow::ensure!(
            disabled <= 1,
            "native-only sequence fence must be 0 or 1, got {disabled}"
        );
        seq.disable_mtp = disabled != 0;
        Ok(())
    }

    /// Worker side of `0xFFFFFFF5`: width-generic verify.
    pub(super) fn ep_worker_generic_verify(
        &self,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        // Width-generic verify, used by four-draft MTP (K=5) and
        // DFlash. Keep this protocol separate from the fixed-width
        // commands so existing ranks remain byte-for-byte unchanged.
        // Nothing staged before anything that can fail (`glm_verify_masks`).
        self.verify_row_masks.reset();
        let (k, masked) = super::glm_verify_masks::split_verify_width(self.ep_broadcast_u32(0)?);
        anyhow::ensure!(
            (2..=32).contains(&k),
            "EP generic verify width must be 2..=32, got {k}"
        );
        let tokens = self.ep_broadcast_tokens(&vec![0u32; k])?;
        if masked {
            // Strict structured output: the head's per-row grammar masks.
            self.send_row_masks(k)?;
        }
        self.sync_secondary()?;
        self.decode_verify_graphed_kgamma(&tokens, seq, stream)?;
        let num_accepted = self.ep_broadcast_u32(0)? as usize;
        anyhow::ensure!(
            num_accepted < k,
            "EP generic verify accepted {num_accepted} drafts for K={k}"
        );
        self.ep_worker_apply_verdict(seq, &tokens, num_accepted)?;
        Ok(())
    }

    /// Worker side of a verified K-row step once the head's accepted-draft
    /// count has arrived: roll back the rejected rows, trim the proposer and
    /// commit the accepted SSM prefix.
    pub(super) fn ep_worker_apply_verdict(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
        num_accepted: usize,
    ) -> Result<()> {
        let k = tokens.len();
        let committed = num_accepted + 1;
        anyhow::ensure!(committed <= k, "EP verdict {num_accepted} exceeds K={k}");
        let to_drop = k - committed;
        if to_drop > 0 {
            anyhow::ensure!(
                seq.seq_len >= to_drop && seq.tokens.len() >= to_drop,
                "EP generic verify rollback underflow: seq_len={}, tokens={}, drop={to_drop}",
                seq.seq_len,
                seq.tokens.len(),
            );
            seq.seq_len -= to_drop;
            for _ in 0..to_drop {
                seq.tokens.pop();
            }
        }
        self.trim_proposer_state(seq, num_accepted, 0)?;
        self.commit_accepted_prefix(seq, committed, k)?;
        Ok(())
    }

    /// GLM whole-worker commands that route by their own payload rather
    /// than the preamble slot. `None` means the command is not one of them.
    pub(super) fn ep_worker_glm_cmd(
        &self,
        seq_id: u32,
        cmd: u32,
        slots: &mut [Option<SequenceState>],
    ) -> Result<Option<bool>> {
        if cmd == super::glm_long_verify::EP_CMD_GLM_LONG_VERIFY {
            return self.glm_long_receive_verify(slots).map(Some);
        }
        if cmd == super::glm_fused_chunk::EP_CMD_GLM_FUSED_CHUNK {
            return self.glm_fused_receive(seq_id, slots).map(Some);
        }
        if cmd == super::draft_assist::EP_CMD_DRAFT_ASSIST {
            return self.draft_assist_serve(seq_id).map(Some);
        }
        Ok(None)
    }

    /// GLM independent-decode guard on the batched-decode (`0xFFFFFFE0`) width.
    pub(super) fn ep_worker_check_independent_width(
        &self,
        n: usize,
        slots: &[Option<SequenceState>],
    ) -> Result<()> {
        if super::glm_independent::enabled(&self.config.model_type)? {
            anyhow::ensure!(
                (2..=8).contains(&n)
                    && n <= slots.len()
                    && n <= self.levers.max_decode_seqs as usize,
                "independent E0 width exceeds actual worker slots"
            );
        }
        Ok(())
    }
}
