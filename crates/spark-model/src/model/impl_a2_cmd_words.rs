// SPDX-License-Identifier: AGPL-3.0-only

//! Command words over the comm backend's host channel
//! (`ATLAS_GLM_CMD_RDMA=1`, read by the RDMA pair).
//!
//! The head-to-worker wire protocol is a stream of u32 words (`impl_a2`,
//! `impl_a2_ep_worker`): slot, command, arguments, tokens, verdict. On the
//! broadcast path every word and every token payload is an NCCL broadcast.
//! When the backend has a host command channel
//! ([`spark_comm::CommBackend::command_words_max`]), the messages that fit
//! take it instead: the same words, in the same order, from the same calls.
//! Payloads above the channel's message size (prompt tokens) and
//! [`TransformerModel::ep_min_u32`], which is rooted at each rank in turn,
//! stay broadcasts. Both ranks choose by the word count alone, and the pair
//! agrees on the channel at bootstrap, so they always choose alike.

use anyhow::Result;

use super::types::TransformerModel;

impl TransformerModel {
    /// Whether an `n`-word command message takes the host channel.
    pub(super) fn ep_cmd_words_on_host(&self, n: usize) -> bool {
        self.comm
            .as_ref()
            .is_some_and(|comm| (1..=comm.command_words_max()).contains(&n))
    }

    /// Rank 0 sends `words`; another rank receives as many in their place.
    /// Returns the message on every rank.
    ///
    /// Each side still drains its stream at the message, as the broadcast
    /// path does (the head's synchronous upload, the worker's sync before its
    /// copy back), so the host work that follows a command word sees every
    /// earlier kernel finished on either path.
    pub(super) fn ep_cmd_words(&self, words: &[u32]) -> Result<Vec<u32>> {
        let comm = self.comm.as_ref().expect("ep_cmd_words without comm");
        let stream = self.gpu.default_stream();
        if comm.rank() == 0 {
            self.gpu.synchronize(stream)?;
            comm.send_command_words(words)?;
            Ok(words.to_vec())
        } else {
            let mut received = vec![0u32; words.len()];
            comm.recv_command_words(&mut received)?;
            self.gpu.synchronize(stream)?;
            Ok(received)
        }
    }
}
