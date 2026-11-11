// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_WARM_CTX` on the model side: the prefix lookup and the
//! prefill checkpoints hand the drafter's accumulator its restore point and
//! its store entries (`layers::dflash_head::warm_ctx`). Rank 0 only, like
//! the capture; no collective.

use anyhow::Result;

use super::types::TransformerModel;
use crate::layers::DflashProposerState;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Install the store `build_model` reserved before sizing the KV pool.
    pub(crate) fn set_dflash_warm_store(
        &mut self,
        store: crate::layers::dflash_head::WarmCtxStore,
    ) {
        self.dflash_warm_store.free(self.gpu.as_ref());
        self.dflash_warm_store = store;
    }

    /// This rank's DFlash proposer state of `seq`, when it owns the drafter
    /// context (rank 0 of a multi-rank world).
    fn dflash_ctx_owner(
        dflash: bool,
        rank0: bool,
        seq: &mut SequenceState,
    ) -> Option<&mut DflashProposerState> {
        if !dflash || !rank0 {
            return None;
        }
        seq.proposer_state.as_mut()?.as_any_mut().downcast_mut()
    }

    fn dflash_ctx_rank0(&self) -> (bool, bool) {
        let rank0 = self.comm.as_ref().is_none_or(|c| c.rank() == 0);
        (!self.dflash_capture_layers.is_empty(), rank0)
    }

    /// Chunk 0, after the prefix lookup chose `skip` for a `total`-token
    /// prompt whose key tokens are `key_tokens`.
    pub(in crate::model) fn dflash_warm_begin(
        &self,
        seq: &mut SequenceState,
        key_tokens: Option<&[u32]>,
        skip: usize,
        total: usize,
        stream: u64,
    ) -> Result<()> {
        let (dflash, rank0) = self.dflash_ctx_rank0();
        let adapter_id = seq.adapter_id;
        let Some(d) = Self::dflash_ctx_owner(dflash, rank0, seq) else {
            return Ok(());
        };
        d.begin_warm_ctx(
            &self.dflash_warm_store,
            adapter_id,
            key_tokens,
            skip,
            total,
            self.gpu.as_ref(),
            stream,
        )
    }

    /// A prefill checkpoint covering the key tokens `prefix` was saved.
    pub(in crate::model) fn dflash_warm_save(
        &self,
        seq: &mut SequenceState,
        prefix: &[u32],
        stream: u64,
    ) -> Result<()> {
        let (dflash, rank0) = self.dflash_ctx_rank0();
        let adapter_id = seq.adapter_id;
        let Some(d) = Self::dflash_ctx_owner(dflash, rank0, seq) else {
            return Ok(());
        };
        if d.save_warm_ctx(
            &self.dflash_warm_store,
            adapter_id,
            prefix,
            self.gpu.as_ref(),
            stream,
        )? {
            tracing::debug!(
                "DFlash warm ctx: stored the window under {} tokens",
                prefix.len()
            );
        }
        Ok(())
    }
}
