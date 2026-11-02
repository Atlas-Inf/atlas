// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_WARM_CTX`, the sequence side: what a prefill captures,
//! restores at its start and stores at its checkpoints (`warm_ctx`,
//! `warm_ctx_store`).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::DflashProposerState;
use super::warm_ctx::{CapturePlan, CtxRun};
use super::warm_ctx_store::{StoreRefs, WarmCtxStore, store_key};

impl DflashProposerState {
    /// Record a prefill capture of positions `[plan.first, plan.end)`.
    pub(crate) fn note_capture(&mut self, plan: &CapturePlan) {
        CtxRun::note(&mut self.ctx_run, plan.window_start, plan.first, plan.end);
    }

    /// Chunk 0 of a prefill, after the prefix lookup: the target's state was
    /// restored at `skip` (0 = computes from the start) of a
    /// `prompt_len`-token prompt. Sets where the window may start and, under
    /// `restore`, copies the stored rows under `skip` into the accumulator.
    /// `key_tokens` are the prompt's prefix-cache key tokens (`None`: the
    /// prompt has no key, so nothing can have been stored for it).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_warm_ctx(
        &mut self,
        store: &WarmCtxStore,
        adapter_id: u64,
        key_tokens: Option<&[u32]>,
        skip: usize,
        prompt_len: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.ctx_floor = 0;
        self.ctx_run = None;
        self.ctx_store = StoreRefs::default();
        if !self.warm_ctx.seeds() || skip == 0 {
            return Ok(());
        }
        self.ctx_floor = skip;
        let want_lo = prompt_len.max(skip).saturating_sub(self.max_ctx_len);
        if !self.warm_ctx.restores() || want_lo >= skip || self.ctx_hidden_acc.0 == 0 {
            return Ok(());
        }
        let restored = match key_tokens.and_then(|t| t.get(..skip)) {
            Some(prefix) => {
                let key = store_key(adapter_id, prefix);
                let (acc, row) = (self.ctx_hidden_acc, self.ctx_slot_bytes);
                store.restore(gpu, key, skip, want_lo, row, acc, stream)?
            }
            None => None,
        };
        match restored {
            Some((lo, src)) => {
                self.ctx_floor = lo;
                self.ctx_store.src = Some(src);
                // The window now starts at `lo` (>= want_lo), slot 0.
                CtxRun::note(&mut self.ctx_run, lo, lo, skip);
                tracing::info!(
                    "DFlash warm ctx: restored {} rows [{lo}, {skip}) under a {prompt_len}-token \
                     prompt ({} wanted from {want_lo})",
                    skip - lo,
                    skip - want_lo,
                );
            }
            None => tracing::info!(
                "DFlash warm ctx: no stored rows at {skip}; window seeded from there \
                 ({} rows short)",
                skip - want_lo
            ),
        }
        Ok(())
    }

    /// A prefill checkpoint was saved at the end of `prefix` (its
    /// prefix-cache key tokens): store the accumulator's real rows that end
    /// there, in this prefill's one window. `Ok(false)` when nothing new was
    /// stored.
    pub(crate) fn save_warm_ctx(
        &mut self,
        store: &WarmCtxStore,
        adapter_id: u64,
        prefix: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        let end = prefix.len();
        if !self.warm_ctx.restores() || self.ctx_hidden_acc.0 == 0 {
            return Ok(false);
        }
        let Some(run) = CtxRun::rows_to(self.ctx_run, end) else {
            return Ok(false);
        };
        let key = store_key(adapter_id, prefix);
        let src = self
            .ctx_hidden_acc
            .offset((run.lo - run.window_start) * self.ctx_slot_bytes);
        let row = self.ctx_slot_bytes;
        let own = store.save(gpu, key, self.ctx_store, src, run.lo, end, row, stream)?;
        if let Some(own) = own {
            self.ctx_store.own = Some(own);
            // This turn's window now serves the conversation's next turn.
            if let Some(src) = self.ctx_store.src.take().filter(|&s| s != own) {
                store.demote(src);
            }
        }
        Ok(own.is_some())
    }
}
