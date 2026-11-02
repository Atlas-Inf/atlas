// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence private cache lifecycle and canonical target repair hooks.
use super::*;
impl DraftProposer for Glm5MtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }
    fn append_native_target_context(
        &self,
        tokens: &[u32],
        start: usize,
        norm_hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.append_target_context(
            tokens,
            start,
            norm_hiddens,
            native_state(state)?,
            ctx,
            stream,
        )
    }
    fn retain_native_target_hidden(
        &self,
        hidden: DevicePtr,
        end: usize,
        state: &mut dyn ProposerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.retain_target_hidden(hidden, end, native_state(state)?, gpu, stream)
    }
    fn native_target_hidden(
        &self,
        state: &dyn ProposerState,
    ) -> Result<Option<(usize, DevicePtr)>> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("invalid GLM-5 MTP proposer state")?;
        Ok((state.target_end > 0).then_some((state.target_end, state.target_hidden)))
    }
    fn clone_prefix_state(
        &self,
        state: &dyn ProposerState,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Box<dyn ProposerState>>> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("invalid GLM-5 MTP proposer state")?;
        Ok(Some(Box::new(
            self.clone_prefix_state_inner(state, rows, gpu, stream)?,
        )))
    }
    fn native_prefix_free_blocks(&self) -> Option<usize> {
        self.kv_cache
            .lock()
            .as_ref()
            .map(PagedKvCache::num_free_blocks)
    }
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let state = native_state(state)?;
        let original_len = state.seq_len;
        let result = self.propose_inner(
            last_token,
            target_hidden,
            position,
            num_drafts,
            state,
            ctx,
            stream,
            grammar_bitmask,
        );
        if result.is_err() {
            // An incomplete proposal has no verified rows. Hide every row it
            // wrote before the scheduler can retry or resume serial decoding.
            state.seq_len = original_len;
            state.last_num_drafted = 0;
            state.confidence.clear();
        }
        result
    }
    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        native_state(state).map(|s| s.seq_len).unwrap_or(0)
    }
    fn requires_prompt_prefill(&self) -> bool {
        true
    }
    fn prepare_drafter_catchup(&self, state: &mut dyn ProposerState) -> Result<usize> {
        let state = native_state(state)?;
        state.seq_len = crate::layers::glm5_mtp_cursor::abandon_draft_tail(
            state.seq_len,
            state.last_num_drafted,
        )?;
        state.last_num_drafted = 0;
        state.confidence.clear();
        Ok(state.seq_len)
    }
    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let state = native_state(state)?;
        anyhow::ensure!(
            state.last_num_drafted == 0,
            "GLM-5 MTP catch-up cannot run with a pending draft tail"
        );
        self.catchup_rows(tokens, hiddens, row_base, pos_base, state, ctx, stream)
    }
    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let state = native_state(state)?;
        state.confidence.clear();
        self.prefill_rows(prompt_tokens, hiddens, state, ctx, stream)
    }
    fn repair_after_verify(
        &self,
        accepted_draft: u32,
        position: usize,
        target_hidden: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.repair_one(
            accepted_draft,
            target_hidden,
            position,
            native_state(state)?,
            ctx,
            stream,
        )
    }
    fn after_verify(
        &self,
        num_accepted: usize,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let state = native_state(state)?;
        state.confidence.clear();
        if state.last_num_drafted == 0 {
            anyhow::ensure!(num_accepted == 0, "GLM-5 MTP accepted rows without a draft");
            return Ok(());
        }
        state.seq_len =
            rewind_len_native_gamma(state.seq_len, state.last_num_drafted, num_accepted)?;
        state.last_num_drafted = 0;
        Ok(())
    }
    fn free_state(
        &self,
        gpu: &dyn GpuBackend,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        state: &mut dyn ProposerState,
    ) -> Result<()> {
        let state = native_state(state)?;
        if !state.block_table.is_empty() {
            self.cache()?.free_blocks(&state.block_table);
            state.block_table.clear();
        }
        if !state.target_hidden.is_null() {
            gpu.free(state.target_hidden)?;
            state.target_hidden = DevicePtr::NULL;
        }
        state.target_end = 0;
        state.seq_len = 0;
        state.last_num_drafted = 0;
        state.confidence.clear();
        Ok(())
    }
}
fn native_state(state: &mut dyn ProposerState) -> Result<&mut Glm5MtpProposerState> {
    state
        .as_any_mut()
        .downcast_mut::<Glm5MtpProposerState>()
        .context("invalid GLM-5 MTP proposer state")
}
