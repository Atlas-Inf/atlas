// SPDX-License-Identifier: AGPL-3.0-only
//! The draft-proposer face of Ling's predictor: propose one draft, prime the
//! prompt and catch-up rows, and release the private KV.

use super::{LingMtp, State, state};
use crate::layer::{ForwardContext, TransformerLayer};
use crate::layers::dflash_head::SequenceGeneration;
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

impl DraftProposer for LingMtp {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(State {
            blocks: vec![],
            rows: 0,
            capacity_exhausted: false,
            pair: None,
            speculative_rows: 0,
            body: self.body.alloc_state(gpu)?,
        }))
    }
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        value: &mut dyn ProposerState,
        _owner: Option<SequenceGeneration>,
        ctx: &ForwardContext,
        stream: u64,
        _embed: Option<DevicePtr>,
        mask: Option<&[i32]>,
        _stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        ensure!(
            (1..=3).contains(&num_drafts),
            "Ling native MTP supports 1..=3 drafts, got {num_drafts}"
        );
        let s = state(value)?;
        {
            let cache = self.cache.lock();
            let bs = cache.block_size();
            let needed = (s.rows + num_drafts)
                .div_ceil(bs)
                .saturating_sub(s.blocks.len());
            if s.capacity_exhausted || needed > cache.num_free_blocks() {
                s.capacity_exhausted = true;
                return Ok(vec![]);
            }
        }
        let h = ctx.config.hidden_size as u32;
        let vocab = ctx.config.vocab_size as u32;
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut cur_token = last_token;
        let mut cur_hidden = target_hidden;
        for step in 0..num_drafts {
            self.append(
                cur_token,
                cur_hidden,
                position.saturating_sub(1) + step,
                step == 0,
                s,
                ctx,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.norm_k,
                self.hidden,
                &self.norm,
                self.residual,
                1,
                h,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;
            if let Some((weight, kernel)) = &self.draft_head {
                ops::w4a16_gemv(
                    ctx.gpu,
                    *kernel,
                    self.residual,
                    weight,
                    self.logits,
                    vocab,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.gemv,
                    self.residual,
                    &self.head,
                    self.logits,
                    vocab,
                    h,
                    stream,
                )?;
            }
            let id = if let Some(mask) = mask {
                crate::layers::deepseek_v4_mtp::argmax_grammar_masked(
                    ctx.gpu,
                    self.logits,
                    vocab as usize,
                    mask,
                    position + step,
                )?
            } else {
                ops::argmax_bf16(
                    ctx.gpu,
                    self.argmax,
                    self.logits,
                    self.inputs,
                    vocab,
                    stream,
                )?;
                let mut raw = [0; 4];
                ctx.gpu.copy_d2h_on_stream(self.inputs, &mut raw, stream)?;
                u32::from_le_bytes(raw)
            };
            drafts.push(id);
            cur_token = id;
            // `self.residual` holds post-`self.norm` output for the next chained step.
            cur_hidden = self.residual;
        }
        s.speculative_rows = drafts.len().saturating_sub(1);
        Ok(drafts)
    }
    fn propose_batch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        _owners: Option<&[SequenceGeneration]>,
        ctx: &ForwardContext,
        stream: u64,
        _out_conf: Option<&mut Vec<Vec<f32>>>,
        grammar_bitmasks: Option<&[Option<Vec<i32>>]>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        if !(1..=3).contains(&num_drafts)
            || grammar_bitmasks.is_some_and(|m| m.iter().any(Option::is_some))
        {
            return Ok(None);
        }
        self.propose_rows(
            last_tokens,
            target_hiddens,
            positions,
            num_drafts,
            states,
            ctx,
            stream,
        )
    }
    fn propose_batch_max(
        &self,
        _buffers: &spark_runtime::buffers::BufferArena,
        _config: &atlas_core::config::ModelConfig,
    ) -> usize {
        super::batch::MAX_ROWS
    }
    fn prefill_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        value: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let s = state(value)?;
        if s.rows > 0 {
            return Ok(0);
        }
        self.prime(tokens.get(1..).unwrap_or(&[]), hiddens, 0, s, ctx, stream)
    }
    fn drafter_rows(&self, value: &mut dyn ProposerState) -> usize {
        state(value).map_or(0, |s| s.rows)
    }
    fn last_pair_key(&self, value: &mut dyn ProposerState) -> Option<usize> {
        state(value).ok().and_then(|s| s.pair)
    }
    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        value: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let s = state(value)?;
        ensure!(row_base == s.rows, "Ling MTP catch-up boundary mismatch");
        self.prime(
            tokens.get(1..).unwrap_or(&[]),
            hiddens,
            pos_base.saturating_sub(1),
            s,
            ctx,
            stream,
        )
    }
    fn after_verify(
        &self,
        accepted: usize,
        _owner: Option<SequenceGeneration>,
        value: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        // Row 0 consumed the committed anchor token and always remains valid.
        // Any chained speculative draft rows past `accepted` are trimmed.
        self.trim_speculative(state(value)?, accepted);
        Ok(())
    }
    fn free_state(
        &self,
        _gpu: &dyn GpuBackend,
        _owner: Option<SequenceGeneration>,
        value: &mut dyn ProposerState,
    ) -> Result<()> {
        let s = state(value)?;
        self.cache.lock().free_blocks(&s.blocks);
        s.blocks.clear();
        s.rows = 0;
        s.capacity_exhausted = false;
        s.pair = None;
        s.speculative_rows = 0;
        Ok(())
    }
}
