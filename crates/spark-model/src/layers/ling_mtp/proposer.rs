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
        ensure!(num_drafts == 1, "Ling has one native draft stage");
        let s = state(value)?;
        {
            let cache = self.cache.lock();
            if s.capacity_exhausted
                || (s.rows == s.blocks.len() * cache.block_size() && cache.num_free_blocks() == 0)
            {
                s.capacity_exhausted = true;
                return Ok(vec![]);
            }
        }
        self.append(
            last_token,
            target_hidden,
            position.saturating_sub(1),
            s,
            ctx,
            stream,
        )?;
        let h = ctx.config.hidden_size as u32;
        let vocab = ctx.config.vocab_size as u32;
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
                position,
            )?
        } else {
            ops::argmax_bf16(
                ctx.gpu,
                self.argmax,
                self.logits,
                self.residual,
                vocab,
                stream,
            )?;
            let mut raw = [0; 4];
            ctx.gpu
                .copy_d2h_on_stream(self.residual, &mut raw, stream)?;
            u32::from_le_bytes(raw)
        };
        Ok(vec![id])
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
        if num_drafts != 1 || grammar_bitmasks.is_some_and(|m| m.iter().any(Option::is_some)) {
            return Ok(None);
        }
        self.propose_rows(last_tokens, target_hiddens, positions, states, ctx, stream)
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
        _accepted: usize,
        _owner: Option<SequenceGeneration>,
        _state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        // One proposal consumed the committed anchor token. Its private KV row
        // remains valid whether the proposed next token was accepted or rejected.
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
        Ok(())
    }
}
