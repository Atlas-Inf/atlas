// SPDX-License-Identifier: AGPL-3.0-only

//! The draft head as the engine's proposer: the rank that serves a request
//! starts a pass, the others serve their part of it, and the first ends it
//! on its host with the drafts worth verifying.

use anyhow::{Context, Result};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::draft::Draft;
use super::draft_head::worth;
use super::draft_seq::tap;
use super::parts::PARTS;
use crate::layer::ForwardContext;
use crate::layers::dflash_head::SequenceGeneration;
use crate::model::DraftAssist;
use crate::speculative::{DraftProposer, ProposerState};

/// The chance, of a draft and all before it standing, below which a pass
/// stops offering drafts, unless `ATLAS_DSV41_DRAFT_FLOOR` says: a verified
/// row costs a part of a decode step, and a refused one buys nothing. About
/// a row's cost over a token's: at 0.2 prose, code and JSON were each as
/// fast as at any floor from 0.12 to 0.45.
const FLOOR: f32 = 0.2;

fn floor() -> f32 {
    static SAID: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *SAID.get_or_init(|| {
        std::env::var("ATLAS_DSV41_DRAFT_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(FLOOR)
    })
}

/// A sequence's side of the head lives with its layers (`draft_seq.rs`).
struct Stateless;

impl ProposerState for Stateless {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl DraftProposer for Draft {
    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(Stateless))
    }

    /// Every pass of a model on more than one rank is the ranks' together.
    fn rank_split_ready(&self, _comm: &dyn CommBackend, _grammar: bool) -> bool {
        self.held.count < PARTS
    }

    fn propose(
        &self,
        last_token: u32,
        _target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        _state: &mut dyn ProposerState,
        _expected_owner: Option<SequenceGeneration>,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let (gpu, comm) = (ctx.gpu, ctx.comm);
        // The other ranks were told of a pass only if one was announced.
        if self.held.count < PARTS && comm.is_none() {
            return Ok(Vec::new());
        }
        // No pass for a sequence whose layers have not just run, nor under a
        // grammar, which the head knows nothing of, nor from a token the
        // model does not have: the other ranks are told only of a pass that
        // every rank can run.
        let drafts = grammar_bitmask.is_none() && self.holds(last_token);
        let seq = tap().current().filter(|_| drafts);
        let plan = seq
            .as_ref()
            .and_then(|seq| seq.lock().expect("draft sequence").plan(position));
        let said = plan.as_ref().map(|_| (last_token, position));
        self.agree(gpu, comm, Some(said), stream)?;
        let (Some(seq), Some(plan)) = (seq, plan) else {
            return Ok(Vec::new());
        };
        {
            let mut seq = seq.lock().expect("draft sequence");
            self.pass(gpu, comm, last_token, position, &mut seq, &plan, stream)?;
        }
        let host = self
            .host
            .as_ref()
            .context("deepseek_v41: a draft pass started on a rank that serves no requests")?;
        let (drafts, x) = self.read(gpu, stream)?;
        // A row that found no token ends the drafts.
        let mut drafts: Vec<u32> = drafts.into_iter().take_while(|&t| self.holds(t)).collect();
        let confidence = host.confidence(last_token, &drafts, &x);
        let most = num_drafts.min(self.rows);
        drafts.truncate(worth(&confidence, floor(), most));
        tracing::debug!(
            "DSpark: {} of {most} drafts offered after {position} tokens, confidence {confidence:?}",
            drafts.len()
        );
        Ok(drafts)
    }

    /// Nothing of a pass outlasts it, so there is nothing to take back.
    fn after_verify(
        &self,
        _num_accepted: usize,
        _expected_owner: Option<SequenceGeneration>,
        _state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }
}

impl DraftAssist for Draft {
    /// This rank's part of the pass the first rank announced.
    fn serve(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        stream: u64,
        _word: u32,
        _extra: Option<u32>,
    ) -> Result<()> {
        let Some((token, len)) = self.agree(gpu, Some(comm), None, stream)? else {
            return Ok(());
        };
        let seq = tap()
            .current()
            .context("deepseek_v41: a draft pass for a sequence no layer of this rank has run")?;
        let mut seq = seq.lock().expect("draft sequence");
        let plan = seq.plan(len).with_context(|| {
            format!("deepseek_v41: a draft pass after {len} tokens, which this rank's layers have not run")
        })?;
        self.pass(gpu, Some(comm), token, len, &mut seq, &plan, stream)
    }
}
