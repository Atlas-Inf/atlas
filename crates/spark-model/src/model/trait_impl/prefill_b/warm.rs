// SPDX-License-Identifier: AGPL-3.0-only

//! The arena zero and chunk embed of a prefill chunk, and the chunk that
//! needs neither (`ATLAS_GLM_WARM_SKIP_CACHED=1`, default off).
//!
//! A chunk that lies entirely below the restore depth of a warm prefill
//! computes nothing: it reserves its blocks, appends its tokens and returns
//! (`ProcRange::EarlyReturn`). It still zeroed the whole arena and embedded
//! its rows first, because both come before the prefix lookup that decides
//! it. A 45K-token warm turn at 8K-row chunks has five such chunks, a
//! 512K-token one over sixty, and each costs a full arena zero (about 17 ms
//! on GB10) plus an embed of up to a chunk of rows.
//!
//! With the switch, in a multi-rank world, the lookup runs first and a chunk
//! with nothing to compute skips both. Nothing of that request reads what
//! they would have written: no layer runs in that chunk, and the next chunk
//! that computes zeroes the whole arena and embeds its own rows before its
//! first kernel, as every multi-rank chunk does. So every pass of the request
//! sees the same arena, byte for byte, as without the switch: it is exact per
//! request. The lookup itself touches no arena buffer (its collectives use
//! the command word buffer, its restore the SSM pools), so moving it ahead of
//! the zero changes nothing it reads or writes.
//!
//! What does change is what a decode step of ANOTHER sequence finds in the
//! arena when it runs between two such chunks: the leftovers of the step
//! before and the prompt bytes the chunk command staged in scratch, instead
//! of zeros and the chunk's embeddings. Such a step already finds leftovers
//! that depend on the requests before it (after any chunk that computed, and
//! after every decode step), so this is the history dependence base has, not
//! a new kind; whether it can reach a result is the read-before-write
//! question `ATLAS_GLM_ZERO_ROWS` asks. It is why both ranks must run with
//! the same value (`warm_turn`, "Rank parity").
//!
//! A chunk that computes is unchanged: full zero, full embed, then the
//! re-embed of its uncached rows (`proc_range`). How much of the arena that
//! zero covers is `ATLAS_GLM_ZERO_ROWS`'s (`warm_turn::ZeroRows`).

use std::time::{Duration, Instant};

use anyhow::Result;

use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::super::super::warm_turn::{PHASES, RequestShape, TraceMode, ZeroRows, logits_hash};
use crate::traits::SequenceState;

impl TransformerModel {
    /// Whether a chunk looks its prefix up before it zeroes and embeds: the
    /// switch, in a multi-rank world. A single rank zeroes once per request,
    /// at chunk 0, and later chunks find the rows earlier ones embedded; a
    /// chunk 0 that skipped would change what they find, so it keeps its
    /// order.
    pub(super) fn warm_lookup_first(&self) -> bool {
        self.warm.skip_cached && self.comm.is_some()
    }

    /// Zero the arena and embed the chunk when `run`; returns the two host
    /// spans (zero when not).
    ///
    /// EP=2: zero ALL buffers on every chunk (NCCL defense-in-depth).
    /// EP=1, first chunk (chunk_start==0): zero only buffers whose stale
    /// contents can affect prefill; the remaining scratch buffers are
    /// overwritten before read by embedding + layer forward.
    /// EP=1, subsequent chunks: skip zeroing — buffers are overwritten by embedding
    /// + layer forward before read. Saves 7 memsets × (chunks-1) per prefill.
    pub(super) fn prefill_b_zero_and_embed(
        &self,
        run: bool,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<[Duration; 2]> {
        if !run {
            return Ok([Duration::ZERO; 2]);
        }
        let t0 = Instant::now();
        if self.comm.is_some() {
            self.warm_zero_arena(stream)?;
        } else if chunk_start == 0 {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }
        self.warm_trace_sync(stream)?;
        let zero = t0.elapsed();
        // ── Phase 1+1b: embed chunk + vision pad overlay ──
        self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;
        self.warm_trace_sync(stream)?;
        Ok([zero, t0.elapsed() - zero])
    }

    /// The multi-rank zero before a chunk: the whole arena, or with
    /// `ATLAS_GLM_ZERO_ROWS` what may be dirty (`ZeroRows`).
    fn warm_zero_arena(&self, stream: u64) -> Result<()> {
        let (gpu, arena) = (self.gpu.as_ref(), &self.buffers);
        match self.warm.zero_rows {
            ZeroRows::Off => arena.zero_all(gpu, stream),
            ZeroRows::Trim(floor) => arena.zero_dirty(gpu, stream, floor),
            ZeroRows::Check(floor) => {
                let stale = arena.stale_past_dirty(gpu, stream, floor)?;
                for (name, at, kept) in &stale {
                    tracing::error!(
                        "ATLAS_GLM_ZERO_ROWS=check: {name} holds a nonzero byte at {at}, past \
                         the {kept} bytes a trimmed zero covers (floor {floor} rows)"
                    );
                }
                if stale.is_empty() {
                    tracing::info!("ATLAS_GLM_ZERO_ROWS=check: clean (floor {floor} rows)");
                }
                arena.zero_all(gpu, stream)
            }
        }
    }

    /// `ATLAS_GLM_WARM_TRACE=1`: drain `stream`, so the span being timed
    /// holds the device time of its own launches. Nothing otherwise.
    pub(super) fn warm_trace_sync(&self, stream: u64) -> Result<()> {
        if self.warm.trace == TraceMode::Spans {
            self.gpu.synchronize(stream)?;
        }
        Ok(())
    }

    /// `ATLAS_GLM_WARM_TRACE`: add a chunk to its request's trace and log
    /// the request's line after the last chunk ([`Self::warm_trace_line`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn warm_trace_chunk(
        &self,
        seq: &SequenceState,
        prompt: usize,
        began: Instant,
        chunk: (usize, usize),
        pre: [Duration; 2],
        marks: [Duration; 5],
        logits: Option<DevicePtr>,
        stream: u64,
    ) -> Result<()> {
        if self.warm.trace == TraceMode::Off {
            return Ok(());
        }
        if let Some(line) =
            self.warm_trace_line(seq, prompt, began, chunk, pre, marks, logits, stream)?
        {
            tracing::info!("{line}");
        }
        Ok(())
    }

    /// Add a chunk that began at `began` to its request's trace; the
    /// request's line after the last chunk. `chunk` is the chunk's first
    /// token and the rows it computed, `pre` the zero and embed spans;
    /// `marks` are the lookup span and the times since `began` at which the
    /// lookup (with a late zero and embed), the block reservation, the
    /// metadata and the forward were done. What follows the forward is the
    /// finish. `logits` is the last chunk's result: the line carries a hash
    /// of that row, read from the device once `stream` has drained.
    #[allow(clippy::too_many_arguments)]
    fn warm_trace_line(
        &self,
        seq: &SequenceState,
        prompt: usize,
        began: Instant,
        chunk: (usize, usize),
        pre: [Duration; 2],
        marks: [Duration; 5],
        logits: Option<DevicePtr>,
        stream: u64,
    ) -> Result<Option<String>> {
        let shape = match logits {
            Some(ptr) => {
                self.gpu.synchronize(stream)?;
                let fp32 = self.logits_ptr_is_fp32_dispatch(ptr);
                let mut row = vec![0u8; self.config.vocab_size * if fp32 { 4 } else { 2 }];
                if !ptr.is_null() {
                    self.gpu.copy_d2h(ptr, &mut row)?;
                }
                Some(RequestShape {
                    rank: self.comm.as_ref().map_or(0, |c| c.rank()),
                    prompt,
                    matched: seq.cached_prefix_tokens,
                    restored: seq.marconi_skip_to,
                    logits: logits_hash(&row),
                })
            }
            None => {
                self.warm_trace_sync(stream)?;
                None
            }
        };
        let [lookup, looked, reserved, meta, forward] = marks;
        let spans: [Duration; PHASES.len() - 1] = [
            pre[0],
            pre[1],
            lookup,
            reserved.saturating_sub(looked),
            meta.saturating_sub(reserved),
            forward.saturating_sub(meta),
            began.elapsed().saturating_sub(forward),
        ];
        let (start, rows) = chunk;
        Ok(self
            .warm
            .note_chunk(seq.slot_idx, began, start == 0, rows, spans, shape))
    }
}

#[cfg(test)]
#[path = "warm_tests.rs"]
mod tests;
