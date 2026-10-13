// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_chunk_dispatch` orchestrator.
//!
//! Refactor wave-4e split a 1000-LoC monolith into Pattern-B phase fns
//! (siblings under `prefill_b/`). The MutexGuard on `kv_cache` is
//! acquired here once and threaded through each phase as `&mut`.
//!
//! Phases (by section comment in original):
//!   1+1b → embed_chunk     (token embed + vision-pad overlay)
//!   2    → prefix_lookup   (prefix-cache hit + EP-sync + Marconi)
//!   2b   → proc_range      (recompute proc_start/count after skip; may early-return)
//!   3    → upload_meta     (positions + MRoPE + slots staging upload)
//!   3b   → upload_paged    (paged-prefill block_table + seq_len upload)
//!   4    → forward_layers  (per-layer prefill/decode + diagnostics)
//!   5-8  → finalize_last   (final norm + lm_head + snapshot save) — last chunk
//!   9    → save_intermediate_checkpoint — non-last chunk

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

mod batch;
mod batch_kernel;
#[cfg(test)]
mod batch_kernel_tests;
mod batched_layer;
pub(in crate::model) mod embed_chunk;
mod finalize_last;
mod forward_layers;
mod h_state_ptrs;
mod midchunk_capture;
mod multi;
mod multi_ep;
mod multi_head;
pub use multi_ep::{multi_cached, multi_requested};
pub(in crate::model) mod pc_inflight;
pub(in crate::model) mod pc_policy;
mod prefix_lookup;
mod proc_range;
mod prompt_logprobs;
mod qwen4exp_ckpt;
pub(in crate::model) mod qwen4exp_points;
mod save_checkpoint;
mod stage_batched;
mod upload_meta;
mod upload_paged;
mod warm;

impl TransformerModel {
    pub(super) fn prefill_chunk_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_chunk_dispatch_with(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            stream,
            None,
        )
    }

    /// The chunk, optionally carrying DFlash verify owners after its rows
    /// (`glm_fused_chunk`).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::model) fn prefill_chunk_dispatch_with(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
        mut passengers: Option<&mut super::super::glm_fused_chunk::Passengers<'_, '_>>,
    ) -> Result<DevicePtr> {
        let total = tokens.len();
        let passenger_rows = passengers.as_ref().map_or(0, |p| p.total());
        assert!(
            chunk_start + chunk_len <= total,
            "chunk_start({chunk_start}) + chunk_len({chunk_len}) > total({total})"
        );

        // Tail-checkpoint split (issue #15 follow-up, 2026-07-02): a warm
        // multi-turn hit matches the radix at BLOCK granularity, and the
        // divergence point sits at/near the previous prompt end (the chat
        // template's generation-only suffix — e.g. Qwen's forced empty
        // <think> block — is absent from the re-rendered history), so the
        // next turn's `matched` lands at floor(divergence/bs)*bs, which is
        // the prompt's last full-block boundary OR one block below it (when
        // the template suffix crosses that boundary; measured: both occur).
        // Snapshot eligibility requires snap_tok <= matched — the leaf
        // snapshot (at `total`) is PAST both, making warm turns recompute the
        // full SSM state (or fall back an entire turn to the previous tail,
        // measured 1.3-3.2k-token replays). Split the final chunk ONCE, one
        // block below the last block boundary under `total`: that position is
        // <= both possible match points, so the snapshot
        // `prefill_b_save_checkpoint` saves there (independent of
        // --ssm-checkpoint-interval) is always eligible and the warm replay
        // is <= 2 blocks, folded into the suffix prefill pass. A single cut
        // costs one extra small pass at save time (a cut at the boundary
        // itself would need a second pass and is redundant — measured
        // +~160ms/turn for two cuts vs <=31-token replay for one).
        //
        // The extra pass costs ~150ms on this class of MoE model (a tiny-M
        // pass still sweeps most activated expert weights), which is -7% on
        // a cold 2k prefill — so on single-GPU the split only fires when the
        // radix already holds a prefix of this prompt (peek is read-only):
        // single-shot requests never pay; conversations pay from turn 2
        // onward, where the cost is amortized against the warm win. Known
        // residual: turn 2 of a conversation still recomputes the full SSM
        // state (its cold turn 1 saved no tail checkpoint). On EP>1 the
        // split is unconditional instead: rank-local radix contents diverge,
        // and chunk sequences must be deterministic on (tokens, config)
        // across ranks (bug #33 invariant). Skipped for vision prompts (pad
        // runs must not straddle chunk boundaries) and non-SSM models
        // (KV-only cache hits need no snapshot).
        if is_last_chunk
            && self.config.num_ssm_layers() > 0
            && self.ssm_snapshots.is_enabled()
            && self.prefix_cache.is_active()
            && !self.tokens_have_vision_pad(tokens)
        {
            let bs = self.kv_cache.lock().block_size();
            // One block below the last block boundary strictly under `total`.
            let cut = pc_policy::tail_cut(total, bs);
            // UNCONDITIONAL. This used to additionally require
            // `ep_active || peek_matched_tokens(..) > 0`, i.e. it split only on a
            // WARM request (radix already populated) — which made the prompt take a
            // DIFFERENT SHAPE cold vs warm: one N-token pass when cold, two passes
            // [0..cut) + [cut..N) when warm. BF16 accumulation is not associative,
            // so the two shapes produce different hidden states, and at temperature 0
            // a near-tied argmax flips. Same prompt, same seed, different answer.
            //
            // MEASURED on Puzzle-75B (fresh container, temp 0, exact-hit shortcut
            // bypassed so this split is the ONLY cold/warm difference):
            //   34-token prompt (cut=16, split fires warm) : cold "17 barrels"
            //                                                warm "15 barrels"  DIVERGE
            //   16-token prompt (cut<=0, split IMPOSSIBLE) : cold == warm       MATCH
            // The cut threshold predicts the divergence exactly.
            //
            // The invariant is already stated for EP>1 in the comment above —
            // "chunk sequences must be deterministic on (tokens, config)" — it was
            // just never enforced on a single rank. Splitting unconditionally makes
            // cold and warm identical by construction, at the cost of one extra pass
            // on cold prefills that cross the cut.
            //
            // `ATLAS_NO_TAIL_SPLIT=1` disables the split entirely (same-binary A/B).
            // That is the OTHER way to satisfy the invariant — always one pass — and
            // it keeps the single-pass numerics, at the cost of the warm-turn tail
            // checkpoint this split exists to create.
            // ATLAS_QWEN4EXP_PREFILL_ROWINV: a block-grid cut is off the GDN
            // grid; without the in-pass checkpoint the last chunk stays whole.
            let split_disabled =
                pc_policy::tail_split_disabled() || crate::layers::ops::qwen4exp_rowinv::on();
            // ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT: one pass, checkpoint in-pass.
            let in_pass = self.qwen4exp_ckpt_takes_tail();
            if !split_disabled && !in_pass && cut > chunk_start && cut < total {
                anyhow::ensure!(
                    passengers.is_none(),
                    "GLM fused chunk cannot take the tail-checkpoint split"
                );
                self.prefill_chunk_dispatch(
                    tokens,
                    seq,
                    chunk_start,
                    cut - chunk_start,
                    false,
                    stream,
                )?;
                return self.prefill_chunk_dispatch(tokens, seq, cut, total - cut, true, stream);
            }
        }

        // Guard: chunk_len must not exceed buffer arena capacity.
        // Exceeding this causes CUDA illegal memory access (error 700)
        // which permanently corrupts GPU state.
        let arena_cap = self.buffers.max_batch_tokens();
        if chunk_len + passenger_rows > arena_cap {
            anyhow::bail!(
                "Prefill chunk ({chunk_len} tokens) exceeds buffer arena capacity ({arena_cap} tokens). \
                 Reduce --max-prefill-tokens or prompt length."
            );
        }

        let profile = std::env::var_os("ATLAS_PROFILE_PREFILL").is_some();
        let tp = std::time::Instant::now();

        // Use the caller-provided stream for compute-copy overlap, unless
        // a multi-rank world is active (EP or pure TP — NCCL collectives
        // must stay stream-ordered with the cmd broadcasts, which run on
        // the default stream).
        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        // Zero the arena and embed the chunk (phases 1+1b, `warm`): here, or
        // with ATLAS_GLM_WARM_SKIP_CACHED after the prefix lookup and only
        // for a chunk that computes.
        let lookup_first = self.warm_lookup_first();
        let span = (chunk_start, chunk_len);
        let mut t_pre =
            self.prefill_b_zero_and_embed(!lookup_first, tokens, span.0, span.1, stream)?;
        let t_embed = tp.elapsed();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 2: prefix-cache lookup + EP sync + Marconi snapshot restore ──
        let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
            tokens,
            seq,
            chunk_start,
            total,
            &mut kv_cache,
            stream,
            None,
        )?;
        self.warm_trace_sync(stream)?;
        // ATLAS_QWEN4EXP_SNAPSHOT_AUX_MB: after the restore read its slot.
        self.enforce_snapshot_aux_budget(&mut kv_cache);
        let t_lookup = tp.elapsed() - t_embed;
        // ATLAS_GLM_PC_INFLIGHT: a checkpoint the head asked for (`pc_inflight`).
        self.pc_apply_plant(tokens, seq, span.0, kv_cache.block_size());
        // ATLAS_GLM_PC_BRANCH: split at the planned branch checkpoint
        // (ATLAS_QWEN4EXP_PC_BRANCH captures it in-pass instead).
        let split_branch = seq.pc_branch_at.filter(|_| !self.qwen4exp_pc_branch());
        if let Some(at) = pc_policy::branch_split_at(split_branch, span, passengers.is_some()) {
            drop(kv_cache);
            return self.pc_branch_split(tokens, seq, span, at, is_last_chunk, stream);
        }
        let cached = lookup_first && seq.prefill_chunk_cached(span.0 + span.1, is_last_chunk);
        if lookup_first {
            t_pre = self.prefill_b_zero_and_embed(!cached, tokens, span.0, span.1, stream)?;
        }
        let t_prefix = tp.elapsed();

        if std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("chunk_entry start={chunk_start} len={chunk_len} kvws={kv_write_start}"),
            );
        }

        // Allocate blocks needed through end of this chunk: agreed by every
        // rank and rolled back on refusal, so a refused chunk is retryable.
        let bs = kv_cache.block_size();
        self.reserve_prefill_blocks(seq, chunk_start + chunk_len, &mut kv_cache, stream)?;
        let t_blocks = tp.elapsed();

        // ── Phase 2b: compute effective processing range (may early-return) ──
        let (proc_start, proc_count, effective_seq_len_start) = match self.prefill_b_proc_range(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            kv_write_start,
            marconi_skip,
            // Single-stream: hidden lives at offset 0 ⇒ pass base (byte-identical).
            self.buffers.hidden_states(),
            stream,
        )? {
            proc_range::ProcRange::Compute {
                proc_start,
                proc_count,
                effective_seq_len_start,
            } => {
                anyhow::ensure!(!cached, "a chunk left unzeroed as cached must not compute");
                (proc_start, proc_count, effective_seq_len_start)
            }
            proc_range::ProcRange::EarlyReturn(ptr) => {
                // #155 ROOT CAUSE (warm-turn phantom snapshots): fully-cached
                // chunks skipped compute but ALSO skipped the Phase-5 token
                // append, leaving seq.tokens a SUFFIX (short by k*4096) on
                // every warm turn. Every consumer keyed on seq.tokens —
                // decode-ckpt/finish-leaf registration (hashed over a
                // mid-conversation window → unreachable phantom entries that
                // flood the snapshot pool), the radix insert at retire
                // (suffix tokens paired with the full block_table → polluted
                // token→block branches + refcount leaks), and rep-penalty
                // context — operated on the wrong sequence. Cached chunks
                // must record their tokens like any other chunk.
                seq.tokens
                    .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
                seq.seq_len = chunk_start + chunk_len;
                seq.last_decode_ckpt_block = seq.tokens.len() / bs;
                let marks = [t_lookup, t_prefix, t_blocks, t_blocks, t_blocks];
                self.warm_trace_chunk(seq, total, tp, (span.0, 0), t_pre, marks, None, stream)?;
                return Ok(ptr);
            }
        };
        self.buffers.note_rows(proc_count + passenger_rows);
        let t_proc = tp.elapsed();
        let _det = crate::det_trace::enter(self.config.ep_rank, seq.slot_idx, proc_start);

        // PLE: warm the row cache ahead of the layer-1 gather — the ids are
        // a pure function of `tokens`, so the worker streams rows for every
        // position this request still has to run while this chunk computes.
        self.ple_prefill_warm(tokens, proc_start, seq)?;

        // ── Phase 3: upload positions + MRoPE + slot metadata ──
        let upload_meta::MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        } = self.prefill_b_upload_meta(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            passenger_rows,
            effective_seq_len_start,
            &kv_cache,
            stream,
        )?;

        // ── Phase 3b: paged metadata (block_table + seq_len) ──
        if needs_paged {
            self.prefill_b_upload_paged(
                seq,
                total,
                proc_start,
                proc_count,
                meta_base,
                slot_offset,
                &kv_cache,
                stream,
            )?;
        }
        let t_meta = tp.elapsed();

        // Force H2D metadata copy to complete before layer forward.
        // On DGX Spark SM121, the DMA engine may not properly serialize
        // pinned H2D copy with subsequent compute on the same stream,
        // causing CUDA 700 at >9K tokens. This sync adds ~5μs overhead
        // per chunk but prevents the illegal memory access.
        self.gpu.synchronize(stream)?;

        // Verify owners riding this chunk: rows after the chunk's, metadata
        // after the chunk's. Every prefill phase that would change the
        // chunk's rows is refused upfront (`glm_fused_chunk_supported`).
        let passenger_run = match passengers.as_deref_mut() {
            Some(p) => {
                anyhow::ensure!(
                    proc_start == chunk_start
                        && proc_count == chunk_len
                        && kv_write_start == 0
                        && !marconi_skip,
                    "GLM fused chunk needs the whole chunk computed"
                );
                let meta = super::super::glm_fused_chunk::ChunkMeta {
                    base: meta_base,
                    pos_stream_bytes,
                    slot_offset,
                    use_mrope,
                };
                Some(self.glm_passengers_setup(p, proc_count, &meta, &mut kv_cache, stream)?)
            }
            None => None,
        };

        // ── Mid-chunk tail SSM capture (opt-in): plan BEFORE the forward
        // pass so SSM layers split their h/conv kernels at `tb` in-pass.
        // `None` (flag off or pass doesn't span `tb`) => no split. ──
        let midcap_plan = self.plan_pass_capture(
            tokens,
            seq,
            &mut kv_cache,
            [proc_start, proc_count],
            is_last_chunk,
            stream,
        )?;
        anyhow::ensure!(
            midcap_plan.is_none() || passenger_run.is_none(),
            "GLM fused chunk cannot split the SSM recurrence mid-chunk"
        );

        // ── Phase 4: forward through all layers ──
        self.prefill_b_forward_layers(
            seq,
            &mut kv_cache,
            chunk_start,
            chunk_len,
            is_last_chunk,
            proc_count,
            effective_seq_len_start,
            kv_write_start,
            marconi_skip,
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
            midcap_plan.as_ref(),
            passengers.as_deref_mut().zip(passenger_run.as_ref()),
            stream,
        )?;
        let t_fwd = tp.elapsed();
        self.warm_trace_sync(stream)?;
        let marks = [t_lookup, t_prefix, t_blocks, t_meta, tp.elapsed()];
        // Measure the forward's true GPU execution: the launches are async, so
        // `t_fwd` is submission time only. A profile-only sync here isolates
        // real GPU duration from the memset/H2D drain attributed to `embed`.
        if profile {
            let _fs = std::time::Instant::now();
            let _ = self.gpu.synchronize(stream);
            tracing::info!(
                "prefill fwd-exec (chunk {}..{}): submit={:?} gpu_exec={:?}",
                chunk_start,
                chunk_start + chunk_len,
                t_fwd.saturating_sub(t_meta),
                _fs.elapsed(),
            );
        }
        if profile {
            tracing::info!(
                "prefill profile (chunk {}..{} len={}): zero={:?} embed={:?} prefix={:?} blocks={:?} proc={:?} meta={:?} fwd={:?} total={:?}",
                chunk_start,
                chunk_start + chunk_len,
                chunk_len,
                t_pre[0],
                t_pre[1],
                t_lookup,
                t_blocks.saturating_sub(t_prefix),
                t_proc.saturating_sub(t_blocks),
                t_meta.saturating_sub(t_proc),
                t_fwd.saturating_sub(t_meta),
                t_fwd,
            );
        }
        if let Some(p) = passengers {
            // Before the chunk's finalize, which reuses the logits rows.
            self.glm_passengers_finish(p, proc_count, stream)?;
        }

        // Register the reserved slot as the session tail once the full pass has
        // captured the @tb state into it (no-op when no capture was planned).
        if let Some(plan) = midcap_plan.as_ref().filter(|p| !p.ckpt) {
            self.finalize_midchunk_capture(tokens, seq, plan);
        }

        // ── Phase 5: update sequence state incrementally ──
        // Always add chunk tokens exactly once. The early-return path for
        // fully cached non-last chunks doesn't add tokens, so this is the
        // single insertion point for all chunks that reach here.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = chunk_start + chunk_len;
        // #155: prime the decode-checkpoint cadence gate; the last chunk
        // leaves it at the prompt's complete-block count (see prefill_a).
        seq.last_decode_ckpt_block = seq.tokens.len() / bs;

        // ── Legacy echo+logprobs: project prompt positions while this
        // chunk's hidden rows are live, BEFORE finalize_last (which
        // re-derives norm_output + logits for the first sampled token).
        // No-op unless seq.collect_prompt_logprobs is set.
        self.collect_prompt_logprobs_chunk(
            tokens,
            seq,
            chunk_start,
            proc_start,
            proc_count,
            stream,
        )?;

        let _pass_aux =
            self.qwen4exp_ckpt_saves(tokens, seq, &mut kv_cache, &midcap_plan, stream)?;
        let out = if is_last_chunk {
            // ── Phase 6+7+8: final norm, lm_head, prefix-cache + snapshot save ──
            self.prefill_b_finalize_last(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                proc_count,
                stream,
            )?
        } else {
            // ── Phase 9: intermediate Marconi checkpoint ──
            self.prefill_b_save_checkpoint(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                stream,
            )?;
            DevicePtr::NULL
        };
        let (chunk, logits) = ((span.0, proc_count), is_last_chunk.then_some(out));
        self.warm_trace_chunk(seq, total, tp, chunk, t_pre, marks, logits, stream)?;
        Ok(out)
    }
}
