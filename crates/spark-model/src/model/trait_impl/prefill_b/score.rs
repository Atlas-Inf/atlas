// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-only scoring: one forward pass over one or more requests, each a
//! shared prefix and N branch continuations, returning each branch's softmax
//! over its own option tokens at its last row. Nothing is sampled, decoded,
//! cached or snapshotted. Queued requests can share one pass, so they also
//! share one read of the weights.
//!
//! Row layout (VARLEN, `cu_seqlens` order, see `score_plan.rs`): every
//! request's head `prefix[0..F)` (F = floor(L / block) * block; omitted when
//! F == 0), then every request's branches `prefix[F..L) ++ branch_i`, whose
//! positions start at F. A branch's block table shares its head's F / block
//! physical blocks (ref-counted, as a prefix-cache hit would) and owns the
//! rest, so no partly filled block is ever shared; each branch replays the
//! prefix tail `[F, L)` itself (< one block of rows).
//!
//! A request may continue a knowledge pack (`score_pack.rs`): its positions
//! start at the pack's length and the streams that read the pack directly take
//! its blocks and SSM state; the pack's rows are never recomputed.
//!
//! Per layer every weight-bound step (norms, projections, MoE) runs ONCE over
//! all rows. Attention needs no ordering: every row's K/V is written before
//! the batched varlen attention reads it, and the kernel derives each
//! stream's causal offset from its own `kv_lens[b]`
//! (`prefill_paged_compute.cuh`). Only the SSM recurrence is ordered — see
//! `score_ssm.rs`.

#![allow(clippy::too_many_arguments)]

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::score_plan::{ScorePlan, ScoreRequest};
use super::score_profile::{Phase, ScoreProfile};
use super::stage_batched::PerStreamStageInfo;
use crate::layer::{ForwardContext, GdnPrefillBuffers};
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    /// See the module docs. Per request, `options[i]` are branch i's single
    /// option token ids; the result is, per request and branch, the softmax
    /// over exactly those logits.
    pub(in crate::model) fn score_requests_impl(
        &self,
        reqs: &[crate::traits::ScoreInput<'_>],
        stream: u64,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        ensure!(!reqs.is_empty(), "score: no requests");
        let v = self.config.vocab_size as u32;
        for r in reqs {
            ensure!(!r.branches.is_empty(), "score: no branches");
            ensure!(
                r.branches.len() == r.options.len(),
                "score: one option set per branch"
            );
            ensure!(
                r.branches.iter().all(|b| !b.is_empty()),
                "score: empty branch"
            );
            ensure!(
                r.options.iter().all(|o| !o.is_empty()),
                "score: empty option set"
            );
            let tokens = r.prefix.iter().chain(r.branches.iter().flatten());
            if let Some(t) = tokens.chain(r.options.iter().flatten()).find(|&&t| t >= v) {
                anyhow::bail!("score: token id {t} outside the model's vocabulary ({v} rows)");
            }
        }
        ensure!(
            self.score_supported(),
            "score: single-rank models with BF16 logits only"
        );
        ensure!(
            !reqs.iter().any(|r| r.prefix_bidir)
                || self
                    .layers
                    .iter()
                    .all(|l| l.is_ssm_layer() || l.honours_state_end()),
            "score: attention \"prefix_bidir\" is not supported by this model's attention \
             layers (MLA, sliding-window, indexer or hyper-connection attention reads causally)"
        );
        let bs = self.kv_cache.lock().block_size();
        // Held for the whole pass: no pack is evicted while this pass reads it.
        let mut held = self.score_packs.lock();
        let (req_pack, ids) = super::score_pack::resolve(&mut held, reqs)?;
        let packs: Vec<&SequenceState> = ids.iter().map(|id| &held.0[id].seq).collect();
        let pack_tokens: Vec<&[u32]> = ids.iter().map(|id| &held.0[id].tokens[..]).collect();
        let plan_reqs: Vec<ScoreRequest<'_>> = reqs
            .iter()
            .zip(&req_pack)
            .map(|(r, k)| {
                let pack = k.map(|k| (k, pack_tokens[k]));
                (r.prefix, r.branches, r.prefix_bidir, pack)
            })
            .collect();
        let plan = ScorePlan::new(&plan_reqs, bs);
        // The block-table stride bounds a stream's positions (--max-seq-len).
        let max_blocks = self.max_blocks_per_seq as usize;
        if let Some(t) = plan
            .tokens
            .iter()
            .find(|t| t.len().div_ceil(bs) > max_blocks)
        {
            anyhow::bail!(
                "score: {} positions (pack + prefix + branch) exceed {max_blocks} KV blocks \
                 per sequence (raise --max-seq-len)",
                t.len()
            );
        }
        let (max_rows, max_streams) = self.score_capacity_held(held.0.len());
        ensure!(
            plan.total_rows() <= max_rows,
            "score: {} rows exceed one pass ({max_rows} rows: the prefill arena and SSM \
             staging buffers; raise --max-prefill-tokens and --max-seq-len)",
            plan.total_rows()
        );
        ensure!(
            plan.streams() <= max_streams,
            "score: {} sequences exceed the SSM state pool ({max_streams}; raise --max-batch-size)",
            plan.streams()
        );
        let options: Vec<Vec<u32>> = reqs
            .iter()
            .flat_map(|r| r.options.iter().cloned())
            .collect();

        let trace = std::env::var("ATLAS_SCORE_TRACE").is_ok_and(|v| v == "1");
        let t0 = std::time::Instant::now();
        let mut marks = [0u128; 3];
        let mut seqs: Vec<SequenceState> = Vec::with_capacity(plan.tokens.len());
        let run = (|| -> Result<Vec<Vec<f32>>> {
            // Only a stream that starts from the zero state needs one: a
            // branch with a head enters from the head's conv window and state,
            // a stream on a pack from the pack's (`score_ssm.rs`), and MTP
            // scratch is never read here.
            for i in 0..plan.streams() {
                seqs.push(self.alloc_sequence_reset(false)?);
                if (i < plan.heads || plan.head_of[i].is_none()) && plan.reads_pack(i).is_none() {
                    let slot = seqs[i].slot_idx;
                    self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream)?;
                }
            }
            marks[0] = t0.elapsed().as_micros();
            self.score_forward(&plan, &mut seqs, &packs, stream)?;
            if trace {
                self.gpu.synchronize(stream)?;
            }
            marks[1] = t0.elapsed().as_micros();
            let out = self.score_head(&plan, &options, stream);
            marks[2] = t0.elapsed().as_micros();
            out
        })();
        if run.is_err() {
            // Work may still be in flight on the stream: let it finish before
            // its blocks and slots go back to the pools.
            if let Err(e) = self.gpu.synchronize(stream) {
                tracing::error!("score: synchronize after a failed pass: {e:#}");
            }
        }
        for seq in seqs.iter_mut() {
            // Score sequences never populate `tokens`, so the prefix-cache
            // release in `free_sequence` is a no-op; the KV refs (shared head
            // blocks included) and the SSM slot are returned here, the slot
            // unzeroed (its next claim zeroes what it reads).
            if let Err(e) = self.free_sequence_dispatch(seq, false) {
                tracing::error!("score: free_sequence: {e:#}");
            }
        }
        if trace {
            tracing::info!(
                "score trace: rows={} streams={} alloc={}us forward={}us head={}us free={}us",
                plan.total_rows(),
                plan.streams(),
                marks[0],
                marks[1].saturating_sub(marks[0]),
                marks[2].saturating_sub(marks[1]),
                t0.elapsed().as_micros().saturating_sub(marks[2])
            );
        }
        // Per-branch rows (stream order) back to per-request lists.
        let mut flat = run?.into_iter();
        Ok(plan
            .branch_counts
            .iter()
            .map(|&k| flat.by_ref().take(k).collect())
            .collect())
    }

    /// Allocate blocks, embed, stage metadata and run the layer loop. `packs`:
    /// the sequences of the knowledge packs `plan.pack_of` indexes.
    pub(super) fn score_forward(
        &self,
        plan: &ScorePlan,
        seqs: &mut [SequenceState],
        packs: &[&SequenceState],
        stream: u64,
    ) -> Result<()> {
        let n = seqs.len();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();

        // ── Blocks: pre-flight the whole request, then allocate. ──
        let own = plan.own_blocks(bs);
        while kv_cache.num_free_blocks() < own {
            let short = own - kv_cache.num_free_blocks();
            let evicted = self.prefix_cache.evict(short);
            if evicted.is_empty() {
                break;
            }
            self.apply_evicted(evicted, &mut kv_cache);
        }
        ensure!(
            kv_cache.num_free_blocks() >= own,
            "score: needs {own} KV blocks, {} free",
            kv_cache.num_free_blocks()
        );
        for i in 0..n {
            if let Some(h) = plan.head_of[i] {
                // Heads precede branches, so h < i.
                let (left, right) = seqs.split_at_mut(i);
                for &b in &left[h].block_table[..plan.shared_blocks[i]] {
                    kv_cache.inc_ref(b);
                    right[0].block_table.push(b);
                }
            } else if let Some(p) = plan.reads_pack(i) {
                // The pack's whole blocks, shared; its partial last block is
                // copied into this stream's own block below.
                for &b in &packs[p].block_table[..plan.shared_blocks[i]] {
                    kv_cache.inc_ref(b);
                    seqs[i].block_table.push(b);
                }
            }
            let end = plan.tokens[i].len();
            super::super::super::block_mgmt::ensure_blocks_through_prefill(
                &mut seqs[i],
                (end - 1) / bs,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;
        }
        if !packs.is_empty() {
            self.score_pack_tail_copies(plan, seqs, packs, &kv_cache, stream)?;
        }
        // prefix_bidir heads read the state's partial tail block from their
        // first branch (it holds the same tail K/V as every branch).
        for h in 0..plan.heads {
            if let Some(b) = plan.tail_from[h] {
                let tail = seqs[b].block_table[plan.shared_blocks[b]];
                kv_cache.inc_ref(tail);
                seqs[h].block_table.push(tail);
            }
        }

        // ── Embed + per-stream paged metadata at stacked offsets. ──
        let total = plan.total_rows();
        self.buffers
            .zero_prefill_essentials_rows(self.gpu.as_ref(), total, stream)?;
        let topk = self.config.num_experts_per_tok.max(1);
        let mut cursor = (total * topk * 4 * 2 + 63) & !63;
        let hidden = self.buffers.hidden_states();
        let mut off = 0usize;
        for i in 0..n {
            let (start, rows, len) = (plan.starts[i], plan.rows(i), plan.kv_len[i]);
            let toks = &plan.tokens[i];
            // positions (x3 with MRoPE) + slots for this stream's own rows.
            let meta_bytes = (rows * 24 + 127) & !63;
            self.prefill_b_embed_chunk_at(
                &seqs[i],
                toks,
                start,
                rows,
                hidden.offset(off * h * bf16),
                stream,
            )?;
            ensure!(
                cursor + meta_bytes <= self.buffers.scratch_bytes(),
                "score: metadata exceeds scratch ({n} streams)"
            );
            let meta_base = self.buffers.scratch().offset(cursor);
            let layout = self.prefill_b_upload_meta_at(
                toks,
                &mut seqs[i],
                start,
                rows,
                start,
                rows,
                start,
                &kv_cache,
                meta_base,
                self.buffers.scratch_bytes() - cursor,
                stream,
            )?;
            self.prefill_b_upload_paged(
                &mut seqs[i],
                len,
                start,
                rows,
                meta_base,
                layout.slot_offset,
                &kv_cache,
                stream,
            )?;
            cursor += meta_bytes;
            off += rows;
        }
        // H2D barrier before kernel compute (GB10 DMA quirk, as batch_kernel).
        self.gpu.synchronize(stream)?;
        let mut prof = ScoreProfile::new();

        let infos: Vec<PerStreamStageInfo<'_>> = seqs
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let page = seq.chunked_prefill_meta.as_ref();
                PerStreamStageInfo {
                    proc_start: plan.starts[i],
                    proc_count: plan.rows(i),
                    block_table_dev: page.map_or(DevicePtr::NULL, |m| m.block_table),
                    seq_len_dev: page.map_or(DevicePtr::NULL, |m| m.seq_len),
                    num_blocks: seq.block_table.len(),
                    seq,
                }
            })
            .collect();
        let mut meta = self.stage_batched_attn_metadata(
            &infos,
            &kv_cache,
            self.config.mrope_interleaved,
            cursor,
            stream,
        )?;
        drop(infos);
        cursor += meta.staged_bytes;
        if plan.any_bidir() {
            cursor = self.score_stage_bidir(plan, &mut meta, cursor, stream)?;
        }
        let ptrs_off = (cursor + 63) & !63;
        // h_state_ptrs[n] then the branches' rebased cu_seqlens[n+1].
        let branch_cu_off = ptrs_off + ((n * 8 + 63) & !63);
        ensure!(
            branch_cu_off + (n + 1) * 4 <= self.buffers.scratch_bytes(),
            "score: SSM staging exceeds scratch ({n} streams)"
        );
        let branch_cu = self.score_stage_branch_cu(plan, &meta, branch_cu_off, stream)?;
        let tabs_off = (branch_cu_off + (n + 1) * 4 + 63) & !63;
        let ssm_tabs = self.score_stage_ssm_tables(plan, seqs, packs, tabs_off, stream)?;

        let gdn = GdnPrefillBuffers {
            qkv: self.gdn_buf_qkv,
            gate_beta: self.gdn_buf_gate_beta,
            output: self.gdn_buf_out,
            // Z read in place from the projection output: no per-layer copy.
            z: self.buffers.ssm_deinterleaved().offset(
                (self.config.linear_num_key_heads * self.config.linear_key_head_dim * 2
                    + self.config.linear_num_value_heads * self.config.linear_value_head_dim)
                    * 2,
            ),
            z_stride: self.config.ssm_qkvz_size(),
            total_len: total,
        };
        // The dense BF16-MMA kernels at every row count, so a request's
        // rows take the same kernels alone as merged into a bigger pass.
        let levers = crate::layers::ops::ModelLevers {
            dense_prefill_any_rows: true,
            ..self.levers
        };
        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Refuse,
        };
        // Any non-zero start keeps the layers on the paged path; the kernels
        // take each stream's real offset from `kv_lens` / `cu_seqlens`.
        let seq_lens_start = plan.starts.iter().copied().max().unwrap_or(0).max(1);
        // No layer reads the residual copy back: alias it to `hidden` rather
        // than write a second activation-sized buffer per norm.
        let residual = if self.layers.iter().any(|l| l.prefill_reads_residual()) {
            self.buffers.residual()
        } else {
            hidden
        };
        prof.mark(self, stream, Phase::Setup)?;
        let mut ssm_k = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer.is_ssm_layer() {
                let tabs = ssm_tabs.map(|t| t.offset(ssm_k * super::score_ssm::TABLES * n * 8));
                ssm_k += 1;
                let mut refs: Vec<&mut SequenceState> = seqs.iter_mut().collect();
                self.score_ssm_layer(
                    layer.as_ref(),
                    layer_idx,
                    hidden,
                    residual,
                    &mut refs,
                    packs,
                    plan,
                    &meta,
                    &gdn,
                    ptrs_off,
                    branch_cu,
                    tabs,
                    &ctx,
                    stream,
                    &mut prof,
                )?;
            } else {
                layer.prefill_inner_batched_q12(
                    hidden,
                    residual,
                    total,
                    &mut kv_cache,
                    seq_lens_start,
                    &meta,
                    &ctx,
                    stream,
                )?;
                prof.mark(self, stream, Phase::Attn)?;
            }
        }
        prof.report(plan.total_rows(), n);
        Ok(())
    }
}

impl TransformerModel {
    /// `prefix_bidir`: upload each stream's state end and the heads' widened
    /// KV extents (a head with a partial tail reads up to S, not F). Returns
    /// the scratch cursor past the staged array.
    fn score_stage_bidir(
        &self,
        plan: &ScorePlan,
        meta: &mut crate::layer::BatchedAttnMetadata,
        cursor: usize,
        stream: u64,
    ) -> Result<usize> {
        let n = plan.streams();
        let off = (cursor + 63) & !63;
        ensure!(
            off + n * 4 <= self.buffers.scratch_bytes(),
            "score: state-end staging exceeds scratch ({n} streams)"
        );
        let as_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let dst = self.buffers.scratch().offset(off);
        self.gpu
            .copy_h2d_async_retained(&as_bytes(&plan.state_end), dst, stream)?;
        meta.state_end = dst;
        let mut changed = false;
        for i in 0..n {
            let want = plan.kv_len[i] as i32;
            if meta.kv_lens_host[i] != want {
                meta.kv_lens_host[i] = want;
                changed = true;
            }
        }
        if changed {
            let bytes = as_bytes(&meta.kv_lens_host);
            self.gpu
                .copy_h2d_async_retained(&bytes, meta.kv_lens, stream)?;
        }
        Ok(off + ((n * 4 + 63) & !63))
    }
}
