// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-only scoring: one forward pass over a shared prefix and N branch
//! continuations, returning each branch's softmax over its own option tokens
//! at its last row. Nothing is sampled, decoded, cached or snapshotted.
//!
//! Row layout (VARLEN, `cu_seqlens` order):
//!   stream 0     = prefix[0..F)               F = floor(L / block) * block
//!   stream 1..=N = prefix[F..L) ++ branch_i   positions start at F
//! Stream 0 is omitted when F == 0. Each branch's block table shares stream
//! 0's F / block physical blocks (ref-counted, as a prefix-cache hit would)
//! and owns the rest, so no partly filled block is ever shared; each branch
//! replays the prefix tail `[F, L)` itself (< one block of rows).
//!
//! Per layer every weight-bound step (norms, projections, MoE) runs ONCE over
//! all rows, so the weights are read once per request. Attention needs no
//! ordering: every row's K/V is written before the batched varlen attention
//! reads it, and the kernel derives each stream's causal offset from its own
//! `kv_lens[b]` (`prefill_paged_compute.cuh`). Only the SSM recurrence is
//! ordered — see `score_ssm.rs`.

#![allow(clippy::too_many_arguments)]

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::score_profile::{Phase, ScoreProfile};
use super::stage_batched::PerStreamStageInfo;
use crate::layer::{ForwardContext, GdnPrefillBuffers};
use crate::traits::{Model, SequenceState};

/// Host-side geometry of one score pass.
pub(super) struct ScorePlan {
    /// Full token list per stream (stream 0 = prefix[..F]; branch = prefix ++ q).
    pub tokens: Vec<Vec<u32>>,
    /// First processed position per stream (0 for stream 0, F for branches).
    pub starts: Vec<usize>,
    /// Whether stream 0 (the shared prefix head) is present.
    pub has_head: bool,
    /// Number of shared blocks (F / block).
    pub shared_blocks: usize,
}

impl ScorePlan {
    fn new(prefix: &[u32], branches: &[Vec<u32>], bs: usize) -> Self {
        let f = (prefix.len() / bs) * bs;
        let has_head = f > 0;
        let mut tokens = Vec::with_capacity(branches.len() + 1);
        let mut starts = Vec::with_capacity(branches.len() + 1);
        if has_head {
            tokens.push(prefix[..f].to_vec());
            starts.push(0);
        }
        for b in branches {
            let mut t = Vec::with_capacity(prefix.len() + b.len());
            t.extend_from_slice(prefix);
            t.extend_from_slice(b);
            tokens.push(t);
            starts.push(f);
        }
        Self {
            tokens,
            starts,
            has_head,
            shared_blocks: f / bs,
        }
    }

    pub(super) fn rows(&self, i: usize) -> usize {
        self.tokens[i].len() - self.starts[i]
    }

    pub(super) fn total_rows(&self) -> usize {
        (0..self.tokens.len()).map(|i| self.rows(i)).sum()
    }
}

impl TransformerModel {
    /// See the module docs. `options[i]` are branch i's single option token
    /// ids; the result is the softmax over exactly those logits.
    pub(in crate::model) fn score_branches_impl(
        &self,
        prefix: &[u32],
        branches: &[Vec<u32>],
        options: &[Vec<u32>],
        stream: u64,
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(!branches.is_empty(), "score: no branches");
        ensure!(branches.len() == options.len(), "score: one option set per branch");
        ensure!(branches.iter().all(|b| !b.is_empty()), "score: empty branch");
        ensure!(options.iter().all(|o| !o.is_empty()), "score: empty option set");
        let v = self.config.vocab_size as u32;
        ensure!(
            options.iter().flatten().all(|&t| t < v),
            "score: option token outside the vocabulary"
        );
        ensure!(
            !self.multi_rank_protocol_active(),
            "score: single-rank models only"
        );
        ensure!(self.comm.is_none(), "score: single-rank models only");
        ensure!(!self.use_fp32_logits, "score: FP32 logits mode is not supported");
        let bs = self.kv_cache.lock().block_size();
        let plan = ScorePlan::new(prefix, branches, bs);
        ensure!(
            plan.total_rows() <= self.buffers.max_batch_tokens(),
            "score: {} rows exceed the prefill arena ({} tokens)",
            plan.total_rows(),
            self.buffers.max_batch_tokens()
        );

        let trace = std::env::var("ATLAS_SCORE_TRACE").is_ok_and(|v| v == "1");
        let t0 = std::time::Instant::now();
        let mut marks = [0u128; 3];
        let mut seqs: Vec<SequenceState> = Vec::with_capacity(plan.tokens.len());
        let run = (|| -> Result<Vec<Vec<f32>>> {
            for _ in 0..plan.tokens.len() {
                seqs.push(self.alloc_sequence()?);
            }
            marks[0] = t0.elapsed().as_micros();
            self.score_forward(&plan, &mut seqs, stream)?;
            if trace {
                self.gpu.synchronize(stream)?;
            }
            marks[1] = t0.elapsed().as_micros();
            let out = self.score_head(&plan, options, stream);
            marks[2] = t0.elapsed().as_micros();
            out
        })();
        for seq in seqs.iter_mut() {
            // Score sequences never populate `tokens`, so the prefix-cache
            // release in `free_sequence` is a no-op; the KV refs (shared head
            // blocks included) and the SSM slot are returned here.
            if let Err(e) = self.free_sequence(seq) {
                tracing::error!("score: free_sequence: {e:#}");
            }
        }
        if trace {
            tracing::info!(
                "score trace: rows={} streams={} alloc={}us forward={}us head={}us free={}us",
                plan.total_rows(),
                plan.tokens.len(),
                marks[0],
                marks[1].saturating_sub(marks[0]),
                marks[2].saturating_sub(marks[1]),
                t0.elapsed().as_micros().saturating_sub(marks[2])
            );
        }
        run
    }

    /// Allocate blocks, embed, stage metadata and run the layer loop.
    fn score_forward(
        &self,
        plan: &ScorePlan,
        seqs: &mut [SequenceState],
        stream: u64,
    ) -> Result<()> {
        let n = seqs.len();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();

        // ── Blocks: pre-flight the whole request, then allocate. ──
        let own: usize = (0..n)
            .map(|i| plan.tokens[i].len().div_ceil(bs) - if plan.has_head && i > 0 {
                plan.shared_blocks
            } else {
                0
            })
            .sum();
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
            if plan.has_head && i > 0 {
                let (head, rest) = seqs.split_at_mut(1);
                for &b in &head[0].block_table[..plan.shared_blocks] {
                    kv_cache.inc_ref(b);
                    rest[i - 1].block_table.push(b);
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

        // ── Embed + per-stream paged metadata at stacked offsets. ──
        self.buffers
            .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        let total = plan.total_rows();
        let topk = self.config.num_experts_per_tok.max(1);
        let mut cursor = (total * topk * 4 * 2 + 63) & !63;
        let hidden = self.buffers.hidden_states();
        let mut off = 0usize;
        for i in 0..n {
            let (start, rows, len) = (plan.starts[i], plan.rows(i), plan.tokens[i].len());
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
        let meta = self.stage_batched_attn_metadata(
            &infos,
            &kv_cache,
            self.config.mrope_interleaved,
            cursor,
            stream,
        )?;
        drop(infos);
        cursor += meta.staged_bytes;
        let ptrs_off = (cursor + 63) & !63;
        // h_state_ptrs[n] then the branches' rebased cu_seqlens[n+1].
        let branch_cu_off = ptrs_off + ((n * 8 + 63) & !63);
        ensure!(
            branch_cu_off + (n + 1) * 4 <= self.buffers.scratch_bytes(),
            "score: SSM staging exceeds scratch ({n} streams)"
        );
        let branch_cu = self.score_stage_branch_cu(plan, &meta, branch_cu_off, stream)?;

        let gdn = GdnPrefillBuffers {
            qkv: self.gdn_buf_qkv,
            gate_beta: self.gdn_buf_gate_beta,
            output: self.gdn_buf_out,
            z: self.gdn_buf_z,
            total_len: total,
        };
        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
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
        let residual = self.buffers.residual();
        prof.mark(self, stream, Phase::Setup)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer.is_ssm_layer() {
                let mut refs: Vec<&mut SequenceState> = seqs.iter_mut().collect();
                self.score_ssm_layer(
                    layer.as_ref(),
                    layer_idx,
                    hidden,
                    residual,
                    &mut refs,
                    plan.has_head,
                    &meta,
                    &gdn,
                    ptrs_off,
                    branch_cu,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_shares_whole_blocks_and_replays_the_tail() {
        let prefix: Vec<u32> = (0..37).collect();
        let plan = ScorePlan::new(&prefix, &[vec![100, 101], vec![200]], 16);
        assert!(plan.has_head);
        assert_eq!(plan.shared_blocks, 2);
        assert_eq!(plan.starts, vec![0, 32, 32]);
        assert_eq!(plan.tokens[0], (0..32).collect::<Vec<u32>>());
        assert_eq!(plan.rows(1), 5 + 2);
        assert_eq!(plan.rows(2), 5 + 1);
        assert_eq!(plan.total_rows(), 32 + 7 + 6);
    }

    #[test]
    fn short_prefix_has_no_head_stream() {
        let plan = ScorePlan::new(&[1, 2, 3], &[vec![4]], 16);
        assert!(!plan.has_head);
        assert_eq!(plan.shared_blocks, 0);
        assert_eq!(plan.starts, vec![0]);
        assert_eq!(plan.rows(0), 4);
    }
}
