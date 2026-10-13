// SPDX-License-Identifier: AGPL-3.0-only

//! The post-prepare remainder of `DraftProposer::propose_batch`: input
//! validation, scratch upload, the staged B×gamma forward, the token
//! readback, and the tail dispatch — plus the per-sequence fallback runner
//! for generic authoritative mode.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::batch_execution;
use super::parity_report;
use super::{
    BlockDiffusionDraftHead, CaptureDescriptor, DflashProposerState, DsparkBatchInput,
    LIGHTNING_SERVED_GAMMA, SequenceGeneration,
};

impl BlockDiffusionDraftHead {
    /// Everything after per-sequence `prepare_drafts_state`: returns the
    /// staged/serial proposals. Errors bubble to `propose_batch`, which for
    /// `generic_auth` turns them into a per-sequence fallback instead of
    /// an Err to the scheduler.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_batch_staged_dispatch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn crate::speculative::ProposerState],
        expected_owners: &[SequenceGeneration],
        native_authoritative: bool,
        generic_auth: bool,
        parity_oracle: Option<Vec<Vec<u32>>>,
        parity_hidden_oracle: Option<Vec<u8>>,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        grammar_bitmasks: Option<&[Option<Vec<i32>>]>,
        batch_split: Option<(&super::rank_split::RankSplit, &dyn spark_comm::CommBackend)>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let n = last_tokens.len();

        // any stream/event dispatch. The current implementation below remains
        // serial-per-sequence or pinned-lane compute; this is only its validated
        // B×gamma input seam. Its capacity is the already-admitted call width,
        // so this does not widen propose_batch_max or allocate batch scratch.
        let mut owners = Vec::with_capacity(n);
        let mut lifecycles = Vec::with_capacity(n);
        let mut block_table_ptrs = Vec::with_capacity(n);
        let mut batch_kv_lens = Vec::with_capacity(n);
        let mut batch_block_tables = Vec::with_capacity(n);
        let mut batch_ctx_counts = Vec::with_capacity(n);
        let mut ban_depths: Vec<u32> = Vec::with_capacity(n);
        for (sequence, state) in states.iter_mut().enumerate() {
            let dstate = state
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
            let lifecycle = dstate.lifecycle.clone();
            let owner = lifecycle
                .as_ref()
                .map(CaptureDescriptor::owner)
                .unwrap_or(expected_owners[owners.len()]);
            owners.push(owner);
            lifecycles.push(lifecycle);
            ban_depths.push(positions.get(sequence).map_or(0, |&anchor| {
                crate::traits::EosBan::banned_draft_depth(
                    dstate.end_floor,
                    anchor,
                    self.gamma.saturating_sub(1),
                )
            }));
            let block_table_dev = dstate.block_table_dev.unwrap_or(DevicePtr::NULL);
            block_table_ptrs.push(block_table_dev.0);
            batch_ctx_counts.push(dstate.ctx_count_drafter);
            batch_block_tables.push(dstate.block_table.clone());
            batch_kv_lens.push(
                dstate
                    .ctx_count_drafter
                    .checked_add(self.gamma)
                    .ok_or_else(|| anyhow::anyhow!("DFlash batch KV length overflow"))?,
            );
        }
        let batch_slot_mapping = batch_execution::paged_slot_mapping(
            &batch_block_tables,
            &batch_ctx_counts,
            self.gamma,
            16,
        )?;
        let batch_slots_ready =
            block_table_ptrs.iter().all(|&pointer| pointer != 0) && batch_slot_mapping.is_some();
        if self.startup.diagnostics.batch_parity {
            tracing::info!(
                "DFlash Bxgamma parity cache gate: batch={} slots_ready={} device_tables={}/{}",
                n,
                batch_slots_ready,
                block_table_ptrs
                    .iter()
                    .filter(|&&pointer| pointer != 0)
                    .count(),
                n
            );
        }
        if native_authoritative {
            anyhow::ensure!(
                batch_slots_ready,
                "Lightning DSpark native batch cache slots are not ready"
            );
            anyhow::ensure!(
                self.lane_count() == 1,
                "Lightning DSpark native batch requires exactly one proposal lane"
            );
        }
        let batch_slot_mapping = batch_slot_mapping.unwrap_or_default();
        // Row contract: the Lightning product pins LIGHTNING_SERVED_GAMMA;
        // generic DFlash validates against its own configured gamma.
        let expected_gamma = if native_authoritative {
            LIGHTNING_SERVED_GAMMA
        } else {
            self.gamma
        };
        let batch_inputs = DsparkBatchInput::validate(
            self.gamma,
            expected_gamma,
            self.batch_capacity,
            &owners,
            last_tokens,
            positions,
            target_hiddens,
            expected_owners,
            &lifecycles,
        )?;
        // Materialize the exact host execution plan now. The next native slice
        // uploads these packed queries and depth rows into batch scratch; the
        // current serial/lane compute below remains the output oracle.
        let packed_query_tokens = batch_inputs.packed_query_tokens(self.mask_token_id);
        let _markov_depth_rows: Vec<Vec<usize>> = (1..batch_inputs.gamma())
            .map(|query| batch_inputs.rows_at_query(query))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let query_bytes: Vec<u8> = packed_query_tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        let packed_positions = batch_inputs.packed_positions()?;
        let position_bytes: Vec<u8> = packed_positions
            .iter()
            .flat_map(|position| position.to_le_bytes())
            .collect();
        let last_token_bytes: Vec<u8> = last_tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        ctx.gpu.copy_h2d(&query_bytes, self.batch_query_ids_dev)?;
        ctx.gpu.copy_h2d(&position_bytes, self.batch_position_ids)?;
        ctx.gpu
            .copy_h2d(&last_token_bytes, self.batch_markov_prev)?;
        // min_tokens: leading draft depths the selector walk may not end on.
        let ban_depth_bytes: Vec<u8> = ban_depths.iter().flat_map(|d| d.to_le_bytes()).collect();
        ctx.gpu.copy_h2d(&ban_depth_bytes, self.batch_ban_depth())?;
        let ptr_bytes: Vec<u8> = block_table_ptrs
            .iter()
            .flat_map(|pointer| pointer.to_le_bytes())
            .collect();
        let cu_seqlens: Vec<i32> = (0..=n)
            .map(|sequence| {
                i32::try_from(sequence * self.gamma)
                    .map_err(|_| anyhow::anyhow!("DFlash batch cu_seqlens overflow"))
            })
            .collect::<Result<_>>()?;
        let cu_bytes: Vec<u8> = cu_seqlens
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let kv_lens_i32: Vec<i32> = batch_kv_lens
            .iter()
            .copied()
            .map(|value| {
                i32::try_from(value)
                    .map_err(|_| anyhow::anyhow!("DFlash batch KV length i32 overflow"))
            })
            .collect::<Result<_>>()?;
        let kv_bytes: Vec<u8> = kv_lens_i32
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut attention_args = Vec::with_capacity(n * 12);
        for sequence in 0..n {
            let kv_len = u32::try_from(batch_kv_lens[sequence])
                .map_err(|_| anyhow::anyhow!("DFlash attention kv_len exceeds u32"))?;
            let q_offset = u32::try_from(batch_ctx_counts[sequence])
                .map_err(|_| anyhow::anyhow!("DFlash attention q_offset exceeds u32"))?;
            let q_rope_pos = u32::try_from(positions[sequence])
                .map_err(|_| anyhow::anyhow!("DFlash attention q_rope_pos exceeds u32"))?;
            attention_args.extend_from_slice(&kv_len.to_le_bytes());
            attention_args.extend_from_slice(&q_offset.to_le_bytes());
            attention_args.extend_from_slice(&q_rope_pos.to_le_bytes());
        }
        ctx.gpu.copy_h2d(&ptr_bytes, self.batch_block_table_ptrs)?;
        ctx.gpu.copy_h2d(&cu_bytes, self.batch_cu_seqlens)?;
        ctx.gpu.copy_h2d(&kv_bytes, self.batch_kv_lens)?;
        ctx.gpu
            .copy_h2d(&attention_args, self.batch_attention_args)?;
        if batch_slots_ready {
            let slot_bytes: Vec<u8> = batch_slot_mapping
                .iter()
                .flat_map(|slot| slot.to_le_bytes())
                .collect();
            ctx.gpu.copy_h2d(&slot_bytes, self.batch_slot_mapping)?;
        }
        crate::layers::ops::batched_embed(
            ctx.gpu,
            self.kernels.batched_embed,
            self.batch_query_ids_dev,
            self.embed_tokens_shared,
            self.batch_query_embed,
            batch_inputs.total_rows() as u32,
            self.hidden_size as u32,
            stream,
        )?;
        let batch_rows = u32::try_from(batch_inputs.total_rows())
            .map_err(|_| anyhow::anyhow!("DFlash batch row count exceeds u32"))?;
        let batch_size =
            u32::try_from(n).map_err(|_| anyhow::anyhow!("DFlash batch width exceeds u32"))?;
        let native_staged = batch_slots_ready && self.lane_count() == 1;
        if native_staged {
            let max_kv_len =
                u32::try_from(batch_kv_lens.iter().copied().max().unwrap_or(self.gamma))
                    .map_err(|_| anyhow::anyhow!("DFlash batched KV length exceeds u32"))?;
            // Generic DFlash2 has no attention sinks and uses the same
            // per-sequence indirect attention kernel as the serial Option-B
            // path; the Lightning product keeps the batched-sink kernel.
            let (serial_tables, serial_args) = if native_authoritative {
                (None, None)
            } else {
                (
                    Some(block_table_ptrs.as_slice()),
                    Some(self.batch_attention_args),
                )
            };
            // ATLAS_GLM_DRAFT_TP_BATCH: the same layers and tail base, the
            // MLP and head halves swapped with the worker (no parity oracle).
            let walked = match batch_split {
                Some((split, comm)) => {
                    let a = super::batch_forward::BatchLayerArgs {
                        batch_rows,
                        batch_size,
                        max_kv_len,
                        serial_block_tables: serial_tables,
                        serial_attention_args: serial_args,
                        ctx,
                        stream,
                    };
                    self.walk_batch_split(split, comm, a)?;
                    true
                }
                None => false,
            };
            if !walked {
                for layer_idx in 0..self.layers.len() {
                    self.run_batched_layer_stage(
                        layer_idx,
                        batch_rows,
                        batch_size,
                        max_kv_len,
                        serial_tables,
                        serial_args,
                        ctx,
                        stream,
                    )?;
                }
            }
            if let Some(expected) = parity_hidden_oracle.as_ref() {
                ctx.gpu.synchronize(stream)?;
                let mut actual = vec![0u8; expected.len()];
                ctx.gpu.copy_d2h(self.batch_query_embed, &mut actual)?;
                if native_authoritative {
                    if actual != *expected {
                        let first = actual
                            .chunks_exact(2)
                            .zip(expected.chunks_exact(2))
                            .position(|(lhs, rhs)| lhs != rhs)
                            .unwrap_or(0);
                        let per_sequence = self.gamma * self.hidden_size;
                        let sequence = first / per_sequence;
                        let local = first % per_sequence;
                        anyhow::bail!(
                            "DFlash Bxgamma backbone parity mismatch at sequence {sequence} BF16 element {local}"
                        );
                    }
                    tracing::info!("DFlash Bxgamma backbone parity PASS: batch={n}");
                } else {
                    // M going γ → B·γ changes GEMM tiling and reduction order,
                    // so backbone bytes are a report, not a bail.
                    let (mismatched, max_abs_diff) = parity_report::bf16_diff(&actual, expected);
                    tracing::info!(
                        "DFlash Bxgamma backbone parity: batch={} mismatched={}/{} max_abs_diff={:.4e}",
                        n,
                        mismatched,
                        actual.len() / 2,
                        max_abs_diff
                    );
                }
            }
            if !walked {
                self.run_batched_tail_base(batch_rows, ctx, stream)?;
            }
            // #102: grammar-mask each masked seq's logits rows 0/1 (anchor +
            // draft 0 — both predict pos+1). The pre-pass writes -inf in
            // place; the selector/Markov arms below stay unchanged.
            if self.kernels.grammar_bitmask.0 != 0
                && let Some(masks) = grammar_bitmasks
                && masks.iter().flatten().next().is_some()
            {
                let words = crate::layers::ops::grammar_bitmask_words(self.vocab_size as u32);
                let seq_logits_bytes = self.gamma * self.vocab_size * 2;
                for (sequence, mask) in masks.iter().take(n).enumerate() {
                    let Some(mask) = mask else { continue };
                    if mask.len() < words {
                        tracing::warn!(
                            "DFlash grammar bitmask short at seq {sequence} ({} < {words} words): skipping mask",
                            mask.len()
                        );
                        continue;
                    }
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(mask.as_ptr() as *const u8, words * 4)
                    };
                    let mask_dev = self.batch_grammar_bitmask.offset(sequence * words * 4);
                    ctx.gpu.copy_h2d(bytes, mask_dev)?;
                    for row in 0..2usize.min(self.gamma) {
                        crate::layers::ops::apply_grammar_bitmask(
                            ctx.gpu,
                            self.kernels.grammar_bitmask,
                            self.batch_logits
                                .offset(sequence * seq_logits_bytes + row * self.vocab_size * 2),
                            mask_dev,
                            self.vocab_size as u32,
                            stream,
                        )?;
                    }
                }
            }
            if self.candidate_selector.is_some() {
                self.run_batched_dflash2_tail(batch_size, last_tokens, ctx, stream)?;
            } else {
                self.run_batched_markov(batch_size, ctx, stream)?;
            }
        }

        // One selector record per sequence: gamma tokens, then its draft
        // confidences when the head reports them (`draft_conf`).
        let record_bytes = self.draft_record_bytes();
        let mut records = Vec::new();
        let native = if native_staged
            && (native_authoritative || self.startup.diagnostics.batch_parity || generic_auth)
        {
            ctx.gpu.synchronize(stream)?;
            records.resize(n * record_bytes, 0u8);
            ctx.gpu.copy_d2h(self.batch_tokens, &mut records)?;
            let row_tokens: Vec<u32> = records
                .chunks_exact(record_bytes)
                .flat_map(|record| record.chunks_exact(4).take(self.gamma))
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            Some(batch_inputs.reorder_sampled_rows(&row_tokens)?)
        } else {
            None
        };

        let lanes_n = self.lane_count();
        if lanes_n == 1 {
            if let Some(oracle) = parity_oracle {
                let native = native.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("DFlash Bxgamma parity did not stage native tokens")
                })?;
                if native_authoritative {
                    for (sequence, (native_tokens, oracle_tokens)) in
                        native.iter().zip(oracle.iter()).enumerate()
                    {
                        anyhow::ensure!(
                            native_tokens.get(..oracle_tokens.len())
                                == Some(oracle_tokens.as_slice()),
                            "DFlash Bxgamma parity mismatch at sequence {sequence}: native={native_tokens:?} oracle={oracle_tokens:?}"
                        );
                    }
                    tracing::info!(
                        "DFlash Bxgamma staged parity PASS: batch={} gamma={} rows={}",
                        n,
                        self.gamma,
                        batch_inputs.total_rows()
                    );
                } else {
                    // GEMM tiling differs at B·γ rows, so drafts are a report;
                    // parity mode still returns the oracle's serial drafts.
                    let (sequences_exact, tokens_equal, tokens_total) =
                        parity_report::draft_agreement(native, &oracle);
                    tracing::info!(
                        "DFlash Bxgamma draft parity: batch={} sequences_exact={}/{} tokens_equal={}/{}",
                        n,
                        sequences_exact,
                        n,
                        tokens_equal,
                        tokens_total
                    );
                }
                return Ok(Some(oracle));
            }
            if native_authoritative {
                let mut out = native.ok_or_else(|| {
                    anyhow::anyhow!("Lightning DSpark native batch returned no staged tokens")
                })?;
                let cap = num_drafts.min(self.gamma.saturating_sub(1)).max(1);
                for (i, tokens) in out.iter_mut().enumerate() {
                    tokens.truncate(cap);
                    let dstate = states[i]
                        .as_any_mut()
                        .downcast_mut::<DflashProposerState>()
                        .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
                    dstate.last_num_drafted = tokens.len();
                }
                return Ok(Some(out));
            }
            if generic_auth {
                // Authoritative: the staged Bxgamma tokens ARE the response —
                // truncated to the draft cap (the diagnostic override stays
                // honoured), with last_num_drafted set like serial.
                let mut out = native.ok_or_else(|| {
                    anyhow::anyhow!(
                        "DFlash batched propose produced no staged tokens \
                         (cache slots not ready or staging skipped)"
                    )
                })?;
                // Plus the optional tighter cap for owner-batched steps.
                let cap = super::batch_plan::batched_draft_cap(
                    self.draft_cap(num_drafts),
                    super::batch_plan::multi_owner_draft_cap(),
                    n,
                );
                for (i, tokens) in out.iter_mut().enumerate() {
                    tokens.truncate(cap);
                    let dstate = states[i]
                        .as_any_mut()
                        .downcast_mut::<DflashProposerState>()
                        .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
                    dstate.last_num_drafted = tokens.len();
                    dstate.last_draft_conf = records
                        .chunks_exact(record_bytes)
                        .nth(i)
                        .map(|record| {
                            super::draft_conf::draft_conf(record, self.gamma, tokens.len())
                        })
                        .unwrap_or_default();
                }
                return Ok(Some(out));
            }
            // Generic single-lane serial path: reached only when the authoritative
            // arm is off (rollback ATLAS_DFLASH_BATCHED_PROPOSE=0, unmet Option B/lane
            // preconditions) or a parity oracle is active.
            let mut serial = Vec::with_capacity(n);
            for i in 0..n {
                serial.push(self.propose_drafts(
                    last_tokens[i],
                    target_hiddens[i],
                    positions[i],
                    num_drafts,
                    states[i],
                    Some(expected_owners[i]),
                    ctx,
                    stream,
                    None,
                    None,
                    Some(target_hiddens[i]),
                )?);
            }
            return Ok(Some(serial));
        }
        if generic_auth {
            // The startup gate requires a single lane; if the invariant is
            // ever broken, the prepared states must not run through the lane
            // path — this Err becomes the per-sequence fallback.
            anyhow::bail!("DFlash batched propose: generic authoritative requires one lane");
        }
        // Multi-lane: each seq proposes on its pinned lane — see
        // `propose_on_lanes` for the enqueue/collect ordering contract.
        self.propose_on_lanes(
            last_tokens,
            target_hiddens,
            positions,
            num_drafts,
            states,
            expected_owners,
            ctx,
            grammar_bitmasks,
        )
        .map(Some)
    }
}
