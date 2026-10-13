// SPDX-License-Identifier: AGPL-3.0-only

//! The per-step entries into the qwen4_exp piecewise graphs
//! (`decode_pieces.rs`): what each layer of a run executes for the single-row
//! decode, a K-row verify of one sequence (K = 2, 3, 4) and the batched
//! verify of several sequences, exactly as each step's eager layer loop
//! calls it. The batched decode (`decode_a2`) calls
//! [`TransformerModel::piece_run`] inline.

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use super::PieceStep;
use crate::layer::{ForwardContext, LayerState};
use crate::traits::SequenceState;

impl TransformerModel {
    /// [`Self::piece_run`] for the single-row decode (`decode_forward_body`):
    /// each layer runs `decode` and the DFlash capture of row 0, as the eager
    /// loop does. A sequence without an SSM pool slot runs eagerly (and
    /// never wide: see [`Self::decode_piece_wide`]).
    pub(in crate::model) fn decode_piece_run(
        &self,
        layer_idx: usize,
        wide: bool,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(slot) = seq.ssm_slot_idx() else {
            return Ok(false);
        };
        let (hidden, residual) = (self.buffers.hidden_states(), self.buffers.residual());
        self.piece_run(
            layer_idx,
            PieceStep::Decode,
            wide,
            &[slot as u32],
            1,
            ctx,
            stream,
            |li, ctx| {
                self.layers[li].decode(
                    hidden,
                    residual,
                    seq.layer_states[li].as_mut(),
                    kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    ctx,
                    stream,
                )?;
                self.try_dflash_capture(li, 0, stream)
            },
        )
    }

    /// Whether a one-sequence step of `rows` rows from `seq.seq_len` runs
    /// wide: pieces admitted, an SSM pool slot (else no layer is captured
    /// and nothing stages), every row inert.
    pub(in crate::model) fn decode_piece_wide(
        &self,
        pieces: bool,
        seq: &SequenceState,
        rows: usize,
    ) -> bool {
        pieces && seq.ssm_slot_idx().is_some() && self.decode_pieces_wide(seq.seq_len + rows - 1)
    }

    /// After a wide one-sequence step: commit rows `0..rows` (positions
    /// `seq.seq_len + r`, so call it before `seq_len` advances).
    pub(in crate::model) fn qsa_commit_staged_seq(
        &self,
        seq: &mut SequenceState,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let rows: Vec<(usize, usize)> = (0..rows).map(|r| (0, seq.seq_len + r)).collect();
        self.qsa_commit_staged(&rows, &mut [&mut seq.layer_states], stream)
    }

    /// [`Self::piece_run`] for a K-row MTP verify of one sequence (`verify_b`
    /// K=2, `verify_c` K=3, `verify_c2` K=4): a GDN layer runs
    /// `decode_batched`, an attention layer (wide runs only) the K rows of
    /// the one sequence through `decode_multi_seq_rows`, and each layer the
    /// DFlash capture of the last row, as the eager loops do. A sequence
    /// without an SSM pool slot runs eagerly.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::model) fn verify_piece_run(
        &self,
        layer_idx: usize,
        k: usize,
        wide: bool,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(slot) = seq.ssm_slot_idx() else {
            return Ok(false);
        };
        let (hidden, residual) = (self.buffers.hidden_states(), self.buffers.residual());
        let row_owner = vec![0usize; k];
        self.piece_run(
            layer_idx,
            PieceStep::Verify,
            wide,
            &[slot as u32],
            k,
            ctx,
            stream,
            |li, ctx| {
                if self.config.layer_type(li) == LayerType::FullAttention {
                    let mut state: [&mut (dyn LayerState + 'static); 1] =
                        [seq.layer_states[li].as_mut()];
                    self.layers[li].decode_multi_seq_rows(
                        hidden,
                        residual,
                        k,
                        &mut state,
                        &row_owner,
                        kv_cache,
                        seq_lens,
                        block_tables,
                        ctx,
                        stream,
                    )?;
                } else {
                    self.layers[li].decode_batched(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[li].as_mut(),
                        kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        ctx,
                        stream,
                    )?;
                }
                self.try_dflash_capture(li, k - 1, stream)
            },
        )
    }

    /// [`Self::piece_run`] for the batched multi-sequence verify (`verify_e`):
    /// a GDN layer runs `decode_verify_multi` over the ragged rows (`ks`)
    /// with its slice of the staged WY tables, an attention layer (wide runs
    /// only) the `r_total` rows through `decode_multi_seq_rows`, as the eager
    /// loop does. `key` is `verify_batched_graph_key`'s slot/row vector.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::model) fn verify_batch_piece_run(
        &self,
        layer_idx: usize,
        wide: bool,
        key: &[u32],
        ks: &[usize],
        row_owner: &[usize],
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        wy_tables: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (hidden, residual) = (self.buffers.hidden_states(), self.buffers.residual());
        let r_total = row_owner.len();
        self.piece_run(
            layer_idx,
            PieceStep::VerifyBatch,
            wide,
            key,
            r_total,
            ctx,
            stream,
            |li, ctx| {
                let mut states: Vec<&mut (dyn LayerState + 'static)> = seqs
                    .iter_mut()
                    .map(|s| s.layer_states[li].as_mut())
                    .collect();
                if self.config.layer_type(li) == LayerType::FullAttention {
                    return self.layers[li].decode_multi_seq_rows(
                        hidden,
                        residual,
                        r_total,
                        &mut states,
                        row_owner,
                        kv_cache,
                        seq_lens,
                        block_tables,
                        ctx,
                        stream,
                    );
                }
                self.layers[li].decode_verify_multi(
                    hidden,
                    residual,
                    ks.len(),
                    ks,
                    &mut states,
                    kv_cache,
                    self.verify_wy_slice(wy_tables, li),
                    ctx,
                    stream,
                )
            },
        )
    }

    /// GDN layer `li`'s slice of the batched verify's WY tables (`verify_e`'s
    /// running `ssm_idx`), NULL without tables or off a GDN layer.
    pub(in crate::model) fn verify_wy_slice(&self, wy_tables: DevicePtr, li: usize) -> DevicePtr {
        if wy_tables.is_null() || self.config.layer_type(li) != LayerType::LinearAttention {
            return DevicePtr::NULL;
        }
        let ssm_idx = (0..li)
            .filter(|&j| self.config.layer_type(j) == LayerType::LinearAttention)
            .count();
        wy_tables.offset(ssm_idx * crate::layer::VERIFY_WY_LAYER_STRIDE_BYTES)
    }
}
