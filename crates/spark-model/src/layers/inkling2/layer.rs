// SPDX-License-Identifier: AGPL-3.0-only

//! One visit of an Inkling 2 physical layer: its weights, its per-sequence
//! conv state, and the engine's entry points. Every entry point runs
//! [`Inkling2Layer::forward`] over a [`Batch`]: a decode step, a prefill chunk
//! or a verify run of one sequence, or one decode token each of several. The
//! model runs layer-major, so every per-token state the visits pass along
//! (block stack, prelude save, messages) is kept per row in `Shared`.

use std::any::Any;
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::batch::Batch;
use super::glue::GlueKernels;
use super::ops::{self, Inkling2Kernels};
use super::schedule::Visit;
use super::shared::{Dims, Shared};
use super::weights::{GlobalWeights, PhysicalWeights};
use crate::layer::{ForwardContext, LayerState, TransformerLayer};

/// Numerics shared by every visit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_n_floor: f32,
    pub log_alpha: f32,
    pub conv_k: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub sliding_window: usize,
    pub gate_rank: usize,
    pub gate_softcap: f32,
    pub up_softcap: f32,
    pub route_scale: f32,
}

/// Per-sequence conv state of one visit: k, v, post-attention and post-MLP
/// convs, `[channels, k - 1]` fp32 each.
pub(crate) struct Inkling2SeqState {
    pub conv: DevicePtr,
    /// Verify journal: the conv state before the last verify, then that
    /// verify's conv inputs (k, v, post-attention, post-MLP), so a rejected
    /// tail can be rolled back.
    pub journal: DevicePtr,
    /// Rows the journal holds for the last verify (0 = none).
    pub journal_rows: usize,
    /// A graph-padding row's state: the shared pad buffer, owned by no one.
    pub pad: bool,
}

impl LayerState for Inkling2SeqState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub(crate) struct Inkling2Layer {
    pub visit_idx: usize,
    pub visit: Visit,
    pub w: PhysicalWeights,
    pub g: GlobalWeights,
    /// `[core, 4, H]` fp32: per loop layer the q, k, v, mlp message queries
    pub msg_queries: DevicePtr,
    pub k: Inkling2Kernels,
    pub gk: GlueKernels,
    pub dims: Dims,
    pub p: Params,
    pub shared: Shared,
    /// The conv-state bases this visit's state table holds on the device.
    pub bound_states: Mutex<Vec<u64>>,
}

impl Inkling2Layer {
    pub(super) fn conv_state_bytes(&self) -> usize {
        self.dims.conv_state_bytes()
    }

    /// Each conv's channels, its offset in the state (fp32 elements) and its
    /// input rows' offset in the journal (bytes past the saved state).
    pub(super) fn conv_layout(&self) -> [(usize, usize, usize); 4] {
        let (kv, h, km1) = (self.dims.kv_dim, self.dims.hidden, self.p.conv_k - 1);
        let rows = MAX_VERIFY_ROWS * 2;
        [
            (kv, 0, 0),
            (kv, kv * km1, kv * rows),
            (h, 2 * kv * km1, 2 * kv * rows),
            (h, (2 * kv + h) * km1, (2 * kv + h) * rows),
        ]
    }

    fn journal_bytes(&self) -> usize {
        self.conv_state_bytes()
            + MAX_VERIFY_ROWS * (2 * self.dims.kv_dim + 2 * self.dims.hidden) * 2
    }
}

/// Verify rows a journal holds (a DSpark block is 8).
pub(super) const MAX_VERIFY_ROWS: usize = 16;

fn seq_state(state: &mut dyn LayerState) -> Result<&mut Inkling2SeqState> {
    state
        .as_any_mut()
        .downcast_mut::<Inkling2SeqState>()
        .context("inkling2 layer state")
}

/// `(first row, rows)` of each of `seqs` sequences in `row_owner`, which must
/// list every sequence's rows together and in order.
fn owner_runs(row_owner: &[usize], seqs: usize) -> Result<Vec<(usize, usize)>> {
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(seqs);
    for (r, &o) in row_owner.iter().enumerate() {
        let seen = runs.len();
        if seen > 0 && o + 1 == seen {
            runs[seen - 1].1 += 1;
        } else if o == seen {
            runs.push((r, 1));
        } else {
            bail!("inkling2: verify row {r} of sequence {o} out of order");
        }
    }
    ensure!(
        runs.len() == seqs && seqs > 0,
        "inkling2: verify rows of {} sequences for {seqs} states",
        runs.len()
    );
    Ok(runs)
}

fn conv_base(state: &mut dyn LayerState) -> Result<DevicePtr> {
    Ok(seq_state(state)?.conv)
}

impl Inkling2Layer {
    /// `rows` consecutive tokens of one sequence from `seq_len`, journaled
    /// (when it fits) so a verify's rejected rows can be rolled back.
    #[allow(clippy::too_many_arguments)]
    fn verify_run(
        &self,
        hidden: DevicePtr,
        rows: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let st = seq_state(state)?;
        let mut batch = Batch::one(seq_len, rows, block_table);
        st.journal_rows = 0;
        if rows <= MAX_VERIFY_ROWS {
            batch.journals.push(st.journal);
            st.journal_rows = rows;
        }
        self.forward(hidden, &batch, &[st.conv], kv_cache, ctx, stream)
    }
}

impl TransformerLayer for Inkling2Layer {
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let batch = Batch::one(seq_len, 1, block_table);
        self.forward(hidden, &batch, &[conv_base(state)?], kv_cache, ctx, stream)
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let batch = Batch::one(seq_len_start, num_tokens, block_table);
        self.forward(hidden, &batch, &[conv_base(state)?], kv_cache, ctx, stream)
    }

    /// Consecutive tokens of one sequence: the prefill shape.
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.verify_run(
            hidden,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
        )
    }

    /// Keep the first `num_accepted` of the last verify's `k` rows in the conv
    /// states: restore the journaled state and replay the accepted rows.
    fn rollback_aux_verify(
        &self,
        state: &mut dyn LayerState,
        num_accepted: usize,
        k: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let st = seq_state(state)?;
        if num_accepted >= k {
            st.journal_rows = 0;
            return Ok(());
        }
        ensure!(
            st.journal_rows == k,
            "inkling2 visit {}: no journal for a {k}-row verify (have {})",
            self.visit_idx,
            st.journal_rows
        );
        gpu.copy_d2d_async(st.journal, st.conv, self.conv_state_bytes(), stream)?;
        if num_accepted > 0 {
            let inputs = st.journal.offset(self.conv_state_bytes());
            for (channels, state_off, journal_off) in self.conv_layout() {
                ops::sconv_rewind(
                    gpu,
                    &self.k,
                    inputs.offset(journal_off),
                    st.conv.offset(state_off * 4),
                    channels,
                    self.p.conv_k,
                    num_accepted,
                    stream,
                )?;
            }
        }
        st.journal_rows = 0;
        Ok(())
    }

    /// One token each of `num_seqs` rows: `active_seqs` sequences, then graph
    /// padding (the model's seq_len 0, dummy block and pad state), so a graph
    /// of this width replays for any batch that pads to it.
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_seqs: usize,
        _active_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let batch = Batch {
            seqs: (0..num_seqs)
                .map(|i| (seq_lens[i], block_tables[i].as_slice()))
                .collect(),
            rows_per_seq: 1,
            journals: Vec::new(),
        };
        let bases = states[..num_seqs]
            .iter_mut()
            .map(|s| conv_base(&mut **s))
            .collect::<Result<Vec<_>>>()?;
        self.forward(hidden, &batch, &bases, kv_cache, ctx, stream)
    }

    /// Verify rows, journaled like [`Self::decode_batched`] (the default
    /// would give each row a throwaway state, losing the conv history). Each
    /// sequence's rows are consecutive tokens, in sequence order; equal row
    /// counts run as one batch, uneven ones one sequence at a time.
    fn decode_multi_seq_rows<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_rows: usize,
        seq_states: &'a mut [&'b mut (dyn LayerState + 'static)],
        row_owner: &[usize],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let runs = owner_runs(
            &row_owner[..num_rows.min(row_owner.len())],
            seq_states.len(),
        )?;
        ensure!(
            runs.iter().map(|r| r.1).sum::<usize>() == num_rows,
            "inkling2: {num_rows} verify rows but {} owned rows",
            row_owner.len()
        );
        let k = runs[0].1;
        if runs.iter().all(|r| r.1 == k) && k <= MAX_VERIFY_ROWS {
            let mut batch = Batch {
                seqs: runs
                    .iter()
                    .map(|&(first, _)| (seq_lens[first], block_tables[first].as_slice()))
                    .collect(),
                rows_per_seq: k,
                journals: Vec::with_capacity(runs.len()),
            };
            let mut bases = Vec::with_capacity(runs.len());
            for state in seq_states.iter_mut() {
                let st = seq_state(&mut **state)?;
                batch.journals.push(st.journal);
                st.journal_rows = k;
                bases.push(st.conv);
            }
            return self.forward(hidden, &batch, &bases, kv_cache, ctx, stream);
        }
        for (s, &(first, rows)) in runs.iter().enumerate() {
            self.verify_run(
                hidden.offset(first * self.dims.hidden * 2),
                rows,
                &mut *seq_states[s],
                kv_cache,
                seq_lens[first],
                &block_tables[first],
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Bind the decode sequence's conv state into this visit's table before
    /// the step's graph capture or replay.
    fn decode_prestage(
        &self,
        _token: u32,
        state: &mut dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.bind_states(gpu, &[conv_base(state)?], false, stream)
            .map(|_| ())
    }

    /// Verify graphs would bake one batch's state tables, and their keys do
    /// not change with the batch's sequences here.
    fn batched_decode_graph_unsupported(&self) -> bool {
        true
    }

    /// Multi-sequence decode refreshes the tables first (`multi_seq_prestage`),
    /// so it graphs although verify does not.
    fn multi_seq_decode_graph_unsupported(&self) -> bool {
        false
    }

    /// Bind the step's conv states, then the pad state up to `padded_n`.
    fn multi_seq_prestage(
        &self,
        states: &mut [&mut dyn LayerState],
        padded_n: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let mut bases = states
            .iter_mut()
            .map(|s| conv_base(&mut **s))
            .collect::<Result<Vec<_>>>()?;
        bases.resize(padded_n.max(bases.len()), self.shared.pad_conv);
        self.bind_states(gpu, &bases, false, stream).map(|_| ())
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let bytes = self.conv_state_bytes();
        let conv = gpu.alloc(bytes).context("allocate inkling2 conv state")?;
        gpu.copy_h2d(&vec![0u8; bytes], conv)?;
        let journal = gpu
            .alloc(self.journal_bytes())
            .context("allocate inkling2 verify journal")?;
        Ok(Box::new(Inkling2SeqState {
            conv,
            journal,
            journal_rows: 0,
            pad: false,
        }))
    }

    /// Prefix-cache snapshots carry each visit's conv state (its only
    /// per-sequence state besides KV) as an aux blob: copied out at a
    /// snapshot, back in on a hit before the remaining prefill runs.
    fn has_aux_state(&self) -> bool {
        true
    }

    fn snapshot_aux_into(
        &self,
        state: &dyn LayerState,
        buf: &mut Vec<u8>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        let st = state
            .as_any()
            .downcast_ref::<Inkling2SeqState>()
            .context("inkling2: foreign layer state")?;
        if st.pad {
            return Ok(false);
        }
        buf.resize(self.conv_state_bytes(), 0);
        crate::layers::aux_d2h::copy(gpu, st.conv, buf, stream)?;
        Ok(true)
    }

    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let bytes = self.conv_state_bytes();
        let st = state
            .as_any_mut()
            .downcast_mut::<Inkling2SeqState>()
            .context("inkling2: foreign layer state")?;
        ensure!(
            !st.pad && blob.len() == bytes,
            "inkling2: conv-state blob of {} bytes for a {bytes}-byte state",
            blob.len()
        );
        gpu.copy_h2d_async(blob, st.conv, stream)?;
        // a restored state has no verify behind it to roll back
        st.journal_rows = 0;
        Ok(())
    }

    /// Every padding row writes the one shared pad state.
    fn padding_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        ensure!(
            self.shared.is_attached(),
            "inkling2: padding state before attach"
        );
        Ok(Box::new(Inkling2SeqState {
            conv: self.shared.pad_conv,
            journal: DevicePtr::NULL,
            journal_rows: 0,
            pad: true,
        }))
    }

    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn LayerState) -> Result<()> {
        if let Some(st) = state.as_any_mut().downcast_mut::<Inkling2SeqState>()
            && !st.pad
            && !st.conv.is_null()
        {
            gpu.free(st.conv)?;
            st.conv = DevicePtr::NULL;
            if !st.journal.is_null() {
                gpu.free(st.journal)?;
                st.journal = DevicePtr::NULL;
            }
        }
        Ok(())
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

#[cfg(test)]
mod owner_run_tests {
    use super::owner_runs;

    #[test]
    fn rows_group_by_sequence_in_order() {
        assert_eq!(owner_runs(&[0, 0, 0], 1).unwrap(), vec![(0, 3)]);
        assert_eq!(
            owner_runs(&[0, 0, 1, 1, 1], 2).unwrap(),
            vec![(0, 2), (2, 3)]
        );
    }

    #[test]
    fn out_of_order_or_missing_rows_fail() {
        assert!(owner_runs(&[0, 1, 0], 2).is_err());
        assert!(owner_runs(&[1, 1], 2).is_err());
        assert!(owner_runs(&[0, 0], 2).is_err());
        assert!(owner_runs(&[], 1).is_err());
    }
}
