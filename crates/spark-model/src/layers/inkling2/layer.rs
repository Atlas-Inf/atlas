// SPDX-License-Identifier: AGPL-3.0-only

//! One visit of an Inkling 2 physical layer: its weights, its per-sequence
//! conv state, and the engine's entry points. Every entry point runs
//! [`Inkling2Layer::forward`] over a [`Batch`]: a decode step, a prefill chunk
//! or a verify run of one sequence, or one decode token each of several. The
//! model runs layer-major, so every per-token state the visits pass along
//! (block stack, prelude save, messages) is kept per row in `Shared`.

use std::any::Any;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::batch::Batch;
use super::glue::GlueKernels;
use super::ops::Inkling2Kernels;
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
    fn conv_state_bytes(&self) -> usize {
        (2 * self.dims.kv_dim + 2 * self.dims.hidden) * (self.p.conv_k - 1) * 4
    }
}

fn conv_base(state: &mut dyn LayerState) -> Result<DevicePtr> {
    Ok(state
        .as_any_mut()
        .downcast_mut::<Inkling2SeqState>()
        .context("inkling2 layer state")?
        .conv)
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
        let batch = Batch::one(seq_len, num_tokens, block_table);
        self.forward(hidden, &batch, &[conv_base(state)?], kv_cache, ctx, stream)
    }

    /// One token each of `active_seqs` sequences (rows past them are graph
    /// padding, which this eager layer never sees used).
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        _num_seqs: usize,
        active_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let batch = Batch {
            seqs: (0..active_seqs)
                .map(|i| (seq_lens[i], block_tables[i].as_slice()))
                .collect(),
            rows_per_seq: 1,
        };
        let bases = states[..active_seqs]
            .iter_mut()
            .map(|s| conv_base(&mut **s))
            .collect::<Result<Vec<_>>>()?;
        self.forward(hidden, &batch, &bases, kv_cache, ctx, stream)
    }

    /// The default would run each row with a throwaway state, losing the conv
    /// history; verify rows wait for the DSpark drafter.
    fn decode_multi_seq_rows<'a, 'b: 'a>(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_rows: usize,
        _seq_states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _row_owner: &[usize],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        bail!("inkling2: multi-sequence verify rows are not supported yet")
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

    /// Single-sequence decode reads the model's staged metadata and the
    /// prestaged state table, so it can be graphed (not with a debug capture).
    fn decode_graph_unsupported(&self) -> bool {
        super::debug::enabled()
    }

    /// Batched decode and verify graphs would bake one batch's state tables,
    /// and their keys do not change with the batch's sequences here.
    fn batched_decode_graph_unsupported(&self) -> bool {
        true
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let bytes = self.conv_state_bytes();
        let conv = gpu.alloc(bytes).context("allocate inkling2 conv state")?;
        gpu.copy_h2d(&vec![0u8; bytes], conv)?;
        Ok(Box::new(Inkling2SeqState { conv }))
    }

    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn LayerState) -> Result<()> {
        if let Some(st) = state.as_any_mut().downcast_mut::<Inkling2SeqState>()
            && !st.conv.is_null()
        {
            gpu.free(st.conv)?;
            st.conv = DevicePtr::NULL;
        }
        Ok(())
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}
