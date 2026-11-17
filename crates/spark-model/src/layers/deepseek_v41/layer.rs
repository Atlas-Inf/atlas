// SPDX-License-Identifier: AGPL-3.0-only

//! A DeepSeek-V4.1 block as the engine's layer, in the plain form: one
//! token at a time, every step its own kernel.
//!
//! The engine hands a layer one row of `[hidden]` BF16 per token. The stream
//! between blocks is four copies of that, so it lives in the arena's
//! Hyper-Connection buffers, one row per token of the pass: the first layer
//! fills a row from the token's embedding, the last collapses it back for
//! the final norm and head. The mix the next block collapses with rides
//! beside it.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::weights::WeightStore;

use super::block::{Block, BlockKernels, Candidates, Handoff, Scratch, SeqState, Shape, Shared};
use super::draft_seq;
use super::engram::{Engram, EngramWeights, held_columns};
use super::engram_table::{EngramHash, EngramTable, Prefetcher};
use super::gemm::{GemmWeights, gemms};
use super::hc;
use super::index::INDEX_DIM;
use super::ops::HEAD_DIM;
use super::parts::Held;
use super::profile;
use super::shape::{batch_rows, ladder};
use super::state::KEPT_ROWS;
use super::tail;
use super::weights::BlockWeights;
use crate::layer::{ForwardContext, LayerState, TransformerLayer};

/// Positions a new sequence's compressed keys have room for; they grow.
const FIRST_POSITIONS: usize = 8192;

/// An Engram in front of a block.
struct Lookup {
    engram: Engram,
    table: Arc<EngramTable>,
    /// The compressed id of every token id.
    token_map: Arc<Vec<u32>>,
}

/// What the first layer does for the Engram layers: as soon as a token is
/// known, the rows they will read for it are fetched in the background. A
/// row comes off the disk in about the time a layer takes.
struct Ahead {
    tables: Vec<Arc<EngramTable>>,
    prefetcher: Prefetcher,
    token_map: Arc<Vec<u32>>,
    /// The columns of a lookup this rank reads.
    columns: std::ops::Range<usize>,
}

pub struct Layer {
    block: Block,
    lookup: Option<Lookup>,
    ahead: Option<Ahead>,
    /// The ratio the layer compresses keys at itself, 0 if it does not.
    compresses: usize,
    first: bool,
    last: bool,
    /// The rows of a chunk's end a prefill may run alone (`tail.rs`).
    tail: Option<usize>,
    /// Whether a rank runs a large batch's mixing for its own rows alone
    /// ([`Owned`]): a layer of every row, with a layer after it that
    /// [`Self::replicates`]. And whether this is that layer, the first that
    /// runs a prompt's last rows alone.
    owns: bool,
    replicates: bool,
    target: Option<Target>,
}

impl Layer {
    /// Which row of the pass `hidden` is: the engine's rows are rows of the
    /// arena's hidden buffer.
    fn row(&self, hidden: DevicePtr, ctx: &ForwardContext) -> Result<usize> {
        let width = (self.block.shared.shape.hidden * 2) as u64;
        let base = ctx.buffers.hidden_states().0;
        let offset = hidden
            .0
            .checked_sub(base)
            .context("a row before the arena's")?;
        let row = (offset / width) as usize;
        ensure!(
            offset % width == 0 && row < ctx.buffers.max_batch_tokens(),
            "deepseek_v41: a row that is not one of the arena's {}",
            ctx.buffers.max_batch_tokens()
        );
        Ok(row)
    }
}

impl Layer {
    /// The layer on `rows` consecutive tokens of one sequence, from
    /// `position`: rows of the arena from `hidden`'s.
    fn run(
        &self,
        hidden: DevicePtr,
        rows: usize,
        state: &mut dyn LayerState,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (gpu, shared) = (ctx.gpu, &self.block.shared);
        let (hidden_bytes, copies) = (shared.shape.hidden * 2, shared.shape.hc.copies);
        let state = state
            .as_any_mut()
            .downcast_mut::<SeqState>()
            .context("deepseek_v41: a layer state that is not this layer's")?;
        let row = self.row(hidden, ctx)?;
        let streams = ctx.buffers.hc_streams().offset(row * copies * hidden_bytes);
        let pre = ctx.buffers.hc_post().offset(row * copies * 4);
        if self.first {
            profile::pass_begins();
        }
        if self.first {
            // Every copy of the stream starts as the embedding, and the
            // first block collapses with the identity mix.
            let (kernels, shape) = (&shared.kernels.hc, &shared.shape.hc);
            hc::start(gpu, kernels, shape, hidden, streams, pre, rows, stream)?;
        }
        if let Some(ahead) = &self.ahead {
            self.look_ahead(ahead, row, rows, state, position, ctx, stream)?;
        }
        if let Some(lookup) = &self.lookup {
            self.look_up(lookup, row, rows, streams, state, position, ctx, stream)?;
        }
        if let Some(target) = &self.target {
            target.take(gpu, &shared.shape, streams, rows, state, position, stream)?;
        }
        let comm = ctx.comm.filter(|_| ctx.config.ep_world_size > 1);
        let own = self.owned(rows, ctx).filter(|_| self.owns);
        self.block.forward_rows(
            gpu, row, rows, streams, pre, state, position, comm, own, stream,
        )?;
        if self.last {
            // Before the head, the stream collapses with the last mix.
            hc::collapse(
                gpu,
                shared.kernels.hc.collapse,
                &shared.shape.hc,
                streams,
                pre,
                hidden,
                rows,
                stream,
            )?;
            (0..rows).for_each(|_| profile::token_done());
            profile::pass_ends(gpu, stream)?;
        }
        Ok(())
    }
}

impl TransformerLayer for Layer {
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.run(hidden, 1, state, seq_len, ctx, stream)
    }

    /// Several consecutive tokens of a sequence in one pass, for a verify of
    /// drafted tokens: each row's bytes are a decode step's, for a batch
    /// this small runs the kernels a row alone runs.
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.run(hidden, num_tokens, state, seq_len, ctx, stream)
    }

    fn decode_batched_is_native(&self) -> bool {
        true
    }

    /// A verify runs its drafts and the token before them.
    fn verify_max_drafts(&self) -> Option<usize> {
        Some(KEPT_ROWS - 1)
    }

    fn rollback_aux_verify(
        &self,
        state: &mut dyn LayerState,
        num_accepted: usize,
        k: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<SeqState>()
            .context("deepseek_v41: a layer state that is not this layer's")?;
        let restore = self.block.shared.kernels.cache.window_restore;
        state.roll_back(gpu, restore, self.compresses, num_accepted, k, stream)
    }

    /// A chunk of a prompt, a batch of rows at a time: the bytes of a
    /// decode step for each row.
    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let width = self.block.shared.shape.hidden * 2;
        let batch = self.block.shared.scratch.rows;
        if self.first {
            self.agree_on_rows(ctx, stream)?;
        }
        let whole = self.block.shared.handoff.lock().expect("handoff").whole;
        let from = tail::first_row(self.tail, num_tokens, whole);
        if self.replicates && from < num_tokens {
            self.replicate(hidden, num_tokens, from, ctx, stream)?;
        }
        for first in (from..num_tokens).step_by(batch) {
            let rows = (num_tokens - first).min(batch);
            let at = hidden.offset(first * width);
            self.run(at, rows, state, seq_len_start + first, ctx, stream)?;
        }
        Ok(())
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let mut state = SeqState::alloc(gpu, self.compresses, FIRST_POSITIONS)?;
        if let Some(target) = &self.target {
            state.draft = target.sequence(gpu, self.block.shared.shape.hidden)?;
        }
        Ok(Box::new(state))
    }

    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn LayerState) -> Result<()> {
        match state.as_any_mut().downcast_mut::<SeqState>() {
            Some(state) => state.free(gpu),
            None => Ok(()),
        }
    }

    /// Every token reads back to the host: positions chosen, expert routing,
    /// table rows.
    fn decode_graph_unsupported(&self) -> bool {
        true
    }

    /// A key source keeps a compressed key and an index key per group, and
    /// while they grow a buffer of up to their size again.
    fn aux_bytes_per_token(&self) -> usize {
        match self.compresses {
            0 => 0,
            ratio => 2 * (HEAD_DIM + INDEX_DIM) * 2 / ratio,
        }
    }
}

/// Every layer of the model, from the release's weights.
pub fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let shape = Shape::from_config(config)?;
    let layers = config.num_hidden_layers;
    // The ranks of an expert-parallel world each hold some of the four parts
    // of every wide projection.
    let held = Held::of_rank(config.ep_rank, config.ep_world_size.max(1))?;
    let stream = gpu.default_stream();
    // A sequence is bounded by what this server was started for, where it
    // says, which is far below what the model allows.
    let positions = match config.max_seq_len {
        0 => config.max_position_embeddings,
        served => served.min(config.max_position_embeddings),
    };
    let shared = Shared {
        shape,
        held,
        kernels: BlockKernels::load(gpu)?,
        // At ratio 1 a query scores one compressed position a token.
        scratch: Arc::new(Scratch::alloc_rows(
            gpu,
            &shape,
            positions,
            batch_rows(),
            held,
        )?),
        handoff: Arc::new(std::sync::Mutex::new(Handoff::default())),
    };
    ensure!(
        config.window_rope_theta > 0.0 && config.rope_theta > 0.0,
        "deepseek_v41: no rotary bases"
    );
    // A layer that attends over its window only is not YaRN-scaled.
    let window = ladder(gpu, config, shape.rope, 0, config.window_rope_theta)?;
    let compressed = ladder(
        gpu,
        config,
        shape.rope,
        config.yarn_original_max_position_embeddings,
        config.rope_theta,
    )?;
    // The layers the draft head reads, when it is served.
    let targets: &[usize] = match draft_seq::requested() {
        true => &config.dspark_target_layer_ids,
        false => &[],
    };
    let mean = match targets {
        [] => None,
        _ => Some(gpu.kernel("dsv41_draft", "dsv41_stream_mean")?),
    };
    let geometry = config.deepseek_v41_engram(config.engram_pad_id)?;
    tracing::info!(
        "DeepSeek-V4.1: {layers} layers, parts {:?} of every wide projection, key sources {:?}, index sources {:?}, Engram at {:?}",
        held.range(),
        config.kv_source_layer_ids,
        config.index_source_layer_ids,
        config.engram_layer_ids
    );
    // Every Engram layer's table, so that one can fetch for those after it.
    let mut tables: Vec<Arc<EngramTable>> = Vec::new();
    if let Some(geometry) = &geometry {
        for (index, &layer) in config.engram_layer_ids.iter().enumerate() {
            tables.push(Arc::new(EngramTable::open(
                store,
                gpu,
                layer,
                EngramHash::new(geometry, index),
                geometry.num_embeddings[index],
            )?));
        }
    }
    let token_map = || {
        config
            .engram_token_map
            .clone()
            .context("deepseek_v41: the Engram token map was not derived from the tokenizer")
    };
    // The layers something reads at every row of a prompt: the ones that
    // make compressed keys, and the Engram layers, which keep a history.
    let per_row: Vec<usize> = config
        .kv_source_layer_ids
        .iter()
        .chain(&config.engram_layer_ids)
        .copied()
        .collect();
    let mut out: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(layers);
    for layer in 0..layers {
        let ratio = config.compress_ratios.get(layer).copied().unwrap_or(0);
        let compresses = match config.kv_source_layer_ids.contains(&layer) {
            true => ratio,
            false => 0,
        };
        let index_source = config.index_source_layer_ids.contains(&layer);
        let candidates = match config.candidate_source_layer_id {
            Some(source) if source == layer => Candidates::Source,
            Some(source) if index_source && source < layer => Candidates::Reader,
            _ => Candidates::None,
        };
        // A layer that runs every row of a prompt shares a batch's rows
        // out between the ranks, and there a rank's GEMMs read wo_b whole.
        let owns = tail::rows_of_layer(layer, layers, &per_row).is_none()
            && tail::rows_of_layer(layers - 1, layers, &per_row).is_some();
        let weights = BlockWeights::load(
            store,
            gpu,
            &shared.kernels.parts,
            &shape,
            layer,
            compresses,
            index_source,
            held,
            owns && gemms(),
            true,
            stream,
        )
        .with_context(|| format!("layer {layer}"))?;
        let gemm = match gemms() {
            false => None,
            true => Some(
                GemmWeights::load(gpu, &shape, &weights, held, stream)
                    .with_context(|| format!("layer {layer}: GEMM operands"))?,
            ),
        };
        let block = Block {
            index: layer,
            ratio,
            candidates,
            freqs: if ratio == 0 { window } else { compressed },
            weights,
            tensor_cores: gemm.is_some(),
            gemm,
            shared: shared.clone(),
        };
        let lookup = match config
            .engram_layer_ids
            .iter()
            .position(|&l| l == layer)
            .filter(|&index| index < tables.len())
        {
            Some(index) => {
                let weights = EngramWeights::load(store, gpu, layer, &shape, held, stream)?;
                let table = tables[index].clone();
                ensure!(
                    table.hash.columns() == weights.columns,
                    "layer {layer}: Engram hashes {} rows a token and projects {}",
                    table.hash.columns(),
                    weights.columns
                );
                Some(Lookup {
                    engram: Engram::new(gpu, shared.kernels, &shape, held, weights)?,
                    table,
                    token_map: token_map()?,
                })
            }
            None => None,
        };
        let ahead = match (layer, tables.first()) {
            (0, Some(table)) => Some(Ahead {
                columns: held_columns(table.hash.columns(), held),
                tables: tables.clone(),
                prefetcher: Prefetcher::start()?,
                token_map: token_map()?,
            }),
            _ => None,
        };
        out.push(Box::new(Layer {
            block,
            lookup,
            ahead,
            compresses,
            first: layer == 0,
            last: layer + 1 == layers,
            tail: tail::rows_of_layer(layer, layers, &per_row),
            owns,
            replicates: tail::rows_of_layer(layer, layers, &per_row).is_some()
                && layer > 0
                && tail::rows_of_layer(layer - 1, layers, &per_row).is_none(),
            target: targets
                .iter()
                .position(|&l| l == layer)
                .zip(mean)
                .map(|(index, mean)| Target {
                    index,
                    of: targets.len(),
                    stages: config.num_mtp_modules,
                    mean,
                }),
        }));
    }
    Ok(out)
}

#[path = "layer_draft.rs"]
mod draft;
use draft::Target;

#[path = "layer_rows.rs"]
mod rows;

#[path = "layer_lookup.rs"]
mod lookup;
