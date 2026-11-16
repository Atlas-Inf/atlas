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
use super::cache_recipe::rope_freqs;
use super::engram::{Engram, EngramWeights, held_columns};
use super::engram_table::{EngramHash, EngramTable, Prefetcher};
use super::gemm::GemmWeights;
use super::hc::{self, IDENTITY_PRE};
use super::index::INDEX_DIM;
use super::ops::HEAD_DIM;
use super::parts::Held;
use super::profile;
use super::shape::batch_rows;
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
            // Every copy of the stream starts as the embedding, and the
            // first block collapses with the identity mix.
            gpu.synchronize(stream)?;
            let identity: Vec<u8> = IDENTITY_PRE.iter().flat_map(|v| v.to_le_bytes()).collect();
            for r in 0..rows {
                let token = hidden.offset(r * hidden_bytes);
                for copy in 0..copies {
                    let to = streams.offset((r * copies + copy) * hidden_bytes);
                    gpu.copy_d2d(token, to, hidden_bytes)?;
                }
                gpu.copy_h2d(&identity, pre.offset(r * copies * 4))?;
            }
        }
        if let Some(ahead) = &self.ahead {
            self.look_ahead(ahead, row, rows, state, position, ctx, stream)?;
        }
        if let Some(lookup) = &self.lookup {
            self.look_up(lookup, row, rows, streams, state, position, ctx, stream)?;
        }
        let comm = ctx.comm.filter(|_| ctx.config.ep_world_size > 1);
        self.block
            .forward_rows(gpu, row, rows, streams, pre, state, position, comm, stream)?;
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
        for first in (0..num_tokens).step_by(batch) {
            let rows = (num_tokens - first).min(batch);
            let at = hidden.offset(first * width);
            self.run(at, rows, state, seq_len_start + first, ctx, stream)?;
        }
        Ok(())
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(SeqState::alloc(
            gpu,
            self.compresses,
            FIRST_POSITIONS,
        )?))
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
    let shared = Shared {
        shape,
        held,
        kernels: BlockKernels::load(gpu)?,
        // At ratio 1 a query scores one compressed position a token.
        scratch: Arc::new(Scratch::alloc_rows(
            gpu,
            &shape,
            config.max_position_embeddings,
            batch_rows(),
            held,
        )?),
        handoff: Arc::new(std::sync::Mutex::new(Handoff::default())),
    };
    let ladder = |original: usize, theta: f64| -> Result<DevicePtr> {
        let freqs = rope_freqs(
            shape.rope,
            original,
            theta as f32,
            config.yarn_factor,
            config.yarn_beta_fast,
            config.yarn_beta_slow,
        );
        let bytes: Vec<u8> = freqs.iter().flat_map(|f| f.to_le_bytes()).collect();
        let p = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, p)?;
        Ok(p)
    };
    ensure!(
        config.window_rope_theta > 0.0 && config.rope_theta > 0.0,
        "deepseek_v41: no rotary bases"
    );
    // A layer that attends over its window only is not YaRN-scaled.
    let window = ladder(0, config.window_rope_theta)?;
    let compressed = ladder(
        config.yarn_original_max_position_embeddings,
        config.rope_theta,
    )?;
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
        let weights = BlockWeights::load(
            store,
            gpu,
            &shared.kernels.parts,
            &shape,
            layer,
            compresses,
            index_source,
            held,
            true,
            stream,
        )
        .with_context(|| format!("layer {layer}"))?;
        // ATLAS_DSV41_NO_GEMM keeps every batch on the row-at-a-time
        // kernels, to compare the two.
        let gemm = match std::env::var_os("ATLAS_DSV41_NO_GEMM") {
            Some(_) => None,
            None => Some(
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
        }));
    }
    Ok(out)
}

#[path = "layer_lookup.rs"]
mod lookup;
