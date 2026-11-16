// SPDX-License-Identifier: AGPL-3.0-only

//! The Engram lookups in front of a DeepSeek-V4.1 layer: the tokens of a
//! batch into the sequence's history, their table rows fetched ahead, and
//! the lookups added into their streams.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::block::SeqState;
use super::super::engram::ENGRAM_ROWS;
use super::super::engram_table::EngramTable;
use super::super::profile;
use super::{Ahead, Layer, Lookup};
use crate::layer::ForwardContext;

impl Layer {
    /// The tokens of `rows` rows from `row`, as the engine staged them.
    fn tokens(
        &self,
        row: usize,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        // What the host was handed is read from the host: reading the ids
        // back from the device waits for the pass's stream, and a decode
        // step's host was held there until the layers before had run, with
        // the GPU idle while it then looked the rows up.
        if let Some(ids) = ctx.host_token_ids.and_then(|ids| ids.get(row..row + rows)) {
            return Ok(ids.to_vec());
        }
        let ids = ctx
            .token_ids
            .with_context(|| format!("deepseek_v41: no token ids for {rows} rows from {row}"))?;
        ctx.gpu.synchronize(stream)?;
        let mut words = vec![0u8; rows * 4];
        ctx.gpu.copy_d2h(ids.offset(row * 4), &mut words)?;
        Ok(words
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect())
    }

    /// Add the tokens of `rows` rows from `row`, at `position` on, to the
    /// sequence's history, as the ids the Engram hashes.
    #[allow(clippy::too_many_arguments)]
    fn remember(
        &self,
        token_map: &[u32],
        row: usize,
        rows: usize,
        state: &mut SeqState,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let tokens = self.tokens(row, rows, ctx, stream)?;
        // A sequence that was rolled back hashes its new tokens over the
        // ones kept.
        state.history.truncate(position);
        ensure!(
            state.history.len() == position,
            "deepseek_v41: Engram at position {position} of a sequence it has seen {} tokens of",
            state.history.len()
        );
        for token in tokens {
            let compressed = *token_map
                .get(token as usize)
                .with_context(|| format!("token {token} is outside the Engram token map"))?;
            state.history.push(compressed);
        }
        Ok(())
    }

    /// The rows `table` holds for the token at `position` of `state`'s
    /// history.
    fn ids(table: &EngramTable, state: &SeqState, position: usize) -> Vec<u64> {
        let from = (position + 1).saturating_sub(table.hash.reach());
        table.hash.rows(&state.history[from..=position])
    }

    /// Start fetching what the Engram layers will read for these tokens.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn look_ahead(
        &self,
        ahead: &Ahead,
        row: usize,
        rows: usize,
        state: &mut SeqState,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.remember(&ahead.token_map, row, rows, state, position, ctx, stream)?;
        for table in &ahead.tables {
            let ids = (position..position + rows)
                .flat_map(|at| Self::ids(table, state, at)[ahead.columns.clone()].to_vec())
                .collect();
            ahead.prefetcher.ask(table, ids);
        }
        Ok(())
    }

    /// Add the Engram lookups of the `rows` tokens from `position` into
    /// their streams, [`ENGRAM_ROWS`] at a time.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn look_up(
        &self,
        lookup: &Lookup,
        row: usize,
        rows: usize,
        streams: DevicePtr,
        state: &mut SeqState,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // The first layer keeps its own history when it is also an Engram.
        if self.ahead.is_none() {
            self.remember(&lookup.token_map, row, rows, state, position, ctx, stream)?;
        }
        let shape = &self.block.shared.shape;
        let (held, per) = (lookup.engram.columns(), lookup.engram.per_part());
        let comm = ctx.comm.filter(|_| ctx.config.ep_world_size > 1);
        for first in (0..rows).step_by(ENGRAM_ROWS) {
            let tokens = (rows - first).min(ENGRAM_ROWS);
            let ids: Vec<Vec<u64>> = (first..first + tokens)
                .map(|r| Self::ids(&lookup.table, state, position + r))
                .collect();
            let found = profile::stage(ctx.gpu, stream, "engram rows", || {
                lookup.table.find(ctx.gpu, &ids, held.clone(), per, stream)
            })?;
            profile::stage(ctx.gpu, stream, "engram", || {
                lookup.engram.apply(
                    ctx.gpu,
                    &self.block.shared.scratch.gemm,
                    found.rows,
                    found.scales,
                    found.row_slots,
                    found.scale_slots,
                    streams.offset(first * shape.hc.copies * shape.hidden * 2),
                    tokens,
                    comm,
                    stream,
                )?;
                // The rows stay in memory until the kernels have read them.
                ctx.gpu.synchronize(stream)
            })?;
            lookup.table.done();
        }
        Ok(())
    }
}
