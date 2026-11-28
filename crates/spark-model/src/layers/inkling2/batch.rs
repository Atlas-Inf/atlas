// SPDX-License-Identifier: AGPL-3.0-only

//! The sequences one Inkling 2 forward carries, and the metadata the visits
//! read for them: a run of consecutive tokens of one sequence (a decode step,
//! a prefill chunk), or one token each of several sequences (a decode batch),
//! plus each visit's table of the sequences' conv states.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::layer::Inkling2Layer;
use crate::layer::ForwardContext;

pub(super) struct Batch<'a> {
    /// Per sequence: the position of its first row, and its block table.
    pub seqs: Vec<(usize, &'a [u32])>,
    /// Rows of each sequence (consecutive tokens from its first position).
    pub rows_per_seq: usize,
    /// A verify run's journals, one per sequence (see
    /// `Inkling2SeqState::journal`); empty when the run is not journaled.
    pub journals: Vec<DevicePtr>,
}

impl<'a> Batch<'a> {
    pub(super) fn one(start: usize, rows: usize, table: &'a [u32]) -> Self {
        Self {
            seqs: vec![(start, table)],
            rows_per_seq: rows,
            journals: Vec::new(),
        }
    }

    pub(super) fn rows(&self) -> usize {
        self.seqs.len() * self.rows_per_seq
    }

    /// Block-table entries between two rows' tables: 0 when every row reads
    /// one sequence's table, else the longest table (one table per row).
    pub(super) fn table_stride(&self) -> usize {
        if self.seqs.len() == 1 {
            0
        } else {
            self.seqs.iter().map(|(_, t)| t.len()).max().unwrap_or(0)
        }
    }
}

/// Where a forward's row metadata lives: the model's staged decode metadata
/// (stable buffers, refreshed before a graph replay), or the rows visit 0
/// uploads itself (prefill and verify runs, which the model stages without a
/// device block table).
#[derive(Clone, Copy)]
pub(super) struct Meta {
    /// `[rows]` positions (u32 or i32)
    pub positions: DevicePtr,
    /// `[rows]` i64 KV slots
    pub slots: DevicePtr,
    /// block tables, one per row `table_stride` apart (0 = one shared table)
    pub block_table: DevicePtr,
    pub table_stride: usize,
}

impl Inkling2Layer {
    /// This forward's metadata; the first visit uploads it when the model has
    /// not staged it (every visit resolves the same way from the same batch).
    pub(super) fn meta(
        &self,
        gpu: &dyn GpuBackend,
        batch: &Batch<'_>,
        ctx: &ForwardContext,
        bs: usize,
        first: bool,
        stream: u64,
    ) -> Result<Meta> {
        let seqs = batch.seqs.len();
        let d = &self.dims;
        ensure!(
            batch.rows() <= d.max_rows && seqs <= d.max_seqs,
            "inkling2: {} rows of {seqs} sequences over the buffers ({} rows, {} sequences)",
            batch.rows(),
            d.max_rows,
            d.max_seqs
        );
        if let Some(m) = ctx.attn_metadata.as_ref()
            && batch.rows_per_seq == 1
            && m.num_seqs as usize >= seqs
            && !m.block_table.is_null()
            && !m.slot.is_null()
        {
            return Ok(Meta {
                positions: m.positions,
                slots: m.slot,
                block_table: m.block_table,
                table_stride: if seqs == 1 {
                    0
                } else {
                    m.max_blocks_per_seq as usize
                },
            });
        }
        let s = &self.shared;
        if first {
            self.upload_meta(gpu, batch, bs, stream)?;
        }
        Ok(Meta {
            positions: s.positions,
            slots: s.slots,
            block_table: s.block_table,
            table_stride: batch.table_stride(),
        })
    }

    /// Upload the batch's positions, KV slots and block tables.
    fn upload_meta(
        &self,
        gpu: &dyn GpuBackend,
        batch: &Batch<'_>,
        bs: usize,
        stream: u64,
    ) -> Result<()> {
        let (s, d) = (&self.shared, &self.dims);
        let (n, seqs) = (batch.rows(), batch.seqs.len());
        ensure!(
            n <= d.max_rows,
            "inkling2: {n} rows over the {}-row buffers",
            d.max_rows
        );
        ensure!(
            seqs <= d.max_seqs,
            "inkling2: {seqs} sequences over the {}-sequence buffers",
            d.max_seqs
        );
        let (mut pos, mut slot) = (Vec::with_capacity(n * 4), Vec::with_capacity(n * 8));
        for &(start, table) in &batch.seqs {
            ensure!(
                table.len() <= d.max_blocks_per_seq,
                "inkling2: block table over its buffer"
            );
            for p in start..start + batch.rows_per_seq {
                let blk = *table
                    .get(p / bs)
                    .with_context(|| format!("inkling2: no KV block for position {p}"))?;
                pos.extend((p as i32).to_le_bytes());
                slot.extend((i64::from(blk) * bs as i64 + (p % bs) as i64).to_le_bytes());
            }
        }
        // one table per row (each sequence's for its rows), or one shared
        let stride = batch.table_stride();
        let tables: Vec<u8> = if stride == 0 {
            batch.seqs[0]
                .1
                .iter()
                .flat_map(|b| b.to_le_bytes())
                .collect()
        } else {
            batch
                .seqs
                .iter()
                .flat_map(|(_, t)| std::iter::repeat_n(*t, batch.rows_per_seq))
                .flat_map(|t| t.iter().copied().chain(std::iter::repeat(0)).take(stride))
                .flat_map(u32::to_le_bytes)
                .collect()
        };
        ensure!(
            tables.len() <= d.block_table_bytes(),
            "inkling2: {n} rows' block tables over the table buffer"
        );
        // earlier kernels of this stream may still read the previous forward's metadata
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&pos, s.positions)?;
        gpu.copy_h2d(&slot, s.slots)?;
        gpu.copy_h2d(&tables, s.block_table)?;
        Ok(())
    }

    /// This visit's table of the sequences' conv-state bases, re-uploaded only
    /// when the batch's sequences change. Under graph capture a change is an
    /// error (the capture falls back to eager): `decode_prestage` binds the
    /// single decode sequence before every capture and replay.
    pub(super) fn bind_states(
        &self,
        gpu: &dyn GpuBackend,
        conv_states: &[DevicePtr],
        capturing: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let d = &self.dims;
        ensure!(
            conv_states.len() <= d.max_seqs,
            "inkling2: {} conv states over the {}-sequence table",
            conv_states.len(),
            d.max_seqs
        );
        let table = self
            .shared
            .state_tables
            .offset(self.visit_idx * d.max_seqs * 8);
        let ptrs: Vec<u64> = conv_states.iter().map(|p| p.0).collect();
        let mut bound = self
            .bound_states
            .lock()
            .map_err(|_| anyhow::anyhow!("inkling2: state table lock poisoned"))?;
        if *bound != ptrs {
            ensure!(
                !capturing,
                "inkling2 visit {}: conv-state table changed under graph capture",
                self.visit_idx
            );
            // stream-ordered after the previous forward's convs
            let bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
            gpu.copy_h2d_async(&bytes, table, stream)?;
            *bound = ptrs;
        }
        Ok(table)
    }

    /// A verify run's conv inputs `xs` (convs `first..`) into its journal.
    pub(super) fn journal_inputs<const N: usize>(
        &self,
        gpu: &dyn GpuBackend,
        batch: &Batch<'_>,
        xs: [DevicePtr; N],
        first: usize,
        stream: u64,
    ) -> Result<()> {
        let layout = self.conv_layout();
        let rows = batch.rows_per_seq;
        for (s, journal) in batch.journals.iter().enumerate() {
            for (i, x) in xs.iter().enumerate() {
                let (channels, _, off) = layout[first + i];
                let src = x.offset(s * rows * channels * 2);
                let dst = journal.offset(self.conv_state_bytes() + off);
                gpu.copy_d2d_async(src, dst, rows * channels * 2, stream)?;
            }
        }
        Ok(())
    }
}
