// SPDX-License-Identifier: AGPL-3.0-only

//! The sequences one Inkling 2 forward carries, and the metadata the visits
//! read for them: a run of consecutive tokens of one sequence (a decode step,
//! a prefill chunk), or one token each of several sequences (a decode batch).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::layer::Inkling2Layer;

pub(super) struct Batch<'a> {
    /// Per sequence: the position of its first row, and its block table.
    pub seqs: Vec<(usize, &'a [u32])>,
    /// Rows of each sequence; more than one only for a single sequence.
    pub rows_per_seq: usize,
}

impl<'a> Batch<'a> {
    pub(super) fn one(start: usize, rows: usize, table: &'a [u32]) -> Self {
        Self {
            seqs: vec![(start, table)],
            rows_per_seq: rows,
        }
    }

    pub(super) fn rows(&self) -> usize {
        self.seqs.len() * self.rows_per_seq
    }

    pub(super) fn position(&self, row: usize) -> usize {
        self.seqs[row / self.rows_per_seq].0 + row % self.rows_per_seq
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

impl Inkling2Layer {
    /// Upload the batch's positions, KV slots, lengths, row-to-sequence map
    /// and block tables (visit 0, for every visit).
    pub(super) fn upload_meta(
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
        ensure!(
            seqs == 1 || batch.rows_per_seq == 1,
            "inkling2: several sequences of several rows each are not supported"
        );
        let mut seq_start = Vec::with_capacity((seqs + 1) * 4);
        let (mut pos, mut slot, mut len, mut row_seq) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, &(start, table)) in batch.seqs.iter().enumerate() {
            ensure!(
                table.len() <= d.max_blocks_per_seq,
                "inkling2: block table over its buffer"
            );
            seq_start.extend(((i * batch.rows_per_seq) as i32).to_le_bytes());
            for p in start..start + batch.rows_per_seq {
                let blk = *table
                    .get(p / bs)
                    .with_context(|| format!("inkling2: no KV block for position {p}"))?;
                pos.extend((p as i32).to_le_bytes());
                slot.extend((i64::from(blk) * bs as i64 + (p % bs) as i64).to_le_bytes());
                len.extend(((p + 1) as i32).to_le_bytes());
                row_seq.extend((i as i32).to_le_bytes());
            }
        }
        seq_start.extend((n as i32).to_le_bytes());
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
                .flat_map(|(_, t)| t.iter().copied().chain(std::iter::repeat(0)).take(stride))
                .flat_map(u32::to_le_bytes)
                .collect()
        };
        // earlier kernels of this stream may still read the previous forward's metadata
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&pos, s.positions)?;
        gpu.copy_h2d(&slot, s.slots)?;
        gpu.copy_h2d(&len, s.seq_lens)?;
        gpu.copy_h2d(&row_seq, s.row_seq)?;
        gpu.copy_h2d(&seq_start, s.seq_start)?;
        gpu.copy_h2d(&tables, s.block_table)?;
        Ok(())
    }

    /// This visit's table of the sequences' conv-state bases, re-uploaded only
    /// when the batch's sequences change.
    pub(super) fn bind_states(
        &self,
        gpu: &dyn GpuBackend,
        conv_states: &[DevicePtr],
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
            // the previous forward's convs may still read the old table
            gpu.synchronize(stream)?;
            let bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
            gpu.copy_h2d(&bytes, table)?;
            *bound = ptrs;
        }
        Ok(table)
    }
}
