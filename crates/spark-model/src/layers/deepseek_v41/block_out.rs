// SPDX-License-Identifier: AGPL-3.0-only

//! The attention's last projection for a few rows.
//!
//! wo_b reads the low-rank output of every group, and the groups are the
//! parts': cut by its rows, as a batch's is, it waits for a gather of that
//! output, and its own output is gathered again. A decode step paid for
//! both in every layer. Cut by its columns, a part multiplies the low-rank
//! output it has just computed by its quarter of every row, which is the
//! same weight bytes, and leaves an FP32 partial of every output value. The
//! ranks gather the partials once, and each adds them in part order and
//! rounds: `BF16(((p0 + p1) + p2) + p3)`, whichever Spark computed them.

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;

use super::block::Block;
use super::gemm::Sent;
use super::ops;
use super::owned::Owned;
use super::parts::{self, PARTIAL_ROWS, PARTS};
use super::profile;

impl Block {
    /// wo_b of `rows` rows whose low-rank output the parts held have left in
    /// the scratch, into the attention's output in its four parts.
    pub(super) fn output_few(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights.attn,
        );
        let held = self.shared.held;
        let width = s.o_groups / PARTS * s.o_rank;
        ensure!(
            w.wo_b.n == s.hidden && w.wo_b.k == PARTS * width,
            "wo_b [{}, {}]",
            w.wo_b.n,
            w.wo_b.k
        );
        for first in (0..rows).step_by(PARTIAL_ROWS) {
            let turn = (rows - first).min(PARTIAL_ROWS);
            let partial = turn * s.hidden * 4;
            let began = profile::begin(gpu, stream)?;
            for part in held.range() {
                // The part's own output rounded to FP8 once, where it lies:
                // a scale block is 32 values of one part.
                let low = sc.o_low.offset((part * rows + first) * width * 2);
                let quant = k.cache.fp8_quant_dequant;
                ops::fp8_quant_dequant(gpu, quant, low, low, turn * width, stream)?;
                let columns = w.wo_b_columns[part].context("a held part of wo_b's columns")?;
                parts::project_rows(
                    gpu,
                    k.parts.fp8_f32,
                    low,
                    columns.data,
                    columns.scales,
                    sc.attn_partial.offset(part * partial),
                    4,
                    s.hidden,
                    turn,
                    s.hidden,
                    width,
                    stream,
                )?;
            }
            profile::end(gpu, stream, "output projections", began)?;
            profile::stage(gpu, stream, "gather", || {
                let recv = sc.gather_recv;
                parts::gather(
                    gpu,
                    comm,
                    held,
                    sc.attn_partial,
                    recv,
                    partial,
                    partial,
                    stream,
                )
            })?;
            parts::round(
                gpu,
                k.parts.round,
                sc.attn_partial,
                sc.attn_out,
                turn * s.hidden,
                turn,
                s.hidden,
                first,
                rows,
                stream,
            )?;
        }
        Ok(())
    }

    /// The other ranks' codes of the rows this rank keeps, of the low-rank
    /// output each left in its part of `scratch.o_low` (`sent`), to where a
    /// gather of the parts would leave them: an all-to-all a group, its
    /// codes and their scales. False, with nothing queued, where that does
    /// not serve; the ranks answer alike.
    pub(super) fn codes_home(
        &self,
        comm: Option<&dyn CommBackend>,
        own: Owned,
        sent: Sent,
        rows: usize,
        stream: u64,
    ) -> Result<bool> {
        let Some(comm) = comm else {
            return Ok(false);
        };
        let (sc, part) = (&self.shared.scratch, self.shared.held.first);
        for group in 0..sent.groups {
            // A group's rows of codes, and of scales, in a part.
            let plane = |at: usize, row_bytes: usize| {
                let at = at + group * rows * row_bytes;
                spark_comm::AllToAllPlane {
                    src: sc.o_low.offset(part * sent.bytes + at).0,
                    src_stride: (own.each * row_bytes) as i64,
                    dst: sc.o_low.offset(at + own.first * row_bytes).0,
                    dst_stride: sent.bytes as i64,
                    rows: own.each,
                    row_bytes,
                    pitch: row_bytes,
                }
            };
            let planes = [plane(0, sent.piece), plane(sent.scales, sent.piece / 32)];
            let queued = comm.all_to_all_async(&planes, stream)?;
            ensure!(
                queued || group == 0,
                "layer {}: the all-to-all took one group's codes and not the next's",
                self.index
            );
            if !queued {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
