// SPDX-License-Identifier: AGPL-3.0-only

//! A large batch's experts on the block-scaled tensor cores
//! (`moe_gemm.rs`), and the gather of their hidden activations beside the
//! launches that do not read it.
//!
//! The ranks gather each other's parts of the hidden activations: 34 MB a
//! rank at 8,192 rows, five milliseconds of a sublayer of twenty-five in
//! which the batch's stream ran nothing. So the pairs go in two spans. The
//! first span's parts are gathered while the second's are computed, and the
//! second's, with the shared expert's, while the first span goes out.
//!
//! The gather hides whole, and most of it comes back: a gather is the
//! memory bus's work too (a piece is copied to its staging, sent, and each
//! rank's copied from where it landed), and the launches beside it read
//! their weights from memory at the bus's pace. They run about four
//! milliseconds longer for the five the gather no longer holds them.

use anyhow::{Context, Result};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;

use super::block::Block;
use super::gemm::Input;
use super::moe_gemm::{ExpertGemm, Runs, Sent, Span};
use super::moe_grouped::{self, GateUp};
use super::parts::{self, PARTS};
use super::profile;

impl Block {
    /// The routed experts of `runs` and the shared expert, of the `rows`
    /// rows whose input is in codes in `big`
    /// ([`ExpertGemm::round_rows`]), into the parts of
    /// [`Block::moe_parts`].
    pub(super) fn experts_big(
        &self,
        gpu: &dyn GpuBackend,
        big: &ExpertGemm,
        runs: &Runs,
        rows: usize,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights.moe,
        );
        let held = self.shared.held;
        let (part_rows, part_n, pairs) = (s.inter / PARTS, s.hidden / PARTS, runs.pairs());
        let sent = Sent::new(pairs, rows, part_rows);
        let slab = |part: usize| w.experts[part].context("a held part of the routed experts");
        let gemm = self.gemm.as_ref().context("the GEMM operands")?;
        let shared = |part: usize| {
            gemm.shared[part]
                .as_ref()
                .context("a held part of the shared expert")
        };
        let inner = |part: usize| sc.inner_parts.offset(part * sent.bytes);
        // A span's bytes of every part: its pairs' codes, and their scales,
        // the last span's with the shared expert's rows after them.
        let gather = |span: &Span, last: bool, on: u64| -> Result<()> {
            let scales = |pair: usize| sent.scales + pair * part_rows / 32;
            let until = if last {
                sent.bytes
            } else {
                scales(span.pairs.end)
            };
            for (from, until) in [
                (span.pairs.start * part_rows, span.pairs.end * part_rows),
                (scales(span.pairs.start), until),
            ] {
                let (base, recv) = (sc.inner_parts.offset(from), sc.gather_recv);
                parts::gather(gpu, comm, held, base, recv, sent.bytes, until - from, on)?;
            }
            Ok(())
        };
        let spans = match runs.halves(big.first_span) {
            Some(halves) if comm.is_none_or(|comm| comm.supports_gather_async(2)) => {
                halves.to_vec()
            }
            _ => vec![runs.whole()],
        };
        let beside = (spans.len() > 1).then_some(&sc.beside);

        // The way in: a part of every pair's hidden activation, in codes.
        let began = profile::begin(gpu, stream)?;
        for (at, span) in spans.iter().enumerate() {
            let last = at + 1 == spans.len();
            for part in held.range() {
                let tiles = span.tiles.clone();
                big.project_in(
                    gpu,
                    &slab(part)?,
                    runs,
                    tiles,
                    sc.grouped.row_of,
                    sc.grouped.gate_up,
                    part_rows,
                    s.hidden,
                    stream,
                )?;
                // Every pair's gate values, then every pair's up values.
                let first = span.pairs.start;
                let gate_up = GateUp {
                    base: sc.grouped.gate_up.offset(first * part_rows * 2),
                    up_offset: pairs * part_rows,
                    row_stride: part_rows,
                };
                moe_grouped::activate_codes(
                    gpu,
                    &k.moe,
                    gate_up,
                    sc.grouped.weights.offset(first * 4),
                    inner(part).offset(first * part_rows),
                    inner(part).offset(sent.scales + first * part_rows / 32),
                    span.pairs.len(),
                    part_rows,
                    s.swiglu_limit,
                    stream,
                )?;
                if !last {
                    continue;
                }
                // The shared expert, every row through the same weights:
                // every row's gate values, then every row's up values.
                let [w1, w3, _] = shared(part)?;
                let (codes, scales) = big.input();
                let act = sc.gemm.coded(gpu, codes, scales, rows, s.hidden, stream)?;
                let up = sc.shared_gate_up.offset(rows * part_rows * 2);
                w1.gemm(act, sc.shared_gate_up, rows, stream)?;
                w3.gemm(act, up, rows, stream)?;
                let gate_up = GateUp {
                    base: sc.shared_gate_up,
                    up_offset: rows * part_rows,
                    row_stride: part_rows,
                };
                let to = inner(part).offset(sent.shared);
                let limit = s.swiglu_limit;
                moe_grouped::activate(
                    gpu,
                    &k.moe,
                    gate_up,
                    spark_runtime::gpu::DevicePtr::NULL,
                    to,
                    rows,
                    part_rows,
                    limit,
                    stream,
                )?;
            }
            match beside {
                Some(beside) => beside.send(gpu, at, stream, |on| gather(span, last, on))?,
                None => {
                    profile::end(gpu, stream, "experts in", began)?;
                    profile::stage(gpu, stream, "gather", || gather(span, last, stream))?;
                }
            }
        }

        // The way out: a part of every pair's output from the whole of its
        // hidden activation, the shared expert's, and a row's added.
        let began = profile::begin(gpu, stream)?;
        for part in held.range() {
            for (at, span) in spans.iter().enumerate() {
                if let Some(beside) = beside {
                    beside.brought(gpu, at, stream)?;
                }
                let slab = slab(part)?;
                big.project_out(
                    gpu,
                    &slab,
                    span,
                    sc.inner_parts,
                    sent,
                    part_rows,
                    s.hidden,
                    stream,
                )?;
            }
            // Its rows of the hidden activations come after the pairs'.
            let [_, _, w2] = shared(part)?;
            let x = Input {
                x: sc.inner_parts.offset(sent.shared),
                stride: part_rows,
                piece: part_rows,
                piece_stride: sent.bytes / 2,
            };
            sc.gemm.project(gpu, x, w2, sc.shared_out, rows, stream)?;
            let y = sc.moe_out.offset(part * rows * part_n * 2);
            big.sum(gpu, sc.shared_out, y, rows, s.top_k, part_n, stream)?;
        }
        profile::end(gpu, stream, "experts out", began)
    }
}
