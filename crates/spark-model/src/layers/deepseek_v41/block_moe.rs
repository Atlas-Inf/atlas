// SPDX-License-Identifier: AGPL-3.0-only

//! The MoE sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Context, Result};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::Block;
use super::gemm::{GEMM_ROWS, Input};
use super::hc::SublayerParts;
use super::moe::{self, ExpertPart};
use super::moe_gemm::{ExpertGemm, Runs, Sent};
use super::moe_grouped::{self, GateUp, Plan};
use super::owned::{self, Owned};
use super::parts::{self, PARTS};
use super::profile;
use super::weights::row_part;

impl Block {
    /// The MoE's output for a batch of `rows`, as the parts [`Block::moe`]
    /// leaves.
    pub(super) fn moe_parts(&self, rows: usize) -> SublayerParts {
        SublayerParts {
            parts: self.shared.scratch.moe_out,
            rows,
        }
    }

    /// The MoE of the `rows` tokens whose input is in `scratch.normed`, and in
    /// FP8 in `scratch.quantized`, into the parts of [`Block::moe_parts`].
    ///
    /// Every expert is in four parts: a quarter of its hidden rows, and a
    /// quarter of the rows of its down projection. A rank computes the hidden
    /// activations of the parts it holds, for the six chosen experts and the
    /// shared one; the ranks gather each other's (`comm`); a rank computes
    /// its quarter of every expert's output from all of them and adds the
    /// experts; and the ranks gather the quarters.
    pub(super) fn moe(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        comm: Option<&dyn CommBackend>,
        own: Option<Owned>,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights.moe,
        );
        let held = self.shared.held;
        // The chosen in expert order: the reference adds their outputs so.
        // Of the rows whose input is here: every row's, or the rows this
        // rank keeps. The ranks then gather each other's routing, and the
        // input in FP8, which every rank's part of the experts reads.
        let (at, kept) = own.map_or((0, rows), |own| (own.first, own.count));
        let routed = at * s.top_k * 4;
        profile::stage(gpu, stream, "route", || {
            if kept == 0 {
                return Ok(());
            }
            moe::route_fast(
                gpu,
                &k.moe,
                sc.normed.offset(at * s.hidden * 2),
                w.gate,
                w.bias,
                sc.route_scores,
                sc.chosen.offset(routed),
                sc.weights.offset(routed),
                kept,
                s.hidden,
                s.experts,
                s.top_k,
                s.route_scale,
                true,
                self.gemm.is_some() && kept >= GEMM_ROWS,
                stream,
            )
        })?;
        // A large batch's experts go to the block-scaled tensor cores, which
        // read the input in codes: the rows here rounded once, for the routed
        // experts and the shared one.
        let coded = sc.expert_gemm.as_ref().filter(|big| {
            self.gemm.is_some() && rows >= big.from && rows > moe_grouped::DEVICE_ROWS
        });
        if let Some(big) = coded {
            let x = sc.quantized.offset(at * s.hidden * 2);
            big.round_rows(gpu, x, at, kept, s.hidden, stream)?;
        }
        if let Some(own) = own {
            let rows_of = |base: DevicePtr, row_bytes: usize| {
                let recv = sc.gather_recv;
                owned::gather_rows(gpu, comm, held, own, base, recv, row_bytes, stream)
            };
            rows_of(sc.chosen, s.top_k * 4)?;
            rows_of(sc.weights, s.top_k * 4)?;
            match coded.map(ExpertGemm::input) {
                // Half the bytes of the values.
                Some((codes, scales)) => {
                    rows_of(codes, s.hidden)?;
                    rows_of(scales, s.hidden / 32)?;
                }
                None => rows_of(sc.quantized, s.hidden * 2)?,
            }
        }
        // One slot per chosen expert, then the shared expert's.
        let slots = s.top_k + 1;
        let began = profile::begin(gpu, stream)?;
        let part_rows = s.inter / PARTS;
        // More than one row: the rows that chose an expert share its weights.
        // A few rows' pairs are sorted by expert on the device, a batch's on
        // the host.
        let plan = match rows {
            0..=moe_grouped::DEVICE_ROWS => None,
            _ => {
                gpu.synchronize(stream)?;
                let mut chosen = vec![0u8; rows * s.top_k * 4];
                let mut weights = vec![0u8; rows * s.top_k * 4];
                gpu.copy_d2h(sc.chosen, &mut chosen)?;
                gpu.copy_d2h(sc.weights, &mut weights)?;
                let word = |c: &[u8]| [c[0], c[1], c[2], c[3]];
                let chosen: Vec<u32> = chosen
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(word(c)))
                    .collect();
                let weights: Vec<f32> = weights
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(word(c)))
                    .collect();
                let plan = Plan::new(&chosen, &weights, s.top_k, slots, s.experts);
                sc.grouped.upload(gpu, &plan)?;
                Some(plan)
            }
        };
        // A few rows take the shared expert in the routed experts' launches.
        let fused = plan.is_none() && rows > 1;
        let grouped = match (rows, &plan) {
            (_, Some(plan)) => Some(plan.counts()),
            (1, None) => None,
            (_, None) => Some(moe_grouped::plan_on_device(
                gpu,
                &k.moe,
                &sc.grouped,
                sc.chosen,
                sc.weights,
                rows,
                s.top_k,
                slots,
                stream,
            )?),
        };
        // The tensor cores take an expert's pairs in tiles.
        let big = match (&plan, coded) {
            (Some(plan), Some(big)) => {
                let runs = Runs::new(plan, rows, s.top_k, slots);
                big.upload(gpu, &runs)?;
                Some((big, runs))
            }
            _ => None,
        };
        // The way in: a part of every slot's hidden activation. A token
        // alone has a slot an expert; a batch has its pairs in expert order
        // and then every row's shared expert.
        let (part_n, room) = (s.hidden / PARTS, rows * slots * part_rows);
        let gemm = self.gemm.as_ref().filter(|_| rows >= GEMM_ROWS);
        let part_of = |part: usize| -> Result<ExpertPart> {
            Ok(ExpertPart {
                slab: w.experts[part].context("a held part of the routed experts")?,
                shared: [
                    row_part(&w.shared_w1, part),
                    row_part(&w.shared_w3, part),
                    row_part(&w.shared_w2, part),
                ],
            })
        };
        // A part of the hidden activations as the ranks gather it: values,
        // or from the tensor cores codes and scales.
        let sent = match (&big, grouped) {
            (Some(_), Some(grouped)) => Some(Sent::new(grouped.pairs, rows, part_rows)),
            _ => None,
        };
        let part_bytes = sent.map_or(room * 2, |sent| sent.bytes);
        for part in held.range() {
            let of = part_of(part)?;
            let inner = sc.inner_parts.offset(part * part_bytes);
            let Some(grouped) = grouped else {
                moe::experts_in(
                    gpu,
                    &k.moe,
                    &of,
                    sc.quantized,
                    sc.chosen,
                    sc.weights,
                    sc.gate_up,
                    inner,
                    s.top_k,
                    part_rows,
                    s.hidden,
                    s.swiglu_limit,
                    stream,
                )?;
                continue;
            };
            match &big {
                Some((big, runs)) => {
                    big.project_in(
                        gpu,
                        &of.slab,
                        runs,
                        sc.grouped.row_of,
                        sc.grouped.gate_up,
                        part_rows,
                        s.hidden,
                        stream,
                    )?;
                    // Every pair's gate values, then every pair's up values.
                    let gate_up = GateUp {
                        base: sc.grouped.gate_up,
                        up_offset: grouped.pairs * part_rows,
                        row_stride: part_rows,
                    };
                    let sent = sent.context("a part as the ranks send it")?;
                    moe_grouped::activate_codes(
                        gpu,
                        &k.moe,
                        gate_up,
                        sc.grouped.weights,
                        inner,
                        inner.offset(sent.scales),
                        grouped.pairs,
                        part_rows,
                        s.swiglu_limit,
                        stream,
                    )?;
                }
                None => moe_grouped::experts_in(
                    gpu,
                    &k.moe,
                    &of.slab,
                    grouped,
                    &sc.grouped,
                    sc.quantized,
                    inner,
                    part_rows,
                    s.hidden,
                    s.swiglu_limit,
                    fused.then_some(([of.shared[0], of.shared[1]], rows)),
                    stream,
                )?,
            }
            if fused {
                continue;
            }
            // The shared expert, every row through the same weights.
            let gate_up = match gemm.and_then(|g| g.shared[part].as_ref()) {
                // Every row's gate values, then every row's up values.
                Some([w1, w3, _]) => {
                    let act = match coded.map(ExpertGemm::input) {
                        Some((codes, scales)) => {
                            sc.gemm.coded(gpu, codes, scales, rows, s.hidden, stream)?
                        }
                        None => {
                            let x = Input::rows(sc.quantized, s.hidden);
                            sc.gemm.quantize(gpu, x, rows, s.hidden, stream)?
                        }
                    };
                    let up = sc.shared_gate_up.offset(rows * part_rows * 2);
                    w1.gemm(act, sc.shared_gate_up, rows, stream)?;
                    w3.gemm(act, up, rows, stream)?;
                    GateUp {
                        base: sc.shared_gate_up,
                        up_offset: rows * part_rows,
                        row_stride: part_rows,
                    }
                }
                // A row's gate and up values side by side.
                None => {
                    let [w1, w3, _] = of.shared;
                    for (weights, at) in [(w1, 0), (w3, part_rows)] {
                        parts::project_rows(
                            gpu,
                            k.parts.fp8,
                            sc.quantized,
                            weights.data,
                            weights.scales,
                            sc.shared_gate_up.offset(at * 2),
                            2,
                            2 * part_rows,
                            rows,
                            part_rows,
                            s.hidden,
                            stream,
                        )?;
                    }
                    GateUp::paired(sc.shared_gate_up, part_rows)
                }
            };
            moe_grouped::activate(
                gpu,
                &k.moe,
                gate_up,
                DevicePtr::NULL,
                inner.offset(sent.map_or(grouped.pairs * part_rows * 2, |sent| sent.shared)),
                rows,
                part_rows,
                s.swiglu_limit,
                stream,
            )?;
        }
        profile::end(gpu, stream, "experts in", began)?;
        profile::stage(gpu, stream, "gather", || {
            parts::gather(
                gpu,
                comm,
                held,
                sc.inner_parts,
                sc.gather_recv,
                part_bytes,
                part_bytes,
                stream,
            )
        })?;
        // The way out: a part of every slot's output from the whole of its
        // hidden activation, and the slots added.
        let began = profile::begin(gpu, stream)?;
        for part in held.range() {
            let of = part_of(part)?;
            let mut shared_out = DevicePtr::NULL;
            match grouped {
                None => moe::experts_out(
                    gpu,
                    &k.moe,
                    &of,
                    sc.chosen,
                    sc.inner_parts,
                    room,
                    sc.slot_out,
                    s.top_k,
                    part_rows,
                    part_n,
                    stream,
                )?,
                Some(grouped) => {
                    match &big {
                        Some((big, runs)) => big.project_out(
                            gpu,
                            &of.slab,
                            runs,
                            sc.inner_parts,
                            sent.context("the parts as the ranks sent them")?,
                            part_rows,
                            s.hidden,
                            stream,
                        )?,
                        None => moe_grouped::experts_out(
                            gpu,
                            &k.moe,
                            &of.slab,
                            grouped,
                            &sc.grouped,
                            sc.inner_parts,
                            room,
                            sc.slot_out,
                            part_rows,
                            part_n,
                            fused.then_some((of.shared[2], rows)),
                            stream,
                        )?,
                    }
                    // The shared expert's, where it did not ride those
                    // launches: its rows of the hidden activations come
                    // after the pairs'.
                    if !fused {
                        let x = sc
                            .inner_parts
                            .offset(sent.map_or(grouped.pairs * part_rows * 2, |sent| sent.shared));
                        match gemm.and_then(|g| g.shared[part].as_ref()) {
                            Some([_, _, w2]) => {
                                let x = Input {
                                    x,
                                    stride: part_rows,
                                    piece: part_rows,
                                    piece_stride: part_bytes / 2,
                                };
                                sc.gemm.project(gpu, x, w2, sc.shared_out, rows, stream)?;
                                shared_out = sc.shared_out;
                            }
                            // Into each row's last slot.
                            None => parts::project_pieces(
                                gpu,
                                k.parts.fp8_pieces,
                                x,
                                of.shared[2],
                                sc.slot_out.offset(s.top_k * part_n * 2),
                                slots * part_n,
                                rows,
                                part_n,
                                s.inter,
                                part_rows,
                                room,
                                stream,
                            )?,
                        }
                    }
                }
            }
            let y = sc.moe_out.offset(part * rows * part_n * 2);
            match &big {
                Some((big, _)) => big.sum(gpu, shared_out, y, rows, s.top_k, part_n, stream)?,
                None => parts::slots_sum(
                    gpu,
                    k.parts.slots_sum,
                    sc.slot_out,
                    shared_out,
                    y,
                    rows,
                    slots,
                    part_n,
                    stream,
                )?,
            }
        }
        profile::end(gpu, stream, "experts out", began)?;
        profile::stage(gpu, stream, "gather", || {
            let (recv, bytes) = (sc.gather_recv, part_n * 2);
            match own {
                // Only the rank that keeps a row reads its output.
                Some(own) => {
                    owned::to_owners(gpu, comm, held, own, sc.moe_out, recv, rows, bytes, stream)
                }
                None => parts::gather(
                    gpu,
                    comm,
                    held,
                    sc.moe_out,
                    recv,
                    rows * bytes,
                    rows * bytes,
                    stream,
                ),
            }
        })
    }
}

#[cfg(test)]
#[path = "block_moe_test_gpu.rs"]
mod test_gpu;
