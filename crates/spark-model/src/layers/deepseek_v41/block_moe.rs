// SPDX-License-Identifier: AGPL-3.0-only

//! The MoE sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Context, Result};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::Block;
use super::gemm::{GEMM_ROWS, Input};
use super::hc::SublayerParts;
use super::moe::{self, ExpertPart};
use super::moe_gemm::{Runs, Sent};
use super::moe_grouped::{self, GateUp, Plan};
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
        profile::stage(gpu, stream, "route", || {
            moe::route_fast(
                gpu,
                &k.moe,
                sc.normed,
                w.gate,
                w.bias,
                sc.route_scores,
                sc.chosen,
                sc.weights,
                rows,
                s.hidden,
                s.experts,
                s.top_k,
                s.route_scale,
                true,
                self.gemm.is_some() && rows >= GEMM_ROWS,
                stream,
            )
        })?;
        // One slot per chosen expert, then the shared expert's.
        let slots = s.top_k + 1;
        let began = profile::begin(gpu, stream)?;
        let part_rows = s.inter / PARTS;
        // More than one row: the rows that chose an expert share its weights.
        let plan = match rows {
            1 => None,
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
        // A large batch's experts go to the block-scaled tensor cores, an
        // expert's pairs in tiles.
        let big = match (&plan, &sc.expert_gemm) {
            (Some(plan), Some(big)) if self.gemm.is_some() && rows >= big.from => {
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
        let sent = match (&big, &plan) {
            (Some(_), Some(plan)) => Some(Sent::new(plan.pairs(), rows, part_rows)),
            _ => None,
        };
        let part_bytes = sent.map_or(room * 2, |sent| sent.bytes);
        for part in held.range() {
            let of = part_of(part)?;
            let inner = sc.inner_parts.offset(part * part_bytes);
            let Some(plan) = &plan else {
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
                        sc.quantized,
                        rows,
                        sc.grouped.row_of,
                        sc.grouped.gate_up,
                        part_rows,
                        s.hidden,
                        stream,
                    )?;
                    // Every pair's gate values, then every pair's up values.
                    let gate_up = GateUp {
                        base: sc.grouped.gate_up,
                        up_offset: plan.pairs() * part_rows,
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
                        plan.pairs(),
                        part_rows,
                        s.swiglu_limit,
                        stream,
                    )?;
                }
                None => moe_grouped::experts_in(
                    gpu,
                    &k.moe,
                    &of.slab,
                    plan,
                    &sc.grouped,
                    sc.quantized,
                    inner,
                    part_rows,
                    s.hidden,
                    s.swiglu_limit,
                    stream,
                )?,
            }
            // The shared expert, every row through the same weights.
            let gate_up = match gemm.and_then(|g| g.shared[part].as_ref()) {
                // Every row's gate values, then every row's up values.
                Some([w1, w3, _]) => {
                    let x = Input::rows(sc.quantized, s.hidden);
                    let act = sc.gemm.quantize(gpu, x, rows, s.hidden, stream)?;
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
                inner.offset(sent.map_or(plan.pairs() * part_rows * 2, |sent| sent.shared)),
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
            match &plan {
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
                Some(plan) => {
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
                            plan,
                            &sc.grouped,
                            sc.inner_parts,
                            room,
                            sc.slot_out,
                            part_rows,
                            part_n,
                            stream,
                        )?,
                    }
                    // The shared expert's: its rows of the hidden
                    // activations come after the pairs'.
                    let x = sc
                        .inner_parts
                        .offset(sent.map_or(plan.pairs() * part_rows * 2, |sent| sent.shared));
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
            parts::gather(
                gpu,
                comm,
                held,
                sc.moe_out,
                sc.gather_recv,
                rows * part_n * 2,
                rows * part_n * 2,
                stream,
            )
        })
    }
}

#[cfg(test)]
#[path = "block_moe_test_gpu.rs"]
mod test_gpu;
