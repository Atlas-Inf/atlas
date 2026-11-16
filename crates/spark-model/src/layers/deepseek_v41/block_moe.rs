// SPDX-License-Identifier: AGPL-3.0-only

//! The MoE sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Context, Result};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::Block;
use super::hc::SublayerParts;
use super::moe::{self, ExpertPart};
use super::moe_grouped::{self, Plan};
use super::parts::{self, PARTS};
use super::profile;
use super::weights::row_part;

impl Block {
    /// The MoE's output, as the parts [`Block::moe`] leaves.
    pub(super) fn moe_parts(&self) -> SublayerParts {
        SublayerParts {
            parts: self.shared.scratch.moe_out,
            rows: self.shared.scratch.rows,
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
        // The way in: a part of every slot's hidden activation. A token
        // alone has a slot an expert; a batch has its pairs in expert order
        // and then every row's shared expert.
        let (part_n, room) = (s.hidden / PARTS, sc.rows * slots * part_rows);
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
        for part in held.range() {
            let of = part_of(part)?;
            let inner = sc.inner_parts.offset(part * room * 2);
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
            moe_grouped::experts_in(
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
            )?;
            // The shared expert, every row through the same weights: its gate
            // and up rows side by side.
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
            moe_grouped::activate(
                gpu,
                &k.moe,
                sc.shared_gate_up,
                DevicePtr::NULL,
                inner.offset(plan.pairs() * part_rows * 2),
                rows,
                part_rows,
                s.swiglu_limit,
                stream,
            )?;
        }
        profile::end(gpu, stream, "experts", began)?;
        profile::stage(gpu, stream, "gather", || {
            parts::gather(
                gpu,
                comm,
                held,
                sc.inner_parts,
                sc.gather_recv,
                room * 2,
                rows * slots * part_rows * 2,
                stream,
            )
        })?;
        // The way out: a part of every slot's output from the whole of its
        // hidden activation, and the slots added.
        let began = profile::begin(gpu, stream)?;
        for part in held.range() {
            let of = part_of(part)?;
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
                    moe_grouped::experts_out(
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
                    )?;
                    // The shared expert's, in each row's last slot.
                    parts::project_pieces(
                        gpu,
                        k.parts.fp8_pieces,
                        sc.inner_parts.offset(plan.pairs() * part_rows * 2),
                        of.shared[2],
                        sc.slot_out.offset(s.top_k * part_n * 2),
                        slots * part_n,
                        rows,
                        part_n,
                        s.inter,
                        part_rows,
                        room,
                        stream,
                    )?;
                }
            }
            parts::slots_sum(
                gpu,
                k.parts.slots_sum,
                sc.slot_out,
                sc.moe_out.offset(part * sc.rows * part_n * 2),
                rows,
                slots,
                part_n,
                stream,
            )?;
        }
        profile::end(gpu, stream, "experts", began)?;
        profile::stage(gpu, stream, "gather", || {
            parts::gather(
                gpu,
                comm,
                held,
                sc.moe_out,
                sc.gather_recv,
                sc.rows * part_n * 2,
                rows * part_n * 2,
                stream,
            )
        })
    }
}

#[cfg(test)]
#[path = "block_moe_test_gpu.rs"]
mod test_gpu;
