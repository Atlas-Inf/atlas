// SPDX-License-Identifier: AGPL-3.0-only

//! The MoE sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Context, Result};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;

use super::block::Block;
use super::hc::SublayerParts;
use super::moe::{self, ExpertPart, ExpertScratch};
use super::parts::{self, PARTS};
use super::profile;
use super::weights::row_part;

impl Block {
    /// The MoE's output, as the parts [`Block::moe`] leaves.
    pub(super) fn moe_parts(&self) -> SublayerParts {
        SublayerParts {
            parts: self.shared.scratch.moe_parts,
            slots: self.shared.shape.top_k + 1,
            rows: self.shared.scratch.rows,
        }
    }

    /// The MoE of the `rows` tokens whose input is in `scratch.normed`, and in
    /// FP8 in `scratch.quantized`, into the parts of [`Block::moe_parts`].
    ///
    /// Every expert is in four parts: a quarter of its hidden rows, and the
    /// columns of its down projection that read them. A part's output is an
    /// FP32 partial per expert. A rank computes the parts it holds for the
    /// six chosen experts and the shared one, the ranks gather each other's
    /// (`comm`), and every rank adds them in the same order.
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
        for part in held.range() {
            let experts = ExpertPart {
                slab: w.experts[part].context("a held part of the routed experts")?,
                shared: [
                    row_part(&w.shared_w1, part),
                    row_part(&w.shared_w3, part),
                    w.shared_w2[part].context("a held part of the shared expert")?,
                ],
            };
            // A row's experts are its own: a launch set a row.
            for row in 0..rows {
                moe::experts(
                    gpu,
                    &k.moe,
                    &experts,
                    sc.quantized.offset(row * s.hidden * 2),
                    sc.chosen.offset(row * s.top_k * 4),
                    sc.weights.offset(row * s.top_k * 4),
                    ExpertScratch {
                        gate_up: sc.gate_up,
                        inner: sc.inner,
                    },
                    sc.moe_parts
                        .offset((part * sc.rows + row) * slots * s.hidden * 4),
                    s.top_k,
                    s.inter / PARTS,
                    s.hidden,
                    s.swiglu_limit,
                    stream,
                )?;
            }
        }
        profile::end(gpu, stream, "experts", began)?;
        profile::stage(gpu, stream, "gather", || {
            parts::gather(
                gpu,
                comm,
                held,
                sc.moe_parts,
                sc.gather_recv,
                sc.rows * slots * s.hidden * 4,
                rows * slots * s.hidden * 4,
                stream,
            )
        })
    }
}

#[cfg(test)]
#[path = "block_moe_test_gpu.rs"]
mod test_gpu;
