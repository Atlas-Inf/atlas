// SPDX-License-Identifier: AGPL-3.0-only

//! The MoE sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Context, Result};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;

use super::block::Block;
use super::moe;
use super::ops;
use super::parts::{self, PARTS};
use super::profile;
use super::weights::row_part;

impl Block {
    /// SwiGLU of `rows` gate and up values in scratch, times `weight`, as the
    /// FP8 activation the down projection reads, in `scratch.inner`.
    fn activate(&self, gpu: &dyn GpuBackend, rows: usize, weight: f32, stream: u64) -> Result<()> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        moe::swiglu(
            gpu,
            k.moe.swiglu,
            sc.gate,
            sc.up,
            sc.inner,
            rows,
            weight,
            s.swiglu_limit,
            stream,
        )?;
        ops::fp8_quant_dequant(
            gpu,
            k.cache.fp8_quant_dequant,
            sc.inner,
            sc.inner,
            rows,
            stream,
        )
    }

    /// The MoE of the normed input in `scratch.normed` into
    /// `scratch.sublayer_out`.
    ///
    /// Every expert is in four parts: a quarter of its hidden rows, and the
    /// columns of its down projection that read them. A part's output is an
    /// FP32 partial per expert. A rank computes the parts it holds for the
    /// six chosen experts and the shared one, the ranks gather each other's
    /// (`comm`), and every rank adds them in the same order.
    pub(super) fn moe(
        &self,
        gpu: &dyn GpuBackend,
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
        let began = profile::begin(gpu, stream)?;
        moe::route_fast(
            gpu,
            &k.moe,
            sc.normed,
            w.gate,
            w.bias,
            sc.route_scores,
            sc.chosen,
            sc.weights,
            1,
            s.hidden,
            s.experts,
            s.top_k,
            s.route_scale,
            stream,
        )?;
        ops::fp8_quant_dequant(
            gpu,
            k.cache.fp8_quant_dequant,
            sc.normed,
            sc.quantized,
            s.hidden,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let (mut chosen, mut weights) = (vec![0u8; s.top_k * 4], vec![0u8; s.top_k * 4]);
        gpu.copy_d2h(sc.chosen, &mut chosen)?;
        gpu.copy_d2h(sc.weights, &mut weights)?;
        let word = |b: &[u8], i: usize| [b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]];
        let mut routed: Vec<(usize, f32)> = (0..s.top_k)
            .map(|i| {
                (
                    u32::from_le_bytes(word(&chosen, i)) as usize,
                    f32::from_le_bytes(word(&weights, i)),
                )
            })
            .collect();
        // The reference adds the experts' outputs in expert order.
        routed.sort_unstable_by_key(|&(expert, _)| expert);
        profile::end(gpu, stream, "route", began)?;

        // One slot per chosen expert, then the shared expert's.
        let (slots, rows) = (s.top_k + 1, s.inter / PARTS);
        let p = &k.parts;
        for part in held.range() {
            let began = profile::begin(gpu, stream)?;
            let out = |slot: usize| sc.moe_parts.offset((part * slots + slot) * s.hidden * 4);
            let slab = w.experts[part].context("a held part of the routed experts")?;
            for (slot, &(expert, weight)) in routed.iter().enumerate() {
                let (w1, w3, w2) = (slab.w1(expert), slab.w3(expert), slab.w2(expert));
                let x = sc.quantized;
                parts::project(
                    gpu, p.fp4, x, w1.data, w1.scales, sc.gate, rows, s.hidden, stream,
                )?;
                parts::project(
                    gpu, p.fp4, x, w3.data, w3.scales, sc.up, rows, s.hidden, stream,
                )?;
                self.activate(gpu, rows, weight, stream)?;
                parts::project(
                    gpu,
                    p.fp4_f32,
                    sc.inner,
                    w2.data,
                    w2.scales,
                    out(slot),
                    s.hidden,
                    rows,
                    stream,
                )?;
            }
            profile::end(gpu, stream, "routed experts", began)?;
            let began = profile::begin(gpu, stream)?;
            // The shared expert: FP8 weights, no routing weight.
            let (w1, w3) = (row_part(&w.shared_w1, part), row_part(&w.shared_w3, part));
            let w2 = w.shared_w2[part].context("a held part of the shared expert")?;
            let x = sc.quantized;
            parts::project(
                gpu, p.fp8, x, w1.data, w1.scales, sc.gate, rows, s.hidden, stream,
            )?;
            parts::project(
                gpu, p.fp8, x, w3.data, w3.scales, sc.up, rows, s.hidden, stream,
            )?;
            self.activate(gpu, rows, 1.0, stream)?;
            parts::project(
                gpu,
                p.fp8_f32,
                sc.inner,
                w2.data,
                w2.scales,
                out(s.top_k),
                s.hidden,
                rows,
                stream,
            )?;
            profile::end(gpu, stream, "shared expert", began)?;
        }
        profile::stage(gpu, stream, "gather", || {
            parts::gather(
                gpu,
                comm,
                held,
                sc.moe_parts,
                sc.gather_recv,
                slots * s.hidden * 4,
                stream,
            )
        })?;
        parts::combine(
            gpu,
            p.combine,
            sc.moe_parts,
            sc.sublayer_out,
            slots,
            s.hidden,
            stream,
        )
    }
}
