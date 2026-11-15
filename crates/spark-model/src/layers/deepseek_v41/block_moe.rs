// SPDX-License-Identifier: AGPL-3.0-only

//! The MoE sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::Block;
use super::moe;
use super::ops;
use super::weights::Fp8Proj;
use crate::layers::ops::mxfp8_gemv_rows;

impl Block {
    /// One expert on the FP8 input in `scratch.quantized`, added into
    /// `scratch.acc`.
    fn expert(
        &self,
        gpu: &dyn GpuBackend,
        gate_up: impl Fn(DevicePtr, DevicePtr) -> Result<()>,
        down: impl Fn(DevicePtr, DevicePtr) -> Result<()>,
        weight: f32,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        gate_up(sc.gate, sc.up)?;
        moe::swiglu(
            gpu,
            k.moe.swiglu,
            sc.gate,
            sc.up,
            sc.inner,
            s.inter,
            weight,
            s.swiglu_limit,
            stream,
        )?;
        ops::fp8_quant_dequant(
            gpu,
            k.cache.fp8_quant_dequant,
            sc.inner,
            sc.inner,
            s.inter,
            stream,
        )?;
        down(sc.inner, sc.down)?;
        moe::accumulate(gpu, k.moe.accumulate, sc.acc, sc.down, s.hidden, stream)
    }

    /// The MoE of the normed input in `scratch.normed` into
    /// `scratch.sublayer_out`.
    pub(super) fn moe(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights.moe,
        );
        ensure!(
            w.experts.len() == s.experts,
            "layer {}: {} of {} experts are loaded",
            self.index,
            w.experts.len(),
            s.experts
        );
        moe::route(
            gpu,
            k.moe.route,
            sc.normed,
            w.gate,
            w.bias,
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
        gpu.memset_zero_async(sc.acc, s.hidden * 4, stream)?;
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
        for (expert, weight) in routed {
            let e = &w.experts[expert];
            self.expert(
                gpu,
                |gate, up| {
                    moe::fp4_gemv(
                        gpu,
                        k.moe.fp4_gemv,
                        sc.quantized,
                        e.w1.w4,
                        e.w1.scales,
                        gate,
                        s.inter,
                        s.hidden,
                        stream,
                    )?;
                    moe::fp4_gemv(
                        gpu,
                        k.moe.fp4_gemv,
                        sc.quantized,
                        e.w3.w4,
                        e.w3.scales,
                        up,
                        s.inter,
                        s.hidden,
                        stream,
                    )
                },
                |inner, down| {
                    moe::fp4_gemv(
                        gpu,
                        k.moe.fp4_gemv,
                        inner,
                        e.w2.w4,
                        e.w2.scales,
                        down,
                        s.hidden,
                        s.inter,
                        stream,
                    )
                },
                weight,
                stream,
            )?;
        }
        let gemv = |p: &Fp8Proj, x, out| {
            mxfp8_gemv_rows(gpu, &k.tiers, &p.w, x, out, 1, p.n, p.k, p.n, stream)
        };
        self.expert(
            gpu,
            |gate, up| {
                gemv(&w.shared_w1, sc.quantized, gate)?;
                gemv(&w.shared_w3, sc.quantized, up)
            },
            |inner, down| gemv(&w.shared_w2, inner, down),
            1.0,
            stream,
        )?;
        moe::round_bf16(
            gpu,
            k.moe.round_bf16,
            sc.acc,
            sc.sublayer_out,
            s.hidden,
            stream,
        )
    }
}
