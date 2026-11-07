// SPDX-License-Identifier: AGPL-3.0-only

//! Native FP8 q/k/v/o prefill on BF16 tensor cores (`ops::DenseBf16mma::fp8`,
//! lever `ATLAS_DENSE_PREFILL_BF16MMA=1`), as the GDN projections take it
//! (`qwen3_ssm/prefill_bf16mma.rs`): BF16 activations times the exact FP8
//! weights, where the default arm quantizes the activations to per-128 FP8.
//! `ATLAS_DENSE_PREFILL_LOWP` bits `qkv8` / `o8` select W8A8 (E4M3 tokens) on
//! the transposed `[K, N]` prefill copies.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, QuantWeight};

/// `fp8_bf16mma_ok` slot of the o projection (q, k, v are 0, 1, 2).
const O: usize = 3;

impl Qwen3AttentionLayer {
    fn fp8_slots(&self) -> [Option<&Fp8Weight>; 4] {
        [
            &self.q_weight,
            &self.k_weight,
            &self.v_weight,
            &self.o_weight,
        ]
        .map(|w| w.as_ref().and_then(|w| w.as_fp8()))
    }

    /// Load time, after `set_fp8_weights`: with the lever on, record which
    /// native FP8 projections have a block-scale grid constant along K (the
    /// kernel applies it per output feature). No-op without the lever.
    pub fn prepare_fp8_bf16mma(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if !ops::DenseBf16mma::lever() {
            return Ok(());
        }
        let mut ok = [false; 4];
        for (slot, w) in self.fp8_slots().into_iter().enumerate() {
            ok[slot] = match w {
                Some(w) => ops::fp8_scale_k_uniform(gpu, w)?,
                None => false,
            };
        }
        self.fp8_bf16mma_ok = ok;
        Ok(())
    }

    /// The projection whose native FP8 weight is `weight`, on BF16 tensor
    /// cores, when this path serves `m` rows of it. `None`: not FP8, the
    /// lever off, or shapes / scale grid it does not take.
    pub(super) fn try_fp8_bf16mma(
        &self,
        ctx: &ForwardContext,
        weight: Option<&QuantWeight>,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        let w = weight?.as_fp8()?;
        if !ctx.levers.dense_prefill_bf16mma {
            return None;
        }
        // The slot by weight identity: each projection owns its device bytes.
        let slot = self
            .fp8_slots()
            .iter()
            .position(|s| s.is_some_and(|s| s.weight == w.weight))?;
        if !self
            .bf16mma
            .serves_fp8(w, self.fp8_bf16mma_ok[slot], m, ctx.levers)
        {
            return None;
        }
        let wt = [
            &self.q_fp8w_t,
            &self.k_fp8w_t,
            &self.v_fp8w_t,
            &self.o_fp8w_t,
        ][slot];
        let lowp = ops::LowpArgs {
            group: if slot == O {
                ops::LowpMix::ATTN_O_A8
            } else {
                ops::LowpMix::ATTN_QKV_A8
            },
            codes: ctx.buffers.ffn_act_a(),
            scales: ctx.buffers.ffn_act_scale(),
        };
        Some(
            self.bf16mma
                .fp8(ctx.gpu, input, w, wt.as_ref(), out, m, lowp, stream),
        )
    }
}
