// SPDX-License-Identifier: AGPL-3.0-only

//! GDN `in_proj_qkvz` / `out_proj` prefill on BF16 tensor cores (lever
//! `ATLAS_DENSE_PREFILL_BF16MMA=1`): `ops::DenseBf16mma::fp8` for native
//! per-tensor FP8 checkpoints (BF16 activations times the exact FP8 weights,
//! where the default arms quantize the activations to FP8), and
//! `ops::DenseBf16mma::nvfp4` on the transposed tables of NVFP4 ones. With
//! `ATLAS_DENSE_PREFILL_LOWP` the `qkvz` / `gout` bits give either format
//! E4M3 tokens (W8A8 / W4A8).

use super::*;

/// `bf16mma_fp8_ok` slots.
pub(super) const QKVZ: usize = 0;
pub(super) const OUT_PROJ: usize = 1;

impl Qwen3SsmLayer {
    /// Load time, after `set_fp8_decode_weights`: resolve the kernels when the
    /// lever is on and record which FP8 weights have a block-scale grid that
    /// is constant along K (the kernel applies it per output feature).
    pub fn prepare_dense_bf16mma(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if !ops::DenseBf16mma::lever() {
            return Ok(());
        }
        self.bf16mma = ops::DenseBf16mma::load(gpu);
        for (slot, w) in [(QKVZ, &self.qkvz_fp8w), (OUT_PROJ, &self.out_proj_fp8w)] {
            self.bf16mma_fp8_ok[slot] = match w {
                Some(w) if !self.fp8w_decode_only => ops::fp8_scale_k_uniform(gpu, w)?,
                _ => false,
            };
        }
        Ok(())
    }

    /// Runs the projection `slot` of `rows` rows if this path serves it.
    pub(super) fn try_bf16mma(
        &self,
        slot: usize,
        ctx: &ForwardContext,
        [input, out]: [DevicePtr; 2],
        rows: u32,
        stream: u64,
    ) -> Result<bool> {
        let (w, wt) = if slot == QKVZ {
            (&self.qkvz_fp8w, &self.qkvz_fp8w_t)
        } else {
            (&self.out_proj_fp8w, &self.out_proj_fp8w_t)
        };
        if !ctx.levers.dense_prefill_bf16mma {
            return Ok(false);
        }
        let lowp = ops::LowpArgs {
            group: if slot == QKVZ {
                ops::LowpMix::GDN_QKVZ
            } else {
                ops::LowpMix::GDN_OUT
            },
            codes: ctx.buffers.ffn_act_a(),
            scales: ctx.buffers.ffn_act_scale(),
        };
        if let Some(w) = w {
            if !self
                .bf16mma
                .serves_fp8(w, self.bf16mma_fp8_ok[slot], rows, ctx.levers)
            {
                return Ok(false);
            }
            self.bf16mma
                .fp8(ctx.gpu, input, w, wt.as_ref(), out, rows, lowp, stream)?;
            return Ok(true);
        }
        let h = ctx.config.hidden_size as u32;
        let value_dim =
            (ctx.config.linear_num_value_heads * ctx.config.linear_value_head_dim) as u32;
        let (wt, [n, k]) = if slot == QKVZ {
            (&self.qkvz_nvfp4_t, [ctx.config.ssm_qkvz_size() as u32, h])
        } else {
            (&self.out_proj_nvfp4_t, [h, value_dim])
        };
        let Some(wt) = wt else { return Ok(false) };
        if !self.bf16mma.serves_nvfp4(wt, [rows, n, k], ctx.levers) {
            return Ok(false);
        }
        self.bf16mma
            .nvfp4(ctx.gpu, input, wt, out, [rows, n, k], lowp, stream)?;
        Ok(true)
    }

    /// The GDN BA projection + gates (`ops::dense_gemm_ba_gates_prefill`'s
    /// arguments after the kernel), on tensor cores where this path serves it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ba_gates_prefill(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        ba: &DenseWeight,
        a_log: DevicePtr,
        dt_bias: DevicePtr,
        gate_out: DevicePtr,
        dims @ [_, n, k, ..]: [u32; 7],
        stream: u64,
    ) -> Result<()> {
        if ctx.levers.dense_prefill_bf16mma && self.bf16mma.serves_ba([n, k]) {
            return self.bf16mma.ba_gates(
                ctx.gpu,
                [input, ba.weight, a_log, dt_bias, gate_out],
                dims,
                stream,
            );
        }
        let [m, n, k, k_stride, gate_stride, nv, vpg] = dims;
        ops::dense_gemm_ba_gates_prefill(
            ctx.gpu,
            self.ba_gates_prefill_k,
            input,
            ba,
            a_log,
            dt_bias,
            gate_out,
            m,
            n,
            k,
            k_stride,
            gate_stride,
            nv,
            vpg,
            stream,
        )
    }
}
