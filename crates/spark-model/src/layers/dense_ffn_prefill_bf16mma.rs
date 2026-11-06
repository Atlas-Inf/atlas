// SPDX-License-Identifier: AGPL-3.0-only

//! Dense-FFN NVFP4 prefill on BF16 tensor cores (`ops::DenseBf16mma`, lever
//! `ATLAS_DENSE_PREFILL_BF16MMA=1`): gate+up with SiLU·mul in the epilogue,
//! then down, on the transposed `[K/2, N]` prefill tables. Two launches
//! where the W4A16 arm runs three GEMMs and a SiLU·mul, and BF16
//! activations where its tile kernel rounds them to E4M3.

use super::*;

impl DenseFfnLayer {
    /// Runs the whole prefill MLP into `moe_output` if this path serves the
    /// layer; `false` leaves it to the arms below.
    pub(super) fn try_prefill_bf16mma(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (Some(gate), Some(up), Some(down)) = (
            &self.weights.gate_proj_t,
            &self.weights.up_proj_t,
            &self.weights.down_proj_t,
        ) else {
            return Ok(false);
        };
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        if !ctx.levers.dense_prefill_bf16mma
            || !matches!(self.activation, FfnActivation::SiLU)
            || self.lora.is_some()
            || !self.bf16mma.serves_mlp([gate, up, down], [m, h, inter])
        {
            return Ok(false);
        }
        let act = ctx.buffers.expert_gate_out();
        let lowp = |group| ops::LowpArgs {
            group,
            codes: ctx.buffers.ffn_act_a(),
            scales: ctx.buffers.ffn_act_scale(),
        };
        self.bf16mma.gate_up_silu(
            ctx.gpu,
            input,
            [gate, up],
            act,
            [m, inter, h],
            lowp(ops::LowpMix::GATE_UP),
            stream,
        )?;
        self.bf16mma.nvfp4(
            ctx.gpu,
            act,
            down,
            ctx.buffers.moe_output(),
            [m, h, inter],
            lowp(ops::LowpMix::DOWN),
            stream,
        )?;
        Ok(true)
    }
}
