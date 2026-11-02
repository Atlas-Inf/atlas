// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_W4A4=1` (with `ATLAS_QWEN4EXP_PREFILL_MOE=1`;
//! both ranks must agree, `startup_parity`): qwen4_exp's routed-MoE prefill
//! at the reference's numerics -- prompt rows on NVFP4 activations x the
//! checkpoint's NVFP4 weights, both as stored, on native FP4 tensor cores
//! (`kernels/gb10/qwen3.8-flash-next/nvfp4/moe_prefill_w4a4.cu`), the
//! activations quantized under the checkpoint's input scales
//! (`w4a4_scales.rs`) exactly as NVIDIA's b12x A4 prefill does:
//!
//! 1. `moe_w4a4_quant_input`: the token-major input to NVFP4 once (g1);
//! 2. `moe_w4a4_gate_up_silu`: gate + up, `bf16(bf16(silu(g)) * u)`, the
//!    activation to NVFP4 under the expert's g2;
//! 3. `moe_w4a4_down`: into `expert_down_out`.
//!
//! It takes the place of the q38 routed chain's E4M3 GEMMs (unset, that
//! chain runs as before). Same buffers and row order (expert-sorted rows;
//! scratch in `expert_up_out` / `expert_gate_out`, the down rows in
//! `expert_down_out`), so the router, the local unpermute, the SP pipes and
//! the reduce-scatter around it are unchanged. Remote experts (EP) have null
//! tables and are skipped. Row invariant: per-row quantization, one in-order
//! accumulator per output (`scripts/dev/qwen4exp_moe_w4a4_bench.sh check`).
//!
//! A layer without usable input scales (the MTP layer) runs the BF16 TC
//! kernels instead (`forward_prefill_tcp`: W4A16, the reference's arm for
//! non-calibrated layers), as does every layer under
//! `ATLAS_QWEN4EXP_PREFILL_MOE_BF16` / `_PREFILL_BF16_PROJ` (decode
//! numerics). Both read the checkpoint's N-major planes, so the startup
//! transpose pass builds no routed prefill copy (`skips_routed_transpose`):
//! 31.6 GiB a rank at TP=EP=2.
//!
//! GB10 (clock-capped as served), 256 local experts, 16000-token chunk
//! (81K local rows), one layer (`scripts/dev/qwen4exp_moe_w4a4_bench.sh
//! time`, SparkQwen's engine): W4A4 8.36 ms (quant 0.48, gate_up+SiLU 4.09,
//! down 3.79) against the E4M3 chain (q38 W2 on the N-major planes) 14.71
//! and the BF16 TC chain 20.25.

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

/// `ATLAS_QWEN4EXP_PREFILL_W4A4=1` with the q38 prefill arm it replaces the
/// routed chain of (`ATLAS_QWEN4EXP_PREFILL_MOE`). Read once.
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::q38_requested()
            && parse(std::env::var("ATLAS_QWEN4EXP_PREFILL_W4A4").ok().as_deref())
    })
}

fn parse(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Rows of a CTA tile and the gate/up and down column tiles of the kernels.
const W4_BM: u32 = 128;
const W4_GU_COLS: u32 = 64;
const W4_DN_COLS: u32 = 128;

/// Byte offset of the scale plane after a `bytes`-byte packed plane.
fn scales_at(bytes: usize) -> usize {
    bytes.next_multiple_of(256)
}

impl MoeLayer {
    /// The kernel half of [`Self::q38_routed_serves`]: under the switch the
    /// checkpoint planes (no K-major copy) at the BF16 TC kernels' shape (any
    /// layer may fall back to them), else the q38 E4M3 chain's shape and
    /// kernels.
    pub(super) fn routed_chain_kernels_serve(
        &self,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
    ) -> bool {
        if requested() {
            return self.gate_ptrs_t.is_none() && h == 2560 && inter == 640;
        }
        h.is_multiple_of(128)
            && inter.is_multiple_of(64)
            && crate::layers::try_kernel(ctx.gpu, "moe_prefill_q38", "moe_q38_down").0 != 0
    }

    /// The routed arms that read the checkpoint's planes, ahead of the q38
    /// E4M3 chain: the BF16 TC kernels under `_PREFILL_MOE_BF16`; under the
    /// switch W4A4 where the layer has input scales, else the TC kernels,
    /// which must then serve. `Ok(false)`, nothing launched: the q38 chain
    /// runs. `[n, rows]` = the chunk's tokens and routed rows; `dims` =
    /// (hidden, inter, num_experts, rows a grid row covers).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_checkpoint_routed_prefill(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        rows: [u32; 2],
        dims: (u32, u32, u32, u32),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (w4a4, tcp) = (requested(), super::tcp_requested());
        if !tcp
            && w4a4
            && self.try_w4a4_routed_prefill(
                expert_input,
                expert_offsets,
                sorted_token_ids,
                rows,
                dims,
                ctx,
                stream,
            )?
        {
            return Ok(true);
        }
        if !tcp && !w4a4 {
            return Ok(false);
        }
        let served = self.tcp_routed_prefill(
            expert_input,
            expert_offsets,
            sorted_token_ids,
            dims,
            ctx,
            stream,
        )?;
        anyhow::ensure!(
            served || !w4a4,
            "ATLAS_QWEN4EXP_PREFILL_W4A4: the target lacks qwen4exp_moe_tcp"
        );
        Ok(served)
    }

    /// The routed gate/up, SiLU and down of a prefill chunk on the W4A4
    /// kernels; `Ok(false)`, nothing launched, when the layer has no input
    /// scales. `[n, rows]` = the chunk's tokens and routed rows; `dims` =
    /// (hidden, inter, num_experts, rows a grid row covers).
    #[allow(clippy::too_many_arguments)]
    fn try_w4a4_routed_prefill(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        [n, rows]: [u32; 2],
        (h, inter, num_experts, rows_per_expert_grid): (u32, u32, u32, u32),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(s) = &self.w4a4 else {
            return Ok(false);
        };
        let gpu = ctx.gpu;
        let kernel = |name| crate::layers::try_kernel(gpu, "moe_prefill_w4a4", name);
        let (k_q, k_gu, k_dn) = (
            kernel("moe_w4a4_quant_input"),
            kernel("moe_w4a4_gate_up_silu"),
            kernel("moe_w4a4_down"),
        );
        anyhow::ensure!(
            k_q.0 != 0 && k_gu.0 != 0 && k_dn.0 != 0,
            "qwen4_exp W4A4 prefill: the target lacks moe_prefill_w4a4"
        );
        anyhow::ensure!(
            h.is_multiple_of(128) && inter.is_multiple_of(128),
            "qwen4_exp W4A4 prefill: hidden {h} / inter {inter} not multiples of 128"
        );
        // Scratch: the NVFP4 input (+ scales) in `expert_up_out`, the NVFP4
        // activation (+ scales) in `expert_gate_out` -- each well inside its
        // [rows, inter] BF16.
        let a_bytes = n as usize * h as usize / 2;
        let act_bytes = rows as usize * inter as usize / 2;
        let sizes = ctx.buffers.sizes();
        anyhow::ensure!(
            sizes.expert_up_out >= scales_at(a_bytes) + a_bytes / 8
                && sizes.expert_gate_out >= scales_at(act_bytes) + act_bytes / 8,
            "qwen4_exp W4A4 prefill: scratch too small for {n} tokens / {rows} rows"
        );
        let (a4, act) = (ctx.buffers.expert_up_out(), ctx.buffers.expert_gate_out());
        let (a4s, acts) = (
            a4.offset(scales_at(a_bytes)),
            act.offset(scales_at(act_bytes)),
        );
        let groups = n * h / 16;
        KernelLaunch::new(gpu, k_q)
            .grid([groups.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(expert_input)
            .arg_ptr(a4)
            .arg_ptr(a4s)
            .arg_f32(s.g1)
            .arg_u32(groups)
            .launch(stream)?;
        // The kernels stride over row tiles, so any grid computes every row.
        let grid_m = rows_per_expert_grid.div_ceil(W4_BM).max(1);
        KernelLaunch::new(gpu, k_gu)
            .grid([inter / W4_GU_COLS, grid_m, num_experts])
            .block([256, 1, 1])
            .arg_ptr(a4)
            .arg_ptr(a4s)
            .arg_ptr(s.gate.packed_ptrs)
            .arg_ptr(s.gate.scale_ptrs)
            .arg_ptr(s.gate.scale2_vals)
            .arg_ptr(s.up.packed_ptrs)
            .arg_ptr(s.up.scale_ptrs)
            .arg_ptr(s.up.scale2_vals)
            .arg_ptr(s.a2_gscale)
            .arg_ptr(act)
            .arg_ptr(acts)
            .arg_ptr(expert_offsets)
            .arg_ptr(sorted_token_ids)
            .arg_u32(num_experts)
            .arg_u32(inter)
            .arg_u32(h)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_dn)
            .grid([h / W4_DN_COLS, grid_m, num_experts])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_ptr(acts)
            .arg_ptr(s.down.packed_ptrs)
            .arg_ptr(s.down.scale_ptrs)
            .arg_ptr(s.down.scale2_vals)
            .arg_ptr(ctx.buffers.expert_down_out())
            .arg_ptr(expert_offsets)
            .arg_u32(num_experts)
            .arg_u32(h)
            .arg_u32(inter)
            .launch(stream)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{parse, scales_at};

    #[test]
    fn the_switch_is_on_only_at_1() {
        assert!(parse(Some("1")));
        for off in [
            None,
            Some("0"),
            Some(""),
            Some("true"),
            Some("on"),
            Some(" 1"),
        ] {
            assert!(!parse(off), "{off:?}");
        }
    }

    #[test]
    fn scale_planes_start_256_byte_aligned_after_the_codes() {
        assert_eq!(scales_at(0), 0);
        assert_eq!(scales_at(1), 256);
        assert_eq!(scales_at(16000 * 1280), 16000 * 1280);
    }
}
