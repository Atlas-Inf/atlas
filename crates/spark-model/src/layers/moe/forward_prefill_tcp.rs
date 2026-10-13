// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_MOE_BF16=1` (with `ATLAS_QWEN4EXP_PREFILL_MOE=1`):
//! the routed-MoE prefill on tensor cores with the TC decode numerics
//! (`kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_moe_tcp.cu`), in place of
//! the q38 routed chain's three launches (`moe_q38_a_to_e4m3`,
//! `moe_q38[w]_gate_up_silu`, `moe_q38[w]_down`): BF16 weights
//! `lut * dec(scale)` (exact) with scale2 in the epilogue, BF16 activations,
//! SiLU * up rounded to BF16, and the TC decode kernels' per-output MMA
//! schedule, so a row's routed-expert bytes from prefill are the ones
//! `ATLAS_QWEN4EXP_MOE_TC` decode writes for it
//! (`scripts/dev/qwen4exp_moe_tcp_bench.sh check`). With
//! `ATLAS_QWEN4EXP_MOE_NO_CLAMP` neither applies the routed SwiGLU clamp;
//! without it both do.
//!
//! Same buffers and row order as the q38 chain (expert-sorted rows, the BF16
//! activation in `expert_gate_out`, the down rows in `expert_down_out`), so
//! the router, the shared expert, the local unpermute and the SP pipes around
//! it are unchanged (the shared expert keeps its own prefill path).

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

/// `ATLAS_QWEN4EXP_PREFILL_MOE_BF16=1`, implied by
/// `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ` (decode precision).
pub(crate) fn tcp_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::forward_prefill_routed::env_flag("ATLAS_QWEN4EXP_PREFILL_MOE_BF16")
            || crate::layers::ops::qwen4exp_rowinv::bf16_proj()
    })
}

/// Rows of a CTA's block (2 warps x 2 m16 tiles: TCP_GU_MT / TCP_DN_MT) and
/// outputs a CTA (gate/up 32 x {gate, up}, down 64) in qwen4exp_moe_tcp.cu.
const TCP_BLOCK_ROWS: u32 = 64;
const TCP_GU_OUT: u32 = 32;
const TCP_DN_OUT: u32 = 64;

impl MoeLayer {
    /// The q38 routed chain's work on the BF16 tensor-core kernels; `false`,
    /// nothing launched, when the switch is off or the shape or kernels are
    /// not this model's.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_tcp_routed_prefill(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        (h, inter, num_experts, rows_per_expert_grid): (u32, u32, u32, u32),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !tcp_requested() || h != 2560 || inter != 640 {
            return Ok(false);
        }
        let gpu = ctx.gpu;
        let nc = crate::model::qwen4exp_batch_fast::no_clamp_requested(&ctx.config.model_type);
        let k_gu = crate::layers::try_kernel(
            gpu,
            "qwen4exp_moe_tcp",
            if nc {
                "qwen4exp_moe_tcp_gate_up_nc"
            } else {
                "qwen4exp_moe_tcp_gate_up"
            },
        );
        let k_dn = crate::layers::try_kernel(gpu, "qwen4exp_moe_tcp", "qwen4exp_moe_tcp_down");
        if k_gu.0 == 0 || k_dn.0 == 0 {
            return Ok(false);
        }
        // The kernels stride their row blocks: any grid computes every row.
        let grid_m = rows_per_expert_grid.div_ceil(TCP_BLOCK_ROWS).max(1);
        let act = ctx.buffers.expert_gate_out();
        let (gp, up, dp) = (&self.gate_ptrs, &self.up_ptrs, &self.down_ptrs);
        KernelLaunch::new(gpu, k_gu)
            .grid([inter / TCP_GU_OUT, grid_m, num_experts])
            .block([128, 1, 1])
            .arg_ptr(expert_input)
            .arg_ptr(gp.packed_ptrs)
            .arg_ptr(gp.scale_ptrs)
            .arg_ptr(gp.scale2_vals)
            .arg_ptr(up.packed_ptrs)
            .arg_ptr(up.scale_ptrs)
            .arg_ptr(up.scale2_vals)
            .arg_ptr(act)
            .arg_ptr(expert_offsets)
            .arg_ptr(sorted_token_ids)
            .arg_u32(num_experts)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_dn)
            .grid([h / TCP_DN_OUT, grid_m, num_experts])
            .block([128, 1, 1])
            .arg_ptr(act)
            .arg_ptr(dp.packed_ptrs)
            .arg_ptr(dp.scale_ptrs)
            .arg_ptr(dp.scale2_vals)
            .arg_ptr(ctx.buffers.expert_down_out())
            .arg_ptr(expert_offsets)
            .arg_u32(num_experts)
            .launch(stream)?;
        Ok(true)
    }
}

/// The shared expert's one-expert tables for [`MoeLayer::try_tcp_shared`],
/// per layer (keyed by the gate weight): `[gate, gate_scale, up, up_scale,
/// down, down_scale]` u64, `[g2, u2, d2]` f32, then the `[0, n]` offsets.
static SHARED_TABLES: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());
/// Identity row ids `0..len` for the one-expert launch: `(ptr, len)`.
static IDENTITY_IDS: std::sync::Mutex<(u64, usize)> = std::sync::Mutex::new((0, 0));
const TABLE_BYTES: usize = 6 * 8 + 3 * 4 + 4;
const OFFSETS_AT: usize = 64;

impl MoeLayer {
    /// `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ`: the shared expert on the routed
    /// experts' TC prefill kernels (`qwen4exp_moe_tcp.cu`) as one expert over
    /// every row, in place of the q38 arm's E4M3 activations. Decode runs the
    /// shared expert as units of the same TC family (`qwen4exp_moe_c8_tc.cu`
    /// `C8_SHARED`, never clamped), whose per-output schedule these kernels
    /// reproduce, so a row's shared-expert bytes are its decode bytes.
    /// `out` = [activation scratch `[n, inter]`, down output `[n, h]`].
    /// `Ok(false)` launched nothing.
    pub(super) fn try_tcp_shared(
        &self,
        input: DevicePtr,
        (n, h, inter): (u32, u32, u32),
        out: [DevicePtr; 2],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let sh = &self.weights.shared_expert;
        if !crate::layers::ops::qwen4exp_rowinv::decode_active()
            || h != 2560
            || inter != 640
            || n == 0
            || sh.gate_proj.weight.0 == 0
            || self.lora.is_some()
        {
            return Ok(false);
        }
        let gpu = ctx.gpu;
        // Never clamped, like decode's shared units.
        let k_gu =
            crate::layers::try_kernel(gpu, "qwen4exp_moe_tcp", "qwen4exp_moe_tcp_gate_up_nc");
        let k_dn = crate::layers::try_kernel(gpu, "qwen4exp_moe_tcp", "qwen4exp_moe_tcp_down");
        anyhow::ensure!(
            k_gu.0 != 0 && k_dn.0 != 0,
            "ATLAS_QWEN4EXP_PREFILL_ROWINV: the TC prefill MoE kernels are missing"
        );
        let table = {
            let mut tables = SHARED_TABLES.lock().unwrap();
            let key = sh.gate_proj.weight.0;
            match tables.iter().find(|t| t.0 == key) {
                Some(t) => DevicePtr(t.1),
                None => {
                    let buf = gpu.alloc(TABLE_BYTES.max(OFFSETS_AT + 8))?;
                    let mut bytes = Vec::with_capacity(OFFSETS_AT);
                    for w in [&sh.gate_proj, &sh.up_proj, &sh.down_proj] {
                        bytes.extend_from_slice(&w.weight.0.to_le_bytes());
                        bytes.extend_from_slice(&w.weight_scale.0.to_le_bytes());
                    }
                    for w in [&sh.gate_proj, &sh.up_proj, &sh.down_proj] {
                        bytes.extend_from_slice(&w.weight_scale_2.to_le_bytes());
                    }
                    gpu.copy_h2d(&bytes, buf)?;
                    tables.push((key, buf.0));
                    buf
                }
            }
        };
        let ids = {
            let mut ids = IDENTITY_IDS.lock().unwrap();
            if ids.1 < n as usize {
                let len = (n as usize).next_power_of_two().max(4096);
                if ids.0 != 0 {
                    // Process-wide: any stream may still read the old one.
                    gpu.synchronize_device()?;
                    gpu.free(DevicePtr(ids.0))?;
                }
                let buf = gpu.alloc(len * 4)?;
                let bytes: Vec<u8> = (0..len as u32).flat_map(|i| i.to_le_bytes()).collect();
                gpu.copy_h2d(&bytes, buf)?;
                *ids = (buf.0, len);
            }
            DevicePtr(ids.0)
        };
        let offsets = table.offset(OFFSETS_AT);
        let span: Vec<u8> = [0u32, n].iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d_async_retained(&span, offsets, stream)?;
        let (p, f) = (
            |i: usize| table.offset(i * 8),
            |i: usize| table.offset(48 + i * 4),
        );
        let grid_m = n.div_ceil(TCP_BLOCK_ROWS);
        KernelLaunch::new(gpu, k_gu)
            .grid([inter / TCP_GU_OUT, grid_m, 1])
            .block([128, 1, 1])
            .arg_ptr(input)
            .arg_ptr(p(0))
            .arg_ptr(p(1))
            .arg_ptr(f(0))
            .arg_ptr(p(2))
            .arg_ptr(p(3))
            .arg_ptr(f(1))
            .arg_ptr(out[0])
            .arg_ptr(offsets)
            .arg_ptr(ids)
            .arg_u32(1)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_dn)
            .grid([h / TCP_DN_OUT, grid_m, 1])
            .block([128, 1, 1])
            .arg_ptr(out[0])
            .arg_ptr(p(4))
            .arg_ptr(p(5))
            .arg_ptr(f(2))
            .arg_ptr(out[1])
            .arg_ptr(offsets)
            .arg_u32(1)
            .launch(stream)?;
        Ok(true)
    }
}
