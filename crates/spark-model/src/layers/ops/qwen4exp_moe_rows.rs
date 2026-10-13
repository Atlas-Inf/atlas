// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_BATCH_FAST=1`: Qwen3.8-Flash-Next's expert-sorted row grid
//! for the originals-layout MoE decode kernels
//! (`kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_moe_rows.cu`).
//!
//! One plan launch sorts the rows' `(row, slot)` entries by expert and marks
//! the units (up to `QU_RMAX` entries of one expert), then ONE gate/up and
//! ONE silu/down launch. gate/up runs, per entry, the CTA body of
//! `moe_expert_gate_up_shared` in an order that puts every row's pick of one
//! expert, and every row's shared expert, back to back (the union of the
//! rows' experts streams from DRAM once). silu/down runs one CTA per
//! 64-output tile of a unit: the tile's weights are copied to shared once
//! and serve every row of the unit, each output the single-row kernel's
//! operation sequence. Every output byte is what one launch per row of
//! `moe_expert_{gate_up,silu_down}_shared` writes;
//! `scripts/dev/qwen4exp_batch_exact_bench.cu` checks each one.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::try_kernel;
use crate::weight_map::QuantizedWeight;

/// Entries one plan launch sorts (QU_SLOTS_MAX: the plan stages them in
/// shared memory).
pub const QWEN4EXP_MOE_ROWS_MAX_SLOTS: usize = 1024;

// The silu/down unit shape; each must match its define in
// kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_moe_rows.cu.
/// Entries of one expert a unit serves (QU_RMAX).
const QU_RMAX: u32 = 8;
/// Output rows per silu/down CTA (QU_SD_TILE).
pub const QWEN4EXP_MOE_ROWS_SD_TILE: u32 = 64;
/// silu/down warps per CTA (QU_SD_WARPS).
const SD_WARPS: u32 = 8;
/// Unit rows per activation chunk (QU_SD_RC).
const SD_RC: u32 = 2;
/// The only intermediate width the silu/down unit kernel is built for
/// (QU_INTER; the kernel traps on any other).
pub const QWEN4EXP_MOE_ROWS_SD_INTER: u32 = 640;

/// Rows a units launch takes (C8_ROWS_MAX in qwen4exp_moe_c8.cu).
pub const QWEN4EXP_MOE_UNITS_MAX_ROWS: usize = 64;
/// The units plan's workspace, bytes: 16 + 3 * (1024 + 64) unit words, then
/// 1024 sorted slot ids (C8_WS_ROWS + C8_SLOTS_MAX in qwen4exp_moe_c8.cu),
/// then TC v3's 529 unit counters (TC3_UMAX in qwen4exp_moe_c8_tc3.cu).
pub const QWEN4EXP_MOE_UNITS_WS_BYTES: usize = (16 + 3 * (1024 + 64) + 1024 + 529) * 4;
/// Units-grid rows of the TC v2 kernels (each CTA strides the units by it):
/// 80 x 64 gate/up and 40 x 64 down CTAs keep GB10's 48 SMs full; 48-160
/// measured within 1% at real C8 routing (qwen4exp_moe_c8_bench REALBIN).
const UNITS_TC_Y_CAP: u32 = 64;
/// Down outputs per units CTA (32 * OG in qwen4exp_moe_c8.cu).
const UNITS_DOWN_TILE: u32 = 64;

/// The kernel handles, all 0 unless the exact batching lane is on for a
/// qwen4_exp model and the target ships the module; `topk`/`blend` (the
/// common modules' multi-row twins of `moe_topk_softmax` and
/// `moe_weighted_sum_blend`) only under `ATLAS_QWEN4EXP_BATCH_SMALL=1` too,
/// `units_*` (qwen4exp_moe_c8.cu) only under `ATLAS_QWEN4EXP_MOE_UNITS=1`.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpMoeRows {
    pub plan: KernelHandle,
    pub gate_up: KernelHandle,
    pub silu_down: KernelHandle,
    pub topk: KernelHandle,
    pub blend: KernelHandle,
    pub units_plan: KernelHandle,
    pub units_gate_up: KernelHandle,
    pub units_down: KernelHandle,
    /// TC v3's fused gate/up + down (`units_gate_up`/`units_down` unused
    /// then) and its persistent grid width.
    pub units_fused: KernelHandle,
    pub units_ctas: u32,
    /// The units are the tensor-core kernels (`ATLAS_QWEN4EXP_MOE_TC=1`,
    /// contract (b)): serial decode must take them too.
    pub units_tc: bool,
    /// The units launches' grid height cap: the TC v2 kernels stride the
    /// units, so their grid is sized to the GPU (`UNITS_TC_Y_CAP`), not to
    /// the units' bound (`u32::MAX`: one unit a CTA).
    pub units_y_cap: u32,
}

impl Qwen4ExpMoeRows {
    pub const OFF: Self = Self {
        plan: KernelHandle(0),
        gate_up: KernelHandle(0),
        silu_down: KernelHandle(0),
        topk: KernelHandle(0),
        blend: KernelHandle(0),
        units_plan: KernelHandle(0),
        units_gate_up: KernelHandle(0),
        units_down: KernelHandle(0),
        units_fused: KernelHandle(0),
        units_ctas: 0,
        units_tc: false,
        units_y_cap: u32::MAX,
    };

    pub fn resolve(gpu: &dyn GpuBackend, config: &ModelConfig) -> Self {
        // Only looked up when wanted, so no other model or default boot asks.
        if !(crate::model::qwen4exp_batch_fast::requested()
            && config.model_type == "qwen4_exp"
            && !config.expert_tp)
        {
            return Self::OFF;
        }
        let m = "qwen4exp_moe_rows";
        let small = crate::model::qwen4exp_batch_fast::small_requested();
        let tc = crate::model::qwen4exp_batch_fast::tc_requested();
        // ATLAS_QWEN4EXP_MOE_NO_CLAMP: every silu/down and units entry without
        // the routed SwiGLU clamp, as serial decode's (`moe::init`).
        let nc = crate::model::qwen4exp_batch_fast::no_clamp_requested(&config.model_type);
        let units = tc || crate::model::qwen4exp_batch_fast::units_requested();
        let (tc, units) = Self::units_fit(config, tc, units);
        let off = KernelHandle(0);
        if tc {
            tracing::info!(
                "qwen4_exp MoE tensor-core units ON (ATLAS_QWEN4EXP_MOE_TC=1): verify rows AND \
                 serial decode on qwen4exp_moe_c8_tc.cu -- a new numerics baseline (contract (b))"
            );
        }
        // ATLAS_QWEN4EXP_MOE_TC_V1=1: the TC kernels as first shipped (O(n^2)
        // plan, a CTA a unit over the units' bound); ATLAS_QWEN4EXP_MOE_TC_V2=1:
        // the separate gate/up and down launches; else v3 (one persistent
        // launch, qwen4exp_moe_tc3.rs). All write the same bytes
        // (scripts/dev/qwen4exp_moe_c8_bench.sh tc-ident).
        let v1 = crate::model::qwen4exp_batch_fast::tc_v1_requested();
        let v3 = tc && !v1 && !crate::model::qwen4exp_batch_fast::tc_v2_requested();
        let (plan3, fused, units_ctas) = if v3 {
            Self::tc3_kernels(gpu, nc)
        } else {
            (off, off, 0)
        };
        let tck = |v2: &'static str, v1n: &'static str| {
            try_kernel(gpu, "qwen4exp_moe_c8_tc", if v1 { v1n } else { v2 })
        };
        let (units_plan, units_gate_up, units_down) = if v3 {
            (plan3, off, off)
        } else if tc {
            (
                tck("qwen4exp_moe_c8_tc_plan", "qwen4exp_moe_c8_tc_plan_v1"),
                if nc {
                    tck(
                        "qwen4exp_moe_c8_tc_gate_up_nc",
                        "qwen4exp_moe_c8_tc_gate_up_nc_v1",
                    )
                } else {
                    tck(
                        "qwen4exp_moe_c8_tc_gate_up",
                        "qwen4exp_moe_c8_tc_gate_up_v1",
                    )
                },
                tck("qwen4exp_moe_c8_tc_down", "qwen4exp_moe_c8_tc_down_v1"),
            )
        } else if units {
            (
                try_kernel(gpu, "qwen4exp_moe_c8", "qwen4exp_moe_c8_plan"),
                if nc {
                    try_kernel(gpu, "qwen4exp_moe_c8", "qwen4exp_moe_c8_gate_up_nc")
                } else {
                    try_kernel(gpu, "qwen4exp_moe_c8", "qwen4exp_moe_c8_gate_up")
                },
                try_kernel(gpu, "qwen4exp_moe_c8", "qwen4exp_moe_c8_down"),
            )
        } else {
            (off, off, off)
        };
        Self {
            plan: try_kernel(gpu, m, "qwen4exp_moe_rows_plan"),
            gate_up: try_kernel(gpu, m, "qwen4exp_moe_rows_gate_up"),
            silu_down: if nc {
                try_kernel(gpu, m, "qwen4exp_moe_rows_silu_down_nc")
            } else {
                try_kernel(gpu, m, "qwen4exp_moe_rows_silu_down")
            },
            topk: if small {
                try_kernel(gpu, "moe_topk", "moe_topk_softmax_rows")
            } else {
                off
            },
            blend: if small {
                try_kernel(gpu, "moe_expert_gemv", "moe_weighted_sum_blend_rows")
            } else {
                off
            },
            units_plan,
            units_gate_up,
            units_down,
            units_fused: fused,
            units_ctas,
            units_tc: tc,
            units_y_cap: if tc && !v1 { UNITS_TC_Y_CAP } else { u32::MAX },
        }
    }

    /// Whether [`Self::units`] can run (`ATLAS_QWEN4EXP_MOE_UNITS=1` and the
    /// target ships `qwen4exp_moe_c8`).
    pub fn units_ready(&self) -> bool {
        self.units_plan.0 != 0
            && (self.units_fused.0 != 0 || (self.units_gate_up.0 != 0 && self.units_down.0 != 0))
    }

    pub fn ready(&self) -> bool {
        self.plan.0 != 0 && self.gate_up.0 != 0 && self.silu_down.0 != 0
    }

    /// `moe_topk_softmax` on each of `rows` router rows (`logits_stride`
    /// BF16 apart) in one launch, `expert_indices`/`expert_weights`
    /// `[rows, top_k]`, each row the single-row kernel's bytes. `Ok(false)`,
    /// nothing launched, without the kernel: the caller loops.
    #[allow(clippy::too_many_arguments)]
    pub fn topk_rows(
        &self,
        gpu: &dyn GpuBackend,
        gate_logits: DevicePtr,
        expert_indices: DevicePtr,
        expert_weights: DevicePtr,
        (num_experts, top_k, rows, logits_stride): (u32, u32, u32, u32),
        normalize: bool,
        stream: u64,
    ) -> Result<bool> {
        if self.topk.0 == 0 {
            return Ok(false);
        }
        KernelLaunch::new(gpu, self.topk)
            .grid([rows, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(gate_logits)
            .arg_ptr(expert_indices)
            .arg_ptr(expert_weights)
            .arg_u32(num_experts)
            .arg_u32(top_k)
            .arg_u32(normalize as u32)
            .arg_u32(logits_stride)
            .launch(stream)?;
        Ok(true)
    }

    /// `moe_weighted_sum_blend` on each of `rows` rows in one launch: row
    /// `r` blends `expert_out` rows `[r * top_k, (r + 1) * top_k)` with
    /// `expert_weights` row `r` into `output` row `r`, gated by `input` row
    /// `r`, against the shared row at `shared_out + r * shared_stride`
    /// elements (0: one row for all; null reads +0.0). `Ok(false)`, nothing
    /// launched, without the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn blend_rows(
        &self,
        gpu: &dyn GpuBackend,
        output: DevicePtr,
        expert_out: DevicePtr,
        expert_weights: DevicePtr,
        (shared_out, shared_stride): (DevicePtr, u32),
        input: DevicePtr,
        gate_weight: DevicePtr,
        (hidden, top_k, rows): (u32, u32, u32),
        stream: u64,
    ) -> Result<bool> {
        if self.blend.0 == 0 {
            return Ok(false);
        }
        KernelLaunch::new(gpu, self.blend)
            .grid([div_ceil(hidden, 256), rows, 1])
            .block([256, 1, 1])
            .arg_ptr(output)
            .arg_ptr(expert_out)
            .arg_ptr(expert_weights)
            .arg_ptr(shared_out)
            .arg_ptr(input)
            .arg_ptr(gate_weight)
            .arg_u32(hidden)
            .arg_u32(top_k)
            .arg_u32(hidden)
            .arg_u32(shared_stride)
            .launch(stream)?;
        Ok(true)
    }

    /// `order[rank] = q` for the `slots` entries of `expert_indices`, ranked
    /// by `(expert id, q)`.
    pub fn plan(
        &self,
        gpu: &dyn GpuBackend,
        expert_indices: DevicePtr,
        order: DevicePtr,
        slots: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=QWEN4EXP_MOE_ROWS_MAX_SLOTS).contains(&slots),
            "qwen4exp MoE rows plan: {slots} entries"
        );
        KernelLaunch::new(gpu, self.plan)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(expert_indices)
            .arg_ptr(order)
            .arg_u32(slots as u32)
            .launch(stream)
    }

    /// `moe_expert_gate_up_shared` for `rows` rows of `input` (`[rows, k]`)
    /// in one launch: `gate_out`/`up_out` `[rows * top_k, n]` (row `r`'s
    /// `[top_k, n]` block at `r * top_k`), `sh_*_out` `[rows, n]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        gate_ptrs: (DevicePtr, DevicePtr, DevicePtr),
        gate_out: DevicePtr,
        up_ptrs: (DevicePtr, DevicePtr, DevicePtr),
        up_out: DevicePtr,
        expert_indices: DevicePtr,
        order: DevicePtr,
        sh_gate: &QuantizedWeight,
        sh_gate_out: DevicePtr,
        sh_up: &QuantizedWeight,
        sh_up_out: DevicePtr,
        (n, k, top_k, rows): (u32, u32, u32, u32),
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.gate_up)
            .grid([div_ceil(n, 8), rows * top_k + rows, 2])
            .block([128, 1, 1])
            .arg_ptr(input)
            .arg_ptr(gate_ptrs.0)
            .arg_ptr(gate_ptrs.1)
            .arg_ptr(gate_ptrs.2)
            .arg_ptr(gate_out)
            .arg_ptr(up_ptrs.0)
            .arg_ptr(up_ptrs.1)
            .arg_ptr(up_ptrs.2)
            .arg_ptr(up_out)
            .arg_ptr(expert_indices)
            .arg_ptr(order)
            .arg_ptr(sh_gate.weight)
            .arg_ptr(sh_gate.weight_scale)
            .arg_f32(sh_gate.weight_scale_2)
            .arg_ptr(sh_gate_out)
            .arg_ptr(sh_up.weight)
            .arg_ptr(sh_up.weight_scale)
            .arg_f32(sh_up.weight_scale_2)
            .arg_ptr(sh_up_out)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)
    }

    /// `moe_expert_silu_down_shared` for `rows` rows in one launch: `output`
    /// `[rows * top_k, n]`, `sh_down_out` `[rows, n]`, reading the gate/up
    /// rows [`Self::gate_up`] wrote. Needs `n` a multiple of
    /// [`QWEN4EXP_MOE_ROWS_SD_TILE`] and `k == QWEN4EXP_MOE_ROWS_SD_INTER`.
    #[allow(clippy::too_many_arguments)]
    pub fn silu_down(
        &self,
        gpu: &dyn GpuBackend,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_ptrs: (DevicePtr, DevicePtr, DevicePtr),
        output: DevicePtr,
        expert_indices: DevicePtr,
        order: DevicePtr,
        sh_gate_in: DevicePtr,
        sh_up_in: DevicePtr,
        sh_down: &QuantizedWeight,
        sh_down_out: DevicePtr,
        (n, k, top_k, rows): (u32, u32, u32, u32),
        stream: u64,
    ) -> Result<()> {
        ensure!(
            n.is_multiple_of(QWEN4EXP_MOE_ROWS_SD_TILE) && k == QWEN4EXP_MOE_ROWS_SD_INTER,
            "qwen4exp MoE rows silu/down: n {n}, k {k}"
        );
        // A grid row per QU_RMAX rows of the shared expert, then one per entry
        // (the plan's unit heads; the others exit).
        KernelLaunch::new(gpu, self.silu_down)
            .grid([
                n / QWEN4EXP_MOE_ROWS_SD_TILE,
                div_ceil(rows, QU_RMAX) + rows * top_k,
                1,
            ])
            .block([SD_WARPS * 32, 1, 1])
            .shared_mem(QWEN4EXP_MOE_ROWS_SD_TILE * (k / 2 + k / 16) + SD_RC * k * 4)
            .arg_ptr(gate_out)
            .arg_ptr(up_out)
            .arg_ptr(down_ptrs.0)
            .arg_ptr(down_ptrs.1)
            .arg_ptr(down_ptrs.2)
            .arg_ptr(output)
            .arg_ptr(expert_indices)
            .arg_ptr(order)
            .arg_ptr(sh_gate_in)
            .arg_ptr(sh_up_in)
            .arg_ptr(sh_down.weight)
            .arg_ptr(sh_down.weight_scale)
            .arg_f32(sh_down.weight_scale_2)
            .arg_ptr(sh_down_out)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)
    }

    /// [`Self::gate_up`] + [`Self::silu_down`] for `rows` rows through the
    /// expert units of `qwen4exp_moe_c8.cu`: the plan into `ws`
    /// ([`QWEN4EXP_MOE_UNITS_WS_BYTES`]), gate/up + SiLU into the (FP32; TC
    /// v3: BF16) activations `act` (`[rows * top_k + rows, inter]`), down into `output`
    /// (`[rows * top_k, h]`) and `sh_down_out` (`[rows, h]`) -- the bytes the
    /// rows pair writes there. The gate/up BF16 rows are not stored.
    #[allow(clippy::too_many_arguments)]
    pub fn units(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        gate_ptrs: (DevicePtr, DevicePtr, DevicePtr),
        up_ptrs: (DevicePtr, DevicePtr, DevicePtr),
        down_ptrs: (DevicePtr, DevicePtr, DevicePtr),
        expert_indices: DevicePtr,
        (ws, act): (DevicePtr, DevicePtr),
        (sh_gate, sh_up, sh_down): (&QuantizedWeight, &QuantizedWeight, &QuantizedWeight),
        output: DevicePtr,
        sh_down_out: DevicePtr,
        (h, inter, top_k, rows): (u32, u32, u32, u32),
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=QWEN4EXP_MOE_UNITS_MAX_ROWS as u32).contains(&rows)
                && rows * top_k <= QWEN4EXP_MOE_ROWS_MAX_SLOTS as u32
                && inter == QWEN4EXP_MOE_ROWS_SD_INTER
                && h == 2560
                && output.0 != 0
                && sh_down_out.0 != 0,
            "qwen4exp MoE units: rows {rows}, top_k {top_k}, inter {inter}, h {h}"
        );
        if self.units_fused.0 != 0 {
            let (w, sh) = ((gate_ptrs, up_ptrs, down_ptrs), (sh_gate, sh_up, sh_down));
            let (ins, outs) = ((input, expert_indices, ws, act), (output, sh_down_out));
            return self.units_tc3(gpu, ins, w, sh, outs, (top_k, rows), stream);
        }
        KernelLaunch::new(gpu, self.units_plan)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(expert_indices)
            .arg_ptr(ws)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)?;
        // A grid row per possible unit: one per entry, one per 16 shared rows
        // (the plan's count; rows past it exit) -- or, for kernels that stride
        // the units, the cap.
        let units = (rows * top_k + div_ceil(rows, 16)).min(self.units_y_cap);
        KernelLaunch::new(gpu, self.units_gate_up)
            .grid([inter / 8, units, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(gate_ptrs.0)
            .arg_ptr(gate_ptrs.1)
            .arg_ptr(gate_ptrs.2)
            .arg_ptr(up_ptrs.0)
            .arg_ptr(up_ptrs.1)
            .arg_ptr(up_ptrs.2)
            .arg_ptr(sh_gate.weight)
            .arg_ptr(sh_gate.weight_scale)
            .arg_f32(sh_gate.weight_scale_2)
            .arg_ptr(sh_up.weight)
            .arg_ptr(sh_up.weight_scale)
            .arg_f32(sh_up.weight_scale_2)
            .arg_ptr(ws)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(act)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.units_down)
            .grid([h / UNITS_DOWN_TILE, units, 1])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_ptr(down_ptrs.0)
            .arg_ptr(down_ptrs.1)
            .arg_ptr(down_ptrs.2)
            .arg_ptr(sh_down.weight)
            .arg_ptr(sh_down.weight_scale)
            .arg_f32(sh_down.weight_scale_2)
            .arg_ptr(ws)
            .arg_ptr(output)
            .arg_ptr(sh_down_out)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)
    }
}
