// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_MOE_FAST=1`: Qwen3.8-Flash-Next's unified-layout MoE decode
//! pair (`kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_moe_decode.cu`).
//!
//! One launch covers 1..=4 rows and writes exactly what
//! `moe_expert_{gate_up,silu_down}_shared_t` (once per row), `_batch2_t` and
//! `_batch3_t` write, bit for bit: same buffers, same slot-major routed rows,
//! same token-major shared rows, zeros for a NULL (EP-remote or absent)
//! expert. A routed expert several rows picked is read once, and the shared
//! expert once for all rows. `scripts/dev/qwen4exp_moe_decode_bench.cu` checks
//! every output byte against the replaced kernels.
//!
//! Default off until A/B'd end to end on the pair. Not a startup-parity
//! setting: the outputs are bit-identical and no collective changes, so a
//! rank that differs only reorders its own work.

use std::sync::OnceLock;

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::try_kernel;
use crate::weight_map::QuantizedWeight;

/// Rows one launch serves (the kernel's `QX_ROWS_MAX`).
pub const QWEN4EXP_MOE_MAX_ROWS: usize = 4;
/// Threads per CTA; each owns 2 adjacent outputs, so a CTA spans 320 columns.
const BLOCK: u32 = 160;
const COLS: u32 = 320;
/// The only shapes the kernels are compiled for.
const HIDDEN: usize = 2560;
const INTER: usize = 640;
/// Routed slots a launch's per-warp ballot covers (the kernel's `QX_SLOTS_MAX`).
const MAX_SLOTS: usize = 64;

/// `ATLAS_QWEN4EXP_MOE_FAST=1`. Read once per process.
pub fn qwen4exp_moe_fast_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !cfg!(atlas_hip) && std::env::var("ATLAS_QWEN4EXP_MOE_FAST").as_deref() == Ok("1")
    })
}

/// The two kernel handles, both 0 unless the switch is on, the model is
/// `qwen4_exp` at the compiled shapes, and the target ships the module.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpMoeFast {
    pub gate_up: KernelHandle,
    pub silu_down: KernelHandle,
}

impl Qwen4ExpMoeFast {
    pub const OFF: Self = Self {
        gate_up: KernelHandle(0),
        silu_down: KernelHandle(0),
    };

    pub fn resolve(gpu: &dyn GpuBackend, config: &ModelConfig) -> Self {
        let wanted = qwen4exp_moe_fast_requested()
            && config.model_type == "qwen4_exp"
            && !config.expert_tp
            && config.hidden_size == HIDDEN
            && config.moe_intermediate_size == INTER
            && config.shared_expert_intermediate_size == INTER
            && config.num_experts_per_tok * QWEN4EXP_MOE_MAX_ROWS <= MAX_SLOTS;
        // Only looked up when wanted, so no other model or default boot asks.
        let fast = if wanted {
            Self {
                gate_up: try_kernel(gpu, "qwen4exp_moe_decode", "qwen4exp_moe_gate_up_t"),
                silu_down: try_kernel(gpu, "qwen4exp_moe_decode", "qwen4exp_moe_silu_down_t"),
            }
        } else {
            Self::OFF
        };
        if qwen4exp_moe_fast_requested() && !fast.ready() {
            static ONCE: OnceLock<()> = OnceLock::new();
            ONCE.get_or_init(|| {
                tracing::warn!(
                    "ATLAS_QWEN4EXP_MOE_FAST=1 ignored: needs qwen4_exp at hidden {HIDDEN} / \
                     MoE intermediate {INTER} and the qwen4exp_moe_decode kernels \
                     (model_type {}, hidden {}, moe inter {}, expert TP {})",
                    config.model_type,
                    config.hidden_size,
                    config.moe_intermediate_size,
                    config.expert_tp,
                );
            });
        }
        fast
    }

    pub fn ready(&self) -> bool {
        self.gate_up.0 != 0 && self.silu_down.0 != 0
    }
}

fn check(n: u32, rows: usize) -> Result<()> {
    ensure!(
        (1..=QWEN4EXP_MOE_MAX_ROWS).contains(&rows) && n.is_multiple_of(COLS),
        "qwen4exp MoE fast: {rows} rows / N {n} outside the compiled launch shape"
    );
    Ok(())
}

/// Gate+up over `rows` rows of `input` (`[rows, k]`), the transposed tables'
/// replacement for `moe_expert_gate_up_shared_t` / `_batch2_t` / `_batch3_t`.
/// Writes `gate_out`/`up_out` `[rows * top_k, n]` and `sh_*_out` `[rows, n]`.
#[allow(clippy::too_many_arguments)]
pub fn qwen4exp_moe_gate_up_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_t_ptrs: DevicePtr,
    gate_scale_t_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_t_ptrs: DevicePtr,
    up_scale_t_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up_t: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    rows: usize,
    stream: u64,
) -> Result<()> {
    check(n, rows)?;
    let rows = rows as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([n / COLS, rows * top_k + 1, 2])
        .block([BLOCK, 1, 1])
        .shared_mem(rows * k * 4)
        .arg_ptr(input)
        .arg_ptr(gate_packed_t_ptrs)
        .arg_ptr(gate_scale_t_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_t_ptrs)
        .arg_ptr(up_scale_t_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.weight_scale)
        .arg_f32(sh_gate_t.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.weight_scale)
        .arg_f32(sh_up_t.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(rows)
        .launch(stream)
}

/// SiLU(gate)*up then down over `rows` rows, the transposed tables'
/// replacement for `moe_expert_silu_down_shared_t` / `_batch2_t` /
/// `_batch3_t`. Writes `output` `[rows * top_k, n]` and `sh_down_out`
/// `[rows, n]`.
#[allow(clippy::too_many_arguments)]
pub fn qwen4exp_moe_silu_down_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_t_ptrs: DevicePtr,
    scale_t_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    rows: usize,
    stream: u64,
) -> Result<()> {
    check(n, rows)?;
    let rows = rows as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([n / COLS, rows * top_k + 1, 1])
        .block([BLOCK, 1, 1])
        .shared_mem(rows * k * 4)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_t_ptrs)
        .arg_ptr(scale_t_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.weight_scale)
        .arg_f32(sh_down_t.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(rows)
        .launch(stream)
}
