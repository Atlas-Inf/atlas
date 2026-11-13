// SPDX-License-Identifier: AGPL-3.0-only

//! CoopMK: per-expert runtime-K cooperative decode MoE on packed EXL3 trellis
//! (kernels/gb10/qwen3.8-flash-next/exl3/exl3_coopmk.cu, device code vendored
//! from vcruz305/exllamav3 @ 047ce72).
//!
//! Two stage kernels per layer read every selected expert's trellis directly:
//! A = gate/up GEMV + svh + SiLU·up + down-input rotation, B = down GEMV + svh +
//! fixed-order weighted sum. The bitrate of each expert projection comes from
//! a device table (`k_tab`), so there is no host sync and no reconstruct.
//!
//! The shared expert rides along as local expert `n_routed` (slot `top_k`),
//! its routing weight = sigmoid(x · shared_expert_gate) computed on device by
//! `exl3_coopmk_prep`. One A/B pair therefore produces the full MoE output.
//!
//! Upstream launcher: exllamav3_ext/quant/exl3_moe_coopmk.cu (`CoopMK::run`),
//! mirrored for bsz = 1 ([`Exl3CoopMk::run`], in-block input rotation) and
//! bsz = 2..=[`COOPMK_ROWS_MAX`] ([`Exl3CoopMk::run_rows`]: the rotation
//! pre-kernel builds the run table, then ONE A/B pair runs every (row, slot),
//! reading each selected expert once per run of up to 8 slots that picked it).

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::exl3::Exl3Weight;

/// `MOE_COOP_THREADS`.
pub const COOPMK_THREADS: u32 = 512;
/// `MOE_COOP_COLS` (narrow tile: 2 n-tiles x 16 columns per block group).
const COOPMK_COLS: u32 = 32;
/// `exl3_coopmk_ns::KM_REG`: K 2..4 decode in registers.
const KM_REG: u32 = (1 << 2) | (1 << 3) | (1 << 4);
/// `exl3_coopmk_ns::KM_STG`: K 1, 5..8 stage through shared memory.
const KM_STG: u32 = 0x1FE & !KM_REG;
/// `MOE_COOP_ACT_SILU`.
const ACT_SILU: i32 = 0;
/// Rows one call can carry (upstream `MAX_BSZN`). The scratch, counters and
/// run table are sized for it at load; slots = rows x (top_k + 1) <= 256.
pub const COOPMK_ROWS_MAX: usize = 8;

/// Byte-exact mirror of the C `MoeCoopParams` (exl3_vendor/coopmk/moe_coop.cuh),
/// passed to the kernels by value. The C side pins `sizeof == 344` and the
/// field offsets with static_asserts; `layout_matches_c` pins them here.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MoeCoopParams {
    pub x: u64,
    pub x_stride: i32,
    pub _pad0: i32,
    pub sel: u64,
    pub rw: u64,
    pub bsz: i32,
    pub topk: i32,
    pub h: i32,
    pub hi: i32,
    pub i: i32,
    pub ho: i32,
    pub h_out: i32,
    pub min_expert: i32,
    pub max_expert: i32,
    pub _pad1: i32,
    pub g_trellis: u64,
    pub g_suh: u64,
    pub g_svh: u64,
    pub u_trellis: u64,
    pub u_suh: u64,
    pub u_svh: u64,
    pub d_trellis: u64,
    pub d_suh: u64,
    pub d_svh: u64,
    pub g_bias: u64,
    pub u_bias: u64,
    pub d_bias: u64,
    pub act: i32,
    pub act_limit: f32,
    pub gated: u8,
    pub _pad2: [u8; 7],
    pub had_g: u64,
    pub had_u: u64,
    pub a_global: u8,
    pub _pad3: [u8; 7],
    pub gu_g: u64,
    pub gu_u: u64,
    pub gu_f32: u8,
    pub _pad4: [u8; 7],
    pub act_out: u64,
    pub d_out: u64,
    pub ctr_a: u64,
    pub ctr_b: u64,
    pub ctr_a_len: i32,
    pub ctr_b_len: i32,
    pub ksplit_a: i32,
    pub ksplit_b: i32,
    pub dbg: i32,
    pub _pad5: i32,
    pub runs: u64,
    pub slots_max: i32,
    pub rows_max: i32,
    pub n_local: i32,
    pub sh_gate_n: i32,
    pub out: u64,
    pub out_stride: i32,
    pub _pad6: i32,
    pub sh_out: u64,
    pub sh_gate_w: u64,
}

const _: () = assert!(std::mem::size_of::<MoeCoopParams>() == 344);

impl MoeCoopParams {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: repr(C), plain-old-data fields, explicit padding fields
        // (all initialized), so every byte of the struct is initialized.
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// The CoopMK entry points, resolved once from module `exl3_coopmk`.
#[derive(Clone, Copy)]
pub struct Exl3CoopMkKernels {
    pub a_all: KernelHandle,
    pub b_all: KernelHandle,
    pub a_reg: KernelHandle,
    pub b_reg: KernelHandle,
    pub a_stg: KernelHandle,
    pub b_stg: KernelHandle,
    pub a_allmb2: KernelHandle,
    pub b_allmb2: KernelHandle,
    pub a_regmb2: KernelHandle,
    pub b_regmb2: KernelHandle,
    /// WIDE = true instances (128-column tiles), same order as above.
    pub a_all_w: KernelHandle,
    pub b_all_w: KernelHandle,
    pub a_reg_w: KernelHandle,
    pub b_reg_w: KernelHandle,
    pub a_stg_w: KernelHandle,
    pub b_stg_w: KernelHandle,
    pub a_allmb2_w: KernelHandle,
    pub b_allmb2_w: KernelHandle,
    pub a_regmb2_w: KernelHandle,
    pub b_regmb2_w: KernelHandle,
    pub prep: KernelHandle,
    pub prep_rows: KernelHandle,
    pub rot: KernelHandle,
    pub f32_to_bf16: KernelHandle,
}

impl Exl3CoopMkKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: &str| gpu.kernel("exl3_coopmk", name);
        Ok(Self {
            a_all: k("exl3_coopmk_a_all")?,
            b_all: k("exl3_coopmk_b_all")?,
            a_reg: k("exl3_coopmk_a_reg")?,
            b_reg: k("exl3_coopmk_b_reg")?,
            a_stg: k("exl3_coopmk_a_stg")?,
            b_stg: k("exl3_coopmk_b_stg")?,
            a_allmb2: k("exl3_coopmk_a_allmb2")?,
            b_allmb2: k("exl3_coopmk_b_allmb2")?,
            a_regmb2: k("exl3_coopmk_a_regmb2")?,
            b_regmb2: k("exl3_coopmk_b_regmb2")?,
            a_all_w: k("exl3_coopmk_a_all_w")?,
            b_all_w: k("exl3_coopmk_b_all_w")?,
            a_reg_w: k("exl3_coopmk_a_reg_w")?,
            b_reg_w: k("exl3_coopmk_b_reg_w")?,
            a_stg_w: k("exl3_coopmk_a_stg_w")?,
            b_stg_w: k("exl3_coopmk_b_stg_w")?,
            a_allmb2_w: k("exl3_coopmk_a_allmb2_w")?,
            b_allmb2_w: k("exl3_coopmk_b_allmb2_w")?,
            a_regmb2_w: k("exl3_coopmk_a_regmb2_w")?,
            b_regmb2_w: k("exl3_coopmk_b_regmb2_w")?,
            prep: k("exl3_coopmk_prep")?,
            prep_rows: k("exl3_coopmk_prep_rows")?,
            rot: k("exl3_coopmk_rot")?,
            f32_to_bf16: k("exl3_coopmk_f32_to_bf16")?,
        })
    }
}

/// Build the per-layer CoopMK tables at load: `ATLAS_EXL3_COOPMK=1` (arm on
/// by default) or a toggle file configured (arm chosen at run time). Unset:
/// nothing is built and the interim path is byte-for-byte what it was.
pub fn coopmk_requested() -> bool {
    std::env::var("ATLAS_EXL3_COOPMK")
        .map(|v| v.trim() != "0")
        .unwrap_or(true)
        || std::env::var("ATLAS_EXL3_COOPMK_TOGGLE_FILE").is_ok()
}

/// `EXL3_COOPMK_PLAN` (upstream knob name kept under Atlas's prefix):
/// 1 one all-K launch per stage, 2 (default) split by decode kind (reg K2..4 /
/// staged K5..8), 3 all-K capped at 64 regs, 4 reg-capped + staged.
fn plan_from_env() -> u32 {
    std::env::var("ATLAS_EXL3_COOPMK_PLAN")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|p| (1..=4).contains(p))
        .unwrap_or(2)
}

/// `ATLAS_EXL3_COOPMK_WIDE`: 0 narrow (default), 1 wide both stages, 2 wide
/// stage A only, 3 upstream `pick_wide` per call (Blackwell rule).
fn wide_mode_from_env() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        match std::env::var("ATLAS_EXL3_COOPMK_WIDE")
            .as_deref()
            .map(str::trim)
        {
            Ok("1") => 1,
            Ok("a") | Ok("A") => 2,
            Ok("auto") => 3,
            _ => 0,
        }
    })
}

/// Upstream `pick_wide` on Blackwell: kslices >= 256, or >= 128 with >= 32 slots.
fn upstream_pick_wide(kslices: u32, slots: u32) -> bool {
    kslices >= 256 || (kslices >= 128 && slots >= 32)
}

/// (wide_a, wide_b) for a call with `slots` slots.
fn wide_for(hidden: u32, inter: u32, slots: u32) -> (bool, bool) {
    match wide_mode_from_env() {
        1 => (true, true),
        2 => (true, false),
        3 => (
            upstream_pick_wide(hidden / 16, slots),
            upstream_pick_wide(inter / 16, slots),
        ),
        _ => (false, false),
    }
}

/// Kernel variants a stage launches under `plan` for the bitrate set `kset`
/// (bit K), in order — upstream `CoopMK::stage_variants`.
fn stage_variants(
    k: &Exl3CoopMkKernels,
    plan: u32,
    kset: u32,
    stage_a: bool,
    wide: bool,
) -> Vec<(KernelHandle, &'static str)> {
    let has_reg = kset & KM_REG != 0;
    let has_stg = kset & KM_STG != 0;
    let pick = |a: KernelHandle, b: KernelHandle, aw: KernelHandle, bw: KernelHandle| match (
        stage_a, wide,
    ) {
        (true, false) => a,
        (false, false) => b,
        (true, true) => aw,
        (false, true) => bw,
    };
    let all = (pick(k.a_all, k.b_all, k.a_all_w, k.b_all_w), "all");
    let reg = (pick(k.a_reg, k.b_reg, k.a_reg_w, k.b_reg_w), "reg");
    let stg = (pick(k.a_stg, k.b_stg, k.a_stg_w, k.b_stg_w), "stg");
    let allmb2 = (
        pick(k.a_allmb2, k.b_allmb2, k.a_allmb2_w, k.b_allmb2_w),
        "allmb2",
    );
    let regmb2 = (
        pick(k.a_regmb2, k.b_regmb2, k.a_regmb2_w, k.b_regmb2_w),
        "regmb2",
    );
    match plan {
        1 => vec![all],
        3 => vec![allmb2],
        4 => {
            let mut v = Vec::new();
            if has_reg {
                v.push(regmb2);
            }
            if has_stg {
                v.push(stg);
            }
            v
        }
        _ => {
            if has_reg && has_stg {
                vec![reg, stg]
            } else if has_reg {
                vec![reg]
            } else {
                vec![stg]
            }
        }
    }
}

/// Per-layer CoopMK state: device pointer tables, bitrate table, scratch and
/// the static parameter block. Built once at load; no weight copies (the
/// tables hold the addresses of the store-owned trellis / suh / svh buffers).
pub struct Exl3CoopMk {
    pub kernels: Exl3CoopMkKernels,
    pub params: MoeCoopParams,
    pub k_tab: DevicePtr,
    /// fp16 [H] converted input row (= params.x).
    pub xh: DevicePtr,
    /// int64 [slots] selected experts (= params.sel).
    pub sel: DevicePtr,
    /// fp16 [slots] routing weights (= params.rw).
    pub rw: DevicePtr,
    /// f32 [H] MoE output row (= params.out).
    pub out_f32: DevicePtr,
    /// bf16 [H] second output for the parity debug switch.
    pub parity_out: DevicePtr,
    pub top_k: u32,
    pub hidden: u32,
    pub inter: u32,
    /// Rows [`Exl3CoopMk::run_rows`] accepts (scratch sized at load).
    pub rows_max: usize,
    /// Local index of the shared expert in the tables (= number of routed).
    pub shared_local: u32,
    pub launches_a: Vec<(KernelHandle, &'static str)>,
    pub launches_b: Vec<(KernelHandle, &'static str)>,
    pub launches_a_w: Vec<(KernelHandle, &'static str)>,
    pub launches_b_w: Vec<(KernelHandle, &'static str)>,
    pub smem_a: u32,
    pub smem_b: u32,
    pub grid_a: u32,
    pub grid_b: u32,
    pub kset_a: u32,
    pub kset_b: u32,
    /// Every device allocation made here (tables + scratch), freed on drop
    /// only by process exit (the packs live for the whole serve).
    pub owned: Vec<DevicePtr>,
}

fn bits_ok(w: &Exl3Weight) -> Result<u32> {
    let k = w.shape.bits;
    if !(1..=8).contains(&k) {
        bail!("CoopMK: bitrate {k} outside the integer range 1..8");
    }
    Ok(k)
}

fn u64_bytes(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[path = "exl3_coopmk_build.rs"]
mod build;
#[path = "exl3_coopmk_rows.rs"]
mod rows;
