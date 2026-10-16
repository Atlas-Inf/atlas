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

impl Exl3CoopMk {
    /// Build the tables for `routed` experts plus `shared` (appended as the
    /// last local expert). Every routed expert must be present (no expert
    /// parallel split on this path). `hidden` = H = Hi = Ho, `inter` = I.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        gpu: &dyn GpuBackend,
        routed: &[(&Exl3Weight, &Exl3Weight, &Exl3Weight)],
        shared: (&Exl3Weight, &Exl3Weight, &Exl3Weight),
        hidden: usize,
        inter: usize,
        top_k: usize,
    ) -> Result<Self> {
        let kernels = Exl3CoopMkKernels::resolve(gpu)?;
        let n_local = routed.len() + 1;
        let slots = top_k + 1;
        if !hidden.is_multiple_of(128) || !inter.is_multiple_of(128) {
            bail!("CoopMK: hidden {hidden} / intermediate {inter} must be multiples of 128");
        }
        if slots > 256 || slots > COOPMK_THREADS as usize {
            bail!("CoopMK: {slots} slots exceed the kernel's 256-slot bound");
        }
        let all: Vec<(&Exl3Weight, &Exl3Weight, &Exl3Weight)> = routed
            .iter()
            .copied()
            .chain(std::iter::once(shared))
            .collect();
        let mut tabs: [Vec<u64>; 9] = Default::default();
        let (mut kg, mut ku, mut kd) = (Vec::new(), Vec::new(), Vec::new());
        let (mut kset_a, mut kset_b) = (0u32, 0u32);
        for (e, (g, u, d)) in all.iter().enumerate() {
            if g.is_null() || u.is_null() || d.is_null() {
                bail!("CoopMK: expert {e} has no packed weights");
            }
            let ok_gu =
                |w: &Exl3Weight| w.shape.in_features == hidden && w.shape.out_features == inter;
            if !ok_gu(g)
                || !ok_gu(u)
                || d.shape.in_features != inter
                || d.shape.out_features != hidden
            {
                bail!(
                    "CoopMK: expert {e} shapes gate {}x{} up {}x{} down {}x{} (in x out) do not match H {hidden} I {inter}",
                    g.shape.in_features,
                    g.shape.out_features,
                    u.shape.in_features,
                    u.shape.out_features,
                    d.shape.in_features,
                    d.shape.out_features
                );
            }
            let (bg, bu, bd) = (bits_ok(g)?, bits_ok(u)?, bits_ok(d)?);
            kset_a |= (1 << bg) | (1 << bu);
            kset_b |= 1 << bd;
            kg.push(bg as i32);
            ku.push(bu as i32);
            kd.push(bd as i32);
            for (t, p) in tabs.iter_mut().zip([
                g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh,
            ]) {
                t.push(p.0);
            }
        }

        let mut owned: Vec<DevicePtr> = Vec::new();
        let mut alloc = |bytes: usize| -> Result<DevicePtr> {
            let p = gpu.alloc(bytes.max(16))?;
            gpu.memset(p, 0, bytes.max(16))?;
            owned.push(p);
            Ok(p)
        };
        let mut tab_ptrs = [DevicePtr::NULL; 9];
        for (i, t) in tabs.iter().enumerate() {
            let p = alloc(t.len() * 8)?;
            gpu.copy_h2d(&u64_bytes(t), p)?;
            tab_ptrs[i] = p;
        }
        let ktab_host: Vec<u8> = kg
            .iter()
            .chain(ku.iter())
            .chain(kd.iter())
            .flat_map(|k| k.to_le_bytes())
            .collect();
        let k_tab = alloc(ktab_host.len())?;
        gpu.copy_h2d(&ktab_host, k_tab)?;

        let (h, i) = (hidden, inter);
        // Scratch for up to COOPMK_ROWS_MAX rows (bsz 1 uses the first row's
        // slice; the kernels index by slot / row, never by the maxima).
        let rows_max = if slots * COOPMK_ROWS_MAX <= 256 {
            COOPMK_ROWS_MAX
        } else {
            1
        };
        let slots_max = slots * rows_max;
        let xh = alloc(rows_max * h * 2)?;
        let sel = alloc(slots_max * 8)?;
        let rw = alloc(slots_max * 2)?;
        let had_g = alloc(slots_max * h * 2)?;
        let had_u = alloc(slots_max * h * 2)?;
        let gu_g = alloc(slots_max * i * 2)?;
        let gu_u = alloc(slots_max * i * 2)?;
        let act_out = alloc(slots_max * i * 2)?;
        let d_out = alloc(slots_max * h * 4)?;
        let ctr_a_len = slots_max * (i / 128);
        let ctr_b_len = rows_max * (h / 128);
        // exl3_moe_coop_ctr_len: counters + run table.
        let ctr_len = ctr_a_len + ctr_b_len + 2 + (slots_max + 1) + slots_max;
        let ctr = alloc(ctr_len * 4)?;
        let out_f32 = alloc(rows_max * h * 4)?;
        let rows_cap = rows_max;
        let parity_out = alloc(h * 2)?;

        let dbg = std::env::var("ATLAS_EXL3_COOPMK_DBG")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        let params = MoeCoopParams {
            x: xh.0,
            x_stride: h as i32,
            sel: sel.0,
            rw: rw.0,
            bsz: 1,
            topk: slots as i32,
            h: h as i32,
            hi: h as i32,
            i: i as i32,
            ho: h as i32,
            h_out: h as i32,
            min_expert: -1,
            max_expert: -1,
            g_trellis: tab_ptrs[0].0,
            g_suh: tab_ptrs[1].0,
            g_svh: tab_ptrs[2].0,
            u_trellis: tab_ptrs[3].0,
            u_suh: tab_ptrs[4].0,
            u_svh: tab_ptrs[5].0,
            d_trellis: tab_ptrs[6].0,
            d_suh: tab_ptrs[7].0,
            d_svh: tab_ptrs[8].0,
            act: ACT_SILU,
            act_limit: 0.0,
            gated: 1,
            had_g: had_g.0,
            had_u: had_u.0,
            a_global: 0,
            gu_g: gu_g.0,
            gu_u: gu_u.0,
            // Qwen4Exp (Flash Next) interm_dtype is fp16 in the fork.
            gu_f32: 0,
            act_out: act_out.0,
            d_out: d_out.0,
            ctr_a: ctr.0,
            ctr_b: ctr.0 + (ctr_a_len * 4) as u64,
            ctr_a_len: ctr_a_len as i32,
            ctr_b_len: ctr_b_len as i32,
            ksplit_a: 1,
            ksplit_b: 1,
            dbg,
            runs: ctr.0 + ((ctr_a_len + ctr_b_len) * 4) as u64,
            slots_max: slots_max as i32,
            rows_max: rows_max as i32,
            n_local: n_local as i32,
            sh_gate_n: 0,
            out: out_f32.0,
            out_stride: h as i32,
            sh_out: 0,
            sh_gate_w: 0,
            ..Default::default()
        };

        let plan = plan_from_env();
        let launches_a = stage_variants(&kernels, plan, kset_a, true, false);
        let launches_b = stage_variants(&kernels, plan, kset_b, false, false);
        let launches_a_w = stage_variants(&kernels, plan, kset_a, true, true);
        let launches_b_w = stage_variants(&kernels, plan, kset_b, false, true);
        // Upstream geometry at bsz 1, narrow tile (pick_wide on Blackwell:
        // kslices >= 256, or >= 128 with >= 32 slots — neither holds here).
        let grid_a = (slots as u32) * 2 * (i as u32 / COOPMK_COLS);
        let grid_b = (slots as u32) * (h as u32 / COOPMK_COLS);
        // smem_a_bytes_mk(Hi) = Hi*2 + WK*ROWS*COLS*4 + WK*STAGE_WORDS*4
        let red = 16 * 8 * 32 * 4;
        let stage = 16 * 128 * 4;
        let part = 16 * 128 * 4;
        let smem_a = (h * 2 + red + stage) as u32;
        let smem_b = (red + stage + part) as u32;
        Ok(Self {
            kernels,
            params,
            k_tab,
            xh,
            sel,
            rw,
            out_f32,
            parity_out,
            top_k: top_k as u32,
            hidden: h as u32,
            inter: i as u32,
            rows_max: rows_cap,
            shared_local: routed.len() as u32,
            launches_a,
            launches_b,
            launches_a_w,
            launches_b_w,
            smem_a,
            smem_b,
            grid_a,
            grid_b,
            kset_a,
            kset_b,
            owned,
        })
    }

    pub fn describe(&self) -> String {
        let names = |v: &[(KernelHandle, &'static str)]| {
            v.iter().map(|x| x.1).collect::<Vec<_>>().join("+")
        };
        let ks = |m: u32| {
            (1..=8)
                .filter(|k| m & (1 << k) != 0)
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "A[{}] K{{{}}} grid {} smem {} | B[{}] K{{{}}} grid {} smem {} | {} local experts, {} slots",
            names(&self.launches_a),
            ks(self.kset_a),
            self.grid_a,
            self.smem_a,
            names(&self.launches_b),
            ks(self.kset_b),
            self.grid_b,
            self.smem_b,
            self.params.n_local,
            self.params.topk
        )
    }

    /// One token: `x_bf16` [H] MoE input, `indices`/`weights` the router's
    /// top-k (u32 / f32 on device), `shared_gate_w` bf16 [H] (or NULL: weight
    /// 1). Writes bf16 [H] to `out_bf16`. Launch-only, no host sync.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &dyn GpuBackend,
        x_bf16: DevicePtr,
        indices: DevicePtr,
        weights: DevicePtr,
        shared_gate_w: DevicePtr,
        out_bf16: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden;
        KernelLaunch::new(gpu, self.kernels.prep)
            .grid([div_ceil(h, 256).min(64), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x_bf16)
            .arg_ptr(self.xh)
            .arg_i32(h as i32)
            .arg_ptr(indices)
            .arg_ptr(weights)
            .arg_ptr(self.sel)
            .arg_ptr(self.rw)
            .arg_i32(self.top_k as i32)
            .arg_ptr(shared_gate_w)
            .arg_i32(self.shared_local as i32)
            .launch(stream)?;
        let bytes = self.params.as_bytes();
        let (wa, wb) = wide_for(h, self.inter, self.top_k + 1);
        let (la, ga) = if wa {
            (
                &self.launches_a_w,
                (self.top_k + 1) * 2 * (self.inter / 128),
            )
        } else {
            (&self.launches_a, self.grid_a)
        };
        let (lb, gb) = if wb {
            (&self.launches_b_w, (self.top_k + 1) * (h / 128))
        } else {
            (&self.launches_b, self.grid_b)
        };
        for (k, _) in la {
            KernelLaunch::new(gpu, *k)
                .grid([ga, 1, 1])
                .block([COOPMK_THREADS, 1, 1])
                .shared_mem(self.smem_a)
                .arg_bytes(bytes)
                .arg_ptr(self.k_tab)
                .launch(stream)?;
        }
        for (k, _) in lb {
            KernelLaunch::new(gpu, *k)
                .grid([gb, 1, 1])
                .block([COOPMK_THREADS, 1, 1])
                .shared_mem(self.smem_b)
                .arg_bytes(bytes)
                .arg_ptr(self.k_tab)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, self.kernels.f32_to_bf16)
            .grid([div_ceil(h, 256).min(64), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.out_f32)
            .arg_ptr(out_bf16)
            .arg_i32(h as i32)
            .launch(stream)
    }
}

impl Exl3CoopMk {
    /// `m` rows in one call (upstream `CoopMK::run` at bsz = m): `x_bf16`
    /// [m, H], `indices` / `weights` [m, top_k] (u32 / f32, contiguous),
    /// `out_bf16` [m, H]. `m == 1` is exactly [`Self::run`]. For m >= 2 the
    /// rotation pre-kernel writes every slot's rotated input to `had_g/had_u`
    /// and builds the (expert, slot) run table, then kernel A/B run each run of
    /// up to 8 slots that share an expert as the rows of one MMA: the expert's
    /// trellis is read once per run instead of once per (row, slot). Per row the
    /// arithmetic is the bsz-1 arithmetic (same rotation, same k-split and
    /// reduction order; MMA rows do not mix). Launch-only, no host sync.
    #[allow(clippy::too_many_arguments)]
    pub fn run_rows(
        &self,
        gpu: &dyn GpuBackend,
        m: usize,
        x_bf16: DevicePtr,
        indices: DevicePtr,
        weights: DevicePtr,
        shared_gate_w: DevicePtr,
        out_bf16: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if m == 1 {
            return self.run(
                gpu,
                x_bf16,
                indices,
                weights,
                shared_gate_w,
                out_bf16,
                stream,
            );
        }
        if m == 0 || m > self.rows_max {
            bail!("CoopMK run_rows: {m} rows outside 1..={}", self.rows_max);
        }
        let h = self.hidden;
        KernelLaunch::new(gpu, self.kernels.prep_rows)
            .grid([m as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x_bf16)
            .arg_ptr(self.xh)
            .arg_i32(h as i32)
            .arg_ptr(indices)
            .arg_ptr(weights)
            .arg_ptr(self.sel)
            .arg_ptr(self.rw)
            .arg_i32(self.top_k as i32)
            .arg_ptr(shared_gate_w)
            .arg_i32(self.shared_local as i32)
            .launch(stream)?;
        let mut q = self.params;
        q.bsz = m as i32;
        q.a_global = 1;
        let slots = m as u32 * (self.top_k + 1);
        let nproj = 2u32;
        // Narrow tile for both stages (the only instances built; upstream's
        // pick_wide would switch stage A to the wide tile at >= 32 slots).
        let (wa, wb) = wide_for(h, self.inter, slots);
        let grid_a = slots * nproj * (self.inter / if wa { 128 } else { COOPMK_COLS });
        let grid_b = slots * (h / if wb { 128 } else { COOPMK_COLS });
        let la = if wa {
            &self.launches_a_w
        } else {
            &self.launches_a
        };
        let lb = if wb {
            &self.launches_b_w
        } else {
            &self.launches_b
        };
        let rot_items = slots * (h / 128) * nproj;
        let rot_grid = div_ceil(rot_items, COOPMK_THREADS / 32);
        let bytes = q.as_bytes();
        KernelLaunch::new(gpu, self.kernels.rot)
            .grid([rot_grid, 1, 1])
            .block([COOPMK_THREADS, 1, 1])
            .arg_bytes(bytes)
            .launch(stream)?;
        for (k, _) in la {
            KernelLaunch::new(gpu, *k)
                .grid([grid_a, 1, 1])
                .block([COOPMK_THREADS, 1, 1])
                .shared_mem(self.smem_a)
                .arg_bytes(bytes)
                .arg_ptr(self.k_tab)
                .launch(stream)?;
        }
        for (k, _) in lb {
            KernelLaunch::new(gpu, *k)
                .grid([grid_b, 1, 1])
                .block([COOPMK_THREADS, 1, 1])
                .shared_mem(self.smem_b)
                .arg_bytes(bytes)
                .arg_ptr(self.k_tab)
                .launch(stream)?;
        }
        let total = m as u32 * h;
        KernelLaunch::new(gpu, self.kernels.f32_to_bf16)
            .grid([div_ceil(total, 256).min(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.out_f32)
            .arg_ptr(out_bf16)
            .arg_i32(total as i32)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::MoeCoopParams;
    use std::mem::offset_of;

    #[test]
    fn layout_matches_c() {
        // Same numbers as the static_asserts in exl3_vendor/coopmk/moe_coop.cuh.
        assert_eq!(std::mem::size_of::<MoeCoopParams>(), 344);
        assert_eq!(offset_of!(MoeCoopParams, sel), 16);
        assert_eq!(offset_of!(MoeCoopParams, bsz), 32);
        assert_eq!(offset_of!(MoeCoopParams, max_expert), 64);
        assert_eq!(offset_of!(MoeCoopParams, g_trellis), 72);
        assert_eq!(offset_of!(MoeCoopParams, d_bias), 160);
        assert_eq!(offset_of!(MoeCoopParams, act), 168);
        assert_eq!(offset_of!(MoeCoopParams, gated), 176);
        assert_eq!(offset_of!(MoeCoopParams, had_g), 184);
        assert_eq!(offset_of!(MoeCoopParams, a_global), 200);
        assert_eq!(offset_of!(MoeCoopParams, gu_g), 208);
        assert_eq!(offset_of!(MoeCoopParams, gu_f32), 224);
        assert_eq!(offset_of!(MoeCoopParams, act_out), 232);
        assert_eq!(offset_of!(MoeCoopParams, ctr_a_len), 264);
        assert_eq!(offset_of!(MoeCoopParams, dbg), 280);
        assert_eq!(offset_of!(MoeCoopParams, runs), 288);
        assert_eq!(offset_of!(MoeCoopParams, sh_gate_n), 308);
        assert_eq!(offset_of!(MoeCoopParams, out), 312);
        assert_eq!(offset_of!(MoeCoopParams, out_stride), 320);
        assert_eq!(offset_of!(MoeCoopParams, sh_out), 328);
        assert_eq!(offset_of!(MoeCoopParams, sh_gate_w), 336);
    }
}
