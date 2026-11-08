// SPDX-License-Identifier: AGPL-3.0-only
//! Byte-for-byte gate for `qwen35_gdn_verify_tree` (`ATLAS_QWEN38_VERIFY_FUSE`)
//! at Qwen3.8-27B GDN shapes (nk 16, nv 48, head dim 128, d_conv 4).
//!
//! GOLDEN is the `ATLAS_EXACT_VERIFY` GDN chain's step for one row, as
//! `decode_batched_conv_gdn_exact` issues it on the four-kernel FP32 arm
//! (the BA gates are its input): `causal_conv1d_update_l2norm_f32`,
//! `gated_delta_rule_decode_f32`, `gated_rms_norm_f32_input` — serial
//! decode's arithmetic. A row of a tree runs that step from its PARENT's
//! golden state (the state serial decode holds after the path ending at the
//! parent). The fused launch must give every row's normed output, every
//! rollback slot (H and conv windows after row t, t < k - 1) and the final
//! state (after the last row) BYTE-IDENTICAL to golden, for chains (NULL and
//! explicit parent tables) and for trees whose branches start at depth 1..4.
//!
//! NEGATIVE CONTROLS: (1) the fused launch from an initial H one ULP away in
//! one element must differ from golden — the comparison sees a 1-ULP input
//! change; (2) a tree's rows run as a chain (NULL table) must differ from the
//! tree's golden — the parent table is what makes a branch exact.
//!
//! It also times today's exact arm (one fused FP32 conv launch, then per row
//! the recurrence, the gated norm and an H copy to the rollback slot) against
//! the fused launch, per GDN layer.
//!
//! GPU REQUIRED. On the GB10, with the qwen3.8-27b kernel set:
//!   ATLAS_TARGET_MODEL=qwen3.8-27b cargo run --release -p spark-model \
//!       --features cuda,gpu-examples --example gdn_verify_tree_check
//! Exit codes: 0 = pass, 1 = FAIL, 2 = skipped (kernel not in this set).
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::layers::ops::qwen38_decode_fuse::GdnVerifyTree;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const D_CONV: usize = 4;
const KEY_DIM: usize = NK * KD;
const VALUE_DIM: usize = NV * VD;
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;
const QKVZ: usize = CONV_DIM + VALUE_DIM;
const H_ELEMS: usize = NV * KD * VD;
const CONV_ELEMS: usize = CONV_DIM * D_CONV;
const H_BYTES: usize = H_ELEMS * 4;
const CONV_BYTES: usize = CONV_ELEMS * 4;
const L2_EPS: f32 = 1e-6;
const EPS: f32 = 1e-6;
const MAX_ROWS: usize = 16;

struct Lcg(u64);
impl Lcg {
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
    }
    fn bf16(&mut self, n: usize, lo: f64, hi: f64) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f64(self.r(lo, hi)).to_bits().to_le_bytes())
            .collect()
    }
    fn f32(&mut self, n: usize, lo: f64, hi: f64) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (self.r(lo, hi) as f32).to_le_bytes())
            .collect()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len())?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// The rows of one verify: weights, token rows, gates and the initial state.
struct Case {
    qkvz: DevicePtr,  // [MAX_ROWS, QKVZ] BF16
    gates: DevicePtr, // [MAX_ROWS, 2 * NV] FP32: gate | beta
    conv_w: DevicePtr,
    norm_w: DevicePtr,
    h0: Vec<u8>,
    conv0: Vec<u8>,
}

impl Case {
    fn new(g: &dyn GpuBackend, seed: u64) -> Result<Self> {
        let mut r = Lcg(seed);
        let mut qkvz = Vec::with_capacity(MAX_ROWS * QKVZ * 2);
        for _ in 0..MAX_ROWS {
            qkvz.extend(r.bf16(CONV_DIM, -1.0, 1.0));
            qkvz.extend(r.bf16(VALUE_DIM, -3.0, 3.0));
        }
        let mut gates = Vec::with_capacity(MAX_ROWS * 2 * NV * 4);
        for _ in 0..MAX_ROWS {
            gates.extend(r.f32(NV, 0.05, 0.999));
            gates.extend(r.f32(NV, 0.05, 0.95));
        }
        Ok(Self {
            qkvz: up(g, &qkvz)?,
            gates: up(g, &gates)?,
            conv_w: up(g, &r.bf16(CONV_ELEMS, -0.5, 0.5))?,
            norm_w: up(g, &r.bf16(VD, 0.5, 1.5))?,
            h0: r.f32(H_ELEMS, -0.1, 0.1),
            conv0: r.f32(CONV_ELEMS, -0.5, 0.5),
        })
    }
    fn qkvz_row(&self, t: usize) -> DevicePtr {
        self.qkvz.offset(t * QKVZ * 2)
    }
    fn gate_row(&self, t: usize) -> DevicePtr {
        self.gates.offset(t * 2 * NV * 4)
    }
}

struct Kernels {
    conv: KernelHandle,
    conv_kn: KernelHandle,
    gdn: KernelHandle,
    norm: KernelHandle,
}

/// Every row's normed output, and the state after every row.
struct Rows {
    out: Vec<u8>,       // [k, VALUE_DIM] BF16
    h: Vec<Vec<u8>>,    // H after row t
    conv: Vec<Vec<u8>>, // conv state after row t
}

/// Golden: each row is the exact chain's step from its parent's state.
fn golden(g: &dyn GpuBackend, ks: &Kernels, c: &Case, parent: &[i32]) -> Result<Rows> {
    let k = parent.len();
    let w = |weight| DenseWeight { weight };
    let (h, conv) = (g.alloc(H_BYTES)?, g.alloc(CONV_BYTES)?);
    let scratch = g.alloc((CONV_DIM + VALUE_DIM) * 4)?;
    let out = g.alloc(k * VALUE_DIM * 2)?;
    let (mut hs, mut cs): (Vec<Vec<u8>>, Vec<Vec<u8>>) = (Vec::new(), Vec::new());
    for t in 0..k {
        let p = parent[t];
        if p < 0 {
            g.copy_h2d(&c.h0, h)?;
            g.copy_h2d(&c.conv0, conv)?;
        } else {
            g.copy_h2d(&hs[p as usize], h)?;
            g.copy_h2d(&cs[p as usize], conv)?;
        }
        let conv_out = scratch;
        ops::conv1d_update_l2norm(
            g,
            ks.conv,
            conv,
            c.qkvz_row(t),
            &w(c.conv_w),
            conv_out,
            CONV_DIM as u32,
            D_CONV as u32,
            1,
            (2 * KEY_DIM) as u32,
            KD as u32,
            L2_EPS,
            0,
        )?;
        let gdn_out = scratch.offset(CONV_DIM * 4);
        ops::gdn_decode(
            g,
            ks.gdn,
            h,
            conv_out,
            conv_out.offset(KEY_DIM * 4),
            conv_out.offset(2 * KEY_DIM * 4),
            c.gate_row(t),
            c.gate_row(t).offset(NV * 4),
            gdn_out,
            1,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            0,
        )?;
        ops::gated_rms_norm(
            g,
            ks.norm,
            gdn_out,
            c.qkvz_row(t).offset(CONV_DIM * 2),
            &w(c.norm_w),
            out.offset(t * VALUE_DIM * 2),
            NV as u32,
            VD as u32,
            VD as u32,
            EPS,
            VD as u32,
            0,
        )?;
        g.synchronize(0)?;
        hs.push(dn(g, h, H_BYTES)?);
        cs.push(dn(g, conv, CONV_BYTES)?);
    }
    Ok(Rows {
        out: dn(g, out, k * VALUE_DIM * 2)?,
        h: hs,
        conv: cs,
    })
}

/// Device buffers of one fused launch.
struct Fused {
    h: DevicePtr,
    conv: DevicePtr,
    h_snap: DevicePtr,
    conv_snap: DevicePtr,
    parent: DevicePtr,
    out: DevicePtr,
}

impl Fused {
    fn new(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            h: g.alloc(H_BYTES)?,
            conv: g.alloc(CONV_BYTES)?,
            h_snap: g.alloc(MAX_ROWS * H_BYTES)?,
            conv_snap: g.alloc(MAX_ROWS * CONV_BYTES)?,
            parent: g.alloc(MAX_ROWS * 4)?,
            out: g.alloc(MAX_ROWS * VALUE_DIM * 2)?,
        })
    }

    /// The parent table to launch with: uploaded, or NULL for a chain.
    fn table(&self, g: &dyn GpuBackend, parent: Option<&[i32]>) -> Result<DevicePtr> {
        Ok(match parent {
            Some(p) => {
                let b: Vec<u8> = p.iter().flat_map(|x| x.to_le_bytes()).collect();
                g.copy_h2d(&b, self.parent)?;
                self.parent
            }
            None => DevicePtr::NULL,
        })
    }

    fn launch(
        &self,
        g: &dyn GpuBackend,
        c: &Case,
        rows: usize,
        table: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let b = GdnVerifyTree {
            h_state: self.h,
            conv_state: self.conv,
            h_snap: self.h_snap,
            h_snap_bytes: H_BYTES,
            conv_snap: self.conv_snap,
            conv_snap_bytes: CONV_BYTES,
            parent: table,
            qkvz: c.qkvz,
            qkvz_stride: QKVZ as u32,
            conv_w: c.conv_w,
            gates: c.gates,
            norm_w: c.norm_w,
            out: self.out,
            rows: rows as u32,
        };
        let ran = ops::qwen38_decode_fuse::gdn_verify_tree(
            g,
            &b,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            D_CONV as u32,
            L2_EPS,
            EPS,
            stream,
        )?;
        ensure!(ran, "gdn_verify_tree declined the 27B shape");
        Ok(())
    }

    /// Run from (h0, conv0) and read back: outputs, slots 0..k-2, final state.
    fn run(
        &self,
        g: &dyn GpuBackend,
        c: &Case,
        h0: &[u8],
        k: usize,
        parent: Option<&[i32]>,
    ) -> Result<Rows> {
        g.copy_h2d(h0, self.h)?;
        g.copy_h2d(&c.conv0, self.conv)?;
        let table = self.table(g, parent)?;
        self.launch(g, c, k, table, 0)?;
        g.synchronize(0)?;
        let mut h = Vec::new();
        let mut conv = Vec::new();
        for t in 0..k - 1 {
            h.push(dn(g, self.h_snap.offset(t * H_BYTES), H_BYTES)?);
            conv.push(dn(g, self.conv_snap.offset(t * CONV_BYTES), CONV_BYTES)?);
        }
        h.push(dn(g, self.h, H_BYTES)?);
        conv.push(dn(g, self.conv, CONV_BYTES)?);
        Ok(Rows {
            out: dn(g, self.out, k * VALUE_DIM * 2)?,
            h,
            conv,
        })
    }
}

/// Byte mismatches of `got` against `gold` (the fused run's last row's state
/// is its final state; earlier rows' are its slots).
fn mismatches(gold: &Rows, got: &Rows) -> Vec<String> {
    let mut bad = Vec::new();
    if gold.out != got.out {
        let rows: Vec<usize> = (0..gold.h.len())
            .filter(|&t| {
                let r = t * VALUE_DIM * 2..(t + 1) * VALUE_DIM * 2;
                gold.out[r.clone()] != got.out[r]
            })
            .collect();
        bad.push(format!("normed out rows {rows:?}"));
    }
    for t in 0..gold.h.len() {
        if gold.h[t] != got.h[t] {
            bad.push(format!("H after row {t}"));
        }
        if gold.conv[t] != got.conv[t] {
            bad.push(format!("conv after row {t}"));
        }
    }
    bad
}

/// Today's exact arm for one layer (timing only): one fused FP32 conv launch
/// with its slots, then per row the recurrence, the gated norm and an H copy.
fn exact_arm(
    g: &dyn GpuBackend,
    ks: &Kernels,
    c: &Case,
    f: &Fused,
    k: usize,
    scratch: DevicePtr,
) -> Result<()> {
    let w = |weight| DenseWeight { weight };
    KernelLaunch::new(g, ks.conv_kn)
        .grid([CONV_DIM.div_ceil(256) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(f.conv)
        .arg_ptr(c.qkvz)
        .arg_ptr(c.conv_w)
        .arg_ptr(scratch)
        .arg_ptr(f.conv_snap)
        .arg_u32(k as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32((2 * KEY_DIM) as u32)
        .arg_u32(KD as u32)
        .arg_u32(QKVZ as u32)
        .arg_u32(QKVZ as u32)
        .arg_u32(CONV_ELEMS as u32)
        .arg_f32(L2_EPS)
        .launch(0)?;
    for t in 0..k {
        let row = scratch.offset(t * QKVZ * 4);
        let gdn_out = row.offset(CONV_DIM * 4);
        ops::gdn_decode(
            g,
            ks.gdn,
            f.h,
            row,
            row.offset(KEY_DIM * 4),
            row.offset(2 * KEY_DIM * 4),
            c.gate_row(t),
            c.gate_row(t).offset(NV * 4),
            gdn_out,
            1,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            0,
        )?;
        ops::gated_rms_norm(
            g,
            ks.norm,
            gdn_out,
            c.qkvz_row(t).offset(CONV_DIM * 2),
            &w(c.norm_w),
            f.out.offset(t * VALUE_DIM * 2),
            NV as u32,
            VD as u32,
            VD as u32,
            EPS,
            VD as u32,
            0,
        )?;
        if t + 1 < k {
            g.copy_d2d_async(f.h, f.h_snap.offset(t * H_BYTES), H_BYTES, 0)?;
        }
    }
    Ok(())
}

fn time_us(g: &dyn GpuBackend, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..5 {
        f()?;
    }
    g.synchronize(0)?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    g.synchronize(0)?;
    Ok(start.elapsed().as_secs_f64() * 1e6 / iters as f64)
}

fn chain(k: usize) -> Vec<i32> {
    (0..k as i32).map(|t| t - 1).collect()
}

fn main() -> Result<()> {
    // SAFETY: single-threaded here; nothing else reads or writes the env yet.
    unsafe { std::env::set_var("ATLAS_QWEN38_VERIFY_FUSE", "1") };
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b or qwen3.6-27b kernel set in this build");
    eprintln!("kernel set: {}", set.target.model);
    let gpu = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &gpu;
    if g.kernel("qwen35_decode_fuse", "qwen35_gdn_verify_tree")
        .is_err()
    {
        eprintln!("SKIPPED: qwen35_gdn_verify_tree is not in this kernel set");
        std::process::exit(2);
    }
    let ks = Kernels {
        conv: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32")?,
        conv_kn: g.kernel(
            "gdn_verify_fused_conv_kn_f32",
            "gdn_verify_fused_conv_kn_f32",
        )?,
        gdn: g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32")?,
        norm: g.kernel("norm", "gated_rms_norm_f32_input")?,
    };
    // Depth-first pre-order trees: branches at depths 1..4 (rows 8 and 11
    // branch off the root row, so their conv windows reach the pre-verify
    // state), a 16-row bush and a 16-row spine with hedges.
    let tree12: Vec<i32> = vec![-1, 0, 1, 2, 3, 2, 5, 1, 0, 8, 9, 0];
    let bush16: Vec<i32> = vec![-1, 0, 1, 2, 1, 4, 0, 6, 7, 6, 0, 10, 11, 12, 10, 0];
    let spine16: Vec<i32> = vec![-1, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0];
    let cases: Vec<(&str, Vec<i32>, bool)> = vec![
        ("chain k=2 (NULL)", chain(2), false),
        ("chain k=8 (NULL)", chain(8), false),
        ("chain k=16 (NULL)", chain(16), false),
        ("chain k=8 (table)", chain(8), true),
        ("tree12", tree12.clone(), true),
        ("bush16", bush16.clone(), true),
        ("spine16", spine16.clone(), true),
    ];
    let f = Fused::new(g)?;
    let mut ok = true;
    for seed in 0..4u64 {
        let c = Case::new(g, 0x7EE0_0000 + seed)?;
        for (name, parent, table) in &cases {
            let gold = golden(g, &ks, &c, parent)?;
            let got = f.run(g, &c, &c.h0, parent.len(), table.then_some(&parent[..]))?;
            let bad = mismatches(&gold, &got);
            println!(
                "seed {seed} {name:<18} rows {:>2}: {}",
                parent.len(),
                if bad.is_empty() {
                    "byte-identical".to_string()
                } else {
                    format!("MISMATCH {bad:?}")
                }
            );
            ok &= bad.is_empty();
            // Sanity: the step did something.
            ok &= got.h[parent.len() - 1] != c.h0;
        }
        // NEGATIVE 1: one ULP in one element of the initial H.
        let mut h1 = c.h0.clone();
        let at = (5 * KD * VD + 3 * VD + 7) * 4;
        let v = u32::from_le_bytes([h1[at], h1[at + 1], h1[at + 2], h1[at + 3]]) ^ 1;
        h1[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let gold = golden(g, &ks, &c, &tree12)?;
        let got = f.run(g, &c, &h1, tree12.len(), Some(&tree12))?;
        let seen = !mismatches(&gold, &got).is_empty();
        println!("seed {seed} NEGATIVE 1-ULP H0: detected={seen}");
        ok &= seen;
        // NEGATIVE 2: the tree's rows as a chain.
        let got = f.run(g, &c, &c.h0, tree12.len(), None)?;
        let seen = !mismatches(&gold, &got).is_empty();
        println!("seed {seed} NEGATIVE tree rows as a chain: detected={seen}");
        ok &= seen;
    }

    // Timing per GDN layer: today's exact arm vs the fused launch.
    let c = Case::new(g, 0x7EE0_1000)?;
    let scratch = g.alloc(MAX_ROWS * QKVZ * 4)?;
    for (k, parent) in [
        (2usize, None),
        (8, None),
        (16, None),
        (16, Some(&bush16[..])),
    ] {
        g.copy_h2d(&c.h0, f.h)?;
        g.copy_h2d(&c.conv0, f.conv)?;
        let arm = time_us(g, 200, || exact_arm(g, &ks, &c, &f, k, scratch))?;
        let table = f.table(g, parent)?;
        let fused = time_us(g, 200, || f.launch(g, &c, k, table, 0))?;
        println!(
            "rows {k:>2} {:<6} us per GDN layer: exact arm {arm:.1}, fused {fused:.1}; x48 layers = {:.2} ms saved per verify",
            if parent.is_some() { "bush" } else { "chain" },
            (arm - fused) * 48.0 / 1000.0
        );
    }

    if ok {
        println!(
            "PASS: qwen35_gdn_verify_tree is byte-identical to the exact chain (chains and trees); both negative controls detected"
        );
        Ok(())
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
