// SPDX-License-Identifier: AGPL-3.0-only
//! Row-invariance and accuracy gate for the `ATLAS_SKINNY_MMA` kernels
//! (`kernels/gb10/qwen3.6-27b/nvfp4/skinny_mma.cu`, `layers/ops/skinny_mma.rs`).
//!
//! 1. BYTES. On every served Qwen3.8-27B shape, an M-row call (M =
//!    1..=`MAX_ROWS`, past 16 rows several 16-row launches) must write
//!    exactly the bytes of M one-row launches of the same family, leave the
//!    row after the M-th untouched, and write identical rows for rows whose
//!    inputs are identical (rows 5, 12, 15, 70 repeat rows 0, 3, 9, 3).
//!    Two-weight launches must match each weight launched alone, and a
//!    strided launch the packed one. A 1-ULP change to one activation of row
//!    1 (its largest) must change row 1 and no other row (negative control).
//! 2. ACCURACY. Two rows of every shape against an FP64 host reference
//!    (the same dequantized weights), next to the GEMV serial decode runs
//!    today (`w4a16_gemv`, `w4a16_gemv_qg`, `w8a16_gemv`): RMS and max error
//!    relative to the output RMS, and the share of outputs that are not the
//!    correctly rounded BF16. The skinny kernel must be no worse (RMS).
//!
//! Exit: 0 pass, 1 a gate failed, 2 kernels absent.
//!
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example skinny_mma_microtest

use anyhow::Result;
use half::bf16;
use spark_model::layers::ops::skinny_mma::{self, Family, MAX_ROWS, SkinnyKernels};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const ROWS: usize = MAX_ROWS as usize;
const SCALE2: f32 = 0.0123;
const SENTINEL: u8 = 0xA5;
/// Rows that repeat an earlier row's input: (row, source).
const SHARED: [(usize, usize); 4] = [(5, 0), (12, 3), (15, 9), (70, 3)];
/// Rows checked against FP64.
const ACC_ROWS: usize = 2;

/// `(label, family, N, K, heads, head_dim)`; heads/head_dim only for Q+gate.
const SHAPES: &[(&str, Family, usize, usize, u32, u32)] = &[
    (
        "w4 gate/up  [17408 x 5120]",
        Family::W4a16,
        17408,
        5120,
        0,
        0,
    ),
    (
        "w4 down     [5120 x 17408]",
        Family::W4a16,
        5120,
        17408,
        0,
        0,
    ),
    (
        "w4 o_proj   [ 5120 x 6144]",
        Family::W4a16,
        5120,
        6144,
        0,
        0,
    ),
    (
        "w4 k/v      [ 1024 x 5120]",
        Family::W4a16,
        1024,
        5120,
        0,
        0,
    ),
    (
        "w4 lm_head [248077 x 5120]",
        Family::W4a16,
        248_077,
        5120,
        0,
        0,
    ),
    (
        "w4 tail     [10307 x 2560]",
        Family::W4a16,
        10307,
        2560,
        0,
        0,
    ),
    (
        "qg q+gate   [12288 x 5120]",
        Family::Qg,
        12288,
        5120,
        24,
        256,
    ),
    ("qg tail     [  500 x 1024]", Family::Qg, 500, 1024, 5, 50),
    (
        "w8 gdn qkvz [16384 x 5120]",
        Family::W8a16,
        16384,
        5120,
        0,
        0,
    ),
    (
        "w8 gdn out  [ 5120 x 6144]",
        Family::W8a16,
        5120,
        6144,
        0,
        0,
    ),
    (
        "w8 tail     [10308 x 2048]",
        Family::W8a16,
        10308,
        2048,
        0,
        0,
    ),
];

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn byte(&mut self) -> u8 {
        (self.f() * 256.0) as u8
    }
    /// Roughly normal, with a heavy tail (one value in 64 up to ~+-200).
    fn act(&mut self) -> f32 {
        let n = (0..4).map(|_| self.f()).sum::<f64>() - 2.0;
        let big = self.f() < 1.0 / 64.0;
        (n * if big { 120.0 } else { 1.2 }) as f32
    }
}

fn e4m3(b: u8) -> f64 {
    let (s, e, m) = (b >> 7, (b >> 3) & 0xF, b & 7);
    let v = if e == 0 {
        m as f64 / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}
const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
fn e2m1(code: u8) -> f64 {
    let v = E2M1[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

struct Shape {
    family: Family,
    n: usize,
    k: usize,
    heads: u32,
    hd: u32,
}

/// Host and device operands of one shape.
struct Ops {
    a_host: Vec<u16>,
    w_host: Vec<u8>,
    s_host: Vec<u8>, // E4M3 group scales, or FP32 block scales (LE bytes)
    a: DevicePtr,
    w: DevicePtr,
    s: DevicePtr,
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}
fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn a_bytes(a: &[u16]) -> Vec<u8> {
    a.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl Ops {
    fn new(g: &dyn GpuBackend, seed: u64, sh: &Shape) -> Result<Self> {
        let (n, k) = (sh.n, sh.k);
        let mut rng = Lcg(seed ^ (n * 31 + k) as u64);
        let mut a_host: Vec<u16> = (0..ROWS * k)
            .map(|_| bf16::from_f32(rng.act()).to_bits())
            .collect();
        for (dst, src) in SHARED {
            let row = a_host[src * k..(src + 1) * k].to_vec();
            a_host[dst * k..(dst + 1) * k].copy_from_slice(&row);
        }
        let (w_host, s_host): (Vec<u8>, Vec<u8>) = if sh.family == Family::W8a16 {
            // E4M3 codes without the NaNs (0x7F, 0xFF): a checkpoint has none.
            let w = (0..n * k)
                .map(|_| rng.byte())
                .map(|b| if b & 0x7F == 0x7F { b ^ 1 } else { b })
                .collect();
            let s = (0..n.div_ceil(128) * (k / 128))
                .flat_map(|_| ((0.002 + 0.01 * rng.f()) as f32).to_le_bytes())
                .collect();
            (w, s)
        } else {
            // Group scales 0x28..0x47: E4M3 0.25..3.75.
            let s = (0..n * k / 16)
                .map(|_| 0x28 + (rng.f() * 32.0) as u8)
                .collect();
            ((0..n * k / 2).map(|_| rng.byte()).collect(), s)
        };
        Ok(Self {
            a: up(g, &a_bytes(&a_host))?,
            w: up(g, &w_host)?,
            s: up(g, &s_host)?,
            a_host,
            w_host,
            s_host,
        })
    }
    fn nvfp4(&self) -> QuantizedWeight {
        QuantizedWeight {
            weight: self.w,
            weight_scale: self.s,
            weight_scale_2: SCALE2,
            ..QuantizedWeight::null()
        }
    }
    /// FP64 output `j` (deinterleaved column order) of activation row `t`.
    fn reference(&self, sh: &Shape, t: usize) -> Vec<f64> {
        let (n, k) = (sh.n, sh.k);
        let a: Vec<f64> = self.a_host[t * k..(t + 1) * k]
            .iter()
            .map(|&v| bf16::from_bits(v).to_f64())
            .collect();
        let mut out = vec![0f64; n];
        for r in 0..n {
            let mut acc = 0f64;
            if sh.family == Family::W8a16 {
                for (kk, &x) in a.iter().enumerate() {
                    let off = 4 * ((r / 128) * (k / 128) + kk / 128);
                    let bs =
                        f32::from_le_bytes(self.s_host[off..off + 4].try_into().unwrap()) as f64;
                    acc += x * e4m3(self.w_host[r * k + kk]) * bs;
                }
            } else {
                for (kk, &x) in a.iter().enumerate() {
                    let byte = self.w_host[r * k / 2 + kk / 2];
                    let code = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    acc += x * e2m1(code) * e4m3(self.s_host[r * k / 16 + kk / 16]);
                }
                acc *= SCALE2 as f64;
            }
            out[col(sh, r)] = acc;
        }
        out
    }
    fn free(&self, g: &dyn GpuBackend) {
        for p in [self.a, self.w, self.s] {
            g.free(p).ok();
        }
    }
}

fn col(sh: &Shape, r: usize) -> usize {
    if sh.family != Family::Qg {
        return r;
    }
    let (h, hd) = (sh.heads as usize, sh.hd as usize);
    let (head, idx) = (r / (2 * hd), r % (2 * hd));
    if idx < hd {
        head * hd + idx
    } else {
        h * hd + head * hd + idx - hd
    }
}

#[allow(clippy::too_many_arguments)]
/// One skinny launch of `m` rows from `a`; `parts` copies of the weight
/// (two-weight launches write part 1 at `c + c1_off`).
fn skinny(
    g: &dyn GpuBackend,
    ks: &SkinnyKernels,
    sh: &Shape,
    op: &Ops,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    (parts, c1_off, stride): (usize, usize, usize),
) -> Result<()> {
    let (n, k) = (sh.n as u32, sh.k as u32);
    let c1 = c.offset(c1_off);
    match sh.family {
        Family::W4a16 => {
            let w = op.nvfp4();
            let all = [(&w, c, n), (&w, c1, n)];
            skinny_mma::launch_w4a16(g, ks, a, &all[..parts], (m as u32, k), stride as u32, 0)
        }
        Family::Qg => skinny_mma::launch_qg(
            g,
            ks,
            a,
            &op.nvfp4(),
            c,
            (m as u32, n, k),
            (sh.heads, sh.hd),
            0,
        ),
        Family::W8a16 => {
            let all = [((op.w, op.s), c, n), ((op.w, op.s), c1, n)];
            skinny_mma::launch_w8a16(g, ks, a, &all[..parts], (m as u32, k), stride as u32, 0)
        }
    }
}

/// The GEMV serial decode runs today, on row `t` of `a`, into `c`.
fn gemv_today(g: &dyn GpuBackend, sh: &Shape, op: &Ops, t: usize, c: DevicePtr) -> Result<()> {
    let (n, k) = (sh.n as u32, sh.k as u32);
    let at = op.a.offset(t * sh.k * 2);
    let (module, func) = match sh.family {
        Family::W4a16 => ("w4a16_gemv", "w4a16_gemv"),
        Family::Qg => ("w4a16_gemv", "w4a16_gemv_qg"),
        Family::W8a16 => ("w8a16_gemv", "w8a16_gemv"),
    };
    let l = KernelLaunch::new(g, g.kernel(module, func)?)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1]);
    let l = l.arg_ptr(at).arg_ptr(op.w).arg_ptr(op.s);
    let l = if sh.family == Family::W8a16 {
        l
    } else {
        l.arg_f32(SCALE2)
    };
    let l = l.arg_ptr(c).arg_u32(n).arg_u32(k);
    let l = if sh.family == Family::Qg {
        l.arg_u32(sh.heads).arg_u32(sh.hd)
    } else {
        l
    };
    l.launch(0)
}

/// (RMS error / output RMS, max error / output RMS, share not correctly rounded).
fn errors(got: &[u8], want: &[f64]) -> (f64, f64, f64) {
    let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
    let (mut se, mut mx, mut off) = (0f64, 0f64, 0usize);
    for (b, &w) in got.chunks_exact(2).zip(want) {
        let v = bf16::from_bits(u16::from_le_bytes([b[0], b[1]]));
        let e = (v.to_f64() - w).abs();
        se += e * e;
        mx = mx.max(e);
        off += usize::from(v != bf16::from_f64(w));
    }
    (
        (se / want.len() as f64).sqrt() / rms,
        mx / rms,
        off as f64 / want.len() as f64,
    )
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b or qwen3.6-27b ptx set: build with ATLAS_TARGET_MODEL=qwen3.8-27b");
    println!("kernel set: {}", set.target.model);
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let ks = SkinnyKernels::resolve(g);
    if !ks.complete() {
        println!("skinny_mma kernels absent from this target set — SKIP");
        std::process::exit(2);
    }
    let (mut bytes_ok, mut control_ok, mut acc_ok) = (true, true, true);
    for &(label, family, n, k, heads, hd) in SHAPES {
        let sh = Shape {
            family,
            n,
            k,
            heads,
            hd,
        };
        let op = Ops::new(g, 7, &sh)?;
        let row_bytes = n * 2;
        let total = ROWS * row_bytes;
        let (c_ref, c) = (g.alloc(total)?, g.alloc(2 * total)?);
        // The reference: ROWS one-row launches.
        for t in 0..ROWS {
            skinny(
                g,
                &ks,
                &sh,
                &op,
                op.a.offset(t * k * 2),
                c_ref.offset(t * row_bytes),
                1,
                (1, 0, n),
            )?;
        }
        g.synchronize(0)?;
        let want = down(g, c_ref, total)?;
        let mut bad = Vec::new();
        for (dst, src) in SHARED {
            if want[dst * row_bytes..(dst + 1) * row_bytes]
                != want[src * row_bytes..(src + 1) * row_bytes]
            {
                bad.push(format!("shared rows {src}/{dst} differ"));
            }
        }
        for m in 1..=ROWS {
            g.memset(c, SENTINEL, 2 * total)?;
            skinny(g, &ks, &sh, &op, op.a, c, m, (1, 0, n))?;
            g.synchronize(0)?;
            let got = down(g, c, total)?;
            let rows_ok = got[..m * row_bytes] == want[..m * row_bytes];
            let tail_ok = got[m * row_bytes..].iter().all(|&b| b == SENTINEL);
            if !(rows_ok && tail_ok) {
                let diff = (0..m * n)
                    .filter(|&i| got[2 * i..2 * i + 2] != want[2 * i..2 * i + 2])
                    .count();
                bad.push(format!("M={m}: {diff} elems differ, tail intact={tail_ok}"));
            }
        }
        // Two weights in one launch, and a strided launch (rows 2n apart).
        if family != Family::Qg {
            for (m, parts, c1_off, stride) in [
                (16, 2, total, n),
                (9, 2, total, n),
                (1, 2, total, n),
                (41, 2, total, n),
                (16, 1, 0, 2 * n),
                (41, 1, 0, 2 * n),
            ] {
                g.memset(c, SENTINEL, 2 * total)?;
                skinny(g, &ks, &sh, &op, op.a, c, m, (parts, c1_off, stride))?;
                g.synchronize(0)?;
                let got = down(g, c, 2 * total)?;
                for t in 0..m {
                    let w = &want[t * row_bytes..(t + 1) * row_bytes];
                    let p0 = &got[t * stride * 2..t * stride * 2 + row_bytes];
                    let p1 = &got[c1_off + t * stride * 2..c1_off + t * stride * 2 + row_bytes];
                    if p0 != w || (parts == 2 && p1 != w) {
                        bad.push(format!(
                            "parts={parts} stride={stride} M={m}: row {t} differs"
                        ));
                        break;
                    }
                }
            }
        }
        bytes_ok &= bad.is_empty();
        println!(
            "{label}  M=1..={ROWS} byte-identical to M one-row launches={}{}",
            bad.is_empty(),
            if bad.is_empty() {
                String::new()
            } else {
                format!("  [{}]", bad.join("; "))
            }
        );

        // Negative control: one BF16 ULP on row 1 changes row 1 only.
        let mut pert = op.a_host.clone();
        let big = (k..2 * k)
            .max_by(|&i, &j| {
                let mag = |x: usize| bf16::from_bits(pert[x]).to_f32().abs();
                mag(i).total_cmp(&mag(j))
            })
            .unwrap_or(k);
        pert[big] ^= 1;
        let a_pert = up(g, &a_bytes(&pert))?;
        skinny(g, &ks, &sh, &op, a_pert, c, ROWS, (1, 0, n))?;
        g.synchronize(0)?;
        let got = down(g, c, total)?;
        let row = |b: &[u8], t: usize| b[t * row_bytes..(t + 1) * row_bytes].to_vec();
        let fired = row(&got, 1) != row(&want, 1);
        let others = (0..ROWS)
            .filter(|&t| t != 1)
            .all(|t| row(&got, t) == row(&want, t));
        control_ok &= fired && others;
        println!(
            "{label}  CONTROL 1-ULP on row 1: row 1 changed={fired}, other rows unchanged={others}"
        );
        g.free(a_pert).ok();

        // Accuracy against FP64, next to today's GEMV.
        for t in 0..ACC_ROWS {
            let r64 = op.reference(&sh, t);
            gemv_today(g, &sh, &op, t, c)?;
            g.synchronize(0)?;
            let old = errors(&down(g, c, row_bytes)?, &r64);
            let new = errors(&want[t * row_bytes..(t + 1) * row_bytes], &r64);
            let ok = new.0 <= old.0 * 1.000_001;
            acc_ok &= ok;
            println!(
                "{label}  row {t} vs FP64: skinny rms {:.3e} max {:.3e} not-RN {:.2}% | gemv rms {:.3e} max {:.3e} not-RN {:.2}%  no-worse={ok}",
                new.0,
                new.1,
                100.0 * new.2,
                old.0,
                old.1,
                100.0 * old.2
            );
        }
        for p in [c_ref, c] {
            g.free(p).ok();
        }
        op.free(g);
    }
    if !control_ok {
        println!("FAIL — the 1-ULP control did not behave; this harness is VACUOUS or rows leak.");
        std::process::exit(1);
    }
    if !bytes_ok {
        println!("FAIL — an M-row skinny launch is NOT byte-identical to M one-row launches.");
        std::process::exit(1);
    }
    if !acc_ok {
        println!("FAIL — a skinny kernel's RMS error vs FP64 exceeds today's GEMV.");
        std::process::exit(1);
    }
    println!(
        "PASS — skinny_mma rows are M-invariant on every shape, and no less accurate than today's GEMV."
    );
    Ok(())
}
