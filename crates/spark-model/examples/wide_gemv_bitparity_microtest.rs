// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity gate for the `ATLAS_WIDE_GEMV` wide twins
//! (`kernels/gb10/qwen3.6-27b/nvfp4/wide_gemv.cu`, `layers/ops/wide_gemv.rs`).
//!
//! Every tier — `{w4a16_gemv,w4a16_gemv_qg,w8a16_gemv}_wide{8,9,12,16}_nb{2,4}`
//! — at EVERY M it can serve (1..=its rows), launched through the runtime's
//! own launch functions (so the persistent grid and block shape are the
//! ones the lever uses), against M single-row launches of the kernel serial
//! decode runs: `w4a16_gemv`, `w4a16_gemv_qg`, `w8a16_gemv`. RAW BF16 BYTES
//! of all M rows; and the row after the M-th must still hold the sentinel
//! the output was filled with (a tier writes no row it was not given).
//!
//! Shapes: the Qwen3.8-27B projections each family serves — NVFP4 MLP
//! gate/up and down, GDN qkvz (the requantized-NVFP4 checkpoint), attention
//! o_proj and k/v, the LM head; Q+gate (24 heads x 256); FP8 GDN qkvz and
//! out_proj and the native-FP8 attention shapes — plus a tail shape per
//! family whose N is not a multiple of the 16-output tile. A 1-ULP
//! activation perturbation must be seen (negative control).
//!
//! Exit: 0 all legs byte-identical, 1 any leg differs, 2 kernels absent.
//!
//! Run (GB10; builds the qwen3.8-27b kernel set, falls back to qwen3.6-27b):
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example wide_gemv_bitparity_microtest
//! `WIDE_PARITY_QUICK=1` runs one seed and skips the LM head.

use anyhow::Result;
use half::bf16;
use spark_model::layers::ops::wide_gemv::{self, Family, NBS, TIERS, WideKernel};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const ROWS: usize = 16;
const SCALE2: f32 = 0.0123_f32;
const SENTINEL: u8 = 0xA5;

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
        "w4 gdn qkvz [16384 x 5120]",
        Family::W4a16,
        16384,
        5120,
        0,
        0,
    ),
    (
        "w4 o/out    [ 5120 x 6144]",
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
        "w4 tail     [10307 x 2688]",
        Family::W4a16,
        10307,
        2688,
        0,
        0,
    ),
    (
        "w4 lm_head [248320 x 5120]",
        Family::W4a16,
        248_320,
        5120,
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
        "w8 q+gate   [12288 x 5120]",
        Family::W8a16,
        12288,
        5120,
        0,
        0,
    ),
    (
        "w8 k/v      [ 1024 x 5120]",
        Family::W8a16,
        1024,
        5120,
        0,
        0,
    ),
    // N % 16 != 0 and N % 128 != 0; N % 4 == 0 because `w8a16_gemv` itself
    // returns before staging its LUT in a block's dead tail groups.
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
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// Device operands of one shape: activations `[ROWS, K]`, the weight, its
/// scales (E4M3 bytes, or FP32 block scales for W8).
struct Operands {
    a: DevicePtr,
    a_host: Vec<u8>,
    w: DevicePtr,
    s: DevicePtr,
}

impl Operands {
    fn new(g: &dyn GpuBackend, seed: u64, family: Family, n: usize, k: usize) -> Result<Self> {
        let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5157_4944);
        let a_host: Vec<u8> = (0..ROWS * k)
            .flat_map(|_| bf16::from_f32(rng.r(-1.5, 1.5)).to_bits().to_le_bytes())
            .collect();
        let (w, s) = if family == Family::W8a16 {
            let w: Vec<u8> = (0..n * k).map(|_| rng.r(0.0, 256.0) as u8).collect();
            let s: Vec<u8> = (0..n.div_ceil(128) * k.div_ceil(128))
                .flat_map(|_| rng.r(0.01, 0.12).to_le_bytes())
                .collect();
            (w, s)
        } else {
            // Block-scale bytes 0x30..=0x47: finite positive E4M3 (0.5..3.5).
            let w: Vec<u8> = (0..n * k / 2).map(|_| rng.r(0.0, 256.0) as u8).collect();
            let s: Vec<u8> = (0..n * k / 16)
                .map(|_| 0x30 + rng.r(0.0, 24.0) as u8)
                .collect();
            (w, s)
        };
        Ok(Self {
            a: up(g, &a_host)?,
            a_host,
            w: up(g, &w)?,
            s: up(g, &s)?,
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

    fn free(self, g: &dyn GpuBackend) {
        for p in [self.a, self.w, self.s] {
            g.free(p).ok();
        }
    }
}

/// The single-row reference kernels.
struct Refs {
    w4: KernelHandle,
    qg: KernelHandle,
    w8: KernelHandle,
}

/// `rows` single-row launches of the family's reference into `c` (rows `n`
/// apart), reading activations from `a`.
#[allow(clippy::too_many_arguments)]
fn reference(
    g: &dyn GpuBackend,
    refs: &Refs,
    (family, n, k, heads, head_dim): (Family, usize, usize, u32, u32),
    a: DevicePtr,
    op: &Operands,
    c: DevicePtr,
    rows: usize,
) -> Result<()> {
    for t in 0..rows {
        let (at, ct) = (a.offset(t * k * 2), c.offset(t * n * 2));
        let l = |h| {
            KernelLaunch::new(g, h)
                .grid([div_ceil(n as u32, 4), 1, 1])
                .block([256, 1, 1])
        };
        match family {
            Family::W8a16 => l(refs.w8)
                .arg_ptr(at)
                .arg_ptr(op.w)
                .arg_ptr(op.s)
                .arg_ptr(ct)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(0)?,
            Family::W4a16 | Family::Qg => {
                let mut l = l(if family == Family::Qg {
                    refs.qg
                } else {
                    refs.w4
                })
                .arg_ptr(at)
                .arg_ptr(op.w)
                .arg_ptr(op.s)
                .arg_f32(SCALE2)
                .arg_ptr(ct)
                .arg_u32(n as u32)
                .arg_u32(k as u32);
                if family == Family::Qg {
                    l = l.arg_u32(heads).arg_u32(head_dim);
                }
                l.launch(0)?
            }
        }
    }
    Ok(())
}

/// One wide launch of `m` rows through the runtime's launch functions.
#[allow(clippy::too_many_arguments)]
fn wide(
    g: &dyn GpuBackend,
    kernel: &WideKernel,
    sms: u32,
    (family, n, k, heads, head_dim): (Family, usize, usize, u32, u32),
    op: &Operands,
    c: DevicePtr,
    m: usize,
) -> Result<()> {
    let shape = (m as u32, n as u32, k as u32);
    match family {
        Family::W4a16 => {
            wide_gemv::launch_w4a16(g, kernel, sms, op.a, &op.nvfp4(), c, shape, n as u32, 0)
        }
        Family::Qg => wide_gemv::launch_qg(
            g,
            kernel,
            sms,
            op.a,
            &op.nvfp4(),
            c,
            shape,
            (heads, head_dim),
            0,
        ),
        Family::W8a16 => {
            wide_gemv::launch_w8a16(g, kernel, sms, op.a, (op.w, op.s), c, shape, n as u32, 0)
        }
    }
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b or qwen3.6-27b ptx set: build with ATLAS_TARGET_MODEL=qwen3.8-27b");
    println!("kernel set: {}", set.target.model);
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let sms = g.sm_count()?;
    let quick = std::env::var_os("WIDE_PARITY_QUICK").is_some();

    let refs = match (
        g.kernel("w4a16_gemv", "w4a16_gemv"),
        g.kernel("w4a16_gemv", "w4a16_gemv_qg"),
        g.kernel("w8a16_gemv", "w8a16_gemv"),
    ) {
        (Ok(w4), Ok(qg), Ok(w8)) => Refs { w4, qg, w8 },
        _ => {
            println!("single-row reference kernels absent — SKIP");
            std::process::exit(2);
        }
    };
    let tiers: Vec<WideKernel> = Family::ALL
        .iter()
        .flat_map(|&f| NBS.iter().flat_map(move |&nb| TIERS.map(|r| (f, r, nb))))
        .map(|(f, r, nb)| WideKernel::resolve(g, f, r, nb))
        .collect();
    if tiers.iter().any(|t| t.handle.0 == 0) {
        println!("wide_gemv tiers absent from this target set — SKIP");
        std::process::exit(2);
    }

    let (mut clean, mut control_ok) = (true, true);
    let seeds: &[u64] = if quick { &[1] } else { &[1, 99] };
    for &seed in seeds {
        for &(label, family, n, k, heads, head_dim) in SHAPES {
            if quick && n > 100_000 {
                continue;
            }
            let shape = (family, n, k, heads, head_dim);
            let op = Operands::new(g, seed, family, n, k)?;
            let bytes = ROWS * n * 2;
            let (c_ref, c) = (g.alloc(bytes)?, g.alloc(bytes)?);
            // Row t of the reference does not depend on how many rows a
            // launch carries: compute all ROWS once, compare prefixes.
            reference(g, &refs, shape, op.a, &op, c_ref, ROWS)?;
            g.synchronize(0)?;
            let want = down(g, c_ref, bytes)?;

            for t in tiers.iter().filter(|t| t.family == family) {
                let mut bad = Vec::new();
                for m in 1..=t.rows as usize {
                    g.memset(c, SENTINEL, bytes)?;
                    wide(g, t, sms, shape, &op, c, m)?;
                    g.synchronize(0)?;
                    let got = down(g, c, (m + 1).min(ROWS) * n * 2)?;
                    let rows_ok = got[..m * n * 2] == want[..m * n * 2];
                    let tail_ok = got[m * n * 2..].iter().all(|&b| b == SENTINEL);
                    if !(rows_ok && tail_ok) {
                        let diff = got[..m * n * 2]
                            .chunks_exact(2)
                            .zip(want.chunks_exact(2))
                            .filter(|(x, y)| x != y)
                            .count();
                        bad.push(format!("M={m}: {diff} elems differ, tail intact={tail_ok}"));
                    }
                }
                clean &= bad.is_empty();
                let name = wide_gemv::kernel_name(family, t.rows, t.nb);
                println!(
                    "seed {seed:>3}  {label}  {name:<26} M=1..={:<2} byte-identical={}{}",
                    t.rows,
                    bad.is_empty(),
                    if bad.is_empty() {
                        String::new()
                    } else {
                        format!("  [{}]", bad.join("; "))
                    }
                );
            }

            // Negative control: one BF16 ULP on row 1 must change the bytes.
            let mut pert = op.a_host.clone();
            pert[2 * (k + 7)] ^= 1;
            let a_pert = up(g, &pert)?;
            reference(g, &refs, shape, a_pert, &op, c_ref, ROWS)?;
            g.synchronize(0)?;
            let differs = down(g, c_ref, bytes)? != want;
            control_ok &= differs;
            println!("seed {seed:>3}  {label}  CONTROL 1-ULP perturbation detected={differs}");
            for p in [a_pert, c_ref, c] {
                g.free(p).ok();
            }
            op.free(g);
        }
    }

    if !control_ok {
        println!("FAIL — negative control did not mismatch; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if !clean {
        println!("FAIL — a wide tier is NOT byte-identical to its single-row kernel.");
        std::process::exit(1);
    }
    println!(
        "PASS — every wide_gemv tier is byte-identical to M x its single-row kernel \
         (w4a16_gemv / w4a16_gemv_qg / w8a16_gemv) at every M it serves."
    );
    Ok(())
}
