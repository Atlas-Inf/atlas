// SPDX-License-Identifier: AGPL-3.0-only
//! Row-identity gate for the BF16 head of qwen3_5's batch lever
//! (`model/exact_verify.rs`): on the Qwen3.8-27B target, the head's
//! multi-row path (`Qwen4ExpWideRows::dense_rows`: `dense_gemv_bf16_batchm`
//! to 8 rows, `qwen4exp_bf16_rows16/32` past that, built from
//! `kernels/gb10/qwen3.6-27b/nvfp4/qwen4exp_wide_rows.cu`) must write, for
//! every M = 1..=72, exactly the bytes of M one-row `dense_gemv_bf16`
//! launches (serial decode's LM head), leave the row after the M-th
//! untouched, and write identical rows for identical inputs (row 40 repeats
//! row 2). A 1-ULP change to one activation of row 1 must change row 1 and
//! no other row (negative control).
//!
//! Exit: 0 pass, 1 a gate failed, 2 kernels absent.
//!
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example bf16_wide_rows_microtest

use anyhow::Result;
use half::bf16;
use spark_model::layers::ops::{self, Qwen4ExpWideRows};
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const ROWS: usize = 72;
const SENTINEL: u8 = 0xA5;
const SHARED: (usize, usize) = (40, 2);
/// `(label, N, K)`: the LM head, and a narrower shape with a ragged N.
const SHAPES: &[(&str, usize, usize)] = &[
    ("lm_head [248320 x 5120]", 248_320, 5120),
    ("tail    [  10307 x 5120]", 10_307, 5120),
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
    fn act(&mut self) -> u16 {
        let n = (0..4).map(|_| self.f()).sum::<f32>() - 2.0;
        bf16::from_f32(n * if self.f() < 1.0 / 64.0 { 60.0 } else { 1.0 }).to_bits()
    }
}

fn up(g: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(&bytes, p)?;
    Ok(p)
}
fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b ptx set: build with ATLAS_TARGET_MODEL=qwen3.8-27b");
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let wide = Qwen4ExpWideRows::resolve_bf16(g);
    let one = g.kernel("gemv", "dense_gemv_bf16")?;
    let narrow = g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
    if g.kernel("qwen4exp_wide_rows", "qwen4exp_bf16_rows32")
        .is_err()
    {
        println!("qwen4exp_wide_rows absent from this target set — SKIP");
        std::process::exit(2);
    }
    let (mut bytes_ok, mut control_ok) = (true, true);
    for &(label, n, k) in SHAPES {
        let mut rng = Lcg(0x5eed ^ (n * 31 + k) as u64);
        let mut a_host: Vec<u16> = (0..ROWS * k).map(|_| rng.act()).collect();
        let (dst, src) = SHARED;
        let row = a_host[src * k..(src + 1) * k].to_vec();
        a_host[dst * k..(dst + 1) * k].copy_from_slice(&row);
        // Weights small and finite: one value in 256 larger.
        let w_host: Vec<u16> = (0..n * k)
            .map(|_| {
                let v = (rng.f() - 0.5) * if rng.f() < 1.0 / 256.0 { 2.0 } else { 0.05 };
                bf16::from_f32(v).to_bits()
            })
            .collect();
        let (a, w) = (up(g, &a_host)?, up(g, &w_host)?);
        let weight = DenseWeight { weight: w };
        let row_bytes = n * 2;
        let total = ROWS * row_bytes;
        let (c_ref, c) = (g.alloc(total)?, g.alloc(total + row_bytes)?);
        for t in 0..ROWS {
            ops::dense_gemv(
                g,
                one,
                a.offset(t * k * 2),
                &weight,
                c_ref.offset(t * row_bytes),
                n as u32,
                k as u32,
                0,
            )?;
        }
        g.synchronize(0)?;
        let want = down(g, c_ref, total)?;
        let mut bad = Vec::new();
        if want[dst * row_bytes..(dst + 1) * row_bytes]
            != want[src * row_bytes..(src + 1) * row_bytes]
        {
            bad.push(format!("shared rows {src}/{dst} differ"));
        }
        for m in 1..=ROWS {
            g.memset(c, SENTINEL, total + row_bytes)?;
            wide.dense_rows(
                g,
                narrow,
                a,
                &weight,
                c,
                (m as u32, n as u32, k as u32, n as u32),
                0,
            )?;
            g.synchronize(0)?;
            let got = down(g, c, total + row_bytes)?;
            let rows_ok = got[..m * row_bytes] == want[..m * row_bytes];
            let tail_ok = got[m * row_bytes..(m + 1) * row_bytes]
                .iter()
                .all(|&b| b == SENTINEL);
            if !(rows_ok && tail_ok) {
                let diff = (0..m * n)
                    .filter(|&i| got[2 * i..2 * i + 2] != want[2 * i..2 * i + 2])
                    .count();
                bad.push(format!("M={m}: {diff} elems differ, tail intact={tail_ok}"));
            }
        }
        bytes_ok &= bad.is_empty();
        println!(
            "{label}  M=1..={ROWS} byte-identical to M dense_gemv_bf16 launches={}{}",
            bad.is_empty(),
            if bad.is_empty() {
                String::new()
            } else {
                format!("  [{}]", bad.join("; "))
            }
        );
        // Negative control: one BF16 ULP on row 1's largest activation.
        let mut pert = a_host.clone();
        let big = (k..2 * k)
            .max_by(|&i, &j| {
                let mag = |x: usize| bf16::from_bits(pert[x]).to_f32().abs();
                mag(i).total_cmp(&mag(j))
            })
            .unwrap_or(k);
        pert[big] ^= 1;
        let a_pert = up(g, &pert)?;
        wide.dense_rows(
            g,
            narrow,
            a_pert,
            &weight,
            c,
            (ROWS as u32, n as u32, k as u32, n as u32),
            0,
        )?;
        g.synchronize(0)?;
        let got = down(g, c, total)?;
        let r = |b: &[u8], t: usize| b[t * row_bytes..(t + 1) * row_bytes].to_vec();
        let fired = r(&got, 1) != r(&want, 1);
        let others = (0..ROWS)
            .filter(|&t| t != 1)
            .all(|t| r(&got, t) == r(&want, t));
        control_ok &= fired && others;
        println!(
            "{label}  CONTROL 1-ULP on row 1: row 1 changed={fired}, other rows unchanged={others}"
        );
        for p in [a, w, a_pert, c_ref, c] {
            g.free(p).ok();
        }
    }
    if !control_ok {
        println!("FAIL — the 1-ULP control did not behave; this harness is VACUOUS or rows leak.");
        std::process::exit(1);
    }
    if !bytes_ok {
        println!("FAIL — a multi-row head launch is NOT byte-identical to one-row launches.");
        std::process::exit(1);
    }
    println!("PASS — the batch lever's BF16 head rows are serial decode's bytes at every M.");
    Ok(())
}
