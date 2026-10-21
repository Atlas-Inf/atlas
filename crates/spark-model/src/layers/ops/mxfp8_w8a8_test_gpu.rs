// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`; a GB10 and a kernel target with `mxfp8_act_quant`)
//! for the W8A8 prefill arm ([`super::super::mxfp8_proj_w8a8`]) on FP4-Local
//! TP=2 local shapes (GDN qkvz `[8192, 2560]`, attention o_proj `[2560, 3072]`):
//!
//! - every output row is the same bytes whatever the pass height, the row's
//!   position in the pass and the scratch chunking — cached-prefix output
//!   equals uncached output only if a prefill chunk boundary cannot move a
//!   row's bytes;
//! - sampled outputs match an f64 host product of the dequantized operands
//!   (the host quantizes the activation with the ModelOpt rule) within BF16
//!   output rounding, and sit within the documented few-percent of the exact
//!   BF16-activation projection (`mxfp8_proj`'s expansion + GEMM).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::mxfp8_act::gpu_tests::{Rng, backend, down, u16_bytes, upload};
use super::super::mxfp8_act::tests::{bf16, quant_block};
use super::super::{Mxfp8Proj, Mxfp8ProjKernels, Mxfp8Scratch, mx_lt_scale_bytes};
use super::MxW8a8;
use crate::weight_map::Mxfp8Weight;

const SHAPES: [(usize, usize); 2] = [(8192, 2560), (2560, 3072)];
const M_BIG: usize = 4096;

fn e4m3(b: u8) -> f64 {
    let (e, m) = (i32::from((b >> 3) & 0xf), f64::from(b & 7));
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if b & 0x80 != 0 { -v } else { v }
}

/// Finite E4M3 weights, E8M0 scales 2^-9 .. 2^9, BF16 activations within a
/// few binades of 1 (both signs).
fn operands(rng: &mut Rng, n: usize, k: usize, m: usize) -> (Vec<u8>, Vec<u8>, Vec<u16>) {
    let w = (0..n * k)
        .map(|_| {
            let b = rng.next() as u8;
            if b & 0x7f == 0x7f { b & 0xfe } else { b }
        })
        .collect();
    let s = (0..n * k / 32)
        .map(|_| 118 + (rng.next() % 19) as u8)
        .collect();
    let a = (0..m * k)
        .map(|_| {
            let sign = ((rng.next() & 1) as u16) << 15;
            let exp = 120 + (rng.next() % 14) as u16;
            sign | (exp << 7) | (rng.next() & 0x7f) as u16
        })
        .collect();
    (w, s, a)
}

struct Case<'a> {
    gpu: &'a dyn GpuBackend,
    w: MxW8a8,
    quant: spark_runtime::gpu::KernelHandle,
    act: DevicePtr,
    out: DevicePtr,
    stream: u64,
}

impl Case<'_> {
    /// Rows `row0 .. row0 + m` through the W8A8 arm with a scratch sized for
    /// `chunk` rows; their BF16 output bytes.
    fn run(&self, row0: usize, m: usize, chunk: usize) -> Result<Vec<u8>> {
        let (n, k) = (self.w.n, self.w.k);
        let bytes = chunk * k + mx_lt_scale_bytes(chunk, k);
        let scratch = Mxfp8Scratch {
            ptr: self.gpu.alloc(bytes)?,
            bytes,
        };
        self.gpu.memset(self.out, 0x77, m * n * 2)?;
        super::super::mxfp8_proj_w8a8(
            self.gpu,
            self.quant,
            &self.w,
            self.act.offset(row0 * k * 2),
            self.out,
            m,
            scratch,
            self.stream,
        )?;
        self.gpu.synchronize(self.stream)?;
        let v = down(self.gpu, self.out, m * n * 2)?;
        self.gpu.free(scratch.ptr)?;
        Ok(v)
    }
}

fn out_at(v: &[u8], n: usize, r: usize, c: usize) -> f64 {
    let i = (r * n + c) * 2;
    f64::from(bf16(u16::from_le_bytes([v[i], v[i + 1]])))
}

#[test]
#[ignore = "needs a GB10 GPU and the mxfp8_act_quant kernels"]
fn w8a8_rows_are_independent_of_pass_height_position_and_chunking() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let quant = gpu.kernel("mxfp8_act_quant", "mxfp8_quant_act_bf16")?;
    let mut rng = Rng(0xa8a8_0001);
    for (n, k) in SHAPES {
        let (w, s, a) = operands(&mut rng, n, k, M_BIG);
        let twin = Mxfp8Weight {
            data: upload(gpu, &w)?,
            scales: upload(gpu, &s)?,
        };
        let case = Case {
            gpu,
            w: MxW8a8::from_twin(gpu, &twin, [n, k], 0, stream)?,
            quant,
            act: upload(gpu, &u16_bytes(&a))?,
            out: gpu.alloc(M_BIG * n * 2)?,
            stream,
        };
        let full = case.run(0, M_BIG, M_BIG)?;
        let row_bytes = n * 2;
        let mut checks = vec![];
        for m in [33usize, 64, 100, 512] {
            checks.push((format!("first {m} rows"), 0, m, m));
        }
        checks.push(("rows 1000..1100".into(), 1000, 100, 100));
        checks.push(("rows 3001..4096".into(), 3001, M_BIG - 3001, M_BIG));
        checks.push(("all rows, 256-row chunks".into(), 0, M_BIG, 256));
        for (label, row0, m, chunk) in checks {
            let got = case.run(row0, m, chunk)?;
            let want = &full[row0 * row_bytes..(row0 + m) * row_bytes];
            let bad = got
                .chunks(2)
                .zip(want.chunks(2))
                .filter(|(x, y)| x != y)
                .count();
            println!("[{n}, {k}] {label}: {bad}/{} outputs differ", m * n);
            assert_eq!(bad, 0, "[{n}, {k}] {label}: rows moved with the pass");
        }
        case.w.free(gpu)?;
        for p in [twin.data, twin.scales, case.act, case.out] {
            gpu.free(p)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU and the mxfp8_act_quant kernels"]
fn w8a8_matches_the_dequantized_host_product_and_stays_near_exact_bf16() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let quant = gpu.kernel("mxfp8_act_quant", "mxfp8_quant_act_bf16")?;
    let kernels = Mxfp8ProjKernels::resolve(gpu)?;
    let mut rng = Rng(0xa8a8_0002);
    let m = 300;
    for (n, k) in SHAPES {
        let (w, s, a) = operands(&mut rng, n, k, m);
        let twin = Mxfp8Weight {
            data: upload(gpu, &w)?,
            scales: upload(gpu, &s)?,
        };
        let case = Case {
            gpu,
            w: MxW8a8::from_twin(gpu, &twin, [n, k], 0, stream)?,
            quant,
            act: upload(gpu, &u16_bytes(&a))?,
            out: gpu.alloc(m * n * 2)?,
            stream,
        };
        let got = case.run(0, m, m)?;
        // The exact projection: BF16 activations times the expanded weight.
        let proj = Mxfp8Proj {
            w: twin,
            n: n as u32,
            k: k as u32,
        };
        let exact_out = gpu.alloc(m * n * 2)?;
        let scratch = Mxfp8Scratch {
            ptr: gpu.alloc(n * k * 2)?,
            bytes: n * k * 2,
        };
        super::super::mxfp8_proj(
            gpu, &kernels, &proj, case.act, exact_out, m, scratch, stream,
        )?;
        gpu.synchronize(stream)?;
        let exact = down(gpu, exact_out, m * n * 2)?;
        let (mut worst, mut num, mut den) = (0f64, 0f64, 0f64);
        for _ in 0..4096 {
            let (r, c) = ((rng.next() as usize) % m, (rng.next() as usize) % n);
            let (mut acc, mut mag) = (0f64, 0f64);
            for b in 0..k / 32 {
                let vals: Vec<f32> = a[r * k + b * 32..r * k + b * 32 + 32]
                    .iter()
                    .map(|&h| bf16(h))
                    .collect();
                let (sa, qa) = quant_block(&vals);
                let (sa, sw) = (
                    2f64.powi(i32::from(sa) - 127),
                    2f64.powi(i32::from(s[c * k / 32 + b]) - 127),
                );
                for i in 0..32 {
                    let t = e4m3(qa[i]) * sa * e4m3(w[c * k + b * 32 + i]) * sw;
                    acc += t;
                    mag += t.abs();
                }
            }
            let g = out_at(&got, n, r, c);
            let tol = acc.abs() * 2f64.powi(-7) + mag * 1e-5;
            worst = worst.max((g - acc).abs() / tol.max(f64::MIN_POSITIVE));
            let e = out_at(&exact, n, r, c);
            num += (g - e) * (g - e);
            den += e * e;
        }
        let rel = (num / den).sqrt();
        println!(
            "[{n}, {k}] worst |w8a8 - host| / tol = {worst:.3}; ||w8a8 - exact|| / ||exact|| = {rel:.4}"
        );
        assert!(worst <= 1.0, "[{n}, {k}]: W8A8 off the dequantized product");
        assert!(
            rel < 5e-2,
            "[{n}, {k}]: W8A8 {rel} from the exact projection"
        );
        case.w.free(gpu)?;
        for p in [
            proj.w.data,
            proj.w.scales,
            case.act,
            case.out,
            exact_out,
            scratch.ptr,
        ] {
            gpu.free(p)?;
        }
    }
    Ok(())
}
