// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]` per repo convention; a GB10 and a kernel target
//! carrying `mxfp8_gemv`, glm-5.3-flash or qwen3.8-flash-next) for the
//! checkpoint-MXFP8 decode copies, at the Qwen3.8-Flash-Next FP4-Local shapes:
//!
//! - parity: [`mxfp8_gemv_rows`] against the BF16 decode GEMV
//!   (`dense_gemv_bf16`) on the exactly dequantized weight, and both against
//!   an f64 host reference. The two read identical operands (E4M3 x 2^e is
//!   exact in BF16), so they differ only in FP32 summation order.
//! - row invariance: every row of an m-row launch (m = 1..=8, 16, 32, and 40
//!   = 32 + 8) is byte-identical to that row run alone (m = 1), which is what
//!   keeps a verify row equal to the serial token.
//! - a microbench of MXFP8 against the BF16 decode GEMVs (`dense_gemv_bf16`
//!   at m = 1, `dense_gemv_bf16_batchm` at 4 and 8), weights rotated across
//!   copies so each launch streams from DRAM as a decode step does.
//!
//!   cargo test --release -p spark-model --lib mxfp8_rows -- --ignored --nocapture --test-threads=1

use super::*;
use crate::layers::ops::{dense_gemv, dense_gemv_batchm};
use anyhow::Context;
use spark_runtime::cuda_backend::AtlasCudaBackend;

/// Scale codes 111..=118: the range the FP4-Local GDN `weight_scale`s span
/// (layer 0 `in_proj_qkv` 111..118, `out_proj` 112..119; 2^-16 .. 2^-9).
const SCALE_LO: u8 = 111;

/// `(name, n, k)`: every MXFP8 projection shape a TP2 rank decodes, plus the
/// TP1 GDN pair.
const SHAPES: [(&str, usize, usize); 9] = [
    ("gdn qkvz tp2", 8192, 2560),
    ("gdn out tp2", 2560, 3072),
    ("gdn qkvz tp1", 16384, 2560),
    ("gdn out tp1", 2560, 6144),
    ("qsa q tp2", 6144, 2560),
    ("qsa k/v tp2", 256, 2560),
    ("qsa index_qk", 640, 2560),
    ("shared gate/up", 640, 2560),
    ("shared down", 2560, 640),
];

pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

pub(crate) fn backend() -> Result<&'static AtlasCudaBackend> {
    static GPU: std::sync::OnceLock<AtlasCudaBackend> = std::sync::OnceLock::new();
    if GPU.get().is_none() {
        let gpu =
            AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules()).context("CUDA backend")?;
        let _ = GPU.set(gpu);
    }
    let gpu = GPU.get().expect("set above");
    gpu.bind_to_thread()?;
    Ok(gpu)
}

pub(crate) fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn bf16s(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn f(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// A synthetic MXFP8 weight `(values, scales)` and its exact BF16 dequant.
fn weight(rng: &mut Rng, n: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<u16>) {
    let values: Vec<u8> = (0..n * k)
        .map(|_| match rng.next() as u8 {
            b @ (0x7f | 0xff) => b - 1,
            b => b,
        })
        .collect();
    let scales: Vec<u8> = (0..n * k / MXFP8_BLOCK)
        .map(|_| SCALE_LO + (rng.next() % 8) as u8)
        .collect();
    let bf16 = dequant(&values, &scales, k);
    (values, scales, bf16)
}

/// The exact BF16 dequant of MXFP8 `[n, k]`: the loader's own host
/// reference (`weight_map::mxfp8_to_bf16_host`), which refuses any scale
/// outside the range where E4M3 x 2^(s - 127) is exact in BF16.
fn dequant(values: &[u8], scales: &[u8], k: usize) -> Vec<u16> {
    crate::weight_map::mxfp8_to_bf16_host(values, scales, values.len() / k, k).unwrap()
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn mxfp8_tiers(gpu: &dyn GpuBackend) -> Result<[KernelHandle; 3]> {
    Ok([
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc8")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc16")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc32")?,
    ])
}

/// Rows run alone, and in each batch width, must match byte for byte.
const WIDTHS: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 40];
const MAX_ROWS: usize = 40;

#[test]
#[ignore = "needs a GB10 GPU and a kernel target with mxfp8_gemv"]
fn mxfp8_rows_match_bf16_and_are_row_invariant_at_real_shapes() -> Result<()> {
    let mut rng = Rng(0x51ab_cd3f_0e2c_9a77);
    for (name, n, k) in SHAPES {
        let (values, scales, bf16) = weight(&mut rng, n, k);
        parity(&mut rng, name, &values, &scales, &bf16, n, k)?;
    }
    Ok(())
}

/// MXFP8 `values` / `scales` (`[n, k]`) against the BF16 decode GEMV on
/// their dequant `bf16`, and every row of every width against the row alone.
fn parity(
    rng: &mut Rng,
    name: &str,
    values: &[u8],
    scales: &[u8],
    bf16: &[u16],
    n: usize,
    k: usize,
) -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let tiers = mxfp8_tiers(gpu)?;
    let gemv = gpu.kernel("gemv", "dense_gemv_bf16")?;
    // BF16 activations in (-2, 2).
    let acts: Vec<u16> = (0..MAX_ROWS * k)
        .map(|_| {
            let x = 2.0 * ((rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
            (x.to_bits() >> 16) as u16
        })
        .collect();
    let a = upload(gpu, &bytes(&acts))?;
    let w = Mxfp8Weight {
        data: upload(gpu, values)?,
        scales: upload(gpu, scales)?,
        tiled: false,
    };
    let dense = crate::weight_map::DenseWeight {
        weight: upload(gpu, &bytes(bf16))?,
    };
    let c = gpu.alloc(MAX_ROWS * n * 2)?;
    let run = |rows: usize, first: usize| -> Result<Vec<u16>> {
        gpu.memset_async(c, 0xa5, MAX_ROWS * n * 2, stream)?;
        mxfp8_gemv_rows(
            gpu,
            &tiers,
            &w,
            a.offset(first * k * 2),
            c,
            rows,
            n,
            k,
            n,
            stream,
        )?;
        gpu.synchronize(stream)?;
        bf16s(gpu, c, rows * n)
    };
    // Each row alone, as serial decode runs it.
    let alone: Vec<Vec<u16>> = (0..MAX_ROWS).map(|i| run(1, i)).collect::<Result<_>>()?;
    for m in WIDTHS {
        let got = run(m, 0)?;
        for (i, row) in got.chunks_exact(n).enumerate() {
            assert!(
                row == &alone[i][..],
                "{name} {n}x{k}: row {i} of {m} differs from the row alone"
            );
        }
    }
    // The BF16 decode GEMV on the dequantized weight, row by row.
    let mut bf = Vec::with_capacity(MAX_ROWS);
    for i in 0..MAX_ROWS {
        dense_gemv(
            gpu,
            gemv,
            a.offset(i * k * 2),
            &dense,
            c,
            n as u32,
            k as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        bf.push(bf16s(gpu, c, n)?);
    }
    // Against each other and an f64 reference (rows 0..4).
    let (mut max_abs, mut max_ref, mut differ) = (0f64, 0f64, 0usize);
    let (mut err2_mx, mut err2_bf, mut ref2) = (0f64, 0f64, 0f64);
    for i in 0..MAX_ROWS {
        for j in 0..n {
            let (x, y) = (f64::from(f(alone[i][j])), f64::from(f(bf[i][j])));
            max_abs = max_abs.max((x - y).abs());
            max_ref = max_ref.max(y.abs());
            differ += usize::from(alone[i][j] != bf[i][j]);
            if i < 4 {
                let exact: f64 = (0..k)
                    .map(|t| f64::from(f(acts[i * k + t])) * f64::from(f(bf16[j * k + t])))
                    .sum();
                err2_mx += (x - exact).powi(2);
                err2_bf += (y - exact).powi(2);
                ref2 += exact * exact;
            }
        }
    }
    let rel = |e2: f64| (e2 / ref2).sqrt();
    eprintln!(
        "{name} {n}x{k}: mxfp8 vs bf16 max|d| {max_abs:.3e} ({:.2e} of max|y| {max_ref:.3e}), \
         {differ}/{} outputs differ; rel-L2 vs f64: mxfp8 {:.2e}, bf16 {:.2e}; \
         rows alone == rows in 1..8/16/32/40-row launches",
        max_abs / max_ref,
        MAX_ROWS * n,
        rel(err2_mx),
        rel(err2_bf),
    );
    // Both are the f64 result rounded once to BF16 plus FP32 summation
    // noise: a few BF16 ULPs at most (rel-L2 of one BF16 rounding is
    // ~2^-9 / sqrt(3) ~ 1.1e-3).
    assert!(rel(err2_mx) < 3e-3, "{name}: MXFP8 rel-L2 {}", rel(err2_mx));
    assert!(
        max_abs <= max_ref * 2f64.powi(-6),
        "{name}: max |mx - bf16| {max_abs}"
    );
    for p in [a, w.data, w.scales, dense.weight, c] {
        gpu.free(p)?;
    }
    Ok(())
}

/// Copies per shape, rotated so a launch never finds its weight in L2.
const COPIES: usize = 6;
const ITERS: usize = 120;

#[test]
#[ignore = "needs a GB10 GPU and a kernel target with mxfp8_gemv"]
fn mxfp8_rows_bench_against_the_bf16_decode_gemvs() -> Result<()> {
    let mut rng = Rng(0x0dd_ba11);
    for (name, n, k) in &SHAPES[..4] {
        let (values, scales, bf16) = weight(&mut rng, *n, *k);
        bench(name, &values, &scales, &bf16, *n, *k)?;
    }
    Ok(())
}

/// MXFP8 against the BF16 decode GEMVs at 1, 4 and 8 rows.
fn bench(name: &str, values: &[u8], scales: &[u8], bf16: &[u16], n: usize, k: usize) -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let tiers = mxfp8_tiers(gpu)?;
    let gemv = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let batchm = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
    let bf16_bytes = bytes(bf16);
    let mut mx = Vec::new();
    let mut dense = Vec::new();
    for _ in 0..COPIES {
        mx.push(Mxfp8Weight {
            data: upload(gpu, values)?,
            scales: upload(gpu, scales)?,
            tiled: false,
        });
        dense.push(crate::weight_map::DenseWeight {
            weight: upload(gpu, &bf16_bytes)?,
        });
    }
    let a = upload(gpu, &vec![0x3fu8; 8 * k * 2])?;
    let c = gpu.alloc(8 * n * 2)?;
    let time = |f: &dyn Fn(usize) -> Result<()>| -> Result<f64> {
        for i in 0..COPIES {
            f(i)?;
        }
        gpu.synchronize(stream)?;
        let t = std::time::Instant::now();
        for i in 0..ITERS {
            f(i % COPIES)?;
        }
        gpu.synchronize(stream)?;
        Ok(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
    };
    for m in [1usize, 4, 8] {
        let t_mx = time(&|i| mxfp8_gemv_rows(gpu, &tiers, &mx[i], a, c, m, n, k, n, stream))?;
        let t_bf = time(&|i| {
            if m == 1 {
                dense_gemv(gpu, gemv, a, &dense[i], c, n as u32, k as u32, stream)
            } else {
                dense_gemv_batchm(
                    gpu, batchm, a, &dense[i], c, m as u32, n as u32, k as u32, n as u32, stream,
                )
            }
        })?;
        let (b_mx, b_bf) = ((n * k + n * k / 32) as f64, (n * k * 2) as f64);
        eprintln!(
            "bench {name} {n}x{k} m={m}: mxfp8 {t_mx:.1} us ({:.0} GB/s), bf16 {t_bf:.1} us \
             ({:.0} GB/s), mxfp8/bf16 time {:.2}",
            b_mx / t_mx / 1e3,
            b_bf / t_bf / 1e3,
            t_mx / t_bf,
        );
    }
    for w in &mx {
        gpu.free(w.data)?;
        gpu.free(w.scales)?;
    }
    for d in &dense {
        gpu.free(d.weight)?;
    }
    gpu.free(a)?;
    gpu.free(c)?;
    Ok(())
}

#[path = "mxfp8_real_test_gpu.rs"]
pub(crate) mod real;
