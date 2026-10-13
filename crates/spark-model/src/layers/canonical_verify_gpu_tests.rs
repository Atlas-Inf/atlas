// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build, run with `ATLAS_W4A16_TC=1`): one probe row through every
//! verify width 1..8, 16 and 32, at the first, middle and last slot, beside
//! other rows each time. The canonical kernels ([`w4a16`], [`dense`], the
//! touch / strided / pair twins) must give the probe the same bits at every
//! width; the width-tuned kernels they replace are shown to not (the test's
//! sensitivity). `canonical_verify_accuracy` also scores each kernel against
//! an FP64 reference (outputs off the correctly rounded BF16, router top-8
//! sets that differ).
//!
//!   ATLAS_W4A16_TC=1 cargo test --release -p spark-model --lib canonical_verify -- --ignored --nocapture

use super::*;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

pub(super) const WIDTHS: [u32; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32];

pub(super) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

pub(super) fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

pub(super) fn from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// One backend for the whole test process: the tier tables resolve their
/// handles once (`OnceLock`), against the first backend that asks.
pub(super) fn backend() -> Result<&'static AtlasCudaBackend> {
    crate::layers::glm_gpu_test_backend::backend()
}

pub(super) fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub(super) fn rows_bytes(rows: &[Vec<u16>]) -> Vec<u8> {
    rows.iter()
        .flatten()
        .flat_map(|h| h.to_le_bytes())
        .collect()
}

pub(super) fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, elems: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; elems * 2];
    gpu.copy_d2h(ptr, &mut b)?;
    crate::layers::glm_gpu_test_backend::finite_bf16("canonical projection", &b)?;
    Ok(b.chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1(nibble: u8) -> f64 {
    let v = E2M1[(nibble & 7) as usize];
    if nibble & 8 != 0 { -v } else { v }
}

fn e4m3(b: u8) -> f64 {
    let (e, m) = ((b >> 3) & 0xf, (b & 7) as f64);
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e as i32 - 7)
    };
    if b & 0x80 != 0 { -v } else { v }
}

/// A random NVFP4 `[n, k]` weight on the GPU and its FP64 dequantization.
pub(super) struct W4 {
    pub q: QuantizedWeight,
    pub n: u32,
    pub k: u32,
    pub dense: Vec<f64>,
}

pub(super) fn w4(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32, k: u32) -> Result<W4> {
    let (nu, ku) = (n as usize, k as usize);
    let packed: Vec<u8> = (0..nu * ku / 2).map(|_| rng.next() as u8).collect();
    // Positive E4M3 group scales 2^-3 .. 2^1.
    let scales: Vec<u8> = (0..nu * ku / 16)
        .map(|_| 0x20 + rng.below(32) as u8)
        .collect();
    let scale2 = 1.0 / 64.0;
    let mut dense = vec![0f64; nu * ku];
    for r in 0..nu {
        for c in 0..ku {
            let byte = packed[(r * ku + c) / 2];
            let nib = if c % 2 == 0 { byte & 0xf } else { byte >> 4 };
            dense[r * ku + c] = e2m1(nib) * e4m3(scales[(r * ku + c) / 16]) * scale2 as f64;
        }
    }
    let q = QuantizedWeight {
        weight: upload(gpu, &packed)?,
        weight_scale: upload(gpu, &scales)?,
        weight_scale_2: scale2,
        ..QuantizedWeight::null()
    };
    Ok(W4 { q, n, k, dense })
}

/// A random BF16 `[n, k]` weight (values `scale`·U(-1,1)) and its values.
pub(super) fn bf16_weight(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    n: u32,
    k: u32,
    scale: f32,
) -> Result<(DenseWeight, Vec<f64>)> {
    let h: Vec<u16> = (0..n * k).map(|_| bf16(rng.unit() * scale)).collect();
    let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    let dense = h.iter().map(|&v| from_bf16(v) as f64).collect();
    Ok((
        DenseWeight {
            weight: upload(gpu, &bytes)?,
        },
        dense,
    ))
}

pub(super) fn random_row(rng: &mut Rng, k: u32) -> Vec<u16> {
    (0..k).map(|_| bf16(rng.unit())).collect()
}

/// Activations spread over 2^-6 .. 2^6: sums that cancel, so FP32 summation
/// order shows in the BF16 output far more often than with U(-1, 1).
pub(super) fn wide_row(rng: &mut Rng, k: u32) -> Vec<u16> {
    (0..k)
        .map(|_| bf16(rng.unit() * (2f32).powi(rng.below(13) as i32 - 6)))
        .collect()
}

/// Launch the W4A16 kernel `name` (a width-tuned tier) with the geometry
/// its op function uses.
pub(super) fn launch_w4(
    gpu: &dyn GpuBackend,
    name: &str,
    w: &W4,
    a: DevicePtr,
    c: DevicePtr,
    m: u32,
) -> Result<()> {
    let (q, n, k, s) = (&w.q, w.n, w.k, gpu.default_stream());
    let kernel = gpu.kernel("w4a16_gemv", name)?;
    match name {
        "w4a16_gemv" => ops::w4a16_gemv(gpu, kernel, a, q, c, n, k, s),
        "w4a16_gemv_batch2" => ops::w4a16_gemv_batch2(gpu, kernel, a, q, c, n, k, s),
        "w4a16_gemv_batch3" => ops::w4a16_gemv_batch3(gpu, kernel, a, q, c, n, k, s),
        _ => KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n, if name.contains("_tc") { 16 } else { 4 }), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(q.weight)
            .arg_ptr(q.weight_scale)
            .arg_f32(q.weight_scale_2)
            .arg_ptr(c)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(s),
    }
}

/// The probe's output bits at each width and slot: `run(rows)` launches over
/// `rows` (the probe among them) and returns the `[rows.len(), n]` output.
pub(super) fn probe_bits(
    rng: &mut Rng,
    probe: &[u16],
    k: u32,
    widths: &[u32],
    mut run: impl FnMut(&[Vec<u16>]) -> Result<Vec<u16>>,
) -> Result<Vec<(u32, u32, Vec<u16>)>> {
    let mut out = vec![];
    for &m in widths {
        let mut slots = vec![0, m / 2, m - 1];
        slots.dedup();
        for slot in slots {
            let rows: Vec<Vec<u16>> = (0..m)
                .map(|r| {
                    if r == slot {
                        probe.to_vec()
                    } else {
                        random_row(rng, k)
                    }
                })
                .collect();
            let all = run(&rows)?;
            let n = all.len() / m as usize;
            out.push((m, slot, all[slot as usize * n..][..n].to_vec()));
        }
    }
    Ok(out)
}

/// How many (width, slot) results differ from the first, with a report.
pub(super) fn differing(label: &str, results: &[(u32, u32, Vec<u16>)]) -> usize {
    let first = &results[0].2;
    let bad: Vec<_> = results
        .iter()
        .filter(|(_, _, bits)| bits != first)
        .map(|(m, s, bits)| {
            let elems = bits.iter().zip(first).filter(|(a, b)| a != b).count();
            format!("m{m}@{s}:{elems}")
        })
        .collect();
    println!(
        "{} {label}: {} of {} (width, slot) cases differ from m1 {bad:?}",
        if bad.is_empty() {
            "INVARIANT"
        } else {
            "VARIES"
        },
        bad.len(),
        results.len()
    );
    bad.len()
}

/// One upload/launch/download of `rows` through `launch`.
pub(super) fn run_rows(
    gpu: &dyn GpuBackend,
    rows: &[Vec<u16>],
    n: u32,
    launch: impl Fn(DevicePtr, DevicePtr, u32) -> Result<()>,
) -> Result<Vec<u16>> {
    let m = rows.len() as u32;
    let a = upload(gpu, &rows_bytes(rows))?;
    let c = gpu.alloc(m as usize * n as usize * 2)?;
    gpu.memset(c, 0x7f, m as usize * n as usize * 2)?;
    launch(a, c, m)?;
    gpu.synchronize(gpu.default_stream())?;
    let out = download(gpu, c, m as usize * n as usize)?;
    gpu.free(a)?;
    gpu.free(c)?;
    Ok(out)
}

/// The width-tuned W4A16 tier the serial (and E9) verify used for `m` rows.
fn legacy_w4_name(m: u32) -> &'static str {
    match m {
        1 => "w4a16_gemv",
        2 => "w4a16_gemv_batch2",
        3 => "w4a16_gemv_batch3",
        4..=8 => "w4a16_gemv_tc8",
        9..=16 => "w4a16_gemv_tc16",
        _ => "w4a16_gemv_tc32",
    }
}

#[test]
#[ignore = "requires a GB10, a glm-5.3-flash kernel build and ATLAS_W4A16_TC=1"]
fn canonical_w4a16_rows_do_not_depend_on_the_width() -> Result<()> {
    ensure!(enabled(), "run with ATLAS_W4A16_TC=1 and the switch on");
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = gpu;
    W4a16BatchmTiers::resolve(gpu);
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let mut failed = 0;
    let mut legacy_varies = 0;
    // KDA q/k/v/o (4096 x 4096), the shared expert's split gate/up
    // (1024 x 4096) and its down K-slice (4096 x 1024), a dense FFN layer's
    // gate/up and down (6144 x 4096, 4096 x 6144).
    for (n, k) in [
        (4096, 4096),
        (1024, 4096),
        (4096, 1024),
        (6144, 4096),
        (4096, 6144),
    ] {
        let w = w4(gpu, &mut rng, n, k)?;
        let probe = wide_row(&mut rng, k);
        let canon = probe_bits(&mut rng, &probe, k, &WIDTHS, |rows| {
            run_rows(gpu, rows, n, |a, c, m| {
                w4a16(gpu, a, &w.q, c, m, n, k, gpu.default_stream())
            })
        })?;
        failed += differing(&format!("canonical w4a16 {n}x{k}"), &canon);
        let legacy = probe_bits(&mut rng, &probe, k, &WIDTHS, |rows| {
            let name = legacy_w4_name(rows.len() as u32);
            run_rows(gpu, rows, n, |a, c, m| launch_w4(gpu, name, &w, a, c, m))
        })?;
        legacy_varies += differing(&format!("legacy w4a16 {n}x{k}"), &legacy);
        // The twins run the same body: strided (ld = k/2), touch, pair (20
        // rows: a three-tile sweep).
        let rows: Vec<Vec<u16>> = (0..20).map(|_| random_row(&mut rng, k)).collect();
        let plain = run_rows(gpu, &rows, n, |a, c, m| {
            w4a16(gpu, a, &w.q, c, m, n, k, gpu.default_stream())
        })?;
        for twin in [
            "w4a16_gemv_tc8_ld",
            "w4a16_gemv_tc8_touch",
            "w4a16_gemv_tc8_pair_touch",
        ] {
            let kernel = gpu.kernel("w4a16_gemv", twin)?;
            let got = run_rows(gpu, &rows, n, |a, c, m| {
                let mut l = KernelLaunch::new(gpu, kernel)
                    .grid([
                        div_ceil(n, 16),
                        1,
                        if twin.ends_with("pair_touch") { 2 } else { 1 },
                    ])
                    .block([256, 1, 1])
                    .arg_ptr(a);
                if twin.ends_with("pair_touch") {
                    // Both planes the same weight and output: same bits.
                    for _ in 0..2 {
                        l = l
                            .arg_ptr(w.q.weight)
                            .arg_ptr(w.q.weight_scale)
                            .arg_f32(w.q.weight_scale_2)
                            .arg_ptr(c);
                    }
                    l.arg_u32(m)
                        .arg_u32(n)
                        .arg_u32(k)
                        .arg_u32(64)
                        .arg_u32(4)
                        .launch(gpu.default_stream())
                } else {
                    l = l
                        .arg_ptr(w.q.weight)
                        .arg_ptr(w.q.weight_scale)
                        .arg_f32(w.q.weight_scale_2)
                        .arg_ptr(c)
                        .arg_u32(m)
                        .arg_u32(n)
                        .arg_u32(k)
                        .arg_u32(k / 2)
                        .arg_u32(k / 16);
                    if twin.ends_with("touch") {
                        l = l.arg_u32(64).arg_u32(4);
                    }
                    l.launch(gpu.default_stream())
                }
            })?;
            let same = got == plain;
            println!(
                "{} {twin} {n}x{k} 20 rows vs tc8",
                if same { "BITWISE" } else { "DIFFERS" }
            );
            failed += !same as usize;
        }
    }
    println!("legacy cases varying: {legacy_varies}");
    ensure!(
        legacy_varies > 0,
        "the width-tuned kernels showed no variation: test is blind"
    );
    ensure!(failed == 0, "{failed} canonical W4A16 cases differ");
    Ok(())
}

/// Launch a BF16 GEMV kernel by name with its op function's geometry.
pub(super) fn launch_dense(
    gpu: &dyn GpuBackend,
    (module, name): (&str, &str),
    weight: &DenseWeight,
    a: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let kernel = gpu.kernel(module, name)?;
    let s = gpu.default_stream();
    match name {
        "dense_gemm_bf16_router" => ops::dense_gemm_router(gpu, kernel, a, weight, c, m, n, k, s),
        "dense_gemm_bf16_router_rows" => {
            ops::dense_gemm_router_rows(gpu, kernel, a, weight, c, m, n, k, s)
        }
        "dense_gemm_bf16_router_m5" => {
            ops::dense_gemm_router_m5(gpu, kernel, a, weight, c, m, n, k, s)
        }
        "dense_gemv_bf16_batchm" => {
            ops::dense_gemv_batchm(gpu, kernel, a, weight, c, m, n, k, n, s)
        }
        _ => ops::dense_gemv_bf16_tc(gpu, kernel, a, weight, c, m, n, k, n, s),
    }
}

#[path = "canonical_verify_accuracy_tests.rs"]
mod accuracy;
#[path = "canonical_verify_bench_tests.rs"]
mod bench;
#[path = "canonical_verify_chain_tests.rs"]
mod chain;
#[path = "canonical_verify_dense_chain_tests.rs"]
mod dense_chain;
#[path = "canonical_verify_dense_tests.rs"]
mod dense_tests;
