// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests for the int8 "sq" EXL3 GEMV (`exl3_int8_ops`) on random but valid
//! EXL3 weights (any trellis word decodes; suh/svh are random-sign scales):
//! error against an f64 reference, run-to-run determinism, m = 1 vs m = 2 row
//! invariance at a pinned grid, and the self-resetting workspace counters.
//!
//!   cargo test -p spark-model --release --features cuda --lib exl3_int8_ops::gpu_tests \
//!       -- --ignored --nocapture --test-threads=1

use half::{bf16, f16};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::weight_map::exl3::{Exl3Shape, reconstruct_ref};

/// (in, out, bits): the model's decode classes, at test-friendly sizes.
const CASES: [(usize, usize, u32); 7] = [
    (2560, 2560, 6),
    (2560, 10240, 6),
    (6144, 2560, 6),
    (2560, 512, 6),
    (2560, 2560, 5),
    (2560, 2560, 4),
    (2560, 4096, 6),
];

fn gpu() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    spark_runtime::cuda_backend::AtlasCudaBackend::new(
        0,
        &atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
            .expect("build with the qwen3.8-flash-next nvfp4 kernels")
            .modules,
    )
    .expect("CUDA backend")
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len().max(1)).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn zeros(g: &dyn GpuBackend, bytes: usize) -> DevicePtr {
    upload(g, &vec![0u8; bytes])
}

fn download_u32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u32> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// SplitMix64.
fn mix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [0, 1).
fn unit(seed: &mut u64) -> f32 {
    ((mix64(seed) >> 40) as f32) / (1u64 << 24) as f32
}

/// Host copy of a random EXL3 weight plus its device upload.
struct Case {
    trellis: Vec<u16>,
    suh: Vec<f16>,
    svh: Vec<f16>,
    shape: Exl3Shape,
    dev: Exl3Weight,
}

fn random_case(g: &dyn GpuBackend, k: usize, n: usize, bits: u32) -> Case {
    let mut seed = 0xE31A_0000_0000_u64 ^ ((k * 1_000_003 + n * 7 + bits as usize) as u64);
    let words = (k / 16) * (n / 16) * 16 * bits as usize;
    let trellis: Vec<u16> = (0..words).map(|_| mix64(&mut seed) as u16).collect();
    let mut scale = |len: usize| -> Vec<f16> {
        (0..len)
            .map(|_| {
                let mag = 0.5 + unit(&mut seed);
                let sign = if mix64(&mut seed) & 1 == 0 { 1.0 } else { -1.0 };
                f16::from_f32(sign * mag)
            })
            .collect()
    };
    let suh = scale(k);
    let svh = scale(n);
    let shape = Exl3Shape {
        in_features: k,
        out_features: n,
        bits,
    };
    let bytes16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let bytesf16 = |v: &[f16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let dev = Exl3Weight {
        trellis: upload(g, &bytes16(&trellis)),
        suh: upload(g, &bytesf16(&suh)),
        svh: upload(g, &bytesf16(&svh)),
        shape,
    };
    Case {
        trellis,
        suh,
        svh,
        shape,
        dev,
    }
}

/// `rows` bf16 activation rows of width `k`, as raw bits.
fn activations(k: usize, rows: usize, salt: u64) -> Vec<u16> {
    let mut seed = 0x0AC7_0000_0001_u64 ^ salt ^ (k as u64);
    (0..rows * k)
        .map(|_| bf16::from_f32(unit(&mut seed) * 2.0 - 1.0).to_bits())
        .collect()
}

/// Everything one sq call needs, sized for `m <= 2`.
struct Rig<'a> {
    g: &'a dyn GpuBackend,
    k8: Exl3Int8Kernels,
    k: Exl3Kernels,
    ws: Exl3Int8Workspace,
    grid: u32,
}

impl<'a> Rig<'a> {
    fn new(g: &'a dyn GpuBackend) -> Self {
        let grid = sq_grid(g.sm_count().unwrap());
        let ws_ints = CASES
            .iter()
            .map(|&(k, n, bits)| sq_plan(k, n, 2, bits, grid).unwrap().ws_ints)
            .max()
            .unwrap();
        Self {
            g,
            k8: Exl3Int8Kernels::resolve(g).unwrap(),
            k: Exl3Kernels::resolve(g).unwrap(),
            ws: Exl3Int8Workspace::new(g, ws_ints).unwrap(),
            grid,
        }
    }

    /// Runs the bf16 linear on `x_bits` (`m` rows) and returns the fp32
    /// output `[m, n]` as raw bits (the kernel's own result, before bf16).
    fn run(&self, c: &Case, x_bits: &[u16], m: u32) -> Vec<u32> {
        let (k, n) = (c.shape.in_features, c.shape.out_features);
        let g = self.g;
        let x = upload(
            g,
            &x_bits
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let x_f16 = zeros(g, m as usize * k * 2);
        let a_had = zeros(g, m as usize * k * 2);
        let c_f32 = zeros(g, m as usize * n * 4);
        let out = zeros(g, m as usize * n * 2);
        let stream = g.default_stream();
        exl3_int8_linear_bf16(
            g, &self.k8, &self.k, &self.ws, x, m, &c.dev, x_f16, a_had, c_f32, out, self.grid,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let got = download_u32(g, c_f32, m as usize * n);
        for p in [x, x_f16, a_had, c_f32, out] {
            g.free(p).unwrap();
        }
        got
    }
}

#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn int8_linear_matches_f64_reference() {
    let g = gpu();
    let rig = Rig::new(&g);
    for &(k, n, bits) in &CASES {
        let c = random_case(&g, k, n, bits);
        let w = reconstruct_ref(&c.trellis, &c.suh, &c.svh, &c.shape).unwrap();
        for m in [1u32, 2] {
            let x_bits = activations(k, m as usize, 11);
            let got = rig.run(&c, &x_bits, m);
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for r in 0..m as usize {
                for col in 0..n {
                    let mut acc = 0.0f64;
                    for kk in 0..k {
                        let xv = f32::from(bf16::from_bits(x_bits[r * k + kk])) as f64;
                        acc += xv * w[kk * n + col] as f64;
                    }
                    let v = f32::from_bits(got[r * n + col]) as f64;
                    num += (v - acc) * (v - acc);
                    den += acc * acc;
                }
            }
            let rel = (num / den).sqrt();
            eprintln!("int8 sq k={k} n={n} bits={bits} m={m}: rel err {rel:.3e}");
            assert!(
                rel <= 1.5e-2,
                "k={k} n={n} bits={bits} m={m}: rel err {rel:.3e}"
            );
        }
    }
}

#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn int8_gemv_is_deterministic() {
    let g = gpu();
    let rig = Rig::new(&g);
    for &(k, n, bits) in &CASES {
        let c = random_case(&g, k, n, bits);
        for m in [1u32, 2] {
            let x_bits = activations(k, m as usize, 23);
            assert_eq!(
                rig.run(&c, &x_bits, m),
                rig.run(&c, &x_bits, m),
                "k={k} n={n} bits={bits} m={m}: two runs differ"
            );
        }
    }
}

/// Whether row 0 of an m = 2 call equals the m = 1 call on the same row, at the
/// same grid. MTP verify runs m = 2; plain decode runs m = 1. Equality would make
/// the speculative output match serial decode through these linears.
#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn int8_row_invariance_m1_vs_m2() {
    let g = gpu();
    let rig = Rig::new(&g);
    let mut all_equal = true;
    for &(k, n, bits) in &CASES {
        let c = random_case(&g, k, n, bits);
        let two = activations(k, 2, 37);
        let y2 = rig.run(&c, &two, 2);
        let y1 = rig.run(&c, &two[..k], 1);
        let max_abs = (0..n)
            .map(|i| (f32::from_bits(y2[i]) - f32::from_bits(y1[i])).abs())
            .fold(0.0f32, f32::max);
        let equal = y2[..n] == y1[..];
        all_equal &= equal;
        eprintln!(
            "row invariance k={k} n={n} bits={bits}: bitwise equal {equal}, max |diff| {max_abs:.3e}"
        );
    }
    assert!(
        all_equal,
        "m = 2 row 0 differs from m = 1 on at least one shape (see the lines above)"
    );
}

#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn int8_workspace_counters_reset() {
    let g = gpu();
    let rig = Rig::new(&g);
    for &(k, n, bits) in &CASES {
        let c = random_case(&g, k, n, bits);
        for m in [1u32, 2] {
            rig.run(&c, &activations(k, m as usize, 41), m);
            let counters = download_u32(&g, rig.ws.ptr, 4096);
            assert!(
                counters.iter().all(|&v| v == 0),
                "k={k} n={n} bits={bits} m={m}: workspace counters not reset"
            );
        }
    }
}

/// Microseconds per sq GEMV call and the effective weight bandwidth, per case
/// and m. Informational (no assertion): run it with different
/// `ATLAS_EXL3_SQ_BLOCKS_PER_SM` values to pick the grid.
#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn int8_sq_timing() {
    let g = gpu();
    let rig = Rig::new(&g);
    let stream = g.default_stream();
    eprintln!("sq timing, grid {}", rig.grid);
    for &(k, n, bits) in &CASES {
        let c = random_case(&g, k, n, bits);
        for m in [1u32, 2] {
            let x_f16 = upload(
                &g,
                &activations(k, m as usize, 53)
                    .iter()
                    .flat_map(|&b| f16::from_f32(f32::from(bf16::from_bits(b))).to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
            let a_had = zeros(&g, m as usize * k * 2);
            let c_f32 = zeros(&g, m as usize * n * 4);
            let call = || {
                exl3_int8_gemv(
                    &g, &rig.k8, &rig.ws, x_f16, m, &c.dev, a_had, c_f32, rig.grid, stream,
                )
                .unwrap()
            };
            for _ in 0..20 {
                call();
            }
            g.synchronize(stream).unwrap();
            let iters = 200;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                call();
            }
            g.synchronize(stream).unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let gbs = (k * n) as f64 * bits as f64 / 8.0 / (us * 1e-6) / 1e9;
            eprintln!("sq k={k} n={n} bits={bits} m={m}: {us:.1} us/call, {gbs:.0} GB/s weights");
            for p in [x_f16, a_had, c_f32] {
                g.free(p).unwrap();
            }
        }
    }
}
