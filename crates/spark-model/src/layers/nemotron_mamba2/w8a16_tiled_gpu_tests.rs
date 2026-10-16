// SPDX-License-Identifier: AGPL-3.0-only

//! E1 block-aligned `w8a16_gemv` (strix-hip shadow) — GPU tests.
//!
//! (a) M=1 vs an f64 CPU reference at the real shapes (in_proj
//!     18560x4096, out_proj 4096x8192, shared expert up/down sizes);
//! (b) `w8a16_gemv_batch4` / `_batch16` rows bitwise == M=1;
//! (c) a microbench: GB/s at those shapes.
//!
//! Run: `cargo test --release -p spark-model w8a16_tiled -- --test-threads 1`

#![cfg(all(test, atlas_hip))]

use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

fn gpu() -> AtlasCudaBackend {
    AtlasCudaBackend::new(
        0,
        &atlas_kernels::ptx_for_exact_target("nemotron-3.5-super-vl-120b-a12b", "nvfp4")
            .expect("n35vl target PTX — build with ATLAS_TARGET_MODEL='*'")
            .modules,
    )
    .expect("CUDA backend")
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len().max(1)).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn download_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u16> {
    let mut v = vec![0u8; n * 2];
    g.copy_d2h(p, &mut v).unwrap();
    v.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

fn bf16_seq(n: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            half::bf16::from_f32(u * scale).to_bits()
        })
        .collect()
}

fn e4m3_to_f32(b: u8) -> f32 {
    // E4M3: sign(1) + exp(4) + mant(3), bias 7; 0x7F/0xFF (NaN) -> 0.
    let s = (b >> 7) & 1;
    let e = (b >> 3) & 0xF;
    let m = b & 0x7;
    let v = if e == 15 && m == 7 {
        0.0f32
    } else if e == 0 {
        (m as f32) * (1.0 / 512.0)
    } else {
        f32::from_bits(((e + 120) as u32) << 23 | ((m as u32) << 20))
    };
    if s == 1 { -v } else { v }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Buf {
    a: DevicePtr,
    b: DevicePtr,
    scl: DevicePtr,
    c: DevicePtr,
}

fn case_bufs(
    g: &dyn GpuBackend,
    m: usize,
    n: usize,
    k: usize,
    seed: u64,
) -> (Buf, Vec<u16>, Vec<u8>, Vec<f32>) {
    let a = bf16_seq(m * k, seed, 0.5);
    let b = lcg_bytes(n * k, seed ^ 0x9e3779b9);
    let k_blocks = (k + 127) / 128;
    let n_blocks = (n + 127) / 128;
    let scl: Vec<f32> = (0..n_blocks * k_blocks)
        .map(|i| 0.25 + ((i * 37 + 11) % 97) as f32 / 100.0)
        .collect();
    (
        Buf {
            a: upload(
                g,
                &a.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            ),
            b: upload(g, &b),
            scl: upload(g, &f32_bytes(&scl)),
            c: g.alloc(m * n * 2).unwrap(),
        },
        a,
        b,
        scl,
    )
}

/// f64 reference for row m, output n.
fn ref_row(a: &[u16], b: &[u8], scl: &[f32], m: usize, n: usize, k: usize) -> f64 {
    let kb = (k + 127) / 128;
    let nb = n / 128;
    let mut acc = 0.0f64;
    for kk in 0..k {
        acc += (half::bf16::from_bits(a[m * k + kk]).to_f32() as f64)
            * (e4m3_to_f32(b[n * k + kk]) as f64)
            * (scl[nb * kb + kk / 128] as f64);
    }
    acc
}

fn check_rows(
    g: &dyn GpuBackend,
    c: DevicePtr,
    a: &[u16],
    b: &[u8],
    scl: &[f32],
    m: usize,
    n: usize,
    k: usize,
    tag: &str,
) {
    let out = download_u16(g, c, m * n);
    let mut max_rel = 0.0f64;
    let mut worst = 0usize;
    for mm in 0..m {
        for nn in 0..n {
            let got = half::bf16::from_bits(out[mm * n + nn]).to_f32() as f64;
            let want = ref_row(a, b, scl, mm, nn, k);
            let rel = if want.abs() < 1e-6 {
                got.abs()
            } else {
                (got - want).abs() / want.abs()
            };
            if rel > max_rel {
                max_rel = rel;
                worst = mm * n + nn;
            }
        }
    }
    eprintln!("{tag}: max_rel={max_rel:.6} worst_flat={worst}");
    assert!(
        max_rel < 0.02,
        "{tag}: max_rel {max_rel} (flat idx {worst})"
    );
}

fn bench(
    g: &dyn GpuBackend,
    kern: spark_runtime::gpu::KernelHandle,
    bufs: &Buf,
    b_copies: &[DevicePtr],
    m: u32,
    n: u32,
    k: u32,
    iters: usize,
    tag: &str,
) {
    // Rotate >=8 weight copies so each launch streams cold DRAM, not L2.
    for _ in 0..3 {
        ops::w8a16_gemv(g, kern, bufs.a, bufs.b, bufs.scl, bufs.c, n, k, 0).unwrap();
    }
    g.synchronize(0).unwrap();
    let t0 = std::time::Instant::now();
    for i in 0..iters {
        let bb = b_copies[i % b_copies.len()];
        ops::w8a16_gemv(g, kern, bufs.a, bb, bufs.scl, bufs.c, n, k, 0).unwrap();
    }
    g.synchronize(0).unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let kb = (k as u64 + 127) / 128;
    let nb = (n as u64 + 127) / 128;
    let bytes = (n as u64 * k as u64) + nb * kb * 4 + (k as u64 * 2) + (n as u64 * 2);
    let gbs = bytes as f64 * iters as f64 / dt / 1e9;
    eprintln!(
        "{tag}: M={m} N={n} K={k}: {gbs:.1} GB/s ({:.0} us/launch)",
        dt * 1e6 / iters as f64
    );
}

fn b_copies(g: &dyn GpuBackend, b: &[u8], n_copies: usize) -> Vec<DevicePtr> {
    (0..n_copies).map(|_| upload(g, b)).collect()
}

#[test]
fn w8a16_gemv_matches_f64_reference_shapes() {
    let g = gpu();
    let k_kern = g
        .kernel("w8a16_gemv", "w8a16_gemv")
        .expect("w8a16_gemv resolves");
    for (n, k, tag) in [
        (18560usize, 4096usize, "in_proj"),
        (4096, 8192, "out_proj"),
        (5376, 4096, "shared_up"),
        (4096, 2688, "shared_down"),
    ] {
        let (bufs, a, b, s) = case_bufs(&g, 1, n, k, 0xC0FFEE);
        ops::w8a16_gemv(
            &g, k_kern, bufs.a, bufs.b, bufs.scl, bufs.c, n as u32, k as u32, 0,
        )
        .unwrap();
        g.synchronize(0).unwrap();
        check_rows(&g, bufs.c, &a, &b, &s, 1, n, k, tag);
    }
}

#[test]
fn w8a16_batch_twins_bitwise_match_m1() {
    let g = gpu();
    let k1 = g.kernel("w8a16_gemv", "w8a16_gemv").unwrap();
    let k4 = g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4").unwrap();
    let k16 = g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16").unwrap();
    let (n, k) = (4096usize, 8192usize);
    let (bufs, a, _b, _s) = case_bufs(&g, 9, n, k, 0xBADDCAFE);

    for m in [4usize, 9] {
        let kk = if m <= 4 { k4 } else { k16 };
        let cb = g.alloc(m * n * 2).unwrap();
        ops::w8a16_gemv_batch4(
            &g, kk, bufs.a, bufs.b, bufs.scl, cb, m as u32, n as u32, k as u32, 0,
        )
        .unwrap();
        g.synchronize(0).unwrap();
        let bout = download_u16(&g, cb, m * n);
        for t in 0..m {
            let c1 = g.alloc(n * 2).unwrap();
            let a_row = upload(
                &g,
                &a[t * k..(t + 1) * k]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            ops::w8a16_gemv(&g, k1, a_row, bufs.b, bufs.scl, c1, n as u32, k as u32, 0).unwrap();
            g.synchronize(0).unwrap();
            let m1 = download_u16(&g, c1, n);
            assert_eq!(
                &m1[..],
                &bout[t * n..(t + 1) * n],
                "m={m} row {t} not bitwise vs M=1"
            );
        }
        eprintln!("batch m={m}: all {m} rows bitwise == M=1");
    }
}

#[test]
fn w8a16_gemv_microbench() {
    let g = gpu();
    let k_kern = g.kernel("w8a16_gemv", "w8a16_gemv").unwrap();
    let k_old = g.kernel("w8a16_gemv_legacy", "w8a16_gemv_legacy").unwrap();
    for (n, k, tag) in [
        (18560usize, 4096usize, "in_proj"),
        (4096, 8192, "out_proj"),
        (5376, 4096, "shared_up"),
        (4096, 2688, "shared_down"),
    ] {
        // >=512 MB of distinct B so the rotation defeats L2 caching.
        let (bufs, _, b, _) = case_bufs(&g, 1, n, k, 0x1234);
        let n_copies = (512 * 1024 * 1024 / (n * k)).max(8);
        let copies = b_copies(&g, &b, n_copies);
        bench(
            &g,
            k_old,
            &bufs,
            &copies,
            1,
            n as u32,
            k as u32,
            40,
            &format!("{tag} old"),
        );
        bench(
            &g,
            k_kern,
            &bufs,
            &copies,
            1,
            n as u32,
            k as u32,
            40,
            &format!("{tag} new"),
        );
    }
}
